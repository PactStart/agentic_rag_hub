"""硅基流动：BGE-M3 嵌入 + bge-reranker 重排（按接口传入 RPM/TPM）。"""

from __future__ import annotations

import os
from typing import Any

import httpx
from openai import OpenAI

from src.llm.rate_limit import RateBudget, with_retry

EMBED_DIM = 1024
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
_BATCH = 32

# 默认配额按硅基控制台常见档位；不同模型/接口请改传参或覆盖常量
_DEFAULT_EMBED = RateBudget("siliconflow:embed:BAAI/bge-m3", rpm=2000, tpm=500_000)
_DEFAULT_RERANK = RateBudget(
    "siliconflow:rerank:BAAI/bge-reranker-v2-m3", rpm=2000, tpm=500_000
)


def _sf_client() -> OpenAI:
    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 SILICONFLOW_API_KEY，请写入 .env")
    base = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    return OpenAI(api_key=key, base_url=base)


def _meta_cost_tokens(meta_tokens: dict[str, Any] | None) -> int:
    """兼容 input_tokens / input_token 两种字段名。"""
    if not meta_tokens:
        return 0
    inp = meta_tokens.get("input_tokens", meta_tokens.get("input_token", 0)) or 0
    out = meta_tokens.get("output_tokens", meta_tokens.get("output_token", 0)) or 0
    return int(inp) + int(out)


def embed_texts(
    texts: list[str],
    is_query: bool = False,
    *,
    budget: RateBudget | None = None,
) -> list[list[float]]:
    """批量嵌入。is_query=True 时加 BGE 检索前缀。"""
    if not texts:
        return []
    model = os.environ.get("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")
    limit = budget or RateBudget(
        f"siliconflow:embed:{model}",
        rpm=_DEFAULT_EMBED.rpm,
        tpm=_DEFAULT_EMBED.tpm,
        retry_max=_DEFAULT_EMBED.retry_max,
    )
    client = _sf_client()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        batch = texts[start : start + _BATCH]
        payload = [(QUERY_PREFIX + t) if is_query else t for t in batch]

        def _call(payload: list[str] = payload) -> object:
            limit.before_call()
            return client.embeddings.create(model=model, input=payload)

        resp = with_retry(_call, max_attempts=limit.retry_max)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            limit.record(int(getattr(usage, "total_tokens", 0) or 0))
        ordered = sorted(resp.data, key=lambda row: row.index)
        part = [list(row.embedding) for row in ordered]
        if part and len(part[0]) != EMBED_DIM:
            raise RuntimeError(f"期望 {EMBED_DIM} 维，实际 {len(part[0])}")
        vectors.extend(part)
    return vectors


def rerank(
    query: str,
    documents: list[str],
    top_n: int,
    *,
    budget: RateBudget | None = None,
) -> list[dict]:
    """返回 [{index, relevance_score}, ...]，已按分数降序。"""
    if not documents:
        return []
    url = os.environ.get("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1").rstrip("/")
    model = os.environ.get("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    key = os.environ.get("SILICONFLOW_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 SILICONFLOW_API_KEY，请写入 .env")
    limit = budget or RateBudget(
        f"siliconflow:rerank:{model}",
        rpm=_DEFAULT_RERANK.rpm,
        tpm=_DEFAULT_RERANK.tpm,
        retry_max=_DEFAULT_RERANK.retry_max,
    )

    def _call() -> list[dict]:
        limit.before_call()
        r = httpx.post(
            f"{url}/rerank",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "query": query,
                "documents": documents,
                "top_n": min(top_n, len(documents)),
                "return_documents": False,
            },
            timeout=60.0,
        )
        # 429 必须在重试包装内 raise，httpx 默认不抛
        r.raise_for_status()
        body = r.json()
        tokens = (body.get("meta") or {}).get("tokens") or {}
        limit.record(_meta_cost_tokens(tokens))
        return body["results"]

    return with_retry(_call, max_attempts=limit.retry_max)
