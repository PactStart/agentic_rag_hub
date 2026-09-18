"""硅基流动：BGE-M3 嵌入 + bge-reranker 重排（按接口传入 RPM/TPM）。"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import httpx
from openai import OpenAI

from src.llm.rate_limit import RateBudget, with_retry

EMBED_DIM = 1024
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
# 单请求条数：过大易超时/413；过小浪费往返。64 对 BGE-M3 较稳。
_DEFAULT_BATCH = 64
_DEFAULT_WORKERS = 4

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
    batch_size: int | None = None,
    workers: int | None = None,
) -> list[list[float]]:
    """批量嵌入（可多线程并发多个 batch）。is_query=True 时加 BGE 检索前缀。"""
    if not texts:
        return []
    model = os.environ.get("SILICONFLOW_EMBED_MODEL", "BAAI/bge-m3")
    limit = budget or RateBudget(
        f"siliconflow:embed:{model}",
        rpm=_DEFAULT_EMBED.rpm,
        tpm=_DEFAULT_EMBED.tpm,
        retry_max=_DEFAULT_EMBED.retry_max,
    )
    size = max(1, int(batch_size or _DEFAULT_BATCH))
    n_workers = max(1, int(workers or _DEFAULT_WORKERS))
    client = _sf_client()

    spans: list[tuple[int, list[str]]] = []
    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        payload = [(QUERY_PREFIX + t) if is_query else t for t in batch]
        spans.append((start, payload))

    out: list[list[float] | None] = [None] * len(texts)

    def _one(start: int, payload: list[str]) -> tuple[int, list[list[float]]]:
        def _call() -> object:
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
        return start, part

    if n_workers == 1 or len(spans) == 1:
        for start, payload in spans:
            s, part = _one(start, payload)
            for i, vec in enumerate(part):
                out[s + i] = vec
    else:
        with ThreadPoolExecutor(max_workers=min(n_workers, len(spans))) as pool:
            futs = [pool.submit(_one, start, payload) for start, payload in spans]
            for fut in as_completed(futs):
                s, part = fut.result()
                for i, vec in enumerate(part):
                    out[s + i] = vec

    if any(v is None for v in out):
        raise RuntimeError("嵌入结果不完整")
    return out  # type: ignore[return-value]


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
        r.raise_for_status()
        body = r.json()
        tokens = (body.get("meta") or {}).get("tokens") or {}
        limit.record(_meta_cost_tokens(tokens))
        return body["results"]

    return with_retry(_call, max_attempts=limit.retry_max)
