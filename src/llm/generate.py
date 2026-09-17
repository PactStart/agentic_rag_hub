"""DeepSeek（OpenAI 兼容）生成：无资料则短路，不调用 LLM。"""

from __future__ import annotations

import os

from openai import OpenAI

from src.stores.protocol import Hit

SYSTEM = """你是企业内部知识助手。只根据【参考资料】回答。
- 资料里有答案：用自己的话简述，并在句末标出处编号，如 [1]
- 资料不足、互相矛盾、或不包含该问题：回复「信息不足」，不要用常识补全
- 不要编造制度条款、数字、日期
"""


def llm_client() -> OpenAI:
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请写入 .env")
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=key, base_url=base)


def format_context(hits: list[Hit]) -> str:
    lines = []
    for i, hit in enumerate(hits, start=1):
        src = hit.get("source") or "unknown"
        page = hit.get("page")
        loc = f"{src}" + (f" p.{page}" if page is not None else "")
        lines.append(f"[{i}] ({loc})\n{hit['text']}")
    return "\n\n".join(lines)


def generate(query: str, hits: list[Hit], client: OpenAI | None = None) -> dict:
    if not hits:
        return {"answer": "信息不足（无检索结果或无权限）", "hits": []}
    user = f"【参考资料】\n{format_context(hits)}\n\n【用户问题】\n{query}"
    cli = client or llm_client()
    resp = cli.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    usage = {}
    if resp.usage:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    return {
        "answer": resp.choices[0].message.content or "",
        "hits": hits,
        "usage": usage,
    }
