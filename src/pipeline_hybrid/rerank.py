"""对融合候选调用硅基 bge-reranker。"""

from __future__ import annotations

from src.llm.siliconflow import rerank as sf_rerank
from src.stores.protocol import Hit


def rerank_hits(query: str, candidates: list[Hit], top_k: int = 5) -> list[Hit]:
    if not candidates:
        return []
    docs = [c["text"] for c in candidates]
    results = sf_rerank(query, docs, top_n=min(top_k, len(docs)))
    out: list[Hit] = []
    for row in results:
        item = dict(candidates[int(row["index"])])
        item["score"] = float(row["relevance_score"])
        out.append(item)  # type: ignore[arg-type]
    return out
