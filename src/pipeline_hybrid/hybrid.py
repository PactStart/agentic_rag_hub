"""BM25 ∥ 向量 → RRF → 硅基精排。不直接依赖 Qdrant / jsonl。"""

from __future__ import annotations

from src.config import load_config
from src.llm.siliconflow import embed_texts
from src.pipeline_hybrid.rerank import rerank_hits
from src.pipeline_hybrid.score_filter import apply_score_filter
from src.stores.factory import build_sparse, build_vector
from src.stores.protocol import Hit, SparseIndex, VectorIndex


def rrf_fuse(bm25_hits: list[Hit], vector_hits: list[Hit], k: int = 60) -> list[Hit]:
    scores: dict[str, float] = {}
    payloads: dict[str, Hit] = {}

    def key(hit: Hit) -> str:
        return hit.get("chunk_id") or hit.get("text") or ""

    def acc(hits: list[Hit]) -> None:
        for rank, hit in enumerate(hits, start=1):
            kid = key(hit)
            if not kid:
                continue
            scores[kid] = scores.get(kid, 0.0) + 1.0 / (k + rank)
            payloads[kid] = hit

    acc(bm25_hits)
    acc(vector_hits)
    fused: list[Hit] = []
    for kid, s in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        item = dict(payloads[kid])
        item["score"] = s
        fused.append(item)  # type: ignore[arg-type]
    return fused


def hybrid_search(
    query: str,
    roles: list[str],
    recall_k: int | None = None,
    top_k: int | None = None,
    use_rerank: bool = True,
    sparse: SparseIndex | None = None,
    vector: VectorIndex | None = None,
    cfg: dict | None = None,
) -> dict:
    cfg = cfg or load_config()
    retrieve = cfg.get("retrieve") or {}
    recall_k = int(recall_k if recall_k is not None else retrieve.get("recall_k") or 10)
    top_k = int(top_k if top_k is not None else retrieve.get("top_k") or 5)
    rrf_k = int(retrieve.get("rrf_k") or 60)
    sparse = sparse or build_sparse(cfg)
    vector = vector or build_vector(cfg)

    bm25_raw = sparse.search(query, roles, top_k=recall_k)
    bm25_hits = apply_score_filter(bm25_raw, retrieve.get("sparse_score"))
    query_vec = embed_texts([query], is_query=True)[0]
    vector_raw = vector.search(query_vec, roles, top_k=recall_k)
    vector_hits = apply_score_filter(vector_raw, retrieve.get("vector_score"))
    fused = rrf_fuse(bm25_hits, vector_hits, k=rrf_k)

    reranked = False
    if use_rerank:
        try:
            ranked = rerank_hits(query, fused, top_k=len(fused) or top_k)
            ranked = apply_score_filter(ranked, retrieve.get("rerank_score"))
            final = ranked[:top_k]
            reranked = True
        except Exception as exc:
            print(
                f"精排不可用（{exc.__class__.__name__}），本次只用 RRF。"
                "可加 --skip-rerank。"
            )
            final = fused[:top_k]
    else:
        final = fused[:top_k]
    return {
        "bm25": bm25_hits[:top_k],
        "vector": vector_hits[:top_k],
        "rrf": fused[:top_k],
        "final": final,
        "reranked": reranked,
        "dropped": {
            "bm25": len(bm25_raw) - len(bm25_hits),
            "vector": len(vector_raw) - len(vector_hits),
        },
    }
