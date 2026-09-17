"""混合检索：BM25 + 向量 + RRF + 硅基重排。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.pipeline_hybrid.hybrid import hybrid_search

ROLE_MAP = {
    "employee": ["employee"],
    "hr": ["hr", "employee"],
    "finance": ["finance", "employee"],
}


def print_group(title: str, hits: list[dict], score_key: str = "score") -> None:
    print(f"\n=== {title} ===")
    if not hits:
        print("（空）")
        return
    for i, hit in enumerate(hits, start=1):
        preview = hit["text"].replace("\n", " ")[:100]
        print(f"{i}. {score_key}={hit.get(score_key, 0):.4f}  [{hit.get('source')}] {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description="混合检索")
    parser.add_argument("--search", default="试用期几个月")
    parser.add_argument("--role", default="hr", choices=list(ROLE_MAP))
    parser.add_argument("--recall-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()
    load_config()
    roles = ROLE_MAP[args.role]
    print(f"查询：{args.search}  角色：{args.role} {roles}")
    result = hybrid_search(
        args.search,
        roles=roles,
        recall_k=args.recall_k,
        top_k=args.top_k,
        use_rerank=not args.skip_rerank,
    )
    dropped = result.get("dropped") or {}
    extra_bm25 = f"（过滤掉 {dropped['bm25']}）" if dropped.get("bm25") else ""
    extra_vec = f"（过滤掉 {dropped['vector']}）" if dropped.get("vector") else ""
    print_group(f"BM25 召回{extra_bm25}", result["bm25"])
    print_group(f"向量召回{extra_vec}", result["vector"])
    print_group("RRF 融合", result["rrf"])
    title = "精排后" if result.get("reranked") else "最终（RRF，未精排）"
    print_group(title, result["final"])


if __name__ == "__main__":
    main()
