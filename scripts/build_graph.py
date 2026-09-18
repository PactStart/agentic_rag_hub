"""把 data/eval/org_triples.jsonl 写入 Neo4j。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.config import load_config
from src.logging_config import setup_logging
from src.pipeline_graph.extractor import load_triples
from src.pipeline_graph.graph_store import GraphStore


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="导入组织关系三元组")
    parser.add_argument("--triples", type=Path, default=ROOT / "data/eval/org_triples.jsonl")
    args = parser.parse_args()
    load_config()
    rows = load_triples(args.triples)
    store = GraphStore()
    try:
        n = store.write(rows)
    finally:
        store.close()
    logger.info("已写入 {} 条三元组 → Neo4j {}", n, args.triples)


if __name__ == "__main__":
    main()
