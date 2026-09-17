"""入库：解析沙盒文档，增量双写稀疏索引 + Qdrant。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline_hybrid.ingest import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="企业沙盒增量入库")
    parser.add_argument("--config", default=None, help="默认 config/rag.yaml")
    args = parser.parse_args()
    run_ingest(config_path=args.config)


if __name__ == "__main__":
    main()
