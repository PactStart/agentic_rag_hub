"""入库：按配置增量双写稀疏索引 + 向量库 + 账本。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loguru import logger

from src.logging_config import setup_logging
from src.pipeline_hybrid.ingest import run_ingest


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="增量入库（沙盒或 CRUD 评测轨）")
    parser.add_argument(
        "--config",
        default=None,
        help="默认 config/rag.yaml；CRUD 用 config/rag.crud.yaml",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="只处理前 N 篇文档（line_corpus 下一行一篇；不会删除账本中其它文档）",
    )
    args = parser.parse_args()
    logger.info("开始入库 config={} limit={}", args.config or "(default)", args.limit)
    summary = run_ingest(config_path=args.config, limit=args.limit)
    logger.info(
        "结束：added={} updated={} skipped={} failed={} embedded={}",
        len(summary.get("added") or []),
        len(summary.get("updated") or []),
        len(summary.get("skipped") or []),
        len(summary.get("failed") or []),
        summary.get("embedded"),
    )


if __name__ == "__main__":
    main()
