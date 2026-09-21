#!/usr/bin/env python3
"""CRUD-RAG 四任务评测 CLI（DeepSeek 生成；对齐官方 BLEU/ROUGE[/bert][/QuestEval]）。

示例::

  # 每任务 3 条冒烟
  uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task all --limit 3

  # 仅 QA，50 条；对照无检索基线
  uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task qa --limit 50
  uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task qa --limit 50 \\
      --baseline no_retrieve --out results/crud_noretrieve

  # 可选重指标（贵/慢）
  uv add --group eval evaluate rouge-score   # 若尚未安装
  uv run python scripts/eval_crud.py --config config/rag.crud.yaml --task summary --limit 20 \\
      --bert-score --quest-eval
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.config import load_config
from src.eval.crud.runner import run_crud_eval
from src.logging_config import setup_logging


def main() -> None:
    setup_logging()
    p = argparse.ArgumentParser(description="CRUD-RAG 四任务评测（本仓库检索 + DeepSeek）")
    p.add_argument("--config", default="config/rag.crud.yaml")
    p.add_argument(
        "--task",
        default="qa",
        help="all|summary|continue|hallu|qa|qa1|qa2|qa3（默认 qa=1doc）",
    )
    p.add_argument("--limit", type=int, default=50, help="每任务条数；0=全量")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument(
        "--baseline",
        choices=("rag", "no_retrieve"),
        default="rag",
        help="rag=混合检索+生成；no_retrieve=空上下文（消融对照）",
    )
    p.add_argument("--out", type=Path, default=Path("results/crud"))
    p.add_argument("--bert-score", action="store_true", help="启用 text2vec 相似度")
    p.add_argument("--quest-eval", action="store_true", help="启用 RAGQuestEval（额外 DeepSeek 调用）")
    p.add_argument("--sleep", type=float, default=0.0, help="每条样本后休眠秒数（限流）")
    p.add_argument("--no-rerank", action="store_true", help="覆盖配置，关闭精排")
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.no_rerank:
        cfg.setdefault("crud_eval", {})["use_rerank"] = False

    limit = None if args.limit == 0 else args.limit
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise SystemExit("未设置 DEEPSEEK_API_KEY")

    bundle = run_crud_eval(
        cfg=cfg,
        task=args.task,
        limit=limit,
        offset=args.offset,
        baseline=args.baseline,  # type: ignore[arg-type]
        use_bert_score=args.bert_score,
        use_quest_eval=args.quest_eval,
        sleep_s=args.sleep,
        out_root=args.out,
    )
    print(f"summary: {bundle.get('summary_md')}")


if __name__ == "__main__":
    main()
