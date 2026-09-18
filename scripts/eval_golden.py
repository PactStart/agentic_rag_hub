"""跑人写黄金集。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.eval.golden import run_golden
from src.logging_config import setup_logging


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, default=ROOT / "data/eval/golden.jsonl")
    args = parser.parse_args()
    load_config()
    run_golden(args.eval)


if __name__ == "__main__":
    main()
