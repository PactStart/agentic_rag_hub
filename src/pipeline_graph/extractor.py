"""三元组：优先读人工 jsonl（省钱）；需要时再用 LLM 抽。"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import project_root


def load_triples(path: Path | None = None) -> list[dict]:
    path = path or project_root() / "data/eval/org_triples.jsonl"
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("acl"):
                src = row.get("source") or ""
                if src == "finance_reimburse.md":
                    row["acl"] = ["finance"]
                elif src == "hr_confidential.md":
                    row["acl"] = ["hr"]
                else:
                    row["acl"] = ["all"]
            rows.append(row)
    return rows


def extract_triples(text: str) -> list[dict]:
    """阶段 8 可选：对任意正文用 LLM 抽三元组。沙盒请用 load_triples。"""
    raise NotImplementedError("沙盒关系请用 data/eval/org_triples.jsonl + scripts/build_graph.py")
