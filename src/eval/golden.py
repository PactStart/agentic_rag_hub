"""人写黄金集的规则评测。"""

from __future__ import annotations

import json
from pathlib import Path

from src.agent.graph import invoke_rag
from src.config import project_root


def load_golden(path: Path | None = None) -> list[dict]:
    path = path or project_root() / "data/eval/golden.jsonl"
    items = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def grade(item: dict, answer: str, hits: list[dict]) -> dict:
    sources = {h.get("source") for h in hits if h.get("source")}
    expect = item["expect"]
    refused = "信息不足" in (answer or "")
    allowed = set(item.get("allowed_sources") or [])
    leaked = bool(allowed and (sources - allowed))
    ok = True
    reason: list[str] = []
    if expect in ("refuse", "forbidden"):
        if not refused:
            ok, reason = False, ["应拒答却答了"]
        if expect == "forbidden" and leaked:
            ok, reason = False, ["来源越权"]
    elif expect == "answer":
        if refused:
            ok, reason = False, ["应回答却拒答"]
        for word in item.get("must_include") or []:
            if word not in (answer or ""):
                ok = False
                reason.append(f"缺关键词:{word}")
        if allowed and sources and not sources.issubset(allowed):
            ok = False
            reason.append("引用了不允许的文件")
    return {"id": item["id"], "ok": ok, "reason": reason, "sources": sorted(sources)}


def run_golden(path: Path | None = None) -> dict:
    items = load_golden(path)
    rows = []
    for item in items:
        out = invoke_rag(item["query"], roles=list(item.get("roles") or ["employee"]))
        rows.append(grade(item, out.get("answer") or "", list(out.get("hits") or [])))
    passed = sum(1 for r in rows if r["ok"])
    report = {"total": len(rows), "passed": passed, "failed": [r for r in rows if not r["ok"]]}
    out_path = project_root() / "results/golden_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    bad = project_root() / "results/badcases.md"
    lines = ["# Bad cases\n"]
    for row in report["failed"]:
        lines.append(f"- `{row['id']}` {row['reason']} sources={row['sources']}")
    if not report["failed"]:
        lines.append("（本轮无失败）")
    bad.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"黄金集 {passed}/{len(rows)} 通过，报告：{out_path}")
    return report
