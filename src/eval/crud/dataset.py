"""加载 ``split_merged.json``，按任务产出统一样本。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# CLI 短名 → split 顶层键
TASK_KEYS: dict[str, str] = {
    "summary": "event_summary",
    "continue": "continuing_writing",
    "hallu": "hallu_modified",
    "qa": "questanswer_1doc",
    "qa1": "questanswer_1doc",
    "qa2": "questanswer_2docs",
    "qa3": "questanswer_3docs",
}

# 官方四类 + 可选多文档 QA
ALL_SPLIT_KEYS: tuple[str, ...] = (
    "event_summary",
    "continuing_writing",
    "hallu_modified",
    "questanswer_1doc",
    "questanswer_2docs",
    "questanswer_3docs",
)

# 检索 query 字段 / GT 字段 / prompt 文件
TASK_SPEC: dict[str, dict[str, str]] = {
    "event_summary": {
        "query_field": "event",
        "gt_field": "summary",
        "prompt": "summary.txt",
        "slug": "summary",
    },
    "continuing_writing": {
        "query_field": "beginning",
        "gt_field": "continuing",
        "prompt": "continue_writing.txt",
        "slug": "continue",
    },
    "hallu_modified": {
        "query_field": "newsBeginning",
        "gt_field": "hallucinatedMod",
        "prompt": "hallu_mod.txt",
        "slug": "hallu",
    },
    "questanswer_1doc": {
        "query_field": "questions",
        "gt_field": "answers",
        "prompt": "quest_answer.txt",
        "slug": "qa1",
    },
    "questanswer_2docs": {
        "query_field": "questions",
        "gt_field": "answers",
        "prompt": "quest_answer.txt",
        "slug": "qa2",
    },
    "questanswer_3docs": {
        "query_field": "questions",
        "gt_field": "answers",
        "prompt": "quest_answer.txt",
        "slug": "qa3",
    },
}


def resolve_tasks(task: str) -> list[str]:
    """``all|summary|continue|hallu|qa|qa1|qa2|qa3`` → split keys。"""
    t = (task or "all").strip().lower()
    if t == "all":
        return list(ALL_SPLIT_KEYS)
    if t == "qa":
        return ["questanswer_1doc"]
    if t not in TASK_KEYS:
        raise ValueError(
            f"未知任务 {task!r}；可选: all|summary|continue|hallu|qa|qa1|qa2|qa3"
        )
    return [TASK_KEYS[t]]


def load_split(path: Path | str) -> dict[str, list[dict[str, Any]]]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"评测集不存在: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("split_merged.json 应为顶层 object（按任务键分组）")
    return data


def load_task_samples(
    path: Path | str,
    split_key: str,
    *,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    data = load_split(path)
    if split_key not in data:
        raise KeyError(f"split 中无键 {split_key!r}；已有: {sorted(data)}")
    rows = data[split_key]
    if not isinstance(rows, list):
        raise ValueError(f"{split_key} 应为 list")
    # 统一 ID；跳过官方哨兵失败样本
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        sid = item.get("ID") or item.get("id")
        if not sid:
            continue
        item["ID"] = str(sid)
        if split_key == "hallu_modified" and item.get("hallucinatedMod") == (
            '","msg":"request openai failed"'
        ):
            continue
        out.append(item)
    if offset:
        out = out[offset:]
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out
