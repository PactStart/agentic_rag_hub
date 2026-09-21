"""按任务拼 prompt、打分、汇总（对齐官方 scoring / compute_overall）。"""

from __future__ import annotations

import datetime
from typing import Any

from src.eval.crud.dataset import TASK_SPEC
from src.eval.crud.llm import deepseek_chat, extract_response_tag
from src.eval.crud.metrics import score_pair
from src.eval.crud.prompts_io import read_prompt
from src.eval.crud.quest_eval import QuestEval


def build_user_prompt(split_key: str, sample: dict[str, Any], search_documents: str) -> str:
    spec = TASK_SPEC[split_key]
    tpl = read_prompt(spec["prompt"])
    if split_key == "event_summary":
        return tpl.format(event=sample["event"], search_documents=search_documents)
    if split_key == "continuing_writing":
        return tpl.format(
            beginning_text=sample["beginning"],
            search_documents=search_documents,
        )
    if split_key == "hallu_modified":
        return tpl.format(
            begin=sample["newsBeginning"],
            hallu_continue=sample["hallucinatedContinuation"],
            search_documents=search_documents,
        )
    # QA
    return tpl.format(question=sample["questions"], search_documents=search_documents)


def generate_for_sample(
    split_key: str,
    sample: dict[str, Any],
    search_documents: str,
    *,
    temperature: float = 0.1,
) -> str:
    if split_key == "hallu_modified" and sample.get("hallucinatedMod") == (
        '","msg":"request openai failed"'
    ):
        return '","msg":"request openai failed"'
    prompt = build_user_prompt(split_key, sample, search_documents)
    raw = deepseek_chat(prompt, temperature=temperature)
    return extract_response_tag(raw)


def score_sample(
    split_key: str,
    sample: dict[str, Any],
    generated_text: str,
    *,
    use_bert_score: bool = False,
    quest_eval: QuestEval | None = None,
) -> dict[str, Any]:
    gt_field = TASK_SPEC[split_key]["gt_field"]
    ground_truth_text = sample[gt_field]
    sample["ground_truth_text"] = ground_truth_text
    sample["generated_text"] = generated_text

    if quest_eval is not None:
        qa_f1, qa_recall, quest_save = quest_eval.quest_eval(sample)
    else:
        qa_f1, qa_recall, quest_save = 0.0, 0.0, {}

    base = score_pair(
        generated_text,
        ground_truth_text,
        use_bert_score=use_bert_score,
    )
    metrics = {
        **{k: v for k, v in base.items() if k != "length"},
        "QA_avg_F1": float(qa_f1),
        "QA_recall": float(qa_recall),
        "length": int(base["length"]),
    }
    return {
        "metrics": metrics,
        "log": {
            "generated_text": generated_text,
            "ground_truth_text": ground_truth_text,
            "quest_eval_save": quest_save,
            "evaluateDatetime": str(datetime.datetime.now()),
        },
        "valid": len(generated_text.strip()) != 0,
    }


def compute_overall(
    results: list[dict[str, Any]],
    *,
    use_quest_eval: bool = False,
    use_bert_score: bool = False,
) -> dict[str, Any]:
    if not results:
        return {"num": 0}
    keys = [
        "bleu-avg",
        "bleu-1",
        "bleu-2",
        "bleu-3",
        "bleu-4",
        "rouge-L",
        "bertScore",
        "QA_avg_F1",
        "QA_recall",
        "length",
    ]
    overall = {k: 0.0 for k in keys}
    valid_qa = 0
    for r in results:
        m = r["metrics"]
        for k in keys:
            overall[k] += float(m.get(k) or 0.0)
        qsave = (r.get("log") or {}).get("quest_eval_save") or {}
        if use_quest_eval and qsave.get("questions_gt"):
            valid_qa += 1
    n = len(results)
    out: dict[str, Any] = {
        f"avg. {k}": overall[k] / n
        for k in keys
        if k not in ("QA_avg_F1", "QA_recall", "bertScore")
    }
    if use_bert_score:
        out["bertScore"] = overall["bertScore"] / n
    if use_quest_eval:
        denom = valid_qa or 1
        out["QA_avg_F1"] = overall["QA_avg_F1"] / denom
        out["QA_recall"] = overall["QA_recall"] / denom
    out["num"] = n
    return out
