"""对齐 CRUD_RAG RAGQuestEval；评测 LLM 用 DeepSeek（OpenAI 兼容）。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import jieba
import numpy as np
from loguru import logger

from src.eval.crud.llm import deepseek_chat, extract_response_tag
from src.eval.crud.prompts_io import prompt_dir, read_prompt

_JSON_FEWSHOT = """
{"key_info": ["新增并网光伏发电容量1060万千瓦", "四分之一", "全国新增光伏电站855万千瓦", "分布式光伏容量205万千瓦", "2014年中国光伏发电量250亿千瓦。", "同比增长超过200%"],

"question": ["2014年中国新增并网光伏发电容量是多少？", "2014年中国新增并网光伏发电容量约占全球新增容量的几分之几？","全国新增光伏电站的容量是多少？", "分布式光伏容量是多少？", "2014年中国光伏发电量是多少？", "2014年中国光伏发电量相比前一年增长了多少？"]}
""".strip()


def compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = list(jieba.cut(a_gold))
    pred_toks = list(jieba.cut(a_pred))
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def word_based_f1_score(a_gold_list: list[str], a_pred_list: list[str]) -> float:
    if not a_gold_list:
        return 0.0
    return float(
        np.mean([compute_f1(g, p) for g, p in zip(a_gold_list, a_pred_list, strict=False)])
    )


class QuestEval:
    def __init__(
        self,
        *,
        task_name: str,
        temperature: float = 0.1,
        max_tokens: int = 1280,
        cache_dir: Path | None = None,
    ) -> None:
        self.task_name = task_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_dir = Path(cache_dir or (prompt_dir().parent / "quest_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / f"{task_name}_quest_gt_save.json"
        self.quest_gt_save: dict[str, Any] = {}
        if self.cache_path.is_file():
            try:
                self.quest_gt_save = json.loads(
                    self.cache_path.read_text(encoding="utf-8")
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("QuestEval cache load failed: {}", exc)

    def save(self) -> None:
        self.cache_path.write_text(
            json.dumps(self.quest_gt_save, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def question_generation(self, text4gen: str) -> dict[str, Any]:
        prompt = read_prompt("quest_eval_gen.txt").format(
            json_response=_JSON_FEWSHOT,
            news=text4gen,
        )
        raw = deepseek_chat(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        # 允许模型包 markdown code fence
        text = extract_response_tag(raw) or raw
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    def question_answer(self, context: str, questions: list[str]) -> list[str]:
        template = read_prompt("quest_eval_answer.txt")
        query = template.format(context=context, questions=questions)
        answers = deepseek_chat(query, temperature=self.temperature, max_tokens=self.max_tokens)
        pattern = r"<response>\n(.*?)\n</response>"
        found = re.findall(pattern, answers, re.DOTALL)
        if found:
            return [a.strip() for a in found]
        # DeepSeek 偶发不换行：放宽
        return [a.strip() for a in re.findall(r"<response>(.*?)</response>", answers, re.DOTALL)]

    def get_qa_pair(self, data_point: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
        gt = data_point["ground_truth_text"]
        gen = data_point["generated_text"]
        sid = data_point["ID"]
        if sid in self.quest_gt_save:
            questions_gt = self.quest_gt_save[sid]["question"]
            answers_gt4gt = self.quest_gt_save[sid]["answers"]
        else:
            keyinfo = self.question_generation(gt)
            questions_gt = keyinfo["question"]
            answers_gt4gt = self.question_answer(gt, questions_gt)
            keyinfo["answers"] = answers_gt4gt
            self.quest_gt_save[sid] = keyinfo
        answers_gm4gt = self.question_answer(gen, questions_gt)
        return questions_gt, answers_gt4gt, answers_gm4gt

    def quest_eval(self, data_point: dict[str, Any]) -> tuple[float, float, dict[str, Any]]:
        try:
            questions_gt, answers_gt4gt, answers_gm4gt = self.get_qa_pair(data_point)
            save = {
                "questions_gt": questions_gt,
                "answers_gt4gt": answers_gt4gt,
                "answers_gm4gt": answers_gm4gt,
            }
            indices = [i for i, x in enumerate(answers_gt4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices if i < len(answers_gm4gt)]
            answers_gt4gt = [answers_gt4gt[i] for i in indices]
            if not answers_gm4gt:
                return 0.0, 0.0, save
            undetermined_ratio = answers_gm4gt.count("无法推断") / len(answers_gm4gt)
            quest_recall = 1 - undetermined_ratio
            indices2 = [i for i, x in enumerate(answers_gm4gt) if x != "无法推断"]
            answers_gm4gt = [answers_gm4gt[i] for i in indices2]
            answers_gt4gt = [answers_gt4gt[i] for i in indices2]
            if not answers_gm4gt:
                return 0.0, float(quest_recall), save
            return float(word_based_f1_score(answers_gt4gt, answers_gm4gt)), float(quest_recall), save
        except Exception as exc:  # noqa: BLE001
            logger.warning("QuestEval failed: {}", repr(exc))
            return 0.0, 0.0, {"questions_gt": [], "answers_gt4gt": [], "answers_gm4gt": []}
