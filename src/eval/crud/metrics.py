"""对齐 CRUD_RAG ``src/metric/common.py``：BLEU（去 BP）/ ROUGE-L / 可选 text2vec。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jieba
from loguru import logger

_bleu = None
_rouge = None


def catch_all_exceptions(func: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("metric {}: {}", func.__name__, repr(exc))
            return None

    return wrapper


def _load_bleu():
    global _bleu
    if _bleu is None:
        import evaluate

        _bleu = evaluate.load("bleu")
    return _bleu


def _load_rouge():
    global _rouge
    if _rouge is None:
        import evaluate

        _rouge = evaluate.load("rouge")
    return _rouge


@catch_all_exceptions
def bleu_score(
    continuation: str,
    reference: str,
    *,
    with_penalty: bool = False,
) -> tuple[float, float, float, float, float]:
    """返回 (bleu_avg, bleu1..4)。默认去掉 brevity penalty（与官方一致）。"""
    tokenize = lambda text: list(jieba.cut(text))  # noqa: E731
    results = _load_bleu().compute(
        predictions=[continuation],
        references=[[reference]],
        tokenizer=tokenize,
    )
    bleu_avg = float(results["bleu"])
    p = results["precisions"]
    bleu1, bleu2, bleu3, bleu4 = float(p[0]), float(p[1]), float(p[2]), float(p[3])
    bp = float(results["brevity_penalty"])
    if with_penalty:
        return bleu_avg, bleu1, bleu2, bleu3, bleu4
    if bp == 0:
        return 0.0, bleu1, bleu2, bleu3, bleu4
    return bleu_avg / bp, bleu1, bleu2, bleu3, bleu4


@catch_all_exceptions
def rougeL_score(continuation: str, reference: str) -> float:
    tokenize = lambda text: list(jieba.cut(text))  # noqa: E731
    results = _load_rouge().compute(
        predictions=[continuation],
        references=[[reference]],
        tokenizer=tokenize,
        rouge_types=["rougeL"],
    )
    return float(results["rougeL"])


@catch_all_exceptions
def bert_score(continuation: str, reference: str) -> float:
    """官方实现是 text2vec 余弦相似，不是 HuggingFace bert-score。"""
    from text2vec import Similarity

    sim = Similarity(model_name_or_path="shibing624/text2vec-base-chinese")
    return float(sim.get_score(continuation, reference))


def score_pair(
    generated: str,
    reference: str,
    *,
    use_bert_score: bool = False,
) -> dict[str, float]:
    bleu = bleu_score(generated, reference)
    if bleu is None:
        bleu_avg = bleu1 = bleu2 = bleu3 = bleu4 = 0.0
    else:
        bleu_avg, bleu1, bleu2, bleu3, bleu4 = bleu
    rouge = rougeL_score(generated, reference) or 0.0
    bert = 0.0
    if use_bert_score:
        bert = bert_score(generated, reference) or 0.0
    return {
        "bleu-avg": float(bleu_avg or 0.0),
        "bleu-1": float(bleu1 or 0.0),
        "bleu-2": float(bleu2 or 0.0),
        "bleu-3": float(bleu3 or 0.0),
        "bleu-4": float(bleu4 or 0.0),
        "rouge-L": float(rouge),
        "bertScore": float(bert),
        "length": float(len(generated)),
    }
