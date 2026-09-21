"""CRUD 评测主循环：检索 → DeepSeek 生成 → 打分 → 落盘。"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from loguru import logger

from src.eval.crud.dataset import TASK_SPEC, load_task_samples, resolve_tasks
from src.eval.crud.quest_eval import QuestEval
from src.eval.crud.tasks import compute_overall, generate_for_sample, score_sample
from src.pipeline_hybrid.hybrid import hybrid_search
from src.stores.factory import build_sparse, build_vector
from src.stores.protocol import Hit

Baseline = Literal["rag", "no_retrieve"]


def format_hits(hits: list[Hit]) -> str:
    if not hits:
        return ""
    parts: list[str] = []
    for h in hits:
        text = (h.get("text") or "").strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def _load_done_ids(preds_path: Path) -> set[str]:
    done: set[str] = set()
    if not preds_path.is_file():
        return done
    with preds_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("valid") and row.get("id"):
                done.add(str(row["id"]))
    return done


def run_one_task(
    split_key: str,
    *,
    cfg: dict[str, Any],
    split_json: Path,
    out_dir: Path,
    limit: int | None,
    offset: int,
    baseline: Baseline,
    use_bert_score: bool,
    use_quest_eval: bool,
    sleep_s: float,
    sparse=None,
    vector=None,
) -> dict[str, Any]:
    ce = cfg.get("crud_eval") or {}
    recall_k = int(ce.get("recall_k") or cfg.get("retrieve", {}).get("recall_k") or 16)
    top_k = int(
        ce.get("top_k")
        or ce.get("retrieve_top_k")
        or cfg.get("retrieve", {}).get("top_k")
        or 8
    )
    use_rerank = bool(ce.get("use_rerank", True))
    temperature = float(ce.get("temperature", 0.1))
    roles = list(ce.get("roles") or ["all"])

    slug = TASK_SPEC[split_key]["slug"]
    task_dir = out_dir / slug
    task_dir.mkdir(parents=True, exist_ok=True)
    preds_path = task_dir / "preds.jsonl"
    report_path = task_dir / "report.json"

    samples = load_task_samples(split_json, split_key, limit=limit, offset=offset)
    done = _load_done_ids(preds_path)
    logger.info(
        "task={} samples={} skip_done={} baseline={} top_k={} recall_k={} rerank={}",
        split_key,
        len(samples),
        len(done),
        baseline,
        top_k,
        recall_k,
        use_rerank,
    )

    quest = (
        QuestEval(task_name=split_key, temperature=temperature)
        if use_quest_eval
        else None
    )

    results: list[dict[str, Any]] = []
    # 已写入的有效结果也计入 overall（resume）
    if preds_path.is_file():
        with preds_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("valid") and "metrics" in row:
                    results.append(row)

    query_field = TASK_SPEC[split_key]["query_field"]
    n_new = 0
    with preds_path.open("a", encoding="utf-8") as fout:
        for i, sample in enumerate(samples, start=1):
            sid = sample["ID"]
            if sid in done:
                continue
            t0 = time.time()
            query = str(sample.get(query_field) or "")
            if baseline == "no_retrieve":
                docs = ""
                hit_n = 0
            else:
                pack = hybrid_search(
                    query,
                    roles=roles,
                    recall_k=recall_k,
                    top_k=top_k,
                    use_rerank=use_rerank,
                    sparse=sparse,
                    vector=vector,
                    cfg=cfg,
                )
                hits = pack.get("final") or []
                docs = format_hits(hits)
                hit_n = len(hits)

            generated = generate_for_sample(
                split_key, sample, docs, temperature=temperature
            )
            scored = score_sample(
                split_key,
                sample,
                generated,
                use_bert_score=use_bert_score,
                quest_eval=quest,
            )
            row = {
                "id": sid,
                "task": split_key,
                "baseline": baseline,
                "query": query,
                "retrieve_hit_count": hit_n,
                "retrieve_context": docs,
                "metrics": scored["metrics"],
                "log": scored["log"],
                "valid": scored["valid"],
                "elapsed_s": round(time.time() - t0, 3),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(row)
            n_new += 1
            logger.info(
                "[{}/{}] {} bleu-avg={:.4f} rouge-L={:.4f} hits={} {:.1f}s",
                i,
                len(samples),
                sid[:12],
                row["metrics"]["bleu-avg"],
                row["metrics"]["rouge-L"],
                hit_n,
                row["elapsed_s"],
            )
            if sleep_s > 0:
                time.sleep(sleep_s)

    if quest is not None:
        quest.save()

    overall = compute_overall(
        results, use_quest_eval=use_quest_eval, use_bert_score=use_bert_score
    )
    snapshot = {
        "task": split_key,
        "slug": slug,
        "baseline": baseline,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "recall_k": recall_k,
            "top_k": top_k,
            "use_rerank": use_rerank,
            "temperature": temperature,
            "roles": roles,
            "use_bert_score": use_bert_score,
            "use_quest_eval": use_quest_eval,
            "split_json": str(split_json),
            "limit": limit,
            "offset": offset,
            "sparse": (cfg.get("sparse") or {}).get("backend"),
            "vector": (cfg.get("vector") or {}).get("backend"),
            "chunk": (cfg.get("chunking") or {}).get("fixed_size"),
        },
        "overall": overall,
        "new_samples": n_new,
        "total_scored": len(results),
    }
    report_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("wrote {} overall={}", report_path, overall)
    return snapshot


def run_crud_eval(
    *,
    cfg: dict[str, Any],
    task: str = "all",
    limit: int | None = 50,
    offset: int = 0,
    baseline: Baseline = "rag",
    use_bert_score: bool = False,
    use_quest_eval: bool = False,
    sleep_s: float = 0.0,
    out_root: Path | str | None = None,
) -> dict[str, Any]:
    ce = cfg.get("crud_eval") or {}
    split_json = Path(ce.get("split_json") or "data/corpus/crud_split/split_merged.json")
    out_root = Path(out_root or "results/crud")
    out_root.mkdir(parents=True, exist_ok=True)

    keys = resolve_tasks(task)
    sparse = vector = None
    if baseline == "rag":
        sparse = build_sparse(cfg)
        vector = build_vector(cfg)

    reports: dict[str, Any] = {}
    for key in keys:
        reports[key] = run_one_task(
            key,
            cfg=cfg,
            split_json=split_json,
            out_dir=out_root,
            limit=limit,
            offset=offset,
            baseline=baseline,
            use_bert_score=use_bert_score,
            use_quest_eval=use_quest_eval,
            sleep_s=sleep_s,
            sparse=sparse,
            vector=vector,
        )

    summary_path = out_root / "summary.md"
    summary_path.write_text(_markdown_summary(reports, cfg), encoding="utf-8")
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tasks": reports,
        "summary_md": str(summary_path),
    }
    (out_root / "summary.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("summary → {}", summary_path)
    return bundle


def _markdown_summary(reports: dict[str, Any], cfg: dict[str, Any]) -> str:
    lines = [
        "# CRUD-RAG 评测汇总",
        "",
        f"- 生成时间：{datetime.now(timezone.utc).isoformat()}",
        f"- 稀疏：{(cfg.get('sparse') or {}).get('backend')}",
        f"- 向量：{(cfg.get('vector') or {}).get('backend')}",
        f"- 切片：{(cfg.get('chunking') or {}).get('fixed_size')}",
        "",
        "| 任务 | n | bleu-avg | rouge-L | bertScore | QA_F1 | QA_recall | baseline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for key, rep in reports.items():
        o = rep.get("overall") or {}
        lines.append(
            "| {task} | {n} | {bleu:.4f} | {rouge:.4f} | {bert} | {f1} | {rec} | {base} |".format(
                task=key,
                n=o.get("num", 0),
                bleu=float(o.get("avg. bleu-avg") or 0),
                rouge=float(o.get("avg. rouge-L") or 0),
                bert=_fmt_opt(o.get("bertScore")),
                f1=_fmt_opt(o.get("QA_avg_F1")),
                rec=_fmt_opt(o.get("QA_recall")),
                base=rep.get("baseline"),
            )
        )
    lines.extend(
        [
            "",
            "说明：分数勿与论文表直接横比（嵌入/chunk/重排与官方不完全一致）。",
            "Badcase：从各任务 `preds.jsonl` 按 `rouge-L` 升序抽样人工看。",
            "",
        ]
    )
    return "\n".join(lines)


def _fmt_opt(v: Any) -> str:
    if v is None:
        return "-"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "-"
