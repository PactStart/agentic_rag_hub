"""解析 → 切片 → 增量双写 SparseIndex + VectorIndex；账本走 ledger（默认 MySQL）。

加速要点：
- 分波（wave）处理，控制内存
- 多线程解析/切片
- 多线程批量 embed
- ES / Milvus 按波批量删 + 批量写（波末再 refresh/flush）
- ``--limit`` = 文档篇数
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from loguru import logger

from src.config import load_config
from src.ingest.factory import build_router, build_source
from src.ingest.source import IngestDoc
from src.ledger.factory import build_ledger
from src.llm.siliconflow import embed_texts
from src.parsers.chunking.factory import build_chunker
from src.parsers.protocol import ParsedDoc
from src.stores.factory import build_sparse, build_vector
from src.stores.protocol import Chunk


def _parsed_from_text(doc: IngestDoc, *, default_acl: list[str]) -> ParsedDoc:
    text = (doc.text or "").strip()
    body = text + ("\n" if text else "")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    return {
        "source": doc.source,
        "markdown": body,
        "path": str(doc.path) if doc.path else "",
        "content_hash": digest,
        "acl": list(default_acl),
        "page": None,
        "doc_type": "line_corpus",
        "title": doc.title or (text[:120] if text else None),
    }


def _batch_cfg(cfg: dict) -> dict[str, int]:
    raw = ((cfg.get("ingest") or {}).get("batch")) or {}
    return {
        "wave_size": int(raw.get("wave_size") or 500),
        "parse_workers": int(raw.get("parse_workers") or 8),
        "embed_batch": int(raw.get("embed_batch") or 64),
        "embed_workers": int(raw.get("embed_workers") or 4),
        "milvus_bulk": int(raw.get("milvus_bulk") or 1000),
    }


def _delete_sources(sparse: Any, vector: Any, sources: list[str]) -> None:
    if not sources:
        return
    if hasattr(sparse, "delete_by_sources"):
        try:
            sparse.delete_by_sources(sources, refresh=False)
        except TypeError:
            sparse.delete_by_sources(sources)
    else:
        for s in sources:
            sparse.delete_by_source(s)
    if hasattr(vector, "delete_by_sources"):
        try:
            vector.delete_by_sources(sources, flush=False)
        except TypeError:
            vector.delete_by_sources(sources)
    else:
        for s in sources:
            vector.delete_by_source(s)


def _finish_write(sparse: Any, vector: Any, *, flush_vector: bool = False) -> None:
    if hasattr(sparse, "refresh_index"):
        sparse.refresh_index()
    # 每波 flush 会触发 compaction；DataNode 一重启就会 node id 对不上。只在结束时 flush。
    if flush_vector and hasattr(vector, "flush"):
        vector.flush()


def run_ingest(
    config_path: str | None = None,
    *,
    limit: int | None = None,
    after: str | None = None,
) -> dict:
    cfg = load_config(config_path)
    source = build_source(cfg)
    router = build_router(cfg)
    sparse = build_sparse(cfg)
    vector = build_vector(cfg)
    ledger = build_ledger(cfg)
    sparse_name = ((cfg.get("sparse") or {}).get("backend")) or "memory_jsonl"
    vector_name = ((cfg.get("vector") or {}).get("backend")) or "qdrant"
    default_acl = list(((cfg.get("acl") or {}).get("default")) or ["all"])
    default_chunker_name = ((cfg.get("ingest") or {}).get("default") or {}).get("chunker")
    text_chunker = build_chunker(cfg, name=default_chunker_name)
    bc = _batch_cfg(cfg)

    docs = source.list_docs(after=after or None)
    if not docs:
        raise FileNotFoundError(
            "没有发现待入库文档，请检查 ingest.source（glob / dir）"
            + (f" 或 --after {after} 之后是否还有文档" if after else "")
        )
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit 须为正整数")
        docs = docs[:limit]
        logger.info("本轮仅处理前 {} 篇文档（--limit）", len(docs))

    logger.info(
        "入库加速参数：wave={} parse_workers={} embed_batch={} embed_workers={} milvus_bulk={}",
        bc["wave_size"],
        bc["parse_workers"],
        bc["embed_batch"],
        bc["embed_workers"],
        bc["milvus_bulk"],
    )

    active = ledger.list_active()
    current_sources: set[str] = set()
    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    embedded_total = 0

    def _parse_one(item: IngestDoc) -> tuple[str, ParsedDoc | None, list[Chunk], str | None]:
        """返回 (source_hint, parsed|None, chunks, error|None)。"""
        try:
            if item.text is not None:
                parsed = _parsed_from_text(item, default_acl=default_acl)
                chunks = text_chunker.split(parsed)
                return parsed["source"], parsed, chunks, None
            if item.path is None:
                return item.source, None, [], "缺少 path 与 text"
            parser = router.resolve(item.path)
            parsed = parser.parse(item.path)
            chunks = parser.chunk(parsed)
            return parsed["source"], parsed, chunks, None
        except Exception as exc:  # noqa: BLE001
            return item.source, None, [], f"{exc.__class__.__name__}: {exc}"

    wave = max(1, bc["wave_size"])
    n_waves = (len(docs) + wave - 1) // wave

    for w in range(n_waves):
        batch_docs = docs[w * wave : (w + 1) * wave]
        logger.info("—— 波次 {}/{}：{} 篇 ——", w + 1, n_waves, len(batch_docs))

        parsed_rows: list[tuple[ParsedDoc, list[Chunk]]] = []
        with ThreadPoolExecutor(max_workers=max(1, bc["parse_workers"])) as pool:
            futs = [pool.submit(_parse_one, item) for item in batch_docs]
            done = 0
            for fut in as_completed(futs):
                done += 1
                if done == 1 or done % 200 == 0 or done == len(futs):
                    logger.info("  解析 {}/{} …", done, len(futs))
                _hint, parsed, chunks, err = fut.result()
                if err or parsed is None:
                    failed.append(_hint)
                    if err:
                        logger.warning("  跳过 {}: {}", _hint, err)
                    continue
                source_id = parsed["source"]
                current_sources.add(source_id)
                old = active.get(source_id)
                old_hash = old.get("content_hash") if old else None
                if old_hash and old_hash == parsed["content_hash"]:
                    ledger.mark_skipped(
                        source=source_id, content_hash=parsed["content_hash"]
                    )
                    skipped.append(source_id)
                    continue
                parsed_rows.append((parsed, chunks))

        if not parsed_rows:
            continue

        to_embed: list[Chunk] = []
        pending: dict[str, dict] = {}
        update_sources: list[str] = []
        for parsed, chunks in parsed_rows:
            source_id = parsed["source"]
            old = active.get(source_id)
            action = "update" if old else "add"
            if action == "update":
                update_sources.append(source_id)
                updated.append(source_id)
            else:
                added.append(source_id)
            to_embed.extend(chunks)
            pending[source_id] = {
                "action": action,
                "from_hash": old.get("content_hash") if old else None,
                "content_hash": parsed["content_hash"],
                "chunk_ids": [c["chunk_id"] for c in chunks],
                "acl": list(parsed.get("acl") or []),
                "title": parsed.get("title"),
            }

        logger.info("  清理待更新 source {} …", len(update_sources))
        _delete_sources(sparse, vector, update_sources)

        logger.info(
            "  嵌入 {} chunks（batch={} × workers={}）…",
            len(to_embed),
            bc["embed_batch"],
            bc["embed_workers"],
        )
        vectors = embed_texts(
            [c["text"] for c in to_embed],
            is_query=False,
            batch_size=bc["embed_batch"],
            workers=bc["embed_workers"],
        )
        embedded_total += len(to_embed)

        # 并行写 ES + 向量
        def _write_sparse() -> None:
            if hasattr(sparse, "upsert"):
                try:
                    sparse.upsert(to_embed, refresh=False)
                except TypeError:
                    sparse.upsert(to_embed)

        def _write_vector() -> None:
            if hasattr(vector, "upsert"):
                try:
                    vector.upsert(
                        to_embed,
                        vectors,
                        flush=False,
                        batch_size=bc["milvus_bulk"],
                    )
                except TypeError:
                    vector.upsert(to_embed, vectors)

        logger.info("  批量写入 sparse ∥ vector …")
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_s = pool.submit(_write_sparse)
            f_v = pool.submit(_write_vector)
            f_s.result()
            f_v.result()
        _finish_write(sparse, vector, flush_vector=False)

        logger.info("  写账本 {} …", len(pending))
        for source_id, meta in pending.items():
            try:
                ledger.mark_success(
                    source=source_id,
                    content_hash=meta["content_hash"],
                    chunk_ids=meta["chunk_ids"],
                    action=meta["action"],
                    from_hash=meta["from_hash"],
                    sparse_backend=sparse_name,
                    vector_backend=vector_name,
                    acl=meta.get("acl"),
                    title=meta.get("title"),
                )
                # 更新内存 active，供后续波次 hash 判断
                active[source_id] = {
                    "source": source_id,
                    "content_hash": meta["content_hash"],
                    "chunk_ids": meta["chunk_ids"],
                    "status": "active",
                    "chunk_count": len(meta["chunk_ids"]),
                }
            except Exception as exc:
                ledger.mark_failed(
                    source=source_id,
                    content_hash=meta["content_hash"],
                    action=meta["action"],
                    from_hash=meta["from_hash"],
                    error=f"{exc.__class__.__name__}: {exc}",
                )
                failed.append(source_id)
                if source_id in added:
                    added.remove(source_id)
                if source_id in updated:
                    updated.remove(source_id)

    deleted: list[str] = []
    # --limit / --after 都不是全量扫描，不能把没见到的旧文档当成已删除
    if limit is None and not after:
        gone = [sid for sid in active if sid not in current_sources]
        logger.info("全量模式：删除消失文档 {} …", len(gone))
        for source_id in gone:
            row = active[source_id]
            try:
                _delete_sources(sparse, vector, [source_id])
                ledger.mark_deleted(source=source_id, from_hash=row.get("content_hash"))
                deleted.append(source_id)
            except Exception as exc:
                ledger.mark_failed(
                    source=source_id,
                    content_hash=row.get("content_hash"),
                    action="delete",
                    from_hash=row.get("content_hash"),
                    error=f"{exc.__class__.__name__}: {exc}",
                )
                failed.append(source_id)
        if gone:
            _finish_write(sparse, vector, flush_vector=False)

    logger.info("入库结束，刷新 Milvus segment …")
    _finish_write(sparse, vector, flush_vector=True)

    summary = {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "deleted": deleted,
        "failed": failed,
        "embedded": embedded_total,
    }
    logger.info(
        "入库完成：新增 {} / 更新 {} / 跳过 {} / 删除 {} / 失败 {} / 嵌入 {}",
        len(added),
        len(updated),
        len(skipped),
        len(deleted),
        len(failed),
        embedded_total,
    )
    return summary
