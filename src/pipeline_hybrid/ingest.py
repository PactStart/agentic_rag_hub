"""解析 → 切片 → 增量双写 SparseIndex + VectorIndex；账本走 ledger（默认 MySQL）。

流程：``ingest.source`` 列文件 → ``ParserRouter`` 按扩展名选解析器 →
parse/chunk →（hash 未变则跳过）→ 批量 embed → 稀疏/向量 upsert + 账本。
"""

from __future__ import annotations

from src.config import load_config
from src.ingest.factory import build_router, build_source
from src.ledger.factory import build_ledger
from src.llm.siliconflow import embed_texts
from src.stores.factory import build_sparse, build_vector
from src.stores.protocol import Chunk


def run_ingest(config_path: str | None = None) -> dict:
    cfg = load_config(config_path)
    source = build_source(cfg)
    router = build_router(cfg)
    sparse = build_sparse(cfg)
    vector = build_vector(cfg)
    ledger = build_ledger(cfg)
    sparse_name = ((cfg.get("sparse") or {}).get("backend")) or "memory_jsonl"
    vector_name = ((cfg.get("vector") or {}).get("backend")) or "qdrant"

    files = source.list_files()
    if not files:
        raise FileNotFoundError(
            "没有发现待入库文件，请检查 ingest.source.glob / globs"
        )

    active = ledger.list_active()
    current_sources: set[str] = set()
    added: list[str] = []
    updated: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []
    to_embed: list[Chunk] = []
    pending: dict[str, dict] = {}

    for path in files:
        try:
            parser = router.resolve(path)
        except NotImplementedError as exc:
            failed.append(str(path))
            print(f"跳过（无解析器）{path.name}: {exc}")
            continue
        parsed = parser.parse(path)
        source_id = parsed["source"]
        current_sources.add(source_id)
        old = active.get(source_id)
        old_hash = old.get("content_hash") if old else None
        if old_hash and old_hash == parsed["content_hash"]:
            ledger.mark_skipped(source=source_id, content_hash=parsed["content_hash"])
            skipped.append(source_id)
            continue
        chunks = parser.chunk(parsed)
        to_embed.extend(chunks)
        action = "update" if old else "add"
        if action == "update":
            updated.append(source_id)
        else:
            added.append(source_id)
        pending[source_id] = {
            "action": action,
            "from_hash": old_hash,
            "content_hash": parsed["content_hash"],
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "acl": list(parsed.get("acl") or []),
            "title": None,
        }

    deleted: list[str] = []
    for source_id, row in list(active.items()):
        if source_id in current_sources:
            continue
        try:
            sparse.delete_by_source(source_id)
            vector.delete_by_source(source_id)
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

    by_source: dict[str, list[Chunk]] = {}
    for chunk in to_embed:
        by_source.setdefault(chunk["source"], []).append(chunk)

    if to_embed:
        print(f"嵌入 {len(to_embed)} 个 chunk（硅基 BGE-M3）…")
        vectors = embed_texts([c["text"] for c in to_embed], is_query=False)
        offset = 0
        for source_id, chunks in by_source.items():
            n = len(chunks)
            part = vectors[offset : offset + n]
            offset += n
            meta = pending[source_id]
            try:
                sparse.delete_by_source(source_id)
                vector.delete_by_source(source_id)
                sparse.upsert(chunks)
                vector.upsert(chunks, part)
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

    summary = {
        "added": added,
        "updated": updated,
        "skipped": skipped,
        "deleted": deleted,
        "failed": failed,
        "embedded": len(to_embed),
    }
    print(
        "入库完成："
        f"新增 {len(added)} / 更新 {len(updated)} / 跳过 {len(skipped)} / "
        f"删除 {len(deleted)} / 失败 {len(failed)} / 嵌入 {len(to_embed)}"
    )
    return summary
