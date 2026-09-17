"""MySQL 入库账本：SQLAlchemy ORM（kb_document + kb_ingest_job）。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from src.ledger.db import make_session_factory, ping_engine
from src.ledger.models import KbDocument, KbIngestJob
from src.ledger.protocol import DocRecord


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class MysqlLedger:
    def __init__(self, dsn: str, tenant_id: str = "default") -> None:
        self.dsn = dsn
        self.tenant_id = tenant_id or "default"
        self._Session = make_session_factory(dsn)
        try:
            ping_engine(dsn)
        except Exception as exc:
            raise RuntimeError(
                "连不上 MySQL 账本库。请确认容器已起、库已建并完成迁移："
                "uv run python scripts/init_mysql.py ；"
                f"原因：{exc}"
            ) from exc

    def _session(self) -> Session:
        return self._Session()

    def list_active(self) -> dict[str, DocRecord]:
        with self._session() as session:
            rows = session.scalars(
                select(KbDocument).where(
                    KbDocument.tenant_id == self.tenant_id,
                    KbDocument.status == "active",
                )
            ).all()
            out: dict[str, DocRecord] = {}
            for row in rows:
                ids = row.chunk_ids_json
                if isinstance(ids, str):
                    import json

                    ids = json.loads(ids)
                out[row.source] = {
                    "source": row.source,
                    "content_hash": row.content_hash or "",
                    "chunk_ids": list(ids or []),
                    "status": row.status,
                    "chunk_count": int(row.chunk_count or 0),
                }
            return out

    def mark_success(
        self,
        *,
        source: str,
        content_hash: str,
        chunk_ids: list[str],
        action: str,
        from_hash: str | None,
        sparse_backend: str | None = None,
        vector_backend: str | None = None,
        acl: list[str] | None = None,
        title: str | None = None,
    ) -> None:
        now = _utcnow()
        chunk_count = len(chunk_ids)
        with self._session() as session:
            stmt = mysql_insert(KbDocument).values(
                tenant_id=self.tenant_id,
                source=source,
                title=title,
                content_hash=content_hash,
                status="active",
                acl_json=list(acl or []),
                chunk_count=chunk_count,
                chunk_ids_json=list(chunk_ids),
                sparse_backend=sparse_backend,
                vector_backend=vector_backend,
                error_message=None,
                ingested_at=now,
            )
            stmt = stmt.on_duplicate_key_update(
                title=stmt.inserted.title,
                content_hash=stmt.inserted.content_hash,
                status="active",
                acl_json=stmt.inserted.acl_json,
                chunk_count=stmt.inserted.chunk_count,
                chunk_ids_json=stmt.inserted.chunk_ids_json,
                sparse_backend=stmt.inserted.sparse_backend,
                vector_backend=stmt.inserted.vector_backend,
                error_message=None,
                ingested_at=stmt.inserted.ingested_at,
            )
            session.execute(stmt)
            session.add(
                KbIngestJob(
                    tenant_id=self.tenant_id,
                    source=source,
                    trigger_type="manual",
                    action=action,
                    from_hash=from_hash,
                    to_hash=content_hash,
                    status="success",
                    chunk_count=chunk_count,
                    error_message=None,
                    started_at=now,
                    finished_at=now,
                )
            )
            session.commit()

    def mark_deleted(self, *, source: str, from_hash: str | None) -> None:
        now = _utcnow()
        with self._session() as session:
            row = session.scalar(
                select(KbDocument).where(
                    KbDocument.tenant_id == self.tenant_id,
                    KbDocument.source == source,
                )
            )
            if row is None:
                session.add(
                    KbDocument(
                        tenant_id=self.tenant_id,
                        source=source,
                        content_hash=from_hash or "",
                        status="deleted",
                        chunk_count=0,
                        chunk_ids_json=[],
                    )
                )
            else:
                row.status = "deleted"
                row.chunk_count = 0
                row.chunk_ids_json = []
                row.error_message = None
            session.add(
                KbIngestJob(
                    tenant_id=self.tenant_id,
                    source=source,
                    trigger_type="manual",
                    action="delete",
                    from_hash=from_hash,
                    to_hash=None,
                    status="success",
                    chunk_count=0,
                    error_message=None,
                    started_at=now,
                    finished_at=now,
                )
            )
            session.commit()

    def mark_skipped(self, *, source: str, content_hash: str) -> None:
        _ = (source, content_hash)

    def mark_failed(
        self,
        *,
        source: str,
        content_hash: str | None,
        action: str,
        from_hash: str | None,
        error: str,
    ) -> None:
        now = _utcnow()
        msg = error[:2000]
        with self._session() as session:
            if content_hash is not None:
                stmt = mysql_insert(KbDocument).values(
                    tenant_id=self.tenant_id,
                    source=source,
                    content_hash=content_hash,
                    status="failed",
                    error_message=msg,
                    chunk_count=0,
                )
                stmt = stmt.on_duplicate_key_update(
                    status="failed",
                    error_message=stmt.inserted.error_message,
                )
                session.execute(stmt)
            session.add(
                KbIngestJob(
                    tenant_id=self.tenant_id,
                    source=source,
                    trigger_type="manual",
                    action=action,
                    from_hash=from_hash,
                    to_hash=content_hash,
                    status="failed",
                    chunk_count=None,
                    error_message=msg,
                    started_at=now,
                    finished_at=now,
                )
            )
            session.commit()
