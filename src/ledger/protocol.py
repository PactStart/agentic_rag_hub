"""入库账本协议：替代本地 ingest_manifest.json。"""

from __future__ import annotations

from typing import Protocol, TypedDict


class DocRecord(TypedDict):
    source: str
    content_hash: str
    chunk_ids: list[str]
    status: str
    chunk_count: int


class IngestLedger(Protocol):
    """按 source 记 hash / 切片，支撑增量 skip / update / delete。"""

    def list_active(self) -> dict[str, DocRecord]:
        """status=active 的文档，key=source。"""
        ...

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
    ) -> None: ...

    def mark_deleted(self, *, source: str, from_hash: str | None) -> None: ...

    def mark_skipped(self, *, source: str, content_hash: str) -> None: ...

    def mark_failed(
        self,
        *,
        source: str,
        content_hash: str | None,
        action: str,
        from_hash: str | None,
        error: str,
    ) -> None: ...
