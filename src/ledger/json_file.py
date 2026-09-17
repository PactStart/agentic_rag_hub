"""本地 JSON 账本（沙盒回退）。生产请用 mysql。"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import project_root
from src.ledger.protocol import DocRecord


class JsonFileLedger:
    def __init__(self, path: str = "data/corpus/ingest_manifest.json") -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = project_root() / self.path

    def _load_raw(self) -> dict:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _save_raw(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_active(self) -> dict[str, DocRecord]:
        raw = self._load_raw()
        out: dict[str, DocRecord] = {}
        for source, row in raw.items():
            out[source] = {
                "source": source,
                "content_hash": str(row.get("hash") or ""),
                "chunk_ids": list(row.get("chunk_ids") or []),
                "status": "active",
                "chunk_count": len(row.get("chunk_ids") or []),
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
        _ = (action, from_hash, sparse_backend, vector_backend, acl, title)
        data = self._load_raw()
        data[source] = {"hash": content_hash, "chunk_ids": list(chunk_ids)}
        self._save_raw(data)

    def mark_deleted(self, *, source: str, from_hash: str | None) -> None:
        _ = from_hash
        data = self._load_raw()
        data.pop(source, None)
        self._save_raw(data)

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
        _ = (source, content_hash, action, from_hash, error)
