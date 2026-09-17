from __future__ import annotations

from typing import Protocol, TypedDict


class Chunk(TypedDict):
    chunk_id: str
    text: str
    source: str
    page: int | None
    acl: list[str]
    content_hash: str
    section: str


class Hit(TypedDict):
    chunk_id: str
    text: str
    source: str
    score: float
    acl: list[str]
    page: int | None
    section: str


class SparseIndex(Protocol):
    """关键词 / BM25。ACL 必须在 search 内做掉。"""

    def upsert(self, chunks: list[Chunk]) -> None: ...
    def delete_by_source(self, source: str) -> None: ...
    def search(self, query: str, roles: list[str], top_k: int) -> list[Hit]: ...


class VectorIndex(Protocol):
    """稠密向量。传入已 embed 的向量，插件不调硅基。"""

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None: ...
    def delete_by_source(self, source: str) -> None: ...
    def search(self, vector: list[float], roles: list[str], top_k: int) -> list[Hit]: ...


def visible_to(acl: list[str], roles: list[str]) -> bool:
    """能看：acl 与角色有交集，或 acl 含 all。"""
    acl_set = set(acl or [])
    if "all" in acl_set:
        return True
    return bool(acl_set & set(roles or []))
