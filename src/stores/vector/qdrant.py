from __future__ import annotations

import time
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

from src.stores.protocol import Chunk, Hit


def _point_uuid(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def acl_filter(roles: list[str]) -> Filter:
    return Filter(
        should=[
            FieldCondition(key="acl", match=MatchAny(any=roles or [])),
            FieldCondition(key="acl", match=MatchValue(value="all")),
        ]
    )


class QdrantVectorIndex:
    def __init__(
        self,
        url: str = "http://127.0.0.1:6333",
        collection: str = "enterprise_rag",
        api_key: str = "",
        dim: int = 1024,
    ) -> None:
        self.url = url or "http://127.0.0.1:6333"
        self.collection = collection or "enterprise_rag"
        self.dim = dim
        self.client = QdrantClient(
            url=self.url,
            api_key=api_key or None,
            timeout=60,
            prefer_grpc=False,
            check_compatibility=False,
            trust_env=False,
        )
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                names = [c.name for c in self.client.get_collections().collections]
                if self.collection not in names:
                    self.client.create_collection(
                        collection_name=self.collection,
                        vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
                    )
                return
            except (UnexpectedResponse, ConnectionError, TimeoutError, OSError) as exc:
                last_error = exc
                print(f"   Qdrant 暂不可用（第 {attempt}/5 次）：{exc.__class__.__name__}")
                time.sleep(3)
        raise RuntimeError("无法连接 Qdrant，请先 docker compose up -d") from last_error

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        points = [
            PointStruct(
                id=_point_uuid(chunk["chunk_id"]),
                vector=emb,
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk.get("page"),
                    "acl": chunk.get("acl") or [],
                    "chunk_id": chunk["chunk_id"],
                    "section": chunk.get("section") or "",
                    "content_hash": chunk.get("content_hash"),
                },
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def delete_by_source(self, source: str) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="source", match=MatchValue(value=source))]
                )
            ),
        )

    def search(self, vector: list[float], roles: list[str], top_k: int) -> list[Hit]:
        hits = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            query_filter=acl_filter(roles),
            limit=top_k,
            with_payload=True,
        ).points
        out: list[Hit] = []
        for h in hits:
            payload = h.payload or {}
            out.append(
                {
                    "chunk_id": str(payload.get("chunk_id") or ""),
                    "text": str(payload.get("text") or ""),
                    "source": str(payload.get("source") or ""),
                    "score": float(h.score or 0.0),
                    "acl": list(payload.get("acl") or []),
                    "page": payload.get("page"),
                    "section": str(payload.get("section") or ""),
                }
            )
        return out
