"""Milvus 稠密向量插件：ACL 在 search 的 filter 里做掉，与 Qdrant 插件同协议。"""

from __future__ import annotations

import time

from pymilvus import DataType, MilvusClient
from pymilvus.exceptions import MilvusException

from src.stores.protocol import Chunk, Hit

# Milvus VARCHAR 上限；超长正文截断以免入库失败
_TEXT_MAX = 65535
_OUTPUT_FIELDS = ["text", "source", "chunk_id", "acl", "page", "section"]


def _escape(value: str) -> str:
    """Milvus 布尔表达式里的字符串转义。"""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def acl_filter_expr(roles: list[str]) -> str:
    """可见：acl 含 all，或与 roles 有交集（ARRAY 字段）。"""
    cleaned = [r for r in (roles or []) if r]
    if not cleaned:
        return 'array_contains(acl, "all")'
    quoted = ", ".join(f'"{_escape(r)}"' for r in cleaned)
    return f'(array_contains_any(acl, [{quoted}]) or array_contains(acl, "all"))'


class MilvusVectorIndex:
    """``vector.backend=milvus``：URI 连 standalone / 集群，主键用 chunk_id。"""

    def __init__(
        self,
        uri: str = "http://127.0.0.1:19530",
        collection: str = "enterprise_rag",
        token: str = "",
        dim: int = 1024,
    ) -> None:
        self.uri = uri or "http://127.0.0.1:19530"
        self.collection = collection or "enterprise_rag"
        self.dim = int(dim)
        self.client = self._connect(token=token or "")
        self._ensure_collection()

    def _connect(self, *, token: str) -> MilvusClient:
        last_error: Exception | None = None
        for attempt in range(1, 6):
            try:
                client = MilvusClient(uri=self.uri, token=token or None)
                client.list_collections()
                return client
            except (MilvusException, ConnectionError, TimeoutError, OSError) as exc:
                last_error = exc
                print(f"   Milvus 暂不可用（第 {attempt}/5 次）：{exc.__class__.__name__}")
                time.sleep(3)
        raise RuntimeError(
            "无法连接 Milvus，请先按 README 启动 milvus-standalone"
        ) from last_error

    def _ensure_collection(self) -> None:
        if not self.client.has_collection(self.collection):
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("id", DataType.VARCHAR, is_primary=True, max_length=64)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=self.dim)
            schema.add_field("text", DataType.VARCHAR, max_length=_TEXT_MAX)
            schema.add_field("source", DataType.VARCHAR, max_length=512)
            schema.add_field("chunk_id", DataType.VARCHAR, max_length=256)
            schema.add_field("section", DataType.VARCHAR, max_length=512)
            schema.add_field("content_hash", DataType.VARCHAR, max_length=64)
            schema.add_field("page", DataType.INT64, nullable=True)
            schema.add_field(
                "acl",
                DataType.ARRAY,
                element_type=DataType.VARCHAR,
                max_capacity=32,
                max_length=64,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                metric_type="COSINE",
                index_type="AUTOINDEX",
            )
            self.client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=index_params,
            )
            print(f"   Milvus collection 已建：{self.collection}（dim={self.dim}, COSINE）")

        # 未 load 时 search 会空结果
        self.client.load_collection(self.collection)

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        if not chunks:
            return

        rows: list[dict] = []
        for chunk, emb in zip(chunks, embeddings, strict=True):
            if len(emb) != self.dim:
                raise ValueError(f"向量维数 {len(emb)} != 配置 embed_dim={self.dim}")
            text = chunk["text"] or ""
            if len(text) > _TEXT_MAX:
                text = text[:_TEXT_MAX]
            cid = chunk["chunk_id"]
            rows.append(
                {
                    "id": cid,
                    "embedding": list(emb),
                    "text": text,
                    "source": chunk["source"],
                    "chunk_id": cid,
                    "section": chunk.get("section") or "",
                    "content_hash": chunk.get("content_hash") or "",
                    "page": chunk.get("page"),
                    "acl": list(chunk.get("acl") or []),
                }
            )
        self.client.upsert(collection_name=self.collection, data=rows)
        self.client.flush(self.collection)

    def delete_by_source(self, source: str) -> None:
        expr = f'source == "{_escape(source)}"'
        self.client.delete(collection_name=self.collection, filter=expr)
        self.client.flush(self.collection)

    def search(self, vector: list[float], roles: list[str], top_k: int) -> list[Hit]:
        if not vector:
            return []
        raw = self.client.search(
            collection_name=self.collection,
            data=[vector],
            filter=acl_filter_expr(roles),
            limit=max(1, top_k),
            output_fields=_OUTPUT_FIELDS,
            anns_field="embedding",
            consistency_level="Strong",
        )
        # SearchResult → 第一路查询的 HybridHits
        hits = list(raw[0]) if raw else []
        out: list[Hit] = []
        for h in hits:
            entity = h.get("entity") or {}
            page = entity.get("page")
            out.append(
                {
                    "chunk_id": str(entity.get("chunk_id") or h.get("id") or ""),
                    "text": str(entity.get("text") or ""),
                    "source": str(entity.get("source") or ""),
                    # COSINE：distance 越大越相似（与内积/余弦一致）
                    "score": float(h.get("distance") if h.get("distance") is not None else 0.0),
                    "acl": list(entity.get("acl") or []),
                    "page": int(page) if page is not None else None,
                    "section": str(entity.get("section") or ""),
                }
            )
        return out
