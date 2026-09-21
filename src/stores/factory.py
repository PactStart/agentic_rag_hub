from __future__ import annotations

import os

from src.stores.protocol import SparseIndex, VectorIndex


def build_sparse(cfg: dict) -> SparseIndex:
    block = cfg.get("sparse") or {}
    name = os.environ.get("SPARSE_BACKEND") or block.get("backend") or "memory_jsonl"
    if name == "memory_jsonl":
        from src.stores.sparse.memory_jsonl import JsonlBm25Index

        path = (block.get("memory_jsonl") or {}).get("path", "data/corpus/chunks.jsonl")
        return JsonlBm25Index(path=path)
    if name == "elasticsearch":
        from src.stores.sparse.elasticsearch import ElasticBm25Index

        es = block.get("elasticsearch") or {}
        return ElasticBm25Index(
            url=es.get("url") or os.environ.get("ELASTICSEARCH_URL") or "http://127.0.0.1:9200",
            index=es.get("index") or os.environ.get("ELASTICSEARCH_INDEX") or "rag_chunks",
            user=os.environ.get("ELASTICSEARCH_USER") or "",
            password=os.environ.get("ELASTICSEARCH_PASSWORD") or "",
            api_key=os.environ.get("ELASTICSEARCH_API_KEY") or "",
        )
    if name == "postgres_fts":
        from src.stores.sparse.postgres_fts import PostgresFtsIndex

        p = block.get("postgres_fts") or {}
        return PostgresFtsIndex(
            dsn=p.get("dsn") or os.environ.get("POSTGRES_DSN") or "",
            table=p.get("table")
            or os.environ.get("POSTGRES_FTS_TABLE")
            or "rag_chunks",
        )
    if name == "qdrant_sparse":
        from src.stores.sparse.qdrant_sparse import QdrantSparseIndex

        return QdrantSparseIndex(**(block.get("qdrant_sparse") or {}))
    raise ValueError(f"未知 sparse.backend: {name}")


def build_vector(cfg: dict) -> VectorIndex:
    block = cfg.get("vector") or {}
    name = block.get("backend") or "qdrant"
    dim = int(cfg.get("embed_dim") or 1024)
    if name == "qdrant":
        from src.stores.vector.qdrant import QdrantVectorIndex

        q = block.get("qdrant") or {}
        return QdrantVectorIndex(
            url=q.get("url") or "http://127.0.0.1:6333",
            collection=q.get("collection") or "enterprise_rag",
            api_key=q.get("api_key") or "",
            dim=dim,
        )
    if name == "milvus":
        from src.stores.vector.milvus import MilvusVectorIndex

        m = block.get("milvus") or {}
        return MilvusVectorIndex(
            uri=m.get("uri") or os.environ.get("MILVUS_URI") or "http://127.0.0.1:19530",
            collection=m.get("collection")
            or os.environ.get("MILVUS_COLLECTION")
            or "enterprise_rag",
            token=m.get("token") or os.environ.get("MILVUS_TOKEN") or "",
            dim=dim,
        )
    if name == "pgvector":
        from src.stores.vector.pgvector import PgVectorIndex

        p = block.get("pgvector") or {}
        return PgVectorIndex(
            dsn=p.get("dsn") or os.environ.get("POSTGRES_DSN") or "",
            table=p.get("table")
            or os.environ.get("PGVECTOR_TABLE")
            or "rag_embeddings",
            dim=dim,
        )
    raise ValueError(f"未知 vector.backend: {name}")
