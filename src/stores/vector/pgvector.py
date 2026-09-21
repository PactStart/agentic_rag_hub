"""Postgres + pgvector 稠密向量插件。ACL 在 SQL 内过滤；不调硅基。"""

from __future__ import annotations

import os

from loguru import logger
from pgvector.psycopg import register_vector

from src.stores.postgres_util import connect, normalize_dsn, quote_ident
from src.stores.protocol import Chunk, Hit


class PgVectorIndex:
    """``vector.backend=pgvector``：cosine 距离，score = 1 - distance。"""

    def __init__(
        self,
        dsn: str = "",
        table: str = "rag_embeddings",
        dim: int = 1024,
        **_kwargs,
    ) -> None:
        self.dsn = normalize_dsn(dsn or os.environ.get("POSTGRES_DSN"))
        self.table = (table or os.environ.get("PGVECTOR_TABLE") or "rag_embeddings").strip()
        self.dim = int(dim)
        self._qtable = quote_ident(self.table)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        idx_hnsw = quote_ident(self.table + "_embedding_hnsw")
        idx_src = quote_ident(self.table + "_source_idx")
        idx_acl = quote_ident(self.table + "_acl_idx")
        with connect(self.dsn) as conn:
            register_vector(conn)
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qtable} (
                    chunk_id TEXT PRIMARY KEY,
                    embedding vector({self.dim}) NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    acl TEXT[] NOT NULL DEFAULT '{{}}'
                )
                """
            )
            # HNSW：小数据也能建；无需先 analyze 再 lists
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {idx_hnsw}
                ON {self._qtable}
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_src} ON {self._qtable} (source)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_acl} ON {self._qtable} USING GIN (acl)"
            )
            conn.commit()
        logger.info(
            "pgvector 表就绪：{}（dim={}, cosine/HNSW）",
            self.table,
            self.dim,
        )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks 与 embeddings 数量不一致")
        if not chunks:
            return
        for emb in embeddings:
            if len(emb) != self.dim:
                raise ValueError(
                    f"向量维数 {len(emb)} ≠ embed_dim/表维 {self.dim}，请改建表或改 embed_dim"
                )
        sql = f"""
        INSERT INTO {self._qtable}
            (chunk_id, embedding, text, source, page, section, content_hash, acl)
        VALUES
            (%(chunk_id)s, %(embedding)s, %(text)s, %(source)s, %(page)s,
             %(section)s, %(content_hash)s, %(acl)s)
        ON CONFLICT (chunk_id) DO UPDATE SET
            embedding = EXCLUDED.embedding,
            text = EXCLUDED.text,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            content_hash = EXCLUDED.content_hash,
            acl = EXCLUDED.acl
        """
        rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "embedding": emb,
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "section": chunk.get("section") or "",
                "content_hash": chunk.get("content_hash") or "",
                "acl": list(chunk.get("acl") or []),
            }
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]
        with connect(self.dsn) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.executemany(sql, rows)
            conn.commit()

    def delete_by_source(self, source: str) -> None:
        self.delete_by_sources([source])

    def delete_by_sources(self, sources: list[str]) -> None:
        cleaned = [s for s in sources if s]
        if not cleaned:
            return
        sql = f"DELETE FROM {self._qtable} WHERE source = ANY(%s)"
        with connect(self.dsn) as conn:
            conn.execute(sql, (cleaned,))
            conn.commit()

    def search(self, vector: list[float], roles: list[str], top_k: int) -> list[Hit]:
        if len(vector) != self.dim:
            raise ValueError(f"查询向量维数 {len(vector)} ≠ {self.dim}")
        roles = [r for r in (roles or []) if r]
        # <=> cosine distance；相似度用 1 - distance，与 Qdrant COSINE 越大越好对齐
        sql = f"""
        SELECT chunk_id, text, source, page, section, acl,
               (1 - (embedding <=> %(vec)s::vector)) AS score
        FROM {self._qtable}
        WHERE (
            acl && %(roles)s::text[]
            OR 'all' = ANY(acl)
        )
        ORDER BY embedding <=> %(vec)s::vector
        LIMIT %(k)s
        """
        with connect(self.dsn) as conn:
            register_vector(conn)
            cur = conn.execute(
                sql,
                {"vec": vector, "roles": roles or [], "k": int(top_k)},
            )
            rows = cur.fetchall()
        hits: list[Hit] = []
        for row in rows:
            hits.append(
                {
                    "chunk_id": str(row["chunk_id"] or ""),
                    "text": str(row["text"] or ""),
                    "source": str(row["source"] or ""),
                    "score": float(row["score"] or 0.0),
                    "acl": list(row["acl"] or []),
                    "page": row.get("page"),
                    "section": str(row.get("section") or ""),
                }
            )
        return hits
