"""Postgres 全文检索（tsvector + ts_rank）。

中文：用 jieba 切词后再写入 / 查询 ``simple`` 配置的 tsvector
（默认 pgvector 镜像无 zhparser；与仓库其它中文路径一致）。
ACL 在 SQL 内过滤。
"""

from __future__ import annotations

import os

import jieba
from loguru import logger

from src.stores.postgres_util import connect, normalize_dsn, quote_ident
from src.stores.protocol import Chunk, Hit


def _tokenize_text(text: str) -> str:
    """jieba 切词，空格拼接，供 simple tsvector 使用。"""
    parts = [t.strip() for t in jieba.cut_for_search(text or "") if t and t.strip()]
    return " ".join(parts)


def _tsquery_or(text: str) -> str:
    """把切词结果编成 simple tsquery 的 OR（``a | b | c``）。"""
    parts = [t.strip() for t in jieba.cut_for_search(text or "") if t and t.strip()]
    cleaned: list[str] = []
    for p in parts:
        # tsquery 特殊字符去掉，避免语法错误
        tok = "".join(ch for ch in p if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if tok:
            cleaned.append(tok)
    # 去重保序
    seen: set[str] = set()
    uniq: list[str] = []
    for t in cleaned:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return " | ".join(uniq)


class PostgresFtsIndex:
    """``sparse.backend=postgres_fts``。"""

    def __init__(
        self,
        dsn: str = "",
        table: str = "rag_chunks",
        **_kwargs,
    ) -> None:
        self.dsn = normalize_dsn(dsn or os.environ.get("POSTGRES_DSN"))
        self.table = (table or os.environ.get("POSTGRES_FTS_TABLE") or "rag_chunks").strip()
        self._qtable = quote_ident(self.table)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with connect(self.dsn) as conn:
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._qtable} (
                    chunk_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER,
                    section TEXT NOT NULL DEFAULT '',
                    content_hash TEXT NOT NULL DEFAULT '',
                    acl TEXT[] NOT NULL DEFAULT '{{}}',
                    tsv tsvector NOT NULL
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_ident(self.table + "_tsv_idx")}
                ON {self._qtable} USING GIN (tsv)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_ident(self.table + "_source_idx")}
                ON {self._qtable} (source)
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {quote_ident(self.table + "_acl_idx")}
                ON {self._qtable} USING GIN (acl)
                """
            )
            conn.commit()
        logger.info("Postgres FTS 表就绪：{}", self.table)

    def upsert(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        sql = f"""
        INSERT INTO {self._qtable}
            (chunk_id, text, source, page, section, content_hash, acl, tsv)
        VALUES
            (%(chunk_id)s, %(text)s, %(source)s, %(page)s, %(section)s,
             %(content_hash)s, %(acl)s, to_tsvector('simple', %(tokens)s))
        ON CONFLICT (chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            section = EXCLUDED.section,
            content_hash = EXCLUDED.content_hash,
            acl = EXCLUDED.acl,
            tsv = EXCLUDED.tsv
        """
        rows = [
            {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk.get("page"),
                "section": chunk.get("section") or "",
                "content_hash": chunk.get("content_hash") or "",
                "acl": list(chunk.get("acl") or []),
                "tokens": _tokenize_text(chunk["text"]),
            }
            for chunk in chunks
        ]
        with connect(self.dsn) as conn:
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

    def search(self, query: str, roles: list[str], top_k: int) -> list[Hit]:
        tsq = _tsquery_or(query)
        if not tsq:
            return []
        roles = [r for r in (roles or []) if r]
        sql = f"""
        SELECT chunk_id, text, source, page, section, acl,
               ts_rank(tsv, to_tsquery('simple', %(q)s)) AS score
        FROM {self._qtable}
        WHERE tsv @@ to_tsquery('simple', %(q)s)
          AND (
            acl && %(roles)s::text[]
            OR 'all' = ANY(acl)
          )
        ORDER BY score DESC
        LIMIT %(k)s
        """
        with connect(self.dsn) as conn:
            cur = conn.execute(
                sql,
                {"q": tsq, "roles": roles or [], "k": int(top_k)},
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
