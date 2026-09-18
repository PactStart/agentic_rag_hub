"""Elasticsearch BM25 插件。ACL 在查询里 filter，不在 Python 里事后删。"""

from __future__ import annotations

from elasticsearch import Elasticsearch, BadRequestError
from elasticsearch.helpers import bulk
from loguru import logger

from src.stores.protocol import Chunk, Hit


def _text_mapping(analyzer: str) -> dict:
    field: dict = {"type": "text", "analyzer": analyzer}
    if analyzer.startswith("ik_"):
        field["search_analyzer"] = "ik_smart"
    return {
        "properties": {
            "text": field,
            "source": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "acl": {"type": "keyword"},
            "page": {"type": "integer"},
            "section": {"type": "keyword"},
            "content_hash": {"type": "keyword"},
        }
    }


class ElasticBm25Index:
    def __init__(
        self,
        url: str = "http://127.0.0.1:9200",
        index: str = "rag_chunks",
        user: str = "",
        password: str = "",
        api_key: str = "",
    ) -> None:
        self.index = index or "rag_chunks"
        kwargs: dict = {"hosts": [url or "http://127.0.0.1:9200"], "request_timeout": 30}
        if api_key:
            kwargs["api_key"] = api_key
        elif user:
            kwargs["basic_auth"] = (user, password)
        self.client = Elasticsearch(**kwargs)
        if not self.client.ping():
            raise RuntimeError(
                "连不上 Elasticsearch。请先：docker compose --profile es up -d --build"
            )
        self._ensure_index()

    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        try:
            self.client.indices.create(
                index=self.index,
                mappings=_text_mapping("ik_max_word"),
            )
            logger.info("ES 索引已建（analyzer=ik_max_word）")
        except BadRequestError:
            self.client.indices.create(
                index=self.index,
                mappings=_text_mapping("standard"),
            )
            logger.warning(
                "ES 无 IK 插件，回退 standard（中文召回会变差，请用 docker/elasticsearch 镜像）"
            )

    def upsert(self, chunks: list[Chunk], *, refresh: bool | str = False) -> None:
        if not chunks:
            return
        actions = [
            {
                "_op_type": "index",
                "_index": self.index,
                "_id": chunk["chunk_id"],
                "_source": {
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "acl": chunk.get("acl") or [],
                    "page": chunk.get("page"),
                    "section": chunk.get("section") or "",
                    "content_hash": chunk.get("content_hash") or "",
                },
            }
            for chunk in chunks
        ]
        # 入库默认不 wait_for，整波写完再 refresh_index()
        bulk(
            self.client,
            actions,
            chunk_size=2000,
            request_timeout=120,
            refresh=refresh,
        )

    def refresh_index(self) -> None:
        self.client.indices.refresh(index=self.index)

    def delete_by_source(self, source: str) -> None:
        self.delete_by_sources([source])

    def delete_by_sources(self, sources: list[str], *, refresh: bool = False) -> None:
        cleaned = [s for s in sources if s]
        if not cleaned:
            return
        # terms 单次不宜过大；分片删除
        step = 500
        for i in range(0, len(cleaned), step):
            part = cleaned[i : i + step]
            self.client.delete_by_query(
                index=self.index,
                query={"terms": {"source": part}},
                refresh=refresh,
                conflicts="proceed",
                request_timeout=120,
            )

    def search(self, query: str, roles: list[str], top_k: int) -> list[Hit]:
        body_query = {
            "bool": {
                "must": {"match": {"text": query}},
                "filter": {
                    "bool": {
                        "should": [
                            {"terms": {"acl": roles or []}},
                            {"term": {"acl": "all"}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
            }
        }
        resp = self.client.search(index=self.index, query=body_query, size=top_k)
        hits: list[Hit] = []
        for row in resp["hits"]["hits"]:
            src = row["_source"]
            hits.append(
                {
                    "chunk_id": str(src.get("chunk_id") or row["_id"]),
                    "text": str(src.get("text") or ""),
                    "source": str(src.get("source") or ""),
                    "score": float(row.get("_score") or 0.0),
                    "acl": list(src.get("acl") or []),
                    "page": src.get("page"),
                    "section": str(src.get("section") or ""),
                }
            )
        return hits
