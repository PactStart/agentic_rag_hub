from __future__ import annotations

import json
from pathlib import Path

import jieba
from rank_bm25 import BM25Okapi

from src.config import project_root
from src.stores.protocol import Chunk, Hit, visible_to


def tokenize(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text) if token.strip()]


class JsonlBm25Index:
    """沙盒用：jsonl 落盘 + 内存 BM25。生产请换 elasticsearch。"""

    def __init__(self, path: str = "data/corpus/chunks.jsonl") -> None:
        self.path = Path(path)
        if not self.path.is_absolute():
            self.path = project_root() / self.path
        self._chunks: list[Chunk] = []
        self._bm25: BM25Okapi | None = None
        self._reload()

    def _reload(self) -> None:
        self._chunks = []
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    self._chunks.append(row)
        if self._chunks:
            tokenized = [tokenize(c["text"]) for c in self._chunks]
            self._bm25 = BM25Okapi(tokenized)
        else:
            self._bm25 = None

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for chunk in self._chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        self._reload()

    def upsert(self, chunks: list[Chunk]) -> None:
        sources = {c["source"] for c in chunks}
        kept = [c for c in self._chunks if c["source"] not in sources]
        self._chunks = kept + list(chunks)
        self._flush()

    def delete_by_source(self, source: str) -> None:
        self._chunks = [c for c in self._chunks if c["source"] != source]
        self._flush()

    def search(self, query: str, roles: list[str], top_k: int) -> list[Hit]:
        if not self._bm25 or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits: list[Hit] = []
        for i in ranked:
            chunk = self._chunks[i]
            if not visible_to(chunk.get("acl") or [], roles):
                continue
            hits.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "score": float(scores[i]),
                    "acl": list(chunk.get("acl") or []),
                    "page": chunk.get("page"),
                    "section": chunk.get("section") or "",
                }
            )
            if len(hits) >= top_k:
                break
        return hits
