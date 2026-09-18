"""定长开窗切片（CRUD 官方默认 chunk_size=128、overlap=0）。"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.parsers.protocol import ParsedDoc
from src.stores.protocol import Chunk


class FixedSizeChunker:
    """按字符窗切纯文本；分隔符偏向中文标点与换行。"""

    def __init__(self, chunk_size: int = 128, chunk_overlap: int = 0) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        text = (doc.get("markdown") or "").strip()
        if not text:
            return []
        pieces = self._splitter.split_text(text)
        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc['source']}:{i}",
                    "text": piece,
                    "source": doc["source"],
                    "page": doc.get("page"),
                    "acl": list(doc["acl"]),
                    "content_hash": doc["content_hash"],
                    "section": "",
                }
            )
        return chunks
