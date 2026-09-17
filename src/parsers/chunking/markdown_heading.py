"""Markdown：先按标题切节，表整块保留，超长节再开窗。"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.parsers.protocol import ParsedDoc
from src.stores.protocol import Chunk


def _looks_like_table(text: str) -> bool:
    return "|" in text and "\n|" in text


class MarkdownHeadingChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

    def split(self, doc: ParsedDoc) -> list[Chunk]:
        header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
        )
        sections = header_splitter.split_text(doc["markdown"])
        window = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", " ", ""],
        )
        pieces: list[Document] = []
        if not sections:
            sections = [Document(page_content=doc["markdown"], metadata={})]
        for sec in sections:
            path = " > ".join(
                p
                for p in [
                    sec.metadata.get("h1"),
                    sec.metadata.get("h2"),
                    sec.metadata.get("h3"),
                ]
                if p
            )
            prefix = f"{doc['source']} > {path}\n" if path else f"{doc['source']}\n"
            body = prefix + sec.page_content
            meta = {"section": path}
            wrapped = Document(page_content=body, metadata=meta)
            if _looks_like_table(sec.page_content):
                pieces.append(wrapped)
            else:
                pieces.extend(window.split_documents([wrapped]))

        chunks: list[Chunk] = []
        for i, piece in enumerate(pieces):
            chunks.append(
                {
                    "chunk_id": f"{doc['source']}:{i}",
                    "text": piece.page_content,
                    "source": doc["source"],
                    "page": doc.get("page"),
                    "acl": list(doc["acl"]),
                    "content_hash": doc["content_hash"],
                    "section": str(piece.metadata.get("section") or ""),
                }
            )
        return chunks
