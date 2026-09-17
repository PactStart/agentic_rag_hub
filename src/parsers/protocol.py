from __future__ import annotations

from pathlib import Path
from typing import NotRequired, Protocol, TypedDict

from src.stores.protocol import Chunk


class ParsedDoc(TypedDict):
    """解析产物。切片层（Chunker）只依赖这些字段。"""

    source: str
    markdown: str
    path: str
    content_hash: str
    acl: list[str]
    page: NotRequired[int | None]
    doc_type: NotRequired[str]


class Parser(Protocol):
    """解析 + 切片：parse 出结构，chunk 用绑定的 Chunker 策略。

    文件发现由 ``ingest.source`` 负责，不在 Parser 上。
    """

    def parse(self, path: Path) -> ParsedDoc: ...
    def chunk(self, doc: ParsedDoc) -> list[Chunk]: ...
