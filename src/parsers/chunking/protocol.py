"""切片策略协议：ParsedDoc → Chunk 列表。与具体 Parser 绑定或按配置覆盖。"""

from __future__ import annotations

from typing import Protocol

from src.parsers.protocol import ParsedDoc
from src.stores.protocol import Chunk


class Chunker(Protocol):
    """不同文档类型用不同实现；ingest 只认 Chunk。"""

    def split(self, doc: ParsedDoc) -> list[Chunk]: ...
