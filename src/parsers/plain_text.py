"""纯文本解析：新闻行 / 无 frontmatter 的 .txt。"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.parsers.chunking.fixed_size import FixedSizeChunker
from src.parsers.chunking.protocol import Chunker
from src.parsers.protocol import ParsedDoc
from src.stores.protocol import Chunk


class PlainTextParser:
    """读 UTF-8 文本为 ParsedDoc，再交给绑定的 Chunker。

    ``source`` 默认用文件名（不含扩展名），便于 CRUD 展开后的稳定 id。
    """

    def __init__(
        self,
        default_acl: list[str] | None = None,
        chunker: Chunker | None = None,
        *,
        source_mode: str = "stem",
    ) -> None:
        self.default_acl = list(default_acl or ["all"])
        self.chunker: Chunker = chunker or FixedSizeChunker()
        self.source_mode = source_mode  # stem | name | path

    def parse(self, path: Path) -> ParsedDoc:
        raw = path.read_text(encoding="utf-8", errors="replace")
        body = raw.strip() + ("\n" if raw.strip() else "")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if self.source_mode == "name":
            source = path.name
        elif self.source_mode == "path":
            source = str(path)
        else:
            source = path.stem
        title = body.splitlines()[0][:120] if body else None
        return {
            "source": source,
            "markdown": body,
            "path": str(path),
            "content_hash": digest,
            "acl": list(self.default_acl),
            "page": None,
            "doc_type": "plain_text",
            "title": title,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        return self.chunker.split(doc)
