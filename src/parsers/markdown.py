from __future__ import annotations

import hashlib
import re
from pathlib import Path

import yaml

from src.parsers.chunking.markdown_heading import MarkdownHeadingChunker
from src.parsers.chunking.protocol import Chunker
from src.parsers.protocol import ParsedDoc
from src.stores.protocol import Chunk

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def split_frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER.match(text)
    if not match:
        return {}, text
    meta = yaml.safe_load(match.group(1)) or {}
    if not isinstance(meta, dict):
        meta = {}
    return meta, text[match.end() :]


class MarkdownParser:
    """把 Markdown（含可选 YAML frontmatter）解析为 ParsedDoc，再交给 Chunker。

    frontmatter 可带 ``acl`` / ``doc_type``；文件发现不在本类职责内。
    """

    def __init__(
        self,
        default_acl: list[str] | None = None,
        chunker: Chunker | None = None,
    ) -> None:
        self.default_acl = default_acl or ["all"]
        self.chunker: Chunker = chunker or MarkdownHeadingChunker()

    def parse(self, path: Path) -> ParsedDoc:
        raw = path.read_text(encoding="utf-8")
        meta, body = split_frontmatter(raw)
        acl = meta.get("acl") or self.default_acl
        if isinstance(acl, str):
            acl = [acl]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        doc_type = str(meta.get("doc_type") or "markdown")
        return {
            "source": path.name,
            "markdown": body.strip() + "\n",
            "path": str(path),
            "content_hash": digest,
            "acl": list(acl),
            "page": None,
            "doc_type": doc_type,
        }

    def chunk(self, doc: ParsedDoc) -> list[Chunk]:
        return self.chunker.split(doc)
