"""按配置构造切片策略。也可由各 Parser 在构造时注入。"""

from __future__ import annotations

from src.parsers.chunking.protocol import Chunker


def build_chunker(cfg: dict, name: str | None = None) -> Chunker:
    block = cfg.get("chunking") or {}
    chosen = name or block.get("backend") or "markdown_heading"
    if chosen == "markdown_heading":
        from src.parsers.chunking.markdown_heading import MarkdownHeadingChunker

        opts = block.get("markdown_heading") or {}
        return MarkdownHeadingChunker(
            chunk_size=opts.get("chunk_size", 512),
            chunk_overlap=opts.get("chunk_overlap", 64),
        )
    if chosen == "fixed_size":
        from src.parsers.chunking.fixed_size import FixedSizeChunker

        opts = block.get("fixed_size") or {}
        return FixedSizeChunker(
            chunk_size=opts.get("chunk_size", 128),
            chunk_overlap=opts.get("chunk_overlap", 0),
        )
    if chosen in ("pdf_layout", "table_row", "faq_pair"):
        raise NotImplementedError(
            f"chunking.backend={chosen} 尚未实现。沙盒请用 markdown_heading。"
        )
    raise ValueError(f"未知 chunking.backend: {chosen}")
