"""按名字构造单个 Parser；由 ``ParserRouter`` 调用，不负责选文件。"""

from __future__ import annotations

from src.parsers.chunking.factory import build_chunker
from src.parsers.markdown import MarkdownParser
from src.parsers.protocol import Parser


def build_parser(
    cfg: dict,
    *,
    name: str,
    chunker_name: str | None = None,
) -> Parser:
    """``name`` 为 yaml 里的解析器键（markdown / docling / …）。

    切片策略优先级：路由传入的 ``chunker_name`` → ``chunking.backend``。
    解析器专属参数读 ``parser.<name>``（如 docling.ocr），其中的 chunker 字段忽略。
    """
    if not name:
        raise ValueError("build_parser 需要 name")

    parser_cfg = cfg.get("parser") or {}
    default_acl = ((cfg.get("acl") or {}).get("default")) or ["all"]
    chunker = build_chunker(
        cfg,
        name=chunker_name or (cfg.get("chunking") or {}).get("backend"),
    )
    # 去掉 chunker，避免再传给尚未实现的 **kwargs 构造器
    opts = {
        k: v
        for k, v in (parser_cfg.get(name) or {}).items()
        if k != "chunker"
    }

    if name == "markdown":
        return MarkdownParser(default_acl=default_acl, chunker=chunker)
    if name == "docling":
        from src.parsers.docling import DoclingParser

        return DoclingParser(chunker=chunker, **opts)
    if name == "llamaparse":
        from src.parsers.llamaparse import LlamaParseParser

        return LlamaParseParser(chunker=chunker, **opts)
    if name == "unstructured":
        from src.parsers.unstructured import UnstructuredParser

        return UnstructuredParser(chunker=chunker, **opts)
    raise ValueError(f"未知 parser: {name}")
