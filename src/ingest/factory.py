"""从 ``config/rag.yaml`` 的 ``ingest`` 段构造 Source 与 ParserRouter。

必填结构（无旧字段回退）::

    ingest:
      source:
        type: local_glob | line_corpus
        glob: ...                 # local_glob
        dir: data/corpus/80000_docs  # line_corpus
      routing: [{match: {ext: [...]}, parser: ..., chunker: ...}, ...]
      default: {parser: ..., chunker: ...}
"""

from __future__ import annotations

from src.ingest.router import ParserRouter, RouteRule
from src.ingest.source import IngestSource, LineCorpusSource, LocalGlobSource


def build_source(cfg: dict) -> IngestSource:
    ingest = cfg.get("ingest")
    if not isinstance(ingest, dict):
        raise ValueError("缺少 ingest 配置块")

    source = ingest.get("source") or {}
    stype = (source.get("type") or "local_glob").strip().lower()

    if stype == "local_glob":
        patterns = source.get("globs") or source.get("glob")
        if not patterns:
            raise ValueError("ingest.source 需配置 glob（字符串）或 globs（列表）")
        return LocalGlobSource(patterns)

    if stype == "line_corpus":
        directory = source.get("dir") or source.get("path") or source.get("directory")
        if not directory:
            raise ValueError("ingest.source.type=line_corpus 时需配置 dir")
        return LineCorpusSource(
            str(directory),
            encoding=str(source.get("encoding") or "utf-8"),
        )

    raise ValueError(
        f"未知 ingest.source.type: {stype}（支持 local_glob | line_corpus）"
    )


def build_router(cfg: dict) -> ParserRouter:
    ingest = cfg.get("ingest")
    if not isinstance(ingest, dict):
        raise ValueError("缺少 ingest 配置块")

    default = ingest.get("default") or {}
    default_parser = default.get("parser")
    if not default_parser:
        raise ValueError("ingest.default.parser 必填（未命中 routing 时的回退解析器）")

    rules: list[RouteRule] = []
    for item in list(ingest.get("routing") or []):
        match = item.get("match") or {}
        exts = match.get("ext") or match.get("exts") or []
        if isinstance(exts, str):
            exts = [exts]
        parser_name = item.get("parser")
        if not parser_name:
            raise ValueError(f"ingest.routing 项缺少 parser: {item}")
        if not exts:
            raise ValueError(f"ingest.routing 项缺少 match.ext: {item}")
        rules.append(
            RouteRule(
                exts=frozenset(e.lstrip(".").lower() for e in exts),
                parser=str(parser_name),
                chunker=item.get("chunker"),
            )
        )

    return ParserRouter(
        cfg,
        rules,
        default_parser=str(default_parser),
        default_chunker=default.get("chunker"),
    )
