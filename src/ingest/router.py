"""按扩展名把文件路由到 Parser + Chunker。

匹配顺序：``ingest.routing`` 自上而下，先命中先生效；
都未命中则用 ``ingest.default``。后续可扩展 mime / doc_type，不必改 ingest 主流程。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.parsers.factory import build_parser
from src.parsers.protocol import Parser


@dataclass(frozen=True)
class RouteRule:
    """一条路由：扩展名集合 → 解析器名 + 可选切片策略名。"""

    exts: frozenset[str]
    parser: str
    chunker: str | None = None


class ParserRouter:
    """惰性构造解析器，并按 (parser, chunker) 缓存，避免每文件 new 一次。"""

    def __init__(
        self,
        cfg: dict,
        rules: list[RouteRule],
        *,
        default_parser: str,
        default_chunker: str | None = None,
    ) -> None:
        self._cfg = cfg
        self._rules = rules
        self._default_parser = default_parser
        self._default_chunker = default_chunker
        self._cache: dict[tuple[str, str | None], Parser] = {}

    def _match(self, path: Path) -> tuple[str, str | None]:
        ext = path.suffix.lstrip(".").lower()
        for rule in self._rules:
            if ext in rule.exts:
                return rule.parser, rule.chunker
        return self._default_parser, self._default_chunker

    def resolve(self, path: Path) -> Parser:
        """返回该路径应使用的 Parser（已绑定对应 Chunker）。"""
        name, chunker = self._match(path)
        key = (name, chunker)
        if key not in self._cache:
            # 未实现的 parser 在首次 resolve 时抛 NotImplementedError，由 ingest 记失败
            self._cache[key] = build_parser(self._cfg, name=name, chunker_name=chunker)
        return self._cache[key]
