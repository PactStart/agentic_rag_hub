"""入库数据源：发现「逻辑文档」，与解析器解耦。

- ``LocalGlobSource``：每个文件一篇（沙盒 md 等）
- ``LineCorpusSource``：扫描目录下文件，**每一行一篇**（CRUD 80000_docs）
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.config import project_root


@dataclass(frozen=True)
class IngestDoc:
    """一轮待入库的一篇逻辑文档。

    - 有 ``path``、无 ``text``：走 ParserRouter（读文件再切片）
    - 有 ``text``：正文已在内存，只用默认 Chunker 切片
    """

    source: str
    path: Path | None = None
    text: str | None = None
    title: str | None = None


class IngestSource:
    def list_docs(self, *, after: str | None = None) -> list[IngestDoc]:
        """``after`` 为已完成的 source id 时，只返回排在它后面的文档（不含自身）。"""
        raise NotImplementedError


class LocalGlobSource(IngestSource):
    """相对项目根目录的 glob：每个匹配文件 = 一篇文档。"""

    def __init__(self, patterns: str | list[str]) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = [p for p in patterns if p]
        if not self.patterns:
            raise ValueError("LocalGlobSource 至少需要一个 glob 模式")

    def list_docs(self, *, after: str | None = None) -> list[IngestDoc]:
        root = project_root()
        found: set[Path] = set()
        for pattern in self.patterns:
            for path in root.glob(pattern):
                if path.is_file():
                    found.add(path.resolve())
        # source 占位；真正 source 以 Parser.parse 为准
        docs = [
            IngestDoc(source=path.name, path=path)
            for path in sorted(found)
        ]
        if after:
            docs = [d for d in docs if d.source > after]
        return docs


class LineCorpusSource(IngestSource):
    """扫描目录下普通文件，每一非空行 = 一篇文档。

    ``source`` 稳定为 ``{文件名}__{行号六位}``，便于账本与 hit 追溯。
    """

    def __init__(self, directory: str, *, encoding: str = "utf-8") -> None:
        self.directory = (directory or "").strip()
        if not self.directory:
            raise ValueError("line_corpus 需配置 ingest.source.dir")
        self.encoding = encoding or "utf-8"

    def list_docs(self, *, after: str | None = None) -> list[IngestDoc]:
        root = project_root()
        dir_path = Path(self.directory)
        if not dir_path.is_absolute():
            dir_path = root / dir_path
        if not dir_path.is_dir():
            raise FileNotFoundError(f"line_corpus 目录不存在：{dir_path}")

        after_name = ""
        if after:
            if "__" not in after:
                raise ValueError(
                    "line_corpus 的 --after 须为 {文件名}__{行号六位}，"
                    f"例如 documents_dup_part_14_part_3__000985，收到：{after}"
                )
            after_name = after.rsplit("__", 1)[0]

        docs: list[IngestDoc] = []
        files = sorted(p for p in dir_path.iterdir() if p.is_file())
        if not files:
            raise FileNotFoundError(f"line_corpus 目录下没有文件：{dir_path}")

        skipped_files = 0
        for path in files:
            # 文件名排序与 source 前缀一致；更早的合集整文件不读
            if after_name and path.name < after_name:
                skipped_files += 1
                continue
            with path.open(encoding=self.encoding, errors="replace") as fh:
                for i, line in enumerate(fh, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    doc_id = f"{path.name}__{i:06d}"
                    if after and doc_id <= after:
                        continue
                    docs.append(
                        IngestDoc(
                            source=doc_id,
                            path=path,
                            text=text,
                        )
                    )
        if after:
            logger.info(
                "续跑 --after {}：跳过更早文件 {} 个，待处理 {} 篇",
                after,
                skipped_files,
                len(docs),
            )
        return docs
