"""入库数据源：只负责「发现待处理文件」，与解析器解耦。

企业侧常见实现：对象存储事件、上传 API、待处理队列表。
本仓库沙盒用 ``LocalGlobSource``；换来源时实现 ``IngestSource`` 即可，不必改 Parser。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.config import project_root


class IngestSource(Protocol):
    def list_files(self) -> list[Path]:
        """返回本轮待入库的绝对路径列表（去重、排序由实现决定）。"""
        ...


class LocalGlobSource:
    """相对项目根目录的 glob 数据源（沙盒 / 本机评测）。

    ``patterns`` 支持单个字符串或列表，例如::

        data/sandbox/**/*.md
        [data/raw/**/*.txt, data/sandbox/**/*.md]
    """

    def __init__(self, patterns: str | list[str]) -> None:
        if isinstance(patterns, str):
            patterns = [patterns]
        self.patterns = [p for p in patterns if p]
        if not self.patterns:
            raise ValueError("LocalGlobSource 至少需要一个 glob 模式")

    def list_files(self) -> list[Path]:
        root = project_root()
        found: set[Path] = set()
        for pattern in self.patterns:
            # Path.glob 支持 **；多模式取并集，避免同一文件重复入库
            for path in root.glob(pattern):
                if path.is_file():
                    found.add(path.resolve())
        return sorted(found)
