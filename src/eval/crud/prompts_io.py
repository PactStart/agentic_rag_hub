"""读写 ``src/eval/crud/prompts/*.txt``。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from loguru import logger


def prompt_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=32)
def read_prompt(filename: str) -> str:
    path = prompt_dir() / filename
    if not path.is_file():
        logger.error("Prompt missing: {}", path)
        return ""
    return path.read_text(encoding="utf-8")
