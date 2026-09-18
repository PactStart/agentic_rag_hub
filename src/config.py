"""加载 config/rag.yaml，并把 ${ENV} 替换成环境变量。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")
_ROOT = Path(__file__).resolve().parents[1]


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            return os.environ.get(match.group(1), "")

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    from src.logging_config import setup_logging

    setup_logging()
    load_dotenv(_ROOT / ".env")
    cfg_path = Path(path or os.environ.get("RAG_CONFIG", "config/rag.yaml"))
    if not cfg_path.is_absolute():
        cfg_path = _ROOT / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件：{cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    cfg = _expand(raw)
    cfg["_root"] = str(_ROOT)
    cfg["_config_path"] = str(cfg_path)
    return cfg


def project_root() -> Path:
    return _ROOT
