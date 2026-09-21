"""统一日志：loguru。脚本入口或 ``load_config`` 时调用 ``setup_logging()``。"""

from __future__ import annotations

import os
import sys

from loguru import logger

_CONFIGURED = False


def setup_logging(level: str | None = None) -> None:
    """配置 stderr sink；可重复调用（仅首次生效）。

    级别：环境变量 ``LOG_LEVEL``（默认 INFO），或入参覆盖。
    ``enqueue=True`` 便于入库多线程写日志。
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logger.remove()
    logger.add(
        sys.stderr,
        level=lvl,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level:<7}</level> | "
            "<cyan>{name}</cyan> - <level>{message}</level>"
        ),
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
    _CONFIGURED = True


__all__ = ["logger", "setup_logging"]
