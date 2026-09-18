"""初始化 MySQL：库不存在则建库，再 alembic upgrade head。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.ledger.db import (
    database_name_from_dsn,
    normalize_mysql_dsn,
    server_dsn_without_database,
)
from src.logging_config import setup_logging
from loguru import logger
from sqlalchemy import create_engine, text

VERSIONS_DIR = ROOT / "alembic" / "versions"


def _dsn() -> str:
    cfg = load_config()
    dsn = ((cfg.get("ledger") or {}).get("mysql") or {}).get("dsn") or ""
    return dsn or os.environ.get("MYSQL_DSN", "")


def ensure_database(dsn: str) -> None:
    db_name = database_name_from_dsn(dsn)
    engine = create_engine(server_dsn_without_database(dsn), future=True)
    with engine.connect() as conn:
        conn.execute(
            text(
                f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
                "DEFAULT CHARACTER SET utf8mb4 DEFAULT COLLATE utf8mb4_unicode_ci"
            )
        )
        conn.commit()
    engine.dispose()
    logger.info("库已就绪：{}", db_name)


def main() -> None:
    setup_logging()
    dsn = _dsn()
    logger.info("目标：{}", normalize_mysql_dsn(dsn).split("@")[-1])

    if not any(VERSIONS_DIR.glob("*.py")):
        raise SystemExit(
            f"找不到迁移脚本：{VERSIONS_DIR}/*.py。"
            "不要清空 versions；新迁移用：uv run alembic revision --autogenerate -m \"...\""
        )

    ensure_database(dsn)
    cmd = [sys.executable, "-m", "alembic", "upgrade", "head"]
    logger.info("执行：{}", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT)
    logger.info("完成。可执行：uv run python scripts/ingest.py")


if __name__ == "__main__":
    main()
