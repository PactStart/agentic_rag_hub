"""SQLAlchemy 引擎与会话。DSN 统一为 mysql+pymysql://…"""

from __future__ import annotations

from functools import lru_cache
from urllib.parse import unquote, urlparse

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker


def normalize_mysql_dsn(dsn: str) -> str:
    """mysql:// → mysql+pymysql://，供 SQLAlchemy 使用。"""
    raw = (dsn or "").strip()
    if not raw:
        raise ValueError(
            "MYSQL_DSN 为空。请在 .env 填写，例如 mysql://root:password@127.0.0.1:3306/rag_hub"
        )
    if "://" not in raw:
        raw = "mysql://" + raw
    parsed = urlparse(raw)
    if parsed.scheme not in ("mysql", "mysql+pymysql"):
        raise ValueError(f"不支持的 MYSQL_DSN scheme: {parsed.scheme}")
    db = (parsed.path or "").lstrip("/")
    if not db:
        raise ValueError("MYSQL_DSN 必须带库名，例如 .../rag_hub")
    user = unquote(parsed.username or "root")
    password = unquote(parsed.password or "")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 3306)
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"mysql+pymysql://{auth}{host}:{port}/{db}?charset=utf8mb4"


def database_name_from_dsn(dsn: str) -> str:
    url = normalize_mysql_dsn(dsn)
    return urlparse(url).path.lstrip("/").split("?")[0]


def server_dsn_without_database(dsn: str) -> str:
    """连到 MySQL 实例（无库），用于 CREATE DATABASE。"""
    url = normalize_mysql_dsn(dsn)
    parsed = urlparse(url)
    user = unquote(parsed.username or "root")
    password = unquote(parsed.password or "")
    host = parsed.hostname or "127.0.0.1"
    port = int(parsed.port or 3306)
    auth = f"{user}:{password}@" if password else f"{user}@"
    return f"mysql+pymysql://{auth}{host}:{port}/?charset=utf8mb4"


@lru_cache(maxsize=4)
def get_engine(dsn: str) -> Engine:
    url = normalize_mysql_dsn(dsn)
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )


def make_session_factory(dsn: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(dsn), autoflush=False, autocommit=False, future=True)


def ping_engine(dsn: str) -> None:
    engine = get_engine(dsn)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
