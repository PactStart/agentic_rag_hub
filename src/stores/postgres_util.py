"""Postgres 连接辅助：统一 DSN。"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row


def normalize_dsn(dsn: str | None) -> str:
    """接受 postgresql:// 或 SQLAlchemy 风格 postgresql+psycopg://。"""
    raw = (dsn or os.environ.get("POSTGRES_DSN") or "").strip()
    if not raw:
        raise RuntimeError(
            "未设置 Postgres DSN。请在 yaml 写 dsn，或 .env 写 POSTGRES_DSN="
            "postgresql://postgres:postgres@127.0.0.1:5432/rag"
        )
    if raw.startswith("postgresql+psycopg://"):
        raw = "postgresql://" + raw[len("postgresql+psycopg://") :]
    elif raw.startswith("postgresql+psycopg2://"):
        raw = "postgresql://" + raw[len("postgresql+psycopg2://") :]
    elif raw.startswith("postgres://"):
        raw = "postgresql://" + raw[len("postgres://") :]
    return raw


def quote_ident(name: str) -> str:
    """安全引用表名（仅允许字母数字下划线）。"""
    cleaned = (name or "").strip()
    if not cleaned or not cleaned.replace("_", "").isalnum():
        raise ValueError(f"非法表名: {name!r}")
    return f'"{cleaned}"'


@contextmanager
def connect(dsn: str) -> Iterator[psycopg.Connection]:
    with psycopg.connect(dsn, row_factory=dict_row) as conn:
        yield conn
