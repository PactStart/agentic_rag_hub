"""知识库文档与入库流水 ORM。DDL 由 Alembic 管理。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, DateTime, Index, Integer, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.mysql import JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class KbDocument(Base):
    """当前生效的文档账本：一租户一 source 一行，用 content_hash 判断是否需要重嵌。"""

    __tablename__ = "kb_document"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source", name="uk_tenant_source"),
        Index("idx_kb_doc_status", "tenant_id", "status"),
        Index("idx_kb_doc_hash", "content_hash"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "comment": "知识库文档账本"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default",
        comment="租户 ID，多租户隔离；默认 default",
    )
    source: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        comment="文档来源标识（相对路径或 URI），与 ingest Source 产出一致",
    )
    title: Mapped[str | None] = mapped_column(
        String(512), nullable=True, comment="文档标题（可选，展示用）"
    )
    content_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        comment="正文内容哈希；未变则增量入库跳过",
    )
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        comment="状态：active | deleted | failed",
    )
    acl_json: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="可见角色列表 JSON，如 [\"all\",\"finance\"]"
    )
    chunk_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, comment="当前写入索引的切片数"
    )
    chunk_ids_json: Mapped[Any | None] = mapped_column(
        JSON, nullable=True, comment="切片 ID 列表 JSON，删除/更新时用于对账清理"
    )
    sparse_backend: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="写入时的稀疏后端名，如 elasticsearch"
    )
    vector_backend: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="写入时的向量后端名，如 milvus / qdrant"
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="最近一次失败原因（status=failed 时）"
    )
    ingested_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="最近一次成功入库时间"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        comment="行创建时间",
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
        comment="行更新时间",
    )


class KbIngestJob(Base):
    """每次入库作业流水：成功/失败都记一条，便于审计与排错。"""

    __tablename__ = "kb_ingest_job"
    __table_args__ = (
        Index("idx_kb_job_source", "tenant_id", "source", "started_at"),
        Index("idx_kb_job_status", "status", "started_at"),
        {"mysql_engine": "InnoDB", "mysql_charset": "utf8mb4", "comment": "入库任务流水"},
    )

    id: Mapped[int] = mapped_column(
        BigInteger, primary_key=True, autoincrement=True, comment="主键"
    )
    tenant_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="default",
        comment="租户 ID，与 kb_document.tenant_id 对应",
    )
    source: Mapped[str] = mapped_column(
        String(512), nullable=False, comment="本次作业处理的文档 source"
    )
    trigger_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="manual",
        comment="触发方式：manual | schedule | webhook 等",
    )
    action: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="动作：add | update | delete | skip 等",
    )
    from_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="变更前 content_hash（新增可空）"
    )
    to_hash: Mapped[str | None] = mapped_column(
        String(64), nullable=True, comment="变更后 content_hash（删除可空）"
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, comment="作业结果：success | failed"
    )
    chunk_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, comment="本次写入/删除涉及的切片数"
    )
    error_message: Mapped[str | None] = mapped_column(
        Text, nullable=True, comment="失败详情"
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, comment="作业开始时间"
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime, nullable=True, comment="作业结束时间"
    )
