"""按 config/rag.yaml（及 SPARSE_BACKEND / LEDGER_BACKEND）探活当前栈。

覆盖：硅基 embed/rerank + sparse / vector / ledger 所选后端。
未实现的 backend 记为失败并提示，不会静默跳过。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config, project_root
from src.llm.siliconflow import embed_texts, rerank
from src.logging_config import setup_logging
from loguru import logger


def _ok(name: str, detail: str) -> None:
    logger.info("OK  {}: {}", name, detail)


def _fail(name: str, detail: str) -> None:
    logger.error("FAIL {}: {}", name, detail)


def probe_siliconflow(cfg: dict) -> bool:
    dim = int(cfg.get("embed_dim") or 1024)
    try:
        vec = embed_texts(["试用期几个月"], is_query=True)[0]
        if len(vec) != dim:
            _fail("siliconflow.embed", f"维数={len(vec)}，期望 embed_dim={dim}")
            return False
        ranked = rerank("苹果", ["苹果是水果", "笔记本电脑"], top_n=2)
        if not ranked or int(ranked[0]["index"]) != 0:
            _fail("siliconflow.rerank", f"相关句应排第一，得到 {ranked}")
            return False
        _ok("siliconflow", f"embed dim={len(vec)}；rerank 正常")
        return True
    except Exception as exc:  # noqa: BLE001 — 探活要汇总所有失败
        _fail("siliconflow", f"{exc.__class__.__name__}: {exc}")
        return False


def probe_sparse(cfg: dict) -> bool:
    block = cfg.get("sparse") or {}
    name = os.environ.get("SPARSE_BACKEND") or block.get("backend") or "memory_jsonl"
    label = f"sparse[{name}]"
    try:
        if name == "memory_jsonl":
            path = Path((block.get("memory_jsonl") or {}).get("path") or "data/corpus/chunks.jsonl")
            if not path.is_absolute():
                path = project_root() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            _ok(label, f"path={path}（父目录可写）")
            return True

        if name == "elasticsearch":
            from elasticsearch import Elasticsearch

            es = block.get("elasticsearch") or {}
            url = es.get("url") or os.environ.get("ELASTICSEARCH_URL") or "http://127.0.0.1:9200"
            kwargs: dict = {"hosts": [url], "request_timeout": 15}
            api_key = os.environ.get("ELASTICSEARCH_API_KEY") or ""
            user = os.environ.get("ELASTICSEARCH_USER") or ""
            password = os.environ.get("ELASTICSEARCH_PASSWORD") or ""
            if api_key:
                kwargs["api_key"] = api_key
            elif user:
                kwargs["basic_auth"] = (user, password)
            client = Elasticsearch(**kwargs)
            if not client.ping():
                _fail(label, f"ping 失败 url={url}")
                return False
            info = client.info()
            version = (info.get("version") or {}).get("number", "?")
            plugins = []
            try:
                plugins = [p.get("component") for p in client.cat.plugins(format="json") or []]
            except Exception:  # noqa: BLE001
                pass
            ik = "analysis-ik" in plugins
            index = es.get("index") or os.environ.get("ELASTICSEARCH_INDEX") or "rag_chunks"
            _ok(
                label,
                f"url={url} version={version} index={index} ik={'yes' if ik else 'no（中文召回会差）'}",
            )
            return True

        if name == "postgres_fts":
            from src.stores.factory import build_sparse

            idx = build_sparse(cfg)
            # 触发建表 / 连库
            _ = idx.search("探活", ["all"], top_k=1)
            p = block.get("postgres_fts") or {}
            table = p.get("table") or os.environ.get("POSTGRES_FTS_TABLE") or "rag_chunks"
            _ok(label, f"table={table}（jieba+tsvector simple）")
            return True

        if name == "qdrant_sparse":
            _fail(label, "插件未实现（见 README 局限）")
            return False

        _fail(label, f"未知 sparse.backend: {name}")
        return False
    except Exception as exc:  # noqa: BLE001
        _fail(label, f"{exc.__class__.__name__}: {exc}")
        return False


def probe_vector(cfg: dict) -> bool:
    block = cfg.get("vector") or {}
    name = block.get("backend") or "qdrant"
    label = f"vector[{name}]"
    try:
        if name == "qdrant":
            from qdrant_client import QdrantClient

            q = block.get("qdrant") or {}
            url = q.get("url") or os.environ.get("QDRANT_URL") or "http://127.0.0.1:6333"
            api_key = q.get("api_key") or os.environ.get("QDRANT_API_KEY") or ""
            client = QdrantClient(
                url=url,
                api_key=api_key or None,
                timeout=15,
                prefer_grpc=False,
                check_compatibility=False,
                trust_env=False,
            )
            names = [c.name for c in client.get_collections().collections]
            coll = q.get("collection") or os.environ.get("QDRANT_COLLECTION") or "enterprise_rag"
            _ok(label, f"url={url} collection={coll} existing={names[:8]}")
            return True

        if name == "milvus":
            from pymilvus import MilvusClient

            m = block.get("milvus") or {}
            uri = m.get("uri") or os.environ.get("MILVUS_URI") or "http://127.0.0.1:19530"
            token = m.get("token") or os.environ.get("MILVUS_TOKEN") or ""
            client = MilvusClient(uri=uri, token=token or None)
            names = client.list_collections()
            coll = (
                m.get("collection")
                or os.environ.get("MILVUS_COLLECTION")
                or "enterprise_rag"
            )
            _ok(label, f"uri={uri} collection={coll} existing={names[:8]}")
            return True

        if name == "pgvector":
            from src.stores.factory import build_vector

            idx = build_vector(cfg)
            dim = int(cfg.get("embed_dim") or 1024)
            _ = idx.search([0.0] * dim, ["all"], top_k=1)
            p = block.get("pgvector") or {}
            table = p.get("table") or os.environ.get("PGVECTOR_TABLE") or "rag_embeddings"
            _ok(label, f"table={table} dim={dim}（extension vector）")
            return True

        _fail(label, f"未知 vector.backend: {name}")
        return False
    except Exception as exc:  # noqa: BLE001
        _fail(label, f"{exc.__class__.__name__}: {exc}")
        return False


def probe_ledger(cfg: dict) -> bool:
    block = cfg.get("ledger") or {}
    name = os.environ.get("LEDGER_BACKEND") or block.get("backend") or "mysql"
    label = f"ledger[{name}]"
    try:
        if name == "mysql":
            from src.ledger.db import ping_engine

            mysql = block.get("mysql") or {}
            dsn = mysql.get("dsn") or os.environ.get("MYSQL_DSN") or ""
            ping_engine(dsn)
            # 不打印密码：只显示 @ 后半段
            host = dsn.split("@")[-1] if "@" in dsn else "(见 MYSQL_DSN)"
            _ok(label, f"SELECT 1 通过 → {host}")
            return True

        if name == "json":
            path = Path((block.get("json") or {}).get("path") or "data/corpus/ingest_manifest.json")
            if not path.is_absolute():
                path = project_root() / path
            path.parent.mkdir(parents=True, exist_ok=True)
            _ok(label, f"path={path}（父目录可写）")
            return True

        _fail(label, f"未知 ledger.backend: {name}")
        return False
    except Exception as exc:  # noqa: BLE001
        _fail(label, f"{exc.__class__.__name__}: {exc}")
        return False


def main() -> None:
    import argparse

    setup_logging()
    parser = argparse.ArgumentParser(description="按配置探活硅基 + sparse/vector/ledger")
    parser.add_argument("--config", default=None, help="默认 config/rag.yaml 或 RAG_CONFIG")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sparse = os.environ.get("SPARSE_BACKEND") or (cfg.get("sparse") or {}).get("backend")
    vector = (cfg.get("vector") or {}).get("backend")
    ledger = os.environ.get("LEDGER_BACKEND") or (cfg.get("ledger") or {}).get("backend")
    logger.info("配置: {}", cfg.get("_config_path"))
    logger.info(
        "栈: sparse={}  vector={}  ledger={}  embed_dim={}",
        sparse,
        vector,
        ledger,
        cfg.get("embed_dim"),
    )
    if (cfg.get("ledger") or {}).get("tenant_id"):
        logger.info("ledger.tenant_id={}", (cfg.get("ledger") or {}).get("tenant_id"))
    logger.info("探活…")

    results = [
        ("siliconflow", probe_siliconflow(cfg)),
        ("sparse", probe_sparse(cfg)),
        ("vector", probe_vector(cfg)),
        ("ledger", probe_ledger(cfg)),
    ]
    failed = [name for name, ok in results if not ok]
    if failed:
        raise SystemExit(f"探活失败: {', '.join(failed)}。请对照 README §10 起服务或改配置。")
    logger.info("探活成功（当前 yaml 用到的组件均可连）")


if __name__ == "__main__":
    main()
