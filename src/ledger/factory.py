"""构造入库账本。路径等参数只读 ``ledger.*``，无顶层旧字段回退。"""

from __future__ import annotations

import os

from src.ledger.protocol import IngestLedger


def build_ledger(cfg: dict) -> IngestLedger:
    block = cfg.get("ledger") or {}
    name = os.environ.get("LEDGER_BACKEND") or block.get("backend") or "mysql"
    tenant = block.get("tenant_id") or os.environ.get("RAG_TENANT_ID") or "default"
    if name == "mysql":
        from src.ledger.mysql import MysqlLedger

        mysql = block.get("mysql") or {}
        dsn = mysql.get("dsn") or os.environ.get("MYSQL_DSN") or ""
        return MysqlLedger(dsn=dsn, tenant_id=str(tenant))
    if name == "json":
        from src.ledger.json_file import JsonFileLedger

        path = (block.get("json") or {}).get("path")
        if not path:
            raise ValueError("ledger.backend=json 时必须配置 ledger.json.path")
        return JsonFileLedger(path=path)
    raise ValueError(f"未知 ledger.backend: {name}（支持 mysql | json）")
