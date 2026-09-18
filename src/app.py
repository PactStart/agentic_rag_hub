"""Gradio：切换角色、答案带出处。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _bypass_local_proxy() -> None:
    """macOS 开了系统/终端代理时，Gradio 探活 localhost 会 502。"""
    extra = ("127.0.0.1", "localhost", "0.0.0.0", "::1")
    for key in ("NO_PROXY", "no_proxy"):
        cur = {p.strip() for p in os.environ.get(key, "").split(",") if p.strip()}
        cur.update(extra)
        os.environ[key] = ",".join(sorted(cur))


_bypass_local_proxy()

import gradio as gr

from src.agent.graph import invoke_rag
from src.config import load_config
from src.logging_config import setup_logging
from loguru import logger

ROLES = {
    "员工": ["employee"],
    "人事": ["hr", "employee"],
    "财务": ["finance", "employee"],
}


def format_answer(result: dict) -> str:
    answer = result.get("answer") or ""
    hits = result.get("hits") or []
    if not hits:
        return answer
    cites = "\n".join(
        f"- [{i}] {h.get('source', '')} {h.get('section') or ''}".rstrip()
        for i, h in enumerate(hits, start=1)
    )
    return answer + "\n\n来源：\n" + cites


def chat(message: str, history: list, role_label: str) -> str:
    load_config()
    roles = ROLES.get(role_label, ["employee"])
    out = invoke_rag(message, roles=roles)
    return format_answer(out)


def main() -> None:
    setup_logging()
    load_config()
    # 本机用 127.0.0.1，避免启动探活走 localhost→代理；局域网演示可改 0.0.0.0
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    logger.info("启动 Gradio {}:{}", host, port)
    app = gr.ChatInterface(
        fn=chat,
        type="messages",
        additional_inputs=[gr.Dropdown(list(ROLES), value="员工", label="当前角色")],
        title="企业内部知识助手（沙盒）",
        description="同一问题换角色：员工看不到财务/人事机密。来源以检索结果为准。",
    )
    app.launch(server_name=host, server_port=port)


if __name__ == "__main__":
    main()
