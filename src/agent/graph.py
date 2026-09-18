"""改写 → 检索（含 ACL）→ 生成。有限状态机，不自由规划。"""

from __future__ import annotations

import os

from langgraph.graph import END, StateGraph
from loguru import logger

from src.agent.nodes import generate_node, retrieve_node, rewrite_node
from src.agent.state import RAGState

_compiled = None


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def get_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_graph()
    return _compiled


def invoke_rag(query: str, roles: list[str], use_rerank: bool = True) -> RAGState:
    graph = get_graph()
    config: dict = {}
    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")
    if public and secret and host:
        try:
            # langfuse v4：CallbackHandler 在 langfuse.langchain；密钥由 Langfuse 客户端读环境变量
            from langfuse import Langfuse
            from langfuse.langchain import CallbackHandler

            Langfuse(public_key=public, secret_key=secret, host=host or None)
            config["callbacks"] = [CallbackHandler(public_key=public)]
        except Exception as exc:
            logger.warning("Langfuse 未启用（{}）", exc.__class__.__name__)
    return graph.invoke(
        {"query": query, "roles": roles, "use_rerank": use_rerank},
        config=config,
    )
