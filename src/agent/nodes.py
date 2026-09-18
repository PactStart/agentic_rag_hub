from loguru import logger

from src.agent.state import RAGState
from src.llm.generate import generate
from src.pipeline_graph.query import graph_retrieve, should_use_graph
from src.pipeline_hybrid.hybrid import hybrid_search


def rewrite_node(state: RAGState) -> dict:
    return {"rewritten": state.get("rewritten") or state["query"]}


def retrieve_node(state: RAGState) -> dict:
    query = state.get("rewritten") or state["query"]
    roles = list(state.get("roles") or ["employee"])
    result = hybrid_search(
        query,
        roles=roles,
        use_rerank=bool(state.get("use_rerank", True)),
    )
    hits = list(result["final"])
    if should_use_graph(query):
        try:
            graph_hits = graph_retrieve(query, roles=roles, top_k=5)
            hits = graph_hits + hits
        except Exception as exc:
            logger.warning("图谱检索跳过（{}）", exc.__class__.__name__)
    return {"hits": hits[:8]}


def generate_node(state: RAGState) -> dict:
    out = generate(state.get("rewritten") or state["query"], list(state.get("hits") or []))
    return {"answer": out["answer"]}
