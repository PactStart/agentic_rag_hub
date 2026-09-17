"""从问题里匹配实体名，查 1～2 跳邻居，拼成可生成的段落。"""

from __future__ import annotations

from src.pipeline_graph.extractor import load_triples
from src.pipeline_graph.graph_store import GraphStore
from src.stores.protocol import Hit, visible_to

GRAPH_HINTS = ("谁负责", "哪个部门", "什么关系", "汇报", "负责什么", "和谁有关", "归属")


def should_use_graph(query: str) -> bool:
    return any(h in query for h in GRAPH_HINTS)


def _seed_names(query: str, catalog: list[str]) -> list[str]:
    hits = [name for name in catalog if name and name in query]
    return sorted(hits, key=len, reverse=True)[:8]


def _catalog() -> list[str]:
    names: set[str] = set()
    for row in load_triples():
        names.add(row["h"])
        names.add(row["t"])
    return list(names)


def graph_retrieve(query: str, roles: list[str] | None = None, top_k: int = 10) -> list[Hit]:
    names = _seed_names(query, _catalog())
    if not names:
        return []
    store = GraphStore()
    try:
        rows = store.neighbors(names, hops=2, limit=max(top_k * 3, 12))
    finally:
        store.close()
    hits: list[Hit] = []
    for i, row in enumerate(rows):
        acl = list(row.get("acl") or ["all"])
        if not visible_to(acl, roles or []):
            continue
        text = f"{row['h']} -[{row['rel']}]-> {row['t']}"
        hits.append(
            {
                "chunk_id": f"graph:{row['h']}:{row['rel']}:{row['t']}:{i}",
                "text": text,
                "source": row.get("source") or "org_graph",
                "score": 1.0 / (len(hits) + 1),
                "acl": acl,
                "page": None,
                "section": "图谱",
            }
        )
        if len(hits) >= top_k:
            break
    return hits
