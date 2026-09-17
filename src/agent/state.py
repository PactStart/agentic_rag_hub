from typing import TypedDict

from src.stores.protocol import Hit


class RAGState(TypedDict, total=False):
    query: str
    rewritten: str
    roles: list[str]
    hits: list[Hit]
    answer: str
    use_rerank: bool
