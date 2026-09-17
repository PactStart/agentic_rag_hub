class QdrantSparseIndex:
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "sparse.backend=qdrant_sparse 尚未实现。沙盒请用 memory_jsonl。"
        )
