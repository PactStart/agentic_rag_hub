class PostgresFtsIndex:
    def __init__(self, **kwargs) -> None:
        raise NotImplementedError(
            "sparse.backend=postgres_fts 尚未实现。沙盒请用 memory_jsonl。"
        )
