"""按 yaml 规则过滤一路召回。RRF 不要走这里（量纲已经丢掉了）。"""

from __future__ import annotations

from src.stores.protocol import Hit

_BOUND_KEYS = ("gt", "gte", "lt", "lte", "relative_gte")


def _num(value: object) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def apply_score_filter(hits: list[Hit], spec: dict | None) -> list[Hit]:
    """
    spec 示例（全可空，空则不过滤）::

        gt: 0              # score > 0
        gte: 0.35          # score >= 0.35
        lt / lte: ...
        relative_gte: 0.5  # score >= 本路最高分 * 0.5
        keep_at_least: 0   # 滤完不足时，从原列表按分数补回 N 条
    """
    if not hits:
        return []
    spec = spec or {}
    bounds = {key: _num(spec.get(key)) for key in _BOUND_KEYS}
    if all(v is None for v in bounds.values()):
        return list(hits)

    peak = max(float(h.get("score") or 0.0) for h in hits)
    kept: list[Hit] = []
    for hit in hits:
        score = float(hit.get("score") or 0.0)
        if bounds["gt"] is not None and not score > bounds["gt"]:
            continue
        if bounds["gte"] is not None and not score >= bounds["gte"]:
            continue
        if bounds["lt"] is not None and not score < bounds["lt"]:
            continue
        if bounds["lte"] is not None and not score <= bounds["lte"]:
            continue
        rel = bounds["relative_gte"]
        if rel is not None:
            if peak <= 0 or score < peak * rel:
                continue
        kept.append(hit)

    floor = int(spec.get("keep_at_least") or 0)
    if floor <= 0 or len(kept) >= floor:
        return kept
    seen = {(h.get("chunk_id"), h.get("text")) for h in kept}
    for hit in hits:
        if len(kept) >= floor:
            break
        key = (hit.get("chunk_id"), hit.get("text"))
        if key in seen:
            continue
        seen.add(key)
        kept.append(hit)
    return kept
