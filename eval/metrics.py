"""Retrieval metrics, as plain functions over ranked lists of ids.

Kept free of any pipeline imports so they can be unit-tested against hand
worked examples - a metric you cannot check by hand is a metric you cannot
trust when it tells you a change helped.

All functions take `retrieved` in rank order (best first) and `relevant` as
the set of ids that count as correct, i.e. binary relevance.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _prefix(retrieved: Sequence[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    return list(retrieved[:k])


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1.0 if any relevant document appears in the top k, else 0.0."""
    relevant = set(relevant)
    return 1.0 if relevant & set(_prefix(retrieved, k)) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the relevant documents that made it into the top k."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    found = relevant & set(_prefix(retrieved, k))
    return len(found) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Fraction of the top k that is relevant.

    The denominator is k, not the number retrieved: a system that returns two
    documents when asked for five should not be rewarded for the shortfall.
    """
    relevant = set(relevant)
    top = _prefix(retrieved, k)
    if not top:
        return 0.0
    return sum(1 for doc in top if doc in relevant) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    """1 / rank of the first relevant document; 0.0 if none is retrieved.

    Averaged over a query set this is MRR. It is the right metric when the
    reader only looks at the first good result.
    """
    relevant = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Normalised discounted cumulative gain with binary relevance.

    Unlike recall, nDCG cares *where* in the ranking a hit landed: moving a
    relevant document from position 5 to position 1 improves nDCG while
    leaving recall@5 untouched.
    """
    relevant = set(relevant)
    if not relevant:
        return 0.0
    top = _prefix(retrieved, k)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, doc in enumerate(top, start=1)
        if doc in relevant
    )
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def summarize(
    per_query: list[dict[str, float]], keys: Sequence[str] | None = None
) -> dict[str, float]:
    """Average each metric across queries."""
    if not per_query:
        return {}
    keys = keys or sorted(per_query[0])
    return {key: mean(q.get(key, 0.0) for q in per_query) for key in keys}
