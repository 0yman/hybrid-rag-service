"""Reciprocal Rank Fusion for combining ranked lists.

Dense cosine scores live in [-1, 1]; BM25 scores are unbounded and corpus
dependent. Adding them directly means normalising two incomparable scales and
then re-tuning that normalisation every time the corpus changes.

RRF sidesteps this by throwing away the scores and keeping only the ranks:

    score(d) = sum over lists of  weight / (k + rank(d))

``k`` (60 by default, from Cormack et al. 2009) damps the influence of the top
position so one confident-but-wrong list cannot dominate the fused ordering.
A document ranked well by both retrievers beats one ranked first by only one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import ScoredChunk


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[ScoredChunk]],
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[ScoredChunk]:
    """Fuse named ranked lists into one, best first.

    Args:
        rankings: retriever name -> its results, already in rank order.
        k: rank-damping constant. Larger flattens the contribution of rank 1.
        weights: optional per-retriever multiplier; defaults to 1.0 each.

    Returns:
        Fused results. Each carries its RRF score plus the raw component score
        from every retriever that found it, which is what makes a bad ranking
        debuggable after the fact.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    weights = weights or {}
    fused_scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    representative: dict[str, ScoredChunk] = {}

    for source, results in rankings.items():
        weight = float(weights.get(source, 1.0))
        for rank, scored in enumerate(results, start=1):
            chunk_id = scored.chunk.chunk_id
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k + rank)
            components.setdefault(chunk_id, {})[source] = scored.score
            components[chunk_id][f"{source}_rank"] = float(rank)
            representative.setdefault(chunk_id, scored)

    fused = [
        ScoredChunk(
            chunk=representative[chunk_id].chunk,
            score=score,
            component_scores=components[chunk_id],
        )
        for chunk_id, score in fused_scores.items()
    ]
    # Ties broken by chunk_id so the ordering is deterministic across runs -
    # non-determinism here makes evaluation results impossible to reproduce.
    fused.sort(key=lambda s: (-s.score, s.chunk.chunk_id))
    return fused
