"""Hybrid retrieval: dense + lexical, fused, optionally reranked.

The pipeline is deliberately two-stage. Stage one casts a wide net cheaply
(``dense_top_k`` + ``lexical_top_k`` candidates) because recall lost here can
never be recovered later. Stage two narrows to ``final_top_k``, which is what
actually reaches the model's context window.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .embeddings import Embedder
from .fusion import reciprocal_rank_fusion
from .lexical import BM25Index
from .models import ScoredChunk
from .vectorstore import VectorStore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalDebug:
    """Per-query breakdown, returned alongside results for evaluation."""

    dense_hits: int
    lexical_hits: int
    fused_hits: int
    reranked: bool


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: Embedder,
        settings: Settings,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.settings = settings
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        sources: tuple[str, ...] = ("dense", "lexical"),
    ) -> tuple[list[ScoredChunk], RetrievalDebug]:
        """Retrieve the best passages for a query.

        `sources` selects which retrievers contribute. Restricting it to one
        is how the evaluation harness measures what fusion is actually worth;
        it is also useful when debugging a single bad query.
        """
        settings = self.settings
        top_k = top_k or settings.final_top_k

        dense: list[ScoredChunk] = []
        lexical: list[ScoredChunk] = []
        rankings: dict[str, list[ScoredChunk]] = {}

        if "dense" in sources:
            dense = self.vector_store.search(
                self.embedder.embed_query(query), settings.dense_top_k
            )
            rankings["dense"] = dense
        if "lexical" in sources:
            lexical = self.bm25_index.search(query, settings.lexical_top_k)
            rankings["lexical"] = lexical
        if not rankings:
            raise ValueError("At least one retrieval source must be enabled")

        fused = reciprocal_rank_fusion(
            rankings,
            k=settings.rrf_k,
            weights={
                "dense": settings.dense_weight,
                "lexical": settings.lexical_weight,
            },
        )

        candidates = fused[: max(top_k * 4, top_k)]
        if self.reranker is not None and candidates:
            candidates = self.reranker.rerank(query, candidates)

        debug = RetrievalDebug(
            dense_hits=len(dense),
            lexical_hits=len(lexical),
            fused_hits=len(fused),
            reranked=self.reranker is not None,
        )
        return candidates[:top_k], debug


class CrossEncoderReranker:
    """Optional second-pass reranker.

    A bi-encoder embeds query and document independently, so it never sees
    them together. A cross-encoder scores the pair jointly and is markedly
    more accurate - but it runs one forward pass per candidate, which is why
    it only ever sees the shortlist, never the whole corpus.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)
        self.name = model_name

    def rerank(self, query: str, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        pairs = [(query, c.chunk.text) for c in candidates]
        scores = self._model.predict(pairs)
        for candidate, score in zip(candidates, scores, strict=True):
            candidate.component_scores["rerank"] = float(score)
            candidate.score = float(score)
        return sorted(
            candidates, key=lambda c: (-c.score, c.chunk.chunk_id)
        )
