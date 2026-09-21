from __future__ import annotations

import pytest

from rag.fusion import reciprocal_rank_fusion
from rag.models import Chunk, ScoredChunk


def scored(chunk_id: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="d",
            title="T",
            text=chunk_id,
            ordinal=0,
            source="d.md",
        ),
        score=score,
    )


class TestReciprocalRankFusion:
    def test_agreement_beats_a_single_first_place(self):
        """A document ranked 2nd by both should beat one ranked 1st by one only."""
        fused = reciprocal_rank_fusion(
            {
                "dense": [scored("solo", 0.9), scored("both", 0.8)],
                "lexical": [scored("other", 9.0), scored("both", 5.0)],
            },
            k=60,
        )
        assert fused[0].chunk.chunk_id == "both"

    def test_score_matches_the_formula(self):
        fused = reciprocal_rank_fusion(
            {"a": [scored("x", 1.0)], "b": [scored("x", 1.0)]}, k=60
        )
        assert fused[0].score == pytest.approx(2 * (1 / 61))

    def test_weights_shift_the_ordering(self):
        rankings = {
            "dense": [scored("d_top", 0.9)],
            "lexical": [scored("l_top", 9.0)],
        }
        assert reciprocal_rank_fusion(rankings, weights={"dense": 5.0})[0].chunk.chunk_id == "d_top"
        assert reciprocal_rank_fusion(rankings, weights={"lexical": 5.0})[0].chunk.chunk_id == "l_top"

    def test_component_scores_are_preserved_for_debugging(self):
        fused = reciprocal_rank_fusion(
            {"dense": [scored("x", 0.42)], "lexical": [scored("x", 7.5)]}, k=60
        )
        components = fused[0].component_scores
        assert components["dense"] == pytest.approx(0.42)
        assert components["lexical"] == pytest.approx(7.5)
        assert components["dense_rank"] == 1.0

    def test_raw_scores_do_not_leak_into_the_ranking(self):
        """BM25 scores are unbounded; only ranks may influence the result."""
        modest = reciprocal_rank_fusion(
            {"dense": [scored("a", 0.1)], "lexical": [scored("b", 2.0)]}, k=60
        )
        enormous = reciprocal_rank_fusion(
            {"dense": [scored("a", 0.1)], "lexical": [scored("b", 2000.0)]}, k=60
        )
        assert [s.chunk.chunk_id for s in modest] == [s.chunk.chunk_id for s in enormous]

    def test_ordering_is_deterministic_for_ties(self):
        rankings = {"dense": [scored("b", 1.0)], "lexical": [scored("a", 1.0)]}
        first = [s.chunk.chunk_id for s in reciprocal_rank_fusion(rankings)]
        for _ in range(5):
            assert [s.chunk.chunk_id for s in reciprocal_rank_fusion(rankings)] == first

    def test_single_source_preserves_its_order(self):
        fused = reciprocal_rank_fusion(
            {"dense": [scored("a", 3.0), scored("b", 2.0), scored("c", 1.0)]}
        )
        assert [s.chunk.chunk_id for s in fused] == ["a", "b", "c"]

    def test_empty_input(self):
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"dense": []}) == []

    def test_k_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            reciprocal_rank_fusion({"dense": [scored("a", 1.0)]}, k=0)
