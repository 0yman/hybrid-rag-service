"""Metrics are checked against hand-computed values.

If a metric is only ever compared to its own output, a sign error in it will
silently redefine what "better" means for the whole project.
"""

from __future__ import annotations

import math

import pytest

from metrics import (
    hit_rate_at_k,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarize,
)

RETRIEVED = ["a", "b", "c", "d", "e"]


class TestHitRate:
    def test_hit_inside_k(self):
        assert hit_rate_at_k(RETRIEVED, {"c"}, 3) == 1.0

    def test_hit_outside_k(self):
        assert hit_rate_at_k(RETRIEVED, {"e"}, 3) == 0.0

    def test_no_relevant_documents(self):
        assert hit_rate_at_k(RETRIEVED, set(), 3) == 0.0


class TestRecall:
    def test_partial(self):
        assert recall_at_k(RETRIEVED, {"a", "e"}, 3) == pytest.approx(0.5)

    def test_complete(self):
        assert recall_at_k(RETRIEVED, {"a", "b"}, 3) == pytest.approx(1.0)

    def test_empty_relevant_set(self):
        assert recall_at_k(RETRIEVED, set(), 5) == 0.0


class TestPrecision:
    def test_denominator_is_k_not_hits(self):
        assert precision_at_k(RETRIEVED, {"a"}, 5) == pytest.approx(0.2)

    def test_short_result_list_is_not_rewarded(self):
        """Returning 1 correct document when asked for 5 is 0.2, not 1.0."""
        assert precision_at_k(["a"], {"a"}, 5) == pytest.approx(0.2)


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "relevant,expected", [({"a"}, 1.0), ({"b"}, 0.5), ({"d"}, 0.25)]
    )
    def test_position_of_first_hit(self, relevant, expected):
        assert reciprocal_rank(RETRIEVED, relevant) == pytest.approx(expected)

    def test_uses_the_first_hit_only(self):
        assert reciprocal_rank(RETRIEVED, {"b", "d"}) == pytest.approx(0.5)

    def test_no_hit(self):
        assert reciprocal_rank(RETRIEVED, {"z"}) == 0.0


class TestNDCG:
    def test_perfect_ranking_is_one(self):
        assert ndcg_at_k(RETRIEVED, {"a", "b"}, 5) == pytest.approx(1.0)

    def test_hand_computed_value(self):
        # One relevant document at rank 3: DCG = 1/log2(4), IDCG = 1/log2(2) = 1.
        expected = 1 / math.log2(4)
        assert ndcg_at_k(RETRIEVED, {"c"}, 5) == pytest.approx(expected)

    def test_rewards_earlier_placement_where_recall_cannot(self):
        """Recall@5 is blind to position; nDCG is the metric that is not."""
        early = ["a", "x", "y", "z", "w"]
        late = ["x", "y", "z", "w", "a"]
        assert recall_at_k(early, {"a"}, 5) == recall_at_k(late, {"a"}, 5)
        assert ndcg_at_k(early, {"a"}, 5) > ndcg_at_k(late, {"a"}, 5)

    def test_no_relevant_documents(self):
        assert ndcg_at_k(RETRIEVED, set(), 5) == 0.0


class TestHelpers:
    def test_k_must_be_positive(self):
        for metric in (hit_rate_at_k, recall_at_k, precision_at_k, ndcg_at_k):
            with pytest.raises(ValueError, match="positive"):
                metric(RETRIEVED, {"a"}, 0)

    def test_mean_of_empty_is_zero(self):
        assert mean([]) == 0.0

    def test_summarize_averages_each_key(self):
        result = summarize([{"recall": 1.0, "mrr": 0.5}, {"recall": 0.0, "mrr": 1.0}])
        assert result == {"recall": pytest.approx(0.5), "mrr": pytest.approx(0.75)}

    def test_summarize_of_empty(self):
        assert summarize([]) == {}
