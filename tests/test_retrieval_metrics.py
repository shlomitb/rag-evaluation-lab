
import pytest

from src.retrieval_metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
)


def test_hit_rate_at_k_counts_queries_with_relevant_results() -> None:
    retrieved = [
        ["A", "B", "C"],
        ["X", "Y", "B"],
        ["X", "Y", "Z"],
    ]
    relevant = [{"A"}, {"B"}, {"A"}]

    assert hit_rate_at_k(retrieved, relevant, k=3) == pytest.approx(2 / 3)


def test_hit_rate_at_k_respects_cutoff() -> None:
    retrieved = [["X", "Y", "B"]]
    relevant = [{"B"}]

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0
    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_supports_multiple_relevant_chunks() -> None:
    retrieved = [["X", "B", "C"]]
    relevant = [{"A", "B"}]

    assert hit_rate_at_k(retrieved, relevant, k=3) == 1.0


def test_hit_rate_at_k_returns_zero_when_no_results_match() -> None:
    retrieved = [["X", "Y"], ["P", "Q"]]
    relevant = [{"A"}, {"B"}]

    assert hit_rate_at_k(retrieved, relevant, k=2) == 0.0


def test_mrr_calculates_average_reciprocal_rank() -> None:
    retrieved = [
        ["A", "B", "C"],
        ["X", "Y", "B"],
        ["X", "Y", "Z"],
    ]
    relevant = [{"A"}, {"B"}, {"A"}]

    expected = (1.0 + 1 / 3 + 0.0) / 3

    assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(
        expected
    )


def test_mrr_rewards_relevant_results_ranked_higher() -> None:
    relevant = [{"A"}]

    rank_one = mean_reciprocal_rank([["A", "X", "Y"]], relevant)
    rank_three = mean_reciprocal_rank([["X", "Y", "A"]], relevant)
    no_match = mean_reciprocal_rank([["X", "Y", "Z"]], relevant)

    assert rank_one == 1.0
    assert rank_three == pytest.approx(1 / 3)
    assert no_match == 0.0


def test_metrics_reject_mismatched_query_counts() -> None:
    retrieved = [["A"], ["B"]]
    relevant = [{"A"}]

    with pytest.raises(ValueError, match="same length"):
        hit_rate_at_k(retrieved, relevant)

    with pytest.raises(ValueError, match="same length"):
        mean_reciprocal_rank(retrieved, relevant)


def test_metrics_reject_empty_evaluation_dataset() -> None:
    with pytest.raises(ValueError, match="At least one"):
        hit_rate_at_k([], [], k=3)

    with pytest.raises(ValueError, match="At least one"):
        mean_reciprocal_rank([], [])


def test_metrics_reject_empty_relevance_labels() -> None:
    with pytest.raises(ValueError, match="at least one relevant"):
        hit_rate_at_k([["A"]], [set()])

    with pytest.raises(ValueError, match="at least one relevant"):
        mean_reciprocal_rank([["A"]], [set()])


def test_hit_rate_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        hit_rate_at_k([["A"]], [{"A"}], k=0)
