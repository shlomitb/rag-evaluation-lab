
"""
Deterministic component tests
Tests for cosine similarity calculations.
We know the expected result or behavior for each one.
We can calculate the answers independently of the implementation, so we don't need an LLM or an embedding model to tell us whether the tests pass.
the symmetry test doesn't specify one exact numeric result, but it specifies a known relationship between two results. That is still a deterministic test.
"""

import math

import pytest

from src.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    """The angle is zero."""
    assert math.isclose(
        cosine_similarity([1.0, 2.0], [1.0, 2.0]),
        1.0,
    )


def test_orthogonal_vectors_have_similarity_zero() -> None:
    """The vectors are perpendicular."""
    assert math.isclose(
        cosine_similarity([1.0, 0.0], [0.0, 1.0]),
        0.0,
        abs_tol=1e-12,
    )


def test_opposite_vectors_have_similarity_minus_one() -> None:
    """The vectors point in opposite directions."""
    assert math.isclose(
        cosine_similarity([1.0, 0.0], [-1.0, 0.0]),
        -1.0,
    )


def test_similarity_is_independent_of_vector_magnitude() -> None:
    """The vectors point in the same direction."""
    result = cosine_similarity([1.0, 2.0], [10.0, 20.0])

    assert math.isclose(result, 1.0)


def test_similarity_is_symmetric() -> None:
    """The formula is symmetric."""
    forward = cosine_similarity([1.0, 2.0], [3.0, 4.0])
    reverse = cosine_similarity([3.0, 4.0], [1.0, 2.0])

    assert math.isclose(forward, reverse)


@pytest.mark.parametrize(
    ("vector_a", "vector_b"),
    [
        ([], []),
        ([1.0], []),
        ([], [1.0]),
    ],
)


def test_empty_vectors_raise_value_error(
    vector_a: list[float],
    vector_b: list[float],
) -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        cosine_similarity(vector_a, vector_b)


def test_vectors_with_different_dimensions_raise_value_error() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        cosine_similarity([1.0, 2.0], [1.0])

@pytest.mark.parametrize(
    "vector",
    [
        [0.0, 0.0],
        [0.0],
    ],
)


def test_zero_vector_raises_value_error(
    vector: list[float],
) -> None:
    with pytest.raises(ValueError, match="Zero vectors"):
        cosine_similarity(vector, vector)

@pytest.mark.parametrize(
    "invalid_value",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        "not a number",
        True,
    ],
)


def test_invalid_vector_values_raise_value_error(
    invalid_value: float | str | bool,
) -> None:
    with pytest.raises(ValueError):
        cosine_similarity([1.0, invalid_value], [1.0, 2.0])
