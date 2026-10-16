
"""Tests for embedding output validation."""

import pytest

from src.embeddings import validate_embeddings


def test_accepts_valid_embeddings() -> None:
    embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    result = validate_embeddings(
        texts=["first text", "second text"],
        embeddings=embeddings,
    )

    assert result == embeddings


def test_rejects_mismatched_number_of_embeddings() -> None:
    with pytest.raises(ValueError, match="number of embeddings"):
        validate_embeddings(
            texts=["first text", "second text"],
            embeddings=[[0.1, 0.2, 0.3]],
        )


def test_rejects_inconsistent_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="same dimension"):
        validate_embeddings(
            texts=["first text", "second text"],
            embeddings=[
                [0.1, 0.2, 0.3],
                [0.4, 0.5],
            ],
        )


def test_rejects_empty_embedding_vectors() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_embeddings(
            texts=["first text"],
            embeddings=[[]],
        )


@pytest.mark.parametrize(
    "invalid_value",
    [float("nan"), float("inf"), float("-inf"), "not a number", True],
)
def test_rejects_invalid_embedding_values(
    invalid_value: object,
) -> None:
    with pytest.raises(ValueError):
        validate_embeddings(
            texts=["first text"],
            embeddings=[[invalid_value]],  # type: ignore[list-item]
        )


def test_accepts_empty_input_lists() -> None:
    assert validate_embeddings(texts=[], embeddings=[]) == []
