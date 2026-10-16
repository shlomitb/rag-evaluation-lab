
"""
Tests for ranking document chunks by similarity.

Our test data deliberately makes the shipping chunk rank first.
But notice the vectors are invented test values;
they do not prove that the embedding model will rank real shipping text first for every shipping-related question.
"""

import math

import pytest

from src.retrieval import retrieve_chunks


@pytest.fixture
def sample_chunks() -> list[dict[str, str | int]]:
    return [
        {
            "source": "shipping_policy.txt",
            "content": "Standard shipping takes 3-5 business days.",
            "chunk_index": 0,
            "chunk_id": "shipping_policy.txt:0",
        },
        {
            "source": "return_policy.txt",
            "content": "Customers may return an item within 30 days.",
            "chunk_index": 0,
            "chunk_id": "return_policy.txt:0",
        },
        {
            "source": "product_catalog.txt",
            "content": "The wireless mouse costs $24.99.",
            "chunk_index": 0,
            "chunk_id": "product_catalog.txt:0",
        },
    ]


@pytest.fixture
def sample_embeddings() -> list[list[float]]:
    # The first vector is closest to the query vector.
    return [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]


def test_retrieves_chunks_in_descending_similarity_order(
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    results = retrieve_chunks(
        query_embedding=[1.0, 0.0],
        chunks=sample_chunks,
        chunk_embeddings=sample_embeddings,
    )

    assert [result["chunk_id"] for result in results] == [
        "shipping_policy.txt:0",
        "return_policy.txt:0",
        "product_catalog.txt:0",
    ]

    assert math.isclose(results[0]["similarity"], 1.0)
    assert math.isclose(results[1]["similarity"], 0.0)
    assert math.isclose(results[2]["similarity"], -1.0)


def test_top_k_limits_number_of_results(
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    results = retrieve_chunks(
        query_embedding=[1.0, 0.0],
        chunks=sample_chunks,
        chunk_embeddings=sample_embeddings,
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["chunk_id"] == "shipping_policy.txt:0"
    assert results[1]["chunk_id"] == "return_policy.txt:0"


def test_results_preserve_chunk_metadata(
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    results = retrieve_chunks(
        query_embedding=[1.0, 0.0],
        chunks=sample_chunks,
        chunk_embeddings=sample_embeddings,
        top_k=1,
    )

    assert results[0]["source"] == "shipping_policy.txt"
    assert results[0]["content"] == (
        "Standard shipping takes 3-5 business days."
    )
    assert results[0]["chunk_index"] == 0
    assert results[0]["chunk_id"] == "shipping_policy.txt:0"
    assert "similarity" in results[0]


def test_empty_inputs_return_no_results() -> None:
    results = retrieve_chunks(
        query_embedding=[1.0, 0.0],
        chunks=[],
        chunk_embeddings=[],
    )

    assert results == []


def test_mismatched_chunk_and_embedding_counts_raise_value_error(
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    with pytest.raises(ValueError, match="must match"):
        retrieve_chunks(
            query_embedding=[1.0, 0.0],
            chunks=sample_chunks,
            chunk_embeddings=sample_embeddings[:2],
        )


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_raises_value_error(
    top_k: int,
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        retrieve_chunks(
            query_embedding=[1.0, 0.0],
            chunks=sample_chunks,
            chunk_embeddings=sample_embeddings,
            top_k=top_k,
        )


def test_invalid_embedding_is_rejected(
    sample_chunks: list[dict[str, str | int]],
    sample_embeddings: list[list[float]],
) -> None:
    invalid_embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0],  # Wrong dimension.
    ]

    with pytest.raises(ValueError, match="same dimension"):
        retrieve_chunks(
            query_embedding=[1.0, 0.0],
            chunks=sample_chunks,
            chunk_embeddings=invalid_embeddings,
        )
