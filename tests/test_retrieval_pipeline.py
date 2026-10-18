
"""
Tests for the end-to-end retrieval pipeline.

These tests should not download or load the embedding model: they inject the fake provider.
They do use the real document ingestion, chunking, and retrieval functions, so they test how those components work together.

"""

from pathlib import Path

import pytest

from src.retrieval_pipeline import search_documents


class FakeEmbeddingProvider:
    """Return predictable vectors for pipeline tests."""

    def __init__(self) -> None:
        self.received_texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.received_texts = texts

        vectors = []

        for text in texts:
            if text == "How long does shipping take?":
                vectors.append([1.0, 0.0])
            elif "shipping" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return vectors


def test_pipeline_retrieves_relevant_document_first(
    tmp_path: Path,
) -> None:
    (tmp_path / "shipping.txt").write_text(
        "Standard shipping takes 3-5 business days.",
        encoding="utf-8",
    )
    (tmp_path / "returns.txt").write_text(
        "Customers may return items within 30 days.",
        encoding="utf-8",
    )

    provider = FakeEmbeddingProvider()

    results = search_documents(
        query="How long does shipping take?",
        directory=tmp_path,
        embedding_provider=provider,
    )

    assert results[0]["source"] == "shipping.txt"
    assert results[0]["similarity"] == pytest.approx(1.0)
    assert len(results) == 2


def test_pipeline_preserves_chunk_metadata(
    tmp_path: Path,
) -> None:
    (tmp_path / "shipping.txt").write_text(
        "Standard shipping takes 3-5 business days.",
        encoding="utf-8",
    )

    results = search_documents(
        query="How long does shipping take?",
        directory=tmp_path,
        embedding_provider=FakeEmbeddingProvider(),
    )

    assert results[0]["source"] == "shipping.txt"
    assert results[0]["chunk_index"] == 0
    assert results[0]["chunk_id"] == "shipping.txt:0"
    assert "shipping" in str(results[0]["content"]).lower()
    assert "similarity" in results[0]


def test_pipeline_returns_no_results_for_empty_directory(
    tmp_path: Path,
) -> None:
    provider = FakeEmbeddingProvider()

    results = search_documents(
        query="How long does shipping take?",
        directory=tmp_path,
        embedding_provider=provider,
    )

    assert results == []
    assert provider.received_texts == []


def test_pipeline_rejects_empty_query(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Query cannot be empty"):
        search_documents(
            query="   ",
            directory=tmp_path,
            embedding_provider=FakeEmbeddingProvider(),
        )


def test_pipeline_rejects_wrong_embedding_count(
    tmp_path: Path,
) -> None:
    (tmp_path / "shipping.txt").write_text(
        "Standard shipping takes 3-5 business days.",
        encoding="utf-8",
    )

    class BrokenEmbeddingProvider:
        def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    with pytest.raises(
        ValueError,
        match="one vector per text",
    ):
        search_documents(
            query="How long does shipping take?",
            directory=tmp_path,
            embedding_provider=BrokenEmbeddingProvider(),
        )
