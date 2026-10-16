
"""Tests for the local embedding provider."""

import pytest

from src.embedding_provider import LocalEmbeddingProvider


class FakeEmbeddingModel:
    """Return predictable vectors without loading a real model."""

    def __init__(self) -> None:
        self.received_texts: list[str] | None = None
        self.received_options: dict[str, object] = {}

    def encode(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> list[list[float]]:
        self.received_texts = texts
        self.received_options = kwargs

        return [
            [0.6, 0.8],
            [0.0, 1.0],
        ]


def test_embed_texts_returns_vectors_from_model() -> None:
    fake_model = FakeEmbeddingModel()
    provider = LocalEmbeddingProvider(model=fake_model)

    result = provider.embed_texts(["first text", "second text"])

    assert result == [
        [0.6, 0.8],
        [0.0, 1.0],
    ]


def test_embed_texts_passes_texts_to_model() -> None:
    fake_model = FakeEmbeddingModel()
    provider = LocalEmbeddingProvider(model=fake_model)

    provider.embed_texts(["first text", "second text"])

    assert fake_model.received_texts == ["first text", "second text"]


def test_embed_texts_requests_normalized_embeddings() -> None:
    fake_model = FakeEmbeddingModel()
    provider = LocalEmbeddingProvider(model=fake_model)

    provider.embed_texts(["first text", "second text"])

    assert fake_model.received_options == {
        "convert_to_numpy": False,
        "normalize_embeddings": True,
    }


def test_empty_input_returns_empty_list_without_loading_model() -> None:
    provider = LocalEmbeddingProvider()

    assert provider.embed_texts([]) == []
    assert provider._model is None


def test_embed_texts_rejects_wrong_number_of_vectors() -> None:
    class IncorrectModel:
        def encode(
            self,
            texts: list[str],
            **kwargs: object,
        ) -> list[list[float]]:
            return [[0.1, 0.2]]

    provider = LocalEmbeddingProvider(model=IncorrectModel())

    with pytest.raises(ValueError, match="number of embeddings"):
        provider.embed_texts(["first text", "second text"])
