
"""Generate text embeddings using a local Sentence Transformers model."""

from typing import Any

from sentence_transformers import SentenceTransformer

from src.embeddings import validate_embeddings


class LocalEmbeddingProvider:
    """Generate normalized embeddings without a paid API."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        model: Any | None = None,
    ) -> None:
        """Configure the model, optionally accepting an injected model for tests."""
        self.model_name = model_name
        self._model = model

    def _get_model(self) -> Any:
        """Load the model only when embeddings are first requested."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert texts into validated, normalized embedding vectors."""
        if not texts:
            return []

        model = self._get_model()

        vectors = model.encode(
            texts,
            convert_to_numpy=False,
            normalize_embeddings=True,
        )

        embeddings = [
            [float(value) for value in vector]
            for vector in vectors
        ]

        return validate_embeddings(texts, embeddings)
