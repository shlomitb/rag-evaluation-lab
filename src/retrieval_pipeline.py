
"""
Connect document ingestion, embeddings, and similarity-based retrieval.

This will also allow us to inject a fake embedding provider in tests,
so the pipeline can be tested without downloading a model or making any external calls.

The EmbeddingProvider protocol describes the one operation this pipeline needs: embed_texts().
Your real local provider satisfies that interface, and our test fake can satisfy it too.
This is useful design for a QA framework: we can replace the embedding implementation without rewriting the retrieval pipeline.
We also embed the query and all chunks in a single batch. This reduces repeated calls to the embedding model.
"""

from pathlib import Path
from typing import Protocol

from src.embedding_provider import LocalEmbeddingProvider
from src.ingestion import ingest_documents
from src.retrieval import retrieve_chunks


class EmbeddingProvider(Protocol):
    """Define the embedding interface required by the pipeline."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert text strings into embedding vectors."""
        ...


def search_documents(
    query: str,
    directory: str | Path,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> list[dict[str, str | int | float]]:
    """Find the document chunks most similar to a user's query."""
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    chunks = ingest_documents(
        directory,
        chunk_size=chunk_size,
        overlap=overlap,
        chunk_strategy="semantic",
    )

    if not chunks:
        return []

    provider = embedding_provider or LocalEmbeddingProvider()

    # Embed the query and all chunks in one batch.
    texts = [query] + [str(chunk["content"]) for chunk in chunks]
    embeddings = provider.embed_texts(texts)

    if len(embeddings) != len(texts):
        raise ValueError(
            "The embedding provider must return one vector per text."
        )

    query_embedding = embeddings[0]
    chunk_embeddings = embeddings[1:]

    return retrieve_chunks(
        query_embedding=query_embedding,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
        top_k=top_k,
    )
