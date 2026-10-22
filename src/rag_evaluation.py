
"""Evaluate answers produced by the RAG pipeline."""

from pathlib import Path
from typing import Any

from src.answer_generation import TextGenerator, generate_answer
from src.retrieval_pipeline import EmbeddingProvider, search_documents


def evaluate_rag_question(
    question: str,
    directory: str | Path,
    generator: TextGenerator,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Retrieve evidence, generate an answer, and return both."""
    retrieved_chunks = search_documents(
        query=question,
        directory=directory,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_provider=embedding_provider,
    )

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        generator=generator,
    )

    return {
        "question": question,
        "answer": answer,
        "retrieved_chunks": retrieved_chunks,
    }
