
"""Connect document retrieval with answer generation."""

from pathlib import Path

from src.answer_generation import TextGenerator, generate_answer
from src.retrieval_pipeline import EmbeddingProvider, search_documents


def answer_question(
    question: str,
    directory: str | Path,
    generator: TextGenerator,
    top_k: int = 3,
    chunk_size: int = 500,
    overlap: int = 50,
    embedding_provider: EmbeddingProvider | None = None,
) -> str:
    """Retrieve relevant chunks and generate an answer to a question."""
    retrieved_chunks = search_documents(
        query=question,
        directory=directory,
        top_k=top_k,
        chunk_size=chunk_size,
        overlap=overlap,
        embedding_provider=embedding_provider,
    )

    return generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        generator=generator,
    )
