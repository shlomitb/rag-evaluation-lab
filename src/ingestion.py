
"""
An ingestion pipeline that coordinates existing components instead of duplicating their logic.
Load documents and convert them into metadata-rich chunks.
"""

from pathlib import Path

from src.chunk_metadata import chunk_document
from src.document_loader import load_documents


def ingest_documents(
    directory: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, str | int]]:
    """Load all text documents and return their chunks with metadata."""
    documents = load_documents(directory)

    chunks = []

    for document in documents:
        document_chunks = chunk_document(
            document,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        chunks.extend(document_chunks)

    return chunks
