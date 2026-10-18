
"""
An ingestion pipeline that coordinates existing components instead of duplicating their logic.
Load documents and convert them into metadata-rich chunks.
"""


from pathlib import Path

from src.chunk_metadata import chunk_document
from src.document_loader import load_documents
from src.semantic_chunking import semantic_chunk_text


def ingest_documents(
    directory: str | Path,
    chunk_size: int = 500,
    overlap: int = 50,
    chunk_strategy: str = "character",
) -> list[dict[str, str | int]]:
    """Load documents and return chunks with source metadata.

    Supported strategies:
    - "character": fixed-size chunks with overlap.
    - "semantic": split on numbered items or paragraph boundaries.
    """
    if chunk_strategy not in {"character", "semantic"}:
        raise ValueError(
            "chunk_strategy must be 'character' or 'semantic'."
        )

    documents = load_documents(directory)
    chunks = []

    for document in documents:
        if chunk_strategy == "character":
            document_chunks = chunk_document(
                document,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        else:
            text_chunks = semantic_chunk_text(
                document["content"],
                chunk_size=chunk_size,
                overlap=overlap,
            )

            document_chunks = [
                {
                    "source": document["source"],
                    "content": content,
                    "chunk_index": index,
                    "chunk_id": f"{document['source']}:{index}",
                }
                for index, content in enumerate(text_chunks)
            ]

        chunks.extend(document_chunks)

    return chunks
