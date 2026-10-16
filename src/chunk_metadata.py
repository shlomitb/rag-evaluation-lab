"""Create document chunks with source information and unique identifiers."""

from src.chunking import chunk_text


def chunk_document(
    document: dict[str, str],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[dict[str, str | int]]:
    """Split a document into chunks while preserving source metadata."""
    content = document["content"]
    source = document["source"]

    text_chunks = chunk_text(
        content,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    chunks = []

    for index, text in enumerate(text_chunks):
        chunks.append(
            {
                "source": source,
                "content": text,
                "chunk_index": index,
                "chunk_id": f"{source}:{index}",
            }
        )

    return chunks
