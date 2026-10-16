"""Split text into smaller, overlapping chunks for retrieval."""


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into overlapping character-based chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    step = chunk_size - overlap
    chunks = []

    start = 0

    while start < len(text):
        chunk = text[start: start + chunk_size]
        chunks.append(chunk)

        # Stop when the current chunk reaches the end of the text.
        if start + chunk_size >= len(text):
            break

        start += step

    return chunks
