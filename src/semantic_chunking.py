
"""Split documents into chunks using meaningful text boundaries."""

import re

from src.chunking import chunk_text


_NUMBERED_ITEM = re.compile(r"^\s*\d+[.)]\s+")


def _is_heading(line: str) -> bool:
    """Identify simple all-uppercase document headings."""
    return (
        line == line.upper()
        and any(character.isalpha() for character in line)
        and not _NUMBERED_ITEM.match(line)
    )


def semantic_chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split text into meaningful sections or numbered items.

    Numbered items are kept separate, with the document heading
    repeated in each chunk when a heading is present.
    Other documents are split by paragraphs, with character-based
    chunking used for paragraphs that are too long.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()

    if not text:
        return []

    lines = text.splitlines()
    nonempty_lines = [line.strip() for line in lines if line.strip()]

    if not nonempty_lines:
        return []

    heading = (
        nonempty_lines[0]
        if _is_heading(nonempty_lines[0])
        else None
    )

    numbered_items = [
        line.strip()
        for line in nonempty_lines
        if _NUMBERED_ITEM.match(line)
    ]

    if numbered_items:
        chunks = []

        for item in numbered_items:
            # Keep the policy or catalog heading with each fact.
            unit = f"{heading}\n\n{item}" if heading else item

            if len(unit) <= chunk_size:
                chunks.append(unit)
            else:
                # Fall back to character-based splitting for unusually
                # long individual items.
                item_chunks = chunk_text(
                    item,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
                for item_chunk in item_chunks:
                    chunks.append(
                        f"{heading}\n\n{item_chunk}"
                        if heading
                        else item_chunk
                    )

        return chunks

    # For ordinary prose, use blank lines as paragraph boundaries.
    paragraphs = [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text)
        if paragraph.strip()
    ]

    chunks = []

    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            chunks.append(paragraph)
        else:
            chunks.extend(
                chunk_text(
                    paragraph,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

    return chunks
