
"""Tests for semantic document chunking."""

import pytest

from src.semantic_chunking import semantic_chunk_text


def test_numbered_items_become_separate_chunks() -> None:
    text = (
        "SHIPPING POLICY\n\n"
        "1. Standard shipping costs $5.99.\n"
        "2. Orders over $50 qualify for free shipping.\n"
        "3. Standard shipping takes 3-5 business days."
    )

    chunks = semantic_chunk_text(text)

    assert len(chunks) == 3


def test_heading_is_preserved_in_each_chunk() -> None:
    text = (
        "SHIPPING POLICY\n\n"
        "1. Standard shipping costs $5.99.\n"
        "2. Standard shipping takes 3-5 business days."
    )

    chunks = semantic_chunk_text(text)

    assert all(
        chunk.startswith("SHIPPING POLICY\n\n")
        for chunk in chunks
    )


def test_each_chunk_contains_the_correct_numbered_item() -> None:
    text = (
        "SHIPPING POLICY\n\n"
        "1. Standard shipping costs $5.99.\n"
        "2. Standard shipping takes 3-5 business days."
    )

    chunks = semantic_chunk_text(text)

    assert "1. Standard shipping costs $5.99." in chunks[0]
    assert "2. Standard shipping takes 3-5 business days." in chunks[1]


def test_all_numbered_items_are_preserved() -> None:
    text = (
        "RETURN POLICY\n\n"
        "1. Returns are allowed within 30 days.\n"
        "2. Items must be unused.\n"
        "3. A receipt is required."
    )

    chunks = semantic_chunk_text(text)

    combined = "\n".join(chunks)

    assert "1. Returns are allowed within 30 days." in combined
    assert "2. Items must be unused." in combined
    assert "3. A receipt is required." in combined


def test_paragraphs_are_split_at_blank_lines() -> None:
    text = (
        "First paragraph with some information.\n\n"
        "Second paragraph with different information."
    )

    chunks = semantic_chunk_text(text)

    assert chunks == [
        "First paragraph with some information.",
        "Second paragraph with different information.",
    ]


def test_long_paragraph_uses_character_based_fallback() -> None:
    text = "abcdefghij"

    chunks = semantic_chunk_text(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert chunks == ["abcd", "defg", "ghij"]


def test_empty_text_returns_no_chunks() -> None:
    assert semantic_chunk_text("") == []
    assert semantic_chunk_text("   \n  ") == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
    ],
)
def test_invalid_chunk_settings_raise_value_error(
    chunk_size: int,
    overlap: int,
) -> None:
    with pytest.raises(ValueError):
        semantic_chunk_text(
            "Some text.",
            chunk_size=chunk_size,
            overlap=overlap,
        )
