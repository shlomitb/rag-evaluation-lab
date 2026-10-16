
import pytest

from src.chunking import chunk_text


def test_short_text_produces_one_chunk() -> None:
    chunks = chunk_text("Hello world", chunk_size=20, overlap=5)

    assert chunks == ["Hello world"]


def test_long_text_is_split_into_expected_chunks() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=4, overlap=1)

    assert chunks == ["abcd", "defg", "ghij"]


def test_chunks_overlap_by_the_requested_amount() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=5, overlap=2)

    assert chunks == ["abcde", "defgh", "ghij"]

    for previous, current in zip(chunks, chunks[1:]):
        assert previous[-2:] == current[:2]


def test_empty_or_whitespace_text_returns_no_chunks() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_zero_or_negative_chunk_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("Hello", chunk_size=0)

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        chunk_text("Hello", chunk_size=-1)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError, match="overlap cannot be negative"):
        chunk_text("Hello", chunk_size=5, overlap=-1)

    with pytest.raises(ValueError, match="overlap must be smaller"):
        chunk_text("Hello", chunk_size=5, overlap=5)




def test_chunking_preserves_all_text_when_reconstructed() -> None:
    original_text = "The quick brown fox jumps over the lazy dog."

    overlap = 5
    chunks = chunk_text(
        original_text,
        chunk_size=12,
        overlap=overlap,
    )

    reconstructed = chunks[0] + "".join(
        chunk[overlap:] for chunk in chunks[1:]
    )

    assert reconstructed == original_text

