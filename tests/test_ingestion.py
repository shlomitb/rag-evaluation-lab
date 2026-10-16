
from pathlib import Path

from src.ingestion import ingest_documents


def test_ingestion_processes_multiple_documents(
    tmp_path: Path,
) -> None:
    """
    Files are loaded and their chunks combined into one list.
    """
    (tmp_path / "a.txt").write_text(
        "abcdefghij", encoding="utf-8"
    )
    (tmp_path / "b.txt").write_text(
        "klmnopqrst", encoding="utf-8"
    )

    chunks = ingest_documents(
        tmp_path,
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) == 6
    assert [chunk["source"] for chunk in chunks] == [
        "a.txt", "a.txt", "a.txt",
        "b.txt", "b.txt", "b.txt",
    ]


def test_ingestion_preserves_chunk_metadata(
    tmp_path: Path,
) -> None:
    """
    Content, indexes, and IDs survive the complete pipeline.
    """
    (tmp_path / "policy.txt").write_text(
        "abcdefghij", encoding="utf-8"
    )

    chunks = ingest_documents(
        tmp_path,
        chunk_size=4,
        overlap=1,
    )

    assert [chunk["content"] for chunk in chunks] == [
        "abcd", "defg", "ghij"
    ]
    assert [chunk["chunk_index"] for chunk in chunks] == [
        0, 1, 2
    ]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "policy.txt:0",
        "policy.txt:1",
        "policy.txt:2",
    ]


def test_ingestion_returns_empty_list_for_empty_directory(
    tmp_path: Path,
) -> None:
    """
    The pipeline handles having no input documents.
    """
    assert ingest_documents(tmp_path) == []
