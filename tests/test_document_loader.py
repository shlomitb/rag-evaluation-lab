"""
tmp_path, a built-in Pytest fixture.
It creates a temporary directory for a test, so we can create test files without changing or deleting your actual sample documents.

Run with:
python -m pytest tests/test_document_loader.py -v
"""


from pathlib import Path

import pytest

from src.document_loader import load_documents


def test_loads_text_documents(tmp_path: Path) -> None:
    """
    Normal behavior: multiple documents load correctly.
    """
    (tmp_path / "b.txt").write_text("Second document", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First document", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 2
    assert documents[0] == {
        "source": "a.txt",
        "content": "First document",
    }
    assert documents[1] == {
        "source": "b.txt",
        "content": "Second document",
    }


def test_skips_empty_documents(tmp_path: Path) -> None:
    (tmp_path / "empty.txt").write_text("   \n ", encoding="utf-8")
    (tmp_path / "useful.txt").write_text("Useful content", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert len(documents) == 1
    assert documents[0]["source"] == "useful.txt"


def test_raises_error_for_missing_directory(tmp_path: Path) -> None:
    missing_directory = tmp_path / "does_not_exist"

    with pytest.raises(FileNotFoundError):
        load_documents(missing_directory)


def test_raises_error_when_path_is_a_file(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_directory.txt"
    file_path.write_text("I'm a file", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        load_documents(file_path)
