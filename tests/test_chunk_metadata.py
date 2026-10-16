from src.chunk_metadata import chunk_document


def test_chunk_document_assigns_sequential_indexes() -> None:
    document = {
        "source": "return_policy.txt",
        "content": "abcdefghij",
    }

    chunks = chunk_document(document, chunk_size=4, overlap=1)

    assert [chunk["chunk_index"] for chunk in chunks] == [0, 1, 2]


def test_chunk_document_assigns_unique_ids() -> None:
    document = {
        "source": "return_policy.txt",
        "content": "abcdefghij",
    }

    chunks = chunk_document(document, chunk_size=4, overlap=1)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))



def test_chunk_document_preserves_source() -> None:
    document = {
        "source": "return_policy.txt",
        "content": "abcdefghij",
    }

    chunks = chunk_document(document, chunk_size=4, overlap=1)

    assert all(
        chunk["source"] == "return_policy.txt"
        for chunk in chunks
    )


def test_chunk_document_preserves_chunk_content() -> None:
    document = {
        "source": "return_policy.txt",
        "content": "abcdefghij",
    }

    chunks = chunk_document(document, chunk_size=4, overlap=1)

    assert [chunk["content"] for chunk in chunks] == [
        "abcd",
        "defg",
        "ghij",
    ]


def test_chunk_document_returns_no_chunks_for_empty_content() -> None:
    document = {
        "source": "empty.txt",
        "content": "   ",
    }

    assert chunk_document(document) == []

