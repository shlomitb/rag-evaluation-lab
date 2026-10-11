
from pathlib import Path


def load_documents(directory: str | Path) -> list[dict[str, str]]:
    """Load text documents from a directory.
    Each returned document contains its filename and text content.

    Documents are represented as a list of dictionaries, each with a filename and content.
    sirted - Finds text files and processes them in a predictable order. Predictability helps testing.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f"Document directory not found: {directory}")

    if not directory.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {directory}")

    documents = []


    for file_path in sorted(directory.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()

        if content:
            documents.append(
                {
                    "source": file_path.name,
                    "content": content,
                }
            )

    return documents
