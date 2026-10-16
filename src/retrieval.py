
"""Retrieve the document chunks most similar to a query embedding."""

from src.similarity import cosine_similarity


def retrieve_chunks(
    query_embedding: list[float],
    chunks: list[dict[str, str | int]],
    chunk_embeddings: list[list[float]],
    top_k: int = 3,
) -> list[dict[str, str | int | float]]:
    """Return the top-k chunks ranked by cosine similarity.

    Each result contains the original chunk metadata and a similarity score.
    Results are ordered from highest similarity to lowest similarity.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if len(chunks) != len(chunk_embeddings):
        raise ValueError(
            "The number of chunks must match the number of embeddings."
        )

    scored_chunks = []

    for chunk, embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, embedding)

        scored_chunks.append(
            {
                **chunk,
                "similarity": score,
            }
        )

    scored_chunks.sort(
        key=lambda chunk: chunk["similarity"],
        reverse=True,
    )

    return scored_chunks[:top_k]
