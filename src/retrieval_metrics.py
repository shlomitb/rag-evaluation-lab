
"""Metrics for evaluating ranked document retrieval results."""


def _validate_evaluation_inputs(
    retrieved_chunk_ids: list[list[str]],
    relevant_chunk_ids: list[set[str]],
) -> None:
    """Validate that each query has a non-empty set of relevant chunk IDs."""
    if len(retrieved_chunk_ids) != len(relevant_chunk_ids):
        raise ValueError(
            "Retrieved results and relevance labels must have the same length."
        )

    if not retrieved_chunk_ids:
        raise ValueError("At least one evaluation query is required.")

    if any(not relevant for relevant in relevant_chunk_ids):
        raise ValueError(
            "Each evaluation query must have at least one relevant chunk ID."
        )


def hit_rate_at_k(
    retrieved_chunk_ids: list[list[str]],
    relevant_chunk_ids: list[set[str]],
    k: int = 3,
) -> float:
    """Return the proportion of queries with a relevant chunk in the top k."""
    if k <= 0:
        raise ValueError("k must be positive.")

    _validate_evaluation_inputs(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    )

    hits = 0

    for retrieved, relevant in zip(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    ):
        if any(chunk_id in relevant for chunk_id in retrieved[:k]):
            hits += 1

    return hits / len(retrieved_chunk_ids)


def mean_reciprocal_rank(
    retrieved_chunk_ids: list[list[str]],
    relevant_chunk_ids: list[set[str]],
) -> float:
    """Return the average reciprocal rank of the first relevant result.

    A relevant result at rank 1 contributes 1.0, at rank 2 contributes
    0.5, and at rank 3 contributes approximately 0.333. A query with
    no relevant result contributes 0.
    """
    _validate_evaluation_inputs(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    )

    reciprocal_ranks = []

    for retrieved, relevant in zip(
        retrieved_chunk_ids,
        relevant_chunk_ids,
    ):
        reciprocal_rank = 0.0

        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                reciprocal_rank = 1 / rank
                break

        reciprocal_ranks.append(reciprocal_rank)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
