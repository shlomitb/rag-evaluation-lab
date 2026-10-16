
"""Validate embedding vectors returned by an embedding provider."""

from math import isfinite


def validate_embeddings(
    texts: list[str],
    embeddings: list[list[float]],
) -> list[list[float]]:
    """Validate embedding count, dimensions, and numeric values.

    - Count: Every input text must have a corresponding vector.
    - Dimensions: All vectors must have the same number of values.
    - Non-empty vectors: A vector with zero dimensions is invalid.
    - Numeric, finite values: Embeddings must not contain strings, booleans, NaN, or infinity.

    if both lists are empty, the function returns an empty list. That's reasonable because there are no vectors to validate.

    Return the original embeddings if they are valid.
    Raise ValueError if the output does not match the expected contract.
    """
    if len(texts) != len(embeddings):
        raise ValueError(
            "The number of embeddings must match the number of texts."
        )

    if not embeddings:
        return embeddings

    dimension = len(embeddings[0])

    if dimension == 0:
        raise ValueError("Embedding vectors cannot be empty.")

    for vector in embeddings:
        if len(vector) != dimension:
            raise ValueError(
                "All embedding vectors must have the same dimension."
            )

        for value in vector:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("Embedding values must be numeric.")

            if not isfinite(value):
                raise ValueError(
                    "Embedding values must be finite numbers."
                )

    return embeddings
