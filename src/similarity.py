
"""Calculate cosine similarity between embedding vectors."""

from math import isfinite, sqrt


def cosine_similarity(
    vector_a: list[float],
    vector_b: list[float],
) -> float:
    """Return the cosine similarity between two vectors.

    The result ranges from -1 to 1 for non-zero vectors.
    Higher values indicate a smaller angle between the vectors.
    """
    if not vector_a or not vector_b:
        raise ValueError("Vectors cannot be empty.")

    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimension.")

    for vector in (vector_a, vector_b):
        for value in vector:
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
            ):
                raise ValueError("Vector values must be numeric.")

            if not isfinite(value):
                raise ValueError("Vector values must be finite.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = sqrt(sum(value * value for value in vector_a))
    magnitude_b = sqrt(sum(value * value for value in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Zero vectors do not have a defined cosine similarity.")

    return dot_product / (magnitude_a * magnitude_b)
