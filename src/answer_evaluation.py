
"""
Deterministic checks for expected answer content.
Checks answer quality using deterministic rules.
Checis for:
1. Required phrases: Does the answer contain every required phrase, ignoring capitalization?
2. Forbidden phrases: Does the answer contain any phrase that must not appear?
3. Required-phrase coverage: What proportion of the required phrases were found?

"""

from dataclasses import dataclass


@dataclass
class AnswerEvaluation:
    """Store the results of deterministic answer checks."""

    passed: bool
    required_phrases_found: list[str]
    missing_required_phrases: list[str]
    forbidden_phrases_found: list[str]
    required_phrase_coverage: float


def evaluate_answer(
    answer: str,
    required_phrases: list[str],
    forbidden_phrases: list[str] | None = None,
) -> AnswerEvaluation:
    """Check whether an answer contains expected and avoids forbidden phrases."""
    forbidden_phrases = forbidden_phrases or []

    if any(not phrase.strip() for phrase in required_phrases):
        raise ValueError("Required phrases cannot be empty.")

    if any(not phrase.strip() for phrase in forbidden_phrases):
        raise ValueError("Forbidden phrases cannot be empty.")

    normalized_answer = answer.casefold()

    required_found = [
        phrase
        for phrase in required_phrases
        if phrase.casefold() in normalized_answer
    ]

    required_missing = [
        phrase
        for phrase in required_phrases
        if phrase.casefold() not in normalized_answer
    ]

    forbidden_found = [
        phrase
        for phrase in forbidden_phrases
        if phrase.casefold() in normalized_answer
    ]

    if required_phrases:
        coverage = len(required_found) / len(required_phrases)
    else:
        coverage = 1.0

    passed = not required_missing and not forbidden_found

    return AnswerEvaluation(
        passed=passed,
        required_phrases_found=required_found,
        missing_required_phrases=required_missing,
        forbidden_phrases_found=forbidden_found,
        required_phrase_coverage=coverage,
    )
