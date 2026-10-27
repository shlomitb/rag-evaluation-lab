
"""Tests for deterministic answer evaluation."""

import pytest

from src.answer_evaluation import evaluate_answer


def test_passes_when_all_required_phrases_are_present():
    result = evaluate_answer(
        answer="Standard shipping takes 3-5 business days and costs $5.",
        required_phrases=[
            "3-5 business days",
            "$5",
        ],
    )

    assert result.passed is True
    assert result.required_phrases_found == [
        "3-5 business days",
        "$5",
    ]
    assert result.missing_required_phrases == []
    assert result.required_phrase_coverage == 1.0


def test_fails_when_a_required_phrase_is_missing():
    result = evaluate_answer(
        answer="Standard shipping takes 3-5 business days.",
        required_phrases=[
            "3-5 business days",
            "$5",
        ],
    )

    assert result.passed is False
    assert result.missing_required_phrases == ["$5"]
    assert result.required_phrase_coverage == 0.5


def test_fails_when_a_forbidden_phrase_is_present():
    result = evaluate_answer(
        answer="Standard shipping is free and takes 3-5 business days.",
        required_phrases=["3-5 business days"],
        forbidden_phrases=["shipping is free"],
    )

    assert result.passed is False
    assert result.forbidden_phrases_found == ["shipping is free"]


def test_phrase_matching_is_case_insensitive():
    result = evaluate_answer(
        answer="STANDARD SHIPPING TAKES 3-5 BUSINESS DAYS.",
        required_phrases=["Standard Shipping Takes 3-5 Business Days"],
    )

    assert result.passed is True


def test_fails_for_an_empty_answer_when_facts_are_required():
    result = evaluate_answer(
        answer="",
        required_phrases=["3-5 business days"],
    )

    assert result.passed is False
    assert result.required_phrase_coverage == 0.0


def test_passes_with_no_required_or_forbidden_phrases():
    result = evaluate_answer(
        answer="Any answer.",
        required_phrases=[],
    )

    assert result.passed is True
    assert result.required_phrase_coverage == 1.0


def test_rejects_an_empty_required_phrase():
    with pytest.raises(ValueError, match="Required phrases cannot be empty"):
        evaluate_answer(
            answer="Some answer.",
            required_phrases=[""],
        )


def test_rejects_an_empty_forbidden_phrase():
    with pytest.raises(ValueError, match="Forbidden phrases cannot be empty"):
        evaluate_answer(
            answer="Some answer.",
            required_phrases=[],
            forbidden_phrases=[" "],
        )



def test_phrase_matching_ignores_quotes():
    result = evaluate_answer(
        answer='Standard shipping "typically takes" 3-5 business days.',
        required_phrases=["typically takes"],
    )

    assert result.passed is True
    assert result.required_phrases_found == ["typically takes"]


def test_phrase_matching_ignores_punctuation_differences():
    result = evaluate_answer(
        answer="Standard shipping typically takes 3-5 business days.",
        required_phrases=["typically takes,"],
    )

    assert result.passed is True
    assert result.required_phrases_found == ["typically takes,"]


def test_phrase_matching_ignores_extra_whitespace():
    result = evaluate_answer(
        answer="Standard shipping typically   takes 3-5 business days.",
        required_phrases=["typically takes"],
    )

    assert result.passed is True
    assert result.required_phrases_found == ["typically takes"]
