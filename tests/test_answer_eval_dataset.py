
"""Validate the structure and content of the answer evaluation dataset."""

import json
from pathlib import Path

import pytest


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "answer_eval.json"
)


@pytest.fixture
def answer_eval_cases() -> list[dict]:
    """Load the answer evaluation dataset."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def test_answer_eval_dataset_is_valid_json():
    """The dataset file must contain valid JSON."""
    with DATASET_PATH.open(encoding="utf-8") as file:
        data = json.load(file)

    assert isinstance(data, list)
    assert data, "The answer evaluation dataset must not be empty."


def test_every_case_has_required_fields(answer_eval_cases):
    """Every case must include the fields needed by the evaluator."""
    required_fields = {
        "id",
        "category",
        "question",
        "required_phrases",
        "forbidden_phrases",
    }

    for case in answer_eval_cases:
        assert required_fields.issubset(case.keys()), (
            f"Case is missing fields: {case}"
        )


def test_case_ids_are_unique_and_non_empty(answer_eval_cases):
    """Each case must have a unique, non-empty ID."""
    ids = [case["id"] for case in answer_eval_cases]

    assert all(
        isinstance(case_id, str) and case_id.strip()
        for case_id in ids
    )
    assert len(ids) == len(set(ids)), "Case IDs must be unique."


def test_questions_and_categories_are_non_empty_strings(answer_eval_cases):
    """Questions and categories must be non-empty strings."""
    for case in answer_eval_cases:
        assert isinstance(case["question"], str)
        assert case["question"].strip(), (
            f"Case {case['id']} has an empty question."
        )

        assert isinstance(case["category"], str)
        assert case["category"].strip(), (
            f"Case {case['id']} has an empty category."
        )


def test_phrase_fields_are_lists_of_non_empty_strings(answer_eval_cases):
    """Required and forbidden phrases must be lists of non-empty strings."""
    for case in answer_eval_cases:
        for field in ("required_phrases", "forbidden_phrases"):
            phrases = case[field]

            assert isinstance(phrases, list), (
                f"Case {case['id']}: {field} must be a list."
            )
            assert all(
                isinstance(phrase, str) and phrase.strip()
                for phrase in phrases
            ), (
                f"Case {case['id']}: {field} must contain "
                "only non-empty strings."
            )


def test_each_case_has_at_least_one_required_phrase(answer_eval_cases):
    """Each initial case must check for at least one expected phrase."""
    for case in answer_eval_cases:
        assert case["required_phrases"], (
            f"Case {case['id']} has no required phrases."
        )


def test_dataset_contains_expected_case_count(answer_eval_cases):
    """Protect the initial 13 cases from accidental additions or removals."""
    assert len(answer_eval_cases) == 13
