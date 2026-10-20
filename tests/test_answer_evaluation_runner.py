
"""Tests for the deterministic answer-evaluation runner."""

import json

import pytest

from src.answer_evaluation_runner import run_answer_evaluation


@pytest.fixture
def dataset_path(tmp_path):
    """Create a small answer-evaluation dataset for testing."""
    cases = [
        {
            "id": "case_01",
            "category": "factual",
            "question": "What is the standard shipping cost?",
            "required_phrases": ["$5.99"],
            "forbidden_phrases": [],
        },
        {
            "id": "case_02",
            "category": "factual",
            "question": "What is the express shipping cost?",
            "required_phrases": ["$14.99"],
            "forbidden_phrases": ["$5.99"],
        },
    ]

    path = tmp_path / "answer_eval.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def test_runner_evaluates_all_cases_and_summarizes_results(dataset_path):
    """The report should include every case and correct aggregate counts."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
        "case_02": "Express shipping costs $14.99.",
    }

    report = run_answer_evaluation(dataset_path, answers)

    assert report.total_cases == 2
    assert report.passed_cases == 2
    assert report.failed_cases == 0
    assert report.pass_rate == 1.0
    assert [result.case_id for result in report.results] == [
        "case_01",
        "case_02",
    ]


def test_runner_preserves_details_for_each_case(dataset_path):
    """Each result should retain its case metadata and evaluation details."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
        "case_02": "Express shipping costs $14.99.",
    }

    report = run_answer_evaluation(dataset_path, answers)

    result = report.results[0]

    assert result.case_id == "case_01"
    assert result.category == "factual"
    assert result.question == "What is the standard shipping cost?"
    assert result.evaluation.passed is True
    assert result.evaluation.required_phrases_found == ["$5.99"]


def test_runner_reports_failed_cases_and_pass_rate(dataset_path):
    """Missing required information should lower the pass rate."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
        "case_02": "Express shipping costs $5.99.",
    }

    report = run_answer_evaluation(dataset_path, answers)

    assert report.total_cases == 2
    assert report.passed_cases == 1
    assert report.failed_cases == 1
    assert report.pass_rate == 0.5
    assert report.results[1].evaluation.passed is False


def test_runner_rejects_missing_answers(dataset_path):
    """Every dataset case must have a supplied answer."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
    }

    with pytest.raises(ValueError, match="Missing answers"):
        run_answer_evaluation(dataset_path, answers)


def test_runner_rejects_unexpected_answer_ids(dataset_path):
    """Answers must correspond to IDs in the dataset."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
        "case_02": "Express shipping costs $14.99.",
        "case_03": "Unexpected answer.",
    }

    with pytest.raises(ValueError, match="Unexpected answer case IDs"):
        run_answer_evaluation(dataset_path, answers)


def test_runner_rejects_non_string_answers(dataset_path):
    """An answer must be text before it can be evaluated."""
    answers = {
        "case_01": "Standard shipping costs $5.99.",
        "case_02": 1499,
    }

    with pytest.raises(ValueError, match="must be a string"):
        run_answer_evaluation(dataset_path, answers)


def test_runner_rejects_empty_dataset(tmp_path):
    """An empty dataset should not produce a misleading report."""
    path = tmp_path / "answer_eval.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty list"):
        run_answer_evaluation(path, {})
