
"""Run deterministic answer evaluations using a JSON test dataset."""

import json
from dataclasses import dataclass
from pathlib import Path

from src.answer_evaluation import AnswerEvaluation, evaluate_answer


@dataclass
class AnswerCaseResult:
    """Store the evaluation result for one answer-evaluation case."""

    case_id: str
    category: str
    question: str
    evaluation: AnswerEvaluation


@dataclass
class AnswerEvaluationReport:
    """Store all case results and calculate aggregate statistics."""

    results: list[AnswerCaseResult]

    @property
    def total_cases(self) -> int:
        """Return the number of evaluated cases."""
        return len(self.results)

    @property
    def passed_cases(self) -> int:
        """Return the number of cases that passed."""
        return sum(result.evaluation.passed for result in self.results)

    @property
    def failed_cases(self) -> int:
        """Return the number of cases that failed."""
        return self.total_cases - self.passed_cases

    @property
    def pass_rate(self) -> float:
        """Return the proportion of cases that passed."""
        if not self.results:
            return 0.0

        return self.passed_cases / self.total_cases


def run_answer_evaluation(
    dataset_path: str | Path,
    answers_by_case_id: dict[str, str],
) -> AnswerEvaluationReport:
    """Evaluate predetermined answers against a JSON dataset."""
    with Path(dataset_path).open(encoding="utf-8") as file:
        cases = json.load(file)

    if not isinstance(cases, list) or not cases:
        raise ValueError("The answer evaluation dataset must be a non-empty list.")

    case_ids = [case["id"] for case in cases]

    if len(case_ids) != len(set(case_ids)):
        raise ValueError("The answer evaluation dataset contains duplicate IDs.")

    expected_ids = set(case_ids)
    supplied_ids = set(answers_by_case_id)

    missing_ids = expected_ids - supplied_ids
    unexpected_ids = supplied_ids - expected_ids

    if missing_ids or unexpected_ids:
        problems = []

        if missing_ids:
            problems.append(f"Missing answers for case IDs: {sorted(missing_ids)}")

        if unexpected_ids:
            problems.append(
                f"Unexpected answer case IDs: {sorted(unexpected_ids)}"
            )

        raise ValueError("; ".join(problems))

    results = []

    for case in cases:
        answer = answers_by_case_id[case["id"]]

        if not isinstance(answer, str):
            raise ValueError(
                f"Answer for case '{case['id']}' must be a string."
            )

        evaluation = evaluate_answer(
            answer=answer,
            required_phrases=case["required_phrases"],
            forbidden_phrases=case["forbidden_phrases"],
        )

        results.append(
            AnswerCaseResult(
                case_id=case["id"],
                category=case["category"],
                question=case["question"],
                evaluation=evaluation,
            )
        )

    return AnswerEvaluationReport(results=results)
