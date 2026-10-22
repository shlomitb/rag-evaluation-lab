
"""
Run the answer evaluation dataset through the real RAG pipeline.

To run all test cases:
python -m examples.run_answer_evaluation

To run 1 test case, ex:
python -m examples.run_answer_evaluation --case-id shipping_05


"""
import argparse
import json
from pathlib import Path

from src.answer_evaluation import evaluate_answer
from src.gemini_generator import GeminiTextGenerator
from src.rag_evaluation import evaluate_rag_question


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "answer_eval.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "generated" / "answer_eval_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        help="Run only the evaluation case with this ID",
    )
    args = parser.parse_args()

    with DATASET_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    # NEW: Filter to one case if --case-id was provided
    if args.case_id:
        cases = [
            case for case in cases
            if case["id"] == args.case_id
        ]

        if not cases:
            raise ValueError(
                f"No evaluation case found with ID: {args.case_id}"
            )

    generator = GeminiTextGenerator()
    results = []

    for index, case in enumerate(cases, start=1):
        case_id = case["id"]
        question = case["question"]

        print(f"\n[{index}/{len(cases)}] {case_id}: {question}")

        try:
            rag_result = evaluate_rag_question(
                question=question,
                directory=PROJECT_ROOT / "data" / "sample_documents",
                generator=generator,
                top_k=3,
            )

            answer = rag_result["answer"]
            evaluation = evaluate_answer(
                answer=answer,
                required_phrases=case["required_phrases"],
                forbidden_phrases=case.get("forbidden_phrases", []),
            )

            result = {
                "case_id": case_id,
                "category": case.get("category", "uncategorized"),
                "question": question,
                "answer": answer,
                "retrieved_chunks": rag_result["retrieved_chunks"],
                "passed": evaluation.passed,
                "required_phrases_found": evaluation.required_phrases_found,
                "missing_required_phrases": (
                    evaluation.missing_required_phrases
                ),
                "forbidden_phrases_found": (
                    evaluation.forbidden_phrases_found
                ),
                "required_phrase_coverage": (
                    evaluation.required_phrase_coverage
                ),
            }

        except Exception as exc:
            # Preserve the failure instead of silently dropping this case.
            result = {
                "case_id": case_id,
                "question": question,
                "error": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }

        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"Result: {'PASS' if result['passed'] else 'FAIL'}")
            print(f"Answer: {result['answer']}")

    successful = sum("error" not in result for result in results)
    passed = sum(result.get("passed", False) for result in results)

    report = {
        "model": generator.model,
        "top_k": 3,
        "total_cases": len(cases),
        "successfully_evaluated": successful,
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        "results": results,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Total cases: {len(cases)}")
    print(f"Successfully evaluated: {successful}")
    print(f"Passed: {passed}")
    print(f"Failed or errored: {len(results) - passed}")
    print(f"Report saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
