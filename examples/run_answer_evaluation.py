
"""
Run the answer evaluation dataset through the real RAG pipeline.

To run all test cases:
python -m examples.run_answer_evaluation

To run 1 test case, ex:
python -m examples.run_answer_evaluation --case-id shipping_05

The RAGAS Faithfulness metric evaluates whether the claims in the generated answer are supported by the retrieved context.
Faithfulness does not mean the answer is necessarily complete, relevant, or correct in every possible sense.
It measures support from the retrieved context. Your deterministic evaluator separately checks the required phrase

Use the ID of one of the evaluation cases in your data/answer_eval.json file.
"""
import argparse
import asyncio
import json
from pathlib import Path

from src.answer_evaluation import evaluate_answer
from src.gemini_generator import GeminiTextGenerator
from src.rag_evaluation import evaluate_rag_question
from src.ragas_evaluation import RagasFaithfulnessEvaluator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "answer_eval.json"
REPORT_PATH = PROJECT_ROOT / "reports" / "generated" / "answer_eval_report.json"


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        help="Run only the evaluation case with this ID",
    )
    args = parser.parse_args()

    with DATASET_PATH.open(encoding="utf-8") as file:
        cases = json.load(file)

    # Filter to one case if --case-id was provided.
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

    # Create the RAGAS evaluator once for the entire run.
    ragas_evaluator = RagasFaithfulnessEvaluator()

    try:
        for index, case in enumerate(cases, start=1):
            case_id = case["id"]
            question = case["question"]

            print(f"\n[{index}/{len(cases)}] {case_id}: {question}")

            try:
                # Run the RAG pipeline.
                rag_result = evaluate_rag_question(
                    question=question,
                    directory=PROJECT_ROOT / "data" / "sample_documents",
                    generator=generator,
                    top_k=3,
                )

                answer = rag_result["answer"]

                # Extract the actual text from the retrieved chunks.
                retrieved_contexts = [
                    chunk["content"]
                    for chunk in rag_result["retrieved_chunks"]
                ]

                # Run the deterministic evaluation.
                evaluation = evaluate_answer(
                    answer=answer,
                    required_phrases=case["required_phrases"],
                    forbidden_phrases=case.get("forbidden_phrases", []),
                )

                # Run Faithfulness as a separate evaluation.
                # If RAGAS fails, preserve the deterministic result.
                faithfulness_score = None
                faithfulness_error = None

                try:
                    faithfulness_score = (
                        await ragas_evaluator.evaluate(
                            question=question,
                            answer=answer,
                            retrieved_contexts=retrieved_contexts,
                        )
                    )

                except Exception as exc:
                    faithfulness_error = (
                        f"{type(exc).__name__}: {exc}"
                    )

                # Combine both evaluation results.
                result = {
                    "case_id": case_id,
                    "category": case.get(
                        "category",
                        "uncategorized",
                    ),
                    "question": question,
                    "answer": answer,
                    "retrieved_chunks": rag_result[
                        "retrieved_chunks"
                    ],
                    "passed": evaluation.passed,
                    "required_phrases_found": (
                        evaluation.required_phrases_found
                    ),
                    "missing_required_phrases": (
                        evaluation.missing_required_phrases
                    ),
                    "forbidden_phrases_found": (
                        evaluation.forbidden_phrases_found
                    ),
                    "required_phrase_coverage": (
                        evaluation.required_phrase_coverage
                    ),
                    "faithfulness_score": faithfulness_score,
                }

                # Include the Faithfulness error only if one occurred.
                if faithfulness_error:
                    result["faithfulness_error"] = (
                        faithfulness_error
                    )

            except Exception as exc:
                # Preserve failures in the RAG pipeline or
                # deterministic evaluation.
                result = {
                    "case_id": case_id,
                    "question": question,
                    "error": f"{type(exc).__name__}: {exc}",
                    "passed": False,
                }

            results.append(result)

            # Print the results for this case.
            if "error" in result:
                print(f"ERROR: {result['error']}")

            else:
                print(
                    "Deterministic result: "
                    f"{'PASS' if result['passed'] else 'FAIL'}"
                )

                print(
                    "Faithfulness score: "
                    f"{result['faithfulness_score']}"
                )

                print(f"Answer: {result['answer']}")

                if "faithfulness_error" in result:
                    print(
                        "Faithfulness evaluation error: "
                        f"{result['faithfulness_error']}"
                    )

        # Calculate the summary statistics.
        successful = sum(
            "error" not in result
            for result in results
        )

        passed = sum(
            result.get("passed", False)
            for result in results
        )

        report = {
            "model": generator.model,
            "top_k": 3,
            "total_cases": len(cases),
            "successfully_evaluated": successful,
            "passed_cases": passed,
            "failed_cases": len(results) - passed,
            "pass_rate": (
                passed / len(results)
                if results
                else 0.0
            ),
            "results": results,
        }

        # Save the evaluation report.
        REPORT_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        REPORT_PATH.write_text(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        # Print the final summary.
        print("\n" + "=" * 60)
        print(f"Total cases: {len(cases)}")
        print(f"Successfully evaluated: {successful}")
        print(f"Passed: {passed}")
        print(f"Failed or errored: {len(results) - passed}")
        print(f"Report saved to: {REPORT_PATH}")

    finally:
        # Close the asynchronous client once, after all cases.
        await ragas_evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())