
"""Run retrieval evaluation against a labeled question dataset."""

import json
from pathlib import Path
from typing import Any

from src.retrieval_metrics import hit_rate_at_k, mean_reciprocal_rank
from src.retrieval_pipeline import EmbeddingProvider, search_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVALUATION_PATH = PROJECT_ROOT / "data" / "retrieval_eval.json"
DEFAULT_DOCUMENTS_DIR = PROJECT_ROOT / "data" / "sample_documents"


def load_evaluation_dataset(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
) -> list[dict[str, Any]]:
    """Load the evaluation dataset and validate its labels."""
    path = Path(evaluation_path)
    with path.open("r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("The evaluation dataset must be a non-empty list.")

    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {index} must be an object.")

        question = item.get("question")
        relevant_ids = item.get("relevant_chunk_ids")
        answerable = item.get("answerable")
        case_type = item.get("case_type")
        require_all = item.get("require_all_relevant_chunks", False)

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Evaluation item {index} must have a non-empty question."
            )

        if not isinstance(relevant_ids, list) or not all(
            isinstance(chunk_id, str) and chunk_id.strip()
            for chunk_id in relevant_ids
        ):
            raise ValueError(
                f"Evaluation item {index} must have a list of valid "
                "relevant_chunk_ids."
            )

        if type(answerable) is not bool:
            raise ValueError(
                f"Evaluation item {index} must have a boolean answerable field."
            )

        if not isinstance(case_type, str) or not case_type.strip():
            raise ValueError(
                f"Evaluation item {index} must have a non-empty case_type."
            )

        if type(require_all) is not bool:
            raise ValueError(
                f"Evaluation item {index} must have a boolean "
                "require_all_relevant_chunks field."
            )

        if answerable and not relevant_ids:
            raise ValueError(
                f"Answerable evaluation item {index} must have at least "
                "one relevant chunk ID."
            )

        if not answerable and relevant_ids:
            raise ValueError(
                f"Unanswerable evaluation item {index} must have an empty "
                "relevant_chunk_ids list."
            )

        if require_all and not answerable:
            raise ValueError(
                f"Unanswerable evaluation item {index} cannot require "
                "all relevant chunks."
            )

        if require_all and len(set(relevant_ids)) < 2:
            raise ValueError(
                f"Evaluation item {index} must have at least two distinct "
                "relevant chunk IDs when require_all_relevant_chunks is true."
            )

    return dataset


def _calculate_answerable_metrics(
    results: list[dict[str, Any]],
    top_k: int,
) -> dict[str, Any]:
    """Calculate ranking metrics for answerable questions only."""
    if not results:
        return {
            "question_count": 0,
            "hit_rate_at_k": None,
            "mrr": None,
        }

    retrieved_ids = [result["retrieved_chunk_ids"][:top_k] for result in results]
    relevant_ids = [
        set(result["relevant_chunk_ids"])
        for result in results
    ]

    return {
        "question_count": len(results),
        "hit_rate_at_k": hit_rate_at_k(
            retrieved_ids, relevant_ids, k=top_k
        ),
        "mrr": mean_reciprocal_rank(retrieved_ids, relevant_ids),
    }


def evaluate_retrieval(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    top_k: int = 3,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval and report metrics by question type."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    dataset = load_evaluation_dataset(evaluation_path)
    question_results: list[dict[str, Any]] = []

    for item in dataset:
        question = item["question"]
        relevant_ids = item["relevant_chunk_ids"]
        answerable = item["answerable"]
        require_all = item.get("require_all_relevant_chunks", False)

        retrieved_chunks = search_documents(
            query=question,
            directory=documents_dir,
            top_k=top_k,
            embedding_provider=embedding_provider,
        )
        retrieved_ids = [
            str(chunk["chunk_id"]) for chunk in retrieved_chunks
        ]

        first_relevant_rank = None
        if answerable:
            first_relevant_rank = next(
                (
                    rank
                    for rank, chunk_id in enumerate(retrieved_ids, start=1)
                    if chunk_id in relevant_ids
                ),
                None,
            )

        # For multi-fact cases, a hit on just one chunk is not enough.
        all_required_retrieved = (
            all(chunk_id in retrieved_ids[:top_k] for chunk_id in relevant_ids)
            if require_all
            else None
        )

        question_results.append(
            {
                "question": question,
                "case_type": item["case_type"],
                "answerable": answerable,
                "relevant_chunk_ids": relevant_ids,
                "require_all_relevant_chunks": require_all,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunk_ids": retrieved_ids,
                "hit": first_relevant_rank is not None if answerable else None,
                "first_relevant_rank": first_relevant_rank,
                "all_required_retrieved": all_required_retrieved,
            }
        )

    answerable_results = [
        result for result in question_results if result["answerable"]
    ]
    unanswerable_results = [
        result for result in question_results if not result["answerable"]
    ]
    multi_fact_results = [
        result
        for result in answerable_results
        if result["require_all_relevant_chunks"]
    ]

    answerable_metrics = _calculate_answerable_metrics(
        answerable_results, top_k
    )

    # Keep answerable ranking metrics separate from unanswerable cases.
    metrics_by_case_type: dict[str, dict[str, Any]] = {}
    case_types = sorted({result["case_type"] for result in question_results})

    for case_type in case_types:
        type_results = [
            result
            for result in question_results
            if result["case_type"] == case_type
        ]
        type_answerable_results = [
            result for result in type_results if result["answerable"]
        ]
        type_unanswerable_count = sum(
            not result["answerable"] for result in type_results
        )

        metrics_by_case_type[case_type] = {
            **_calculate_answerable_metrics(type_answerable_results, top_k),
            "unanswerable_question_count": type_unanswerable_count,
        }

    multi_fact_complete_rate = (
        sum(result["all_required_retrieved"] for result in multi_fact_results)
        / len(multi_fact_results)
        if multi_fact_results
        else None
    )

    return {
        "total_questions": len(question_results),
        "answerable_question_count": len(answerable_results),
        "unanswerable_question_count": len(unanswerable_results),
        "top_k": top_k,
        "answerable_metrics": answerable_metrics,
        "multi_fact_question_count": len(multi_fact_results),
        "multi_fact_complete_rate_at_k": multi_fact_complete_rate,
        "metrics_by_case_type": metrics_by_case_type,
        "results": question_results,
    }


def print_evaluation_report(report: dict[str, Any]) -> None:
    """Print overall, category-level, and per-question results."""
    top_k = report["top_k"]
    metrics = report["answerable_metrics"]

    print("\nRETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Answerable questions: {report['answerable_question_count']}")
    print(f"Unanswerable questions: {report['unanswerable_question_count']}")

    if metrics["question_count"]:
        print(f"Hit Rate@{top_k} (answerable only): {metrics['hit_rate_at_k']:.3f}")
        print(f"MRR (answerable only): {metrics['mrr']:.3f}")
    else:
        print("No answerable questions; Hit Rate and MRR are not calculated.")

    complete_rate = report["multi_fact_complete_rate_at_k"]
    if complete_rate is not None:
        print(
            f"Multi-fact complete rate@{top_k}: {complete_rate:.3f} "
            f"({report['multi_fact_question_count']} questions)"
        )
    else:
        print("No multi-fact questions to evaluate.")

    print("\nMETRICS BY CASE TYPE")
    print("=" * 60)
    for case_type, category_metrics in report["metrics_by_case_type"].items():
        print(f"\n{case_type}")
        print(f"  Answerable questions: {category_metrics['question_count']}")
        print(
            "  Unanswerable questions: "
            f"{category_metrics['unanswerable_question_count']}"
        )
        if category_metrics["question_count"]:
            print(
                f"  Hit Rate@{top_k}: "
                f"{category_metrics['hit_rate_at_k']:.3f}"
            )
            print(f"  MRR: {category_metrics['mrr']:.3f}")

    print("\nINDIVIDUAL RESULTS")
    print("=" * 60)
    for index, result in enumerate(report["results"], start=1):
        if not result["answerable"]:
            status = "UNANSWERABLE"
            rank_text = "not applicable"
        else:
            status = "HIT" if result["hit"] else "MISS"
            rank_text = (
                str(result["first_relevant_rank"])
                if result["first_relevant_rank"] is not None
                else "not retrieved"
            )

        print(f"\n{index}. [{status}] [{result['case_type']}] {result['question']}")
        print(f"   First relevant result: {rank_text}")

        if result["require_all_relevant_chunks"]:
            complete = result["all_required_retrieved"]
            print(f"   All required chunks retrieved: {complete}")

        for position, chunk in enumerate(result["retrieved_chunks"], start=1):
            print(
                f"   {position}. {chunk['chunk_id']} "
                f"(similarity: {chunk['similarity']:.4f})"
            )
            print(f"      {chunk['content'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    evaluation_report = evaluate_retrieval()
    print_evaluation_report(evaluation_report)
