
"""
Run retrieval evaluation against a labeled question dataset.

1. Loads the evaluation questions and their expected relevant chunk IDs.
2. Sends each question through your existing search_documents() pipeline.
3. Checks whether a relevant chunk appears in the retrieved results.
4. Calculates Hit Rate@K and MRR using the metric functions you already tested.
5. Reports each question as a HIT or MISS, including the rank of its first relevant result.
"""

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
    """Load and validate the basic structure of the evaluation dataset."""
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

        if not isinstance(question, str) or not question.strip():
            raise ValueError(
                f"Evaluation item {index} must have a non-empty question."
            )

        if (
            not isinstance(relevant_ids, list)
            or not relevant_ids
            or not all(
                isinstance(chunk_id, str) and chunk_id.strip()
                for chunk_id in relevant_ids
            )
        ):
            raise ValueError(
                f"Evaluation item {index} must have non-empty "
                "relevant_chunk_ids."
            )

    return dataset


def evaluate_retrieval(
    evaluation_path: str | Path = DEFAULT_EVALUATION_PATH,
    documents_dir: str | Path = DEFAULT_DOCUMENTS_DIR,
    top_k: int = 3,
    embedding_provider: EmbeddingProvider | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval and return aggregate metrics and per-question results."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    dataset = load_evaluation_dataset(evaluation_path)
    question_results = []

    for item in dataset:
        question = item["question"]
        relevant_ids = set(item["relevant_chunk_ids"])

        retrieved_chunks = search_documents(
            query=question,
            directory=documents_dir,
            top_k=top_k,
            embedding_provider=embedding_provider,
        )

        retrieved_ids = [
            str(chunk["chunk_id"])
            for chunk in retrieved_chunks
        ]

        first_relevant_rank = next(
            (
                rank
                for rank, chunk_id in enumerate(retrieved_ids, start=1)
                if chunk_id in relevant_ids
            ),
            None,
        )

        question_results.append(
            {
                "question": question,
                "relevant_chunk_ids": sorted(relevant_ids),
                "retrieved_chunks": retrieved_chunks,
                "retrieved_chunk_ids": retrieved_ids,
                "hit": first_relevant_rank is not None,
                "first_relevant_rank": first_relevant_rank,
            }
        )

    retrieved_ids_by_question = [
        result["retrieved_chunk_ids"]
        for result in question_results
    ]
    relevant_ids_by_question = [
        set(result["relevant_chunk_ids"])
        for result in question_results
    ]

    return {
        "total_questions": len(question_results),
        "top_k": top_k,
        "hit_rate_at_k": hit_rate_at_k(
            retrieved_ids_by_question,
            relevant_ids_by_question,
            k=top_k,
        ),
        "mrr": mean_reciprocal_rank(
            retrieved_ids_by_question,
            relevant_ids_by_question,
        ),
        "results": question_results,
    }


def print_evaluation_report(report: dict[str, Any]) -> None:
    """Print aggregate metrics and the results for each question."""
    top_k = report["top_k"]

    print("\nRETRIEVAL EVALUATION")
    print("=" * 60)
    print(f"Questions evaluated: {report['total_questions']}")
    print(f"Hit Rate@{top_k}: {report['hit_rate_at_k']:.3f}")
    print(f"MRR: {report['mrr']:.3f}")

    print("\nINDIVIDUAL RESULTS")
    print("=" * 60)

    for index, result in enumerate(report["results"], start=1):
        rank = result["first_relevant_rank"]
        status = "HIT" if result["hit"] else "MISS"
        rank_text = str(rank) if rank is not None else "not retrieved"

        print(f"\n{index}. [{status}] {result['question']}")
        print(f"   First relevant result: {rank_text}")

        for position, chunk in enumerate(
            result["retrieved_chunks"],
            start=1,
        ):
            print(
                f"   {position}. {chunk['chunk_id']} "
                f"(similarity: {chunk['similarity']:.4f})"
            )
            print(f"      {chunk['content'].replace(chr(10), ' ')}")


if __name__ == "__main__":
    evaluation_report = evaluate_retrieval()
    print_evaluation_report(evaluation_report)
