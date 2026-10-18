"""
These tests use a fake retrieval function, so they don't load the embedding model or make any external calls.
They check the evaluation logic itself, including the cases where a question needs two chunks and where it has no known answer.
"""


import json

import pytest

import src.run_retrieval_evaluation as evaluation


def write_dataset(tmp_path, dataset):
    """Write a temporary evaluation dataset and return its path."""
    path = tmp_path / "retrieval_eval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def make_item(
    question,
    relevant_chunk_ids,
    *,
    answerable=True,
    case_type="baseline",
    require_all_relevant_chunks=False,
):
    return {
        "question": question,
        "relevant_chunk_ids": relevant_chunk_ids,
        "answerable": answerable,
        "case_type": case_type,
        "require_all_relevant_chunks": require_all_relevant_chunks,
    }


def make_chunk(chunk_id):
    return {
        "chunk_id": chunk_id,
        "content": f"Content for {chunk_id}",
        "similarity": 0.9,
    }


def test_load_evaluation_dataset_accepts_answerable_and_unanswerable_items(
    tmp_path,
):
    dataset = [
        make_item("How long is shipping?", ["shipping_policy.txt:2"]),
        make_item(
            "Does the mouse support Bluetooth 5.3?",
            [],
            answerable=False,
            case_type="unanswerable",
        ),
    ]
    path = write_dataset(tmp_path, dataset)

    loaded = evaluation.load_evaluation_dataset(path)

    assert loaded == dataset


def test_load_evaluation_dataset_rejects_empty_dataset(tmp_path):
    path = write_dataset(tmp_path, [])

    with pytest.raises(ValueError, match="non-empty list"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_answerable_item_without_relevant_ids(
    tmp_path,
):
    dataset = [make_item("How long is shipping?", [])]
    path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="must have at least one relevant chunk"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_unanswerable_item_with_relevant_ids(
    tmp_path,
):
    dataset = [
        make_item(
            "Does the mouse support Bluetooth 5.3?",
            ["product_catalog.txt:0"],
            answerable=False,
            case_type="unanswerable",
        )
    ]
    path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="must have an empty"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_missing_case_type(tmp_path):
    item = make_item("How long is shipping?", ["shipping_policy.txt:2"])
    del item["case_type"]
    path = write_dataset(tmp_path, [item])

    with pytest.raises(ValueError, match="case_type"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_non_boolean_answerable(tmp_path):
    item = make_item("How long is shipping?", ["shipping_policy.txt:2"])
    item["answerable"] = "true"
    path = write_dataset(tmp_path, [item])

    with pytest.raises(ValueError, match="boolean answerable"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_multi_fact_with_only_one_chunk(
    tmp_path,
):
    dataset = [
        make_item(
            "What does shipping cost and how long does it take?",
            ["shipping_policy.txt:3"],
            case_type="multi_fact",
            require_all_relevant_chunks=True,
        )
    ]
    path = write_dataset(tmp_path, dataset)

    with pytest.raises(ValueError, match="at least two distinct"):
        evaluation.load_evaluation_dataset(path)


def test_evaluate_retrieval_reports_answerable_and_multi_fact_metrics(
    tmp_path, monkeypatch
):
    dataset = [
        make_item(
            "How long is standard shipping?",
            ["shipping_policy.txt:2"],
            case_type="baseline",
        ),
        make_item(
            "What warranty does the keyboard have?",
            ["product_catalog.txt:1"],
            case_type="baseline",
        ),
        make_item(
            "What does express shipping cost and how long does it take?",
            ["shipping_policy.txt:3", "shipping_policy.txt:4"],
            case_type="multi_fact",
            require_all_relevant_chunks=True,
        ),
        make_item(
            "Does the mouse support Bluetooth 5.3?",
            [],
            answerable=False,
            case_type="unanswerable",
        ),
    ]
    path = write_dataset(tmp_path, dataset)

    results_by_question = {
        # Relevant chunk ranked first.
        "How long is standard shipping?": [
            make_chunk("shipping_policy.txt:2"),
            make_chunk("shipping_policy.txt:5"),
        ],
        # No relevant chunk retrieved.
        "What warranty does the keyboard have?": [
            make_chunk("product_catalog.txt:0"),
            make_chunk("product_catalog.txt:2"),
        ],
        # One of two required chunks retrieved: a hit, but incomplete.
        "What does express shipping cost and how long does it take?": [
            make_chunk("shipping_policy.txt:3"),
            make_chunk("shipping_policy.txt:5"),
        ],
        # Unanswerable questions can still return candidate chunks.
        "Does the mouse support Bluetooth 5.3?": [
            make_chunk("product_catalog.txt:0"),
            make_chunk("product_catalog.txt:1"),
        ],
    }

    def fake_search_documents(query, directory, top_k, embedding_provider):
        return results_by_question[query][:top_k]

    monkeypatch.setattr(
        evaluation, "search_documents", fake_search_documents
    )

    report = evaluation.evaluate_retrieval(
        evaluation_path=path,
        documents_dir=tmp_path,
        top_k=2,
    )

    assert report["total_questions"] == 4
    assert report["answerable_question_count"] == 3
    assert report["unanswerable_question_count"] == 1

    # Only the three answerable questions contribute to ranking metrics.
    assert report["answerable_metrics"]["question_count"] == 3
    assert report["answerable_metrics"]["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert report["answerable_metrics"]["mrr"] == pytest.approx(2 / 3)

    # The multi-fact question retrieved one required chunk, but not both.
    assert report["multi_fact_question_count"] == 1
    assert report["multi_fact_complete_rate_at_k"] == 0.0

    multi_fact_result = report["results"][2]
    assert multi_fact_result["hit"] is True
    assert multi_fact_result["all_required_retrieved"] is False

    # Unanswerable questions are reported, not counted as ranking hits/misses.
    unanswerable_result = report["results"][3]
    assert unanswerable_result["answerable"] is False
    assert unanswerable_result["hit"] is None
    assert unanswerable_result["first_relevant_rank"] is None

    # Category metrics let us see where retrieval is weaker.
    assert report["metrics_by_case_type"]["baseline"]["question_count"] == 2
    assert report["metrics_by_case_type"]["baseline"]["hit_rate_at_k"] == 0.5
    assert report["metrics_by_case_type"]["multi_fact"]["hit_rate_at_k"] == 1.0
    assert (
        report["metrics_by_case_type"]["unanswerable"][
            "unanswerable_question_count"
        ]
        == 1
    )


def test_evaluate_retrieval_rejects_non_positive_top_k(tmp_path):
    path = write_dataset(
        tmp_path,
        [make_item("How long is shipping?", ["shipping_policy.txt:2"])],
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        evaluation.evaluate_retrieval(evaluation_path=path, top_k=0)
