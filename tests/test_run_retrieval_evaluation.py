
import json

import pytest

import src.run_retrieval_evaluation as evaluation


def write_dataset(tmp_path, dataset):
    path = tmp_path / "retrieval_eval.json"
    path.write_text(json.dumps(dataset), encoding="utf-8")
    return path


def test_load_evaluation_dataset_reads_questions(tmp_path) -> None:
    dataset = [
        {
            "question": "How much is shipping?",
            "relevant_chunk_ids": ["shipping_policy.txt:0"],
        }
    ]
    path = write_dataset(tmp_path, dataset)

    assert evaluation.load_evaluation_dataset(path) == dataset


def test_load_evaluation_dataset_rejects_empty_dataset(tmp_path) -> None:
    path = write_dataset(tmp_path, [])

    with pytest.raises(ValueError, match="non-empty list"):
        evaluation.load_evaluation_dataset(path)


def test_load_evaluation_dataset_rejects_missing_relevance_labels(
    tmp_path,
) -> None:
    path = write_dataset(
        tmp_path,
        [{"question": "How much is shipping?", "relevant_chunk_ids": []}],
    )

    with pytest.raises(ValueError, match="relevant_chunk_ids"):
        evaluation.load_evaluation_dataset(path)


def test_evaluate_retrieval_calculates_metrics_and_ranks(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = [
        {
            "question": "Question one",
            "relevant_chunk_ids": ["doc.txt:0"],
        },
        {
            "question": "Question two",
            "relevant_chunk_ids": ["doc.txt:1"],
        },
        {
            "question": "Question three",
            "relevant_chunk_ids": ["doc.txt:2"],
        },
    ]
    path = write_dataset(tmp_path, dataset)

    fake_results = {
        "Question one": [
            {
                "chunk_id": "doc.txt:0",
                "content": "Relevant immediately",
                "similarity": 0.9,
            },
            {
                "chunk_id": "doc.txt:4",
                "content": "Other text",
                "similarity": 0.5,
            },
        ],
        "Question two": [
            {
                "chunk_id": "doc.txt:4",
                "content": "Other text",
                "similarity": 0.9,
            },
            {
                "chunk_id": "doc.txt:1",
                "content": "Relevant second",
                "similarity": 0.8,
            },
        ],
        "Question three": [
            {
                "chunk_id": "doc.txt:4",
                "content": "Other text",
                "similarity": 0.9,
            },
            {
                "chunk_id": "doc.txt:5",
                "content": "Other text",
                "similarity": 0.8,
            },
        ],
    }

    def fake_search_documents(
        query,
        directory,
        top_k,
        embedding_provider=None,
    ):
        return fake_results[query][:top_k]

    monkeypatch.setattr(
        evaluation,
        "search_documents",
        fake_search_documents,
    )

    report = evaluation.evaluate_retrieval(
        evaluation_path=path,
        documents_dir=tmp_path,
        top_k=2,
    )

    assert report["total_questions"] == 3
    assert report["top_k"] == 2
    assert report["hit_rate_at_k"] == pytest.approx(2 / 3)
    assert report["mrr"] == pytest.approx((1 + 1 / 2 + 0) / 3)

    assert [result["hit"] for result in report["results"]] == [
        True,
        True,
        False,
    ]
    assert [
        result["first_relevant_rank"]
        for result in report["results"]
    ] == [1, 2, None]


def test_evaluate_retrieval_rejects_invalid_top_k(tmp_path) -> None:
    path = write_dataset(
        tmp_path,
        [
            {
                "question": "Question one",
                "relevant_chunk_ids": ["doc.txt:0"],
            }
        ],
    )

    with pytest.raises(ValueError, match="top_k must be positive"):
        evaluation.evaluate_retrieval(
            evaluation_path=path,
            documents_dir=tmp_path,
            top_k=0,
        )
