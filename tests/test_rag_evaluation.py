
"""Tests for the RAG evaluation integration."""

from src import rag_evaluation
from tests.test_retrieval_pipeline import FakeEmbeddingProvider


class FakeTextGenerator:
    """Return a predetermined answer without calling an LLM."""

    def __init__(self, answer: str):
        self.answer = answer
        self.received_prompt = None

    def generate_text(self, prompt: str) -> str:
        self.received_prompt = prompt
        return self.answer


def test_evaluate_rag_question_returns_answer_and_retrieved_evidence(
    monkeypatch, tmp_path
):
    """The result should preserve both the answer and its source chunks."""
    chunks = [
        {
            "chunk_id": "shipping_01",
            "content": "Standard shipping costs $5.99.",
            "source": "shipping_policy.txt",
        }
    ]

    def fake_search_documents(**kwargs):
        assert kwargs["query"] == "How much does standard shipping cost?"
        assert kwargs["directory"] == tmp_path
        return chunks

    monkeypatch.setattr(
        rag_evaluation, "search_documents", fake_search_documents
    )

    generator = FakeTextGenerator("Standard shipping costs $5.99.")

    result = rag_evaluation.evaluate_rag_question(
        question="How much does standard shipping cost?",
        directory=tmp_path,
        generator=generator,
    )

    assert result["question"] == "How much does standard shipping cost?"
    assert result["answer"] == "Standard shipping costs $5.99."
    assert result["retrieved_chunks"] == chunks
    assert "Standard shipping costs $5.99." in generator.received_prompt


def test_evaluate_rag_question_handles_empty_retrieval(
    monkeypatch, tmp_path
):
    """No retrieved chunks should produce the predefined no-context answer."""
    monkeypatch.setattr(
        rag_evaluation, "search_documents", lambda **kwargs: []
    )

    generator = FakeTextGenerator("This should not be generated.")

    result = rag_evaluation.evaluate_rag_question(
        question="What is the return policy?",
        directory=tmp_path,
        generator=generator,
    )

    assert result["retrieved_chunks"] == []
    assert "could not find relevant information" in result["answer"]
    assert generator.received_prompt is None


def test_evaluate_rag_question_passes_retrieval_configuration(
    monkeypatch, tmp_path
):
    """Custom retrieval settings and embedding provider should be forwarded."""
    captured = {}

    def fake_search_documents(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        rag_evaluation, "search_documents", fake_search_documents
    )

    generator = FakeTextGenerator("Unused")


    provider = FakeEmbeddingProvider()

    rag_evaluation.evaluate_rag_question(
        question="What is the return policy?",
        directory=tmp_path,
        generator=generator,
        top_k=5,
        chunk_size=300,
        overlap=25,
        embedding_provider=provider,
    )

    assert captured["top_k"] == 5
    assert captured["chunk_size"] == 300
    assert captured["overlap"] == 25
    assert captured["embedding_provider"] is provider
