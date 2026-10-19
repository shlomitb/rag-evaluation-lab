
"""Tests for the end-to-end RAG pipeline."""

from src import rag_pipeline
from tests.test_retrieval_pipeline import FakeEmbeddingProvider


class FakeTextGenerator:
    """A predictable generator that makes no API calls."""

    def __init__(self, response: str = "Standard shipping takes 3-5 days."):
        self.response = response
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_answer_question_retrieves_context_and_generates_answer(monkeypatch):
    chunks = [
        {
            "chunk_id": "shipping_policy.txt:2",
            "content": "Standard shipping takes 3-5 business days.",
            "score": 0.92,
        }
    ]
    retrieval_calls = []

    def fake_search_documents(**kwargs):
        retrieval_calls.append(kwargs)
        return chunks

    monkeypatch.setattr(
        rag_pipeline,
        "search_documents",
        fake_search_documents,
    )

    generator = FakeTextGenerator(
        "Standard shipping takes 3-5 business days."
    )

    answer = rag_pipeline.answer_question(
        question="How long does standard shipping take?",
        directory="fake_documents",
        generator=generator,
        top_k=5,
    )

    assert answer == "Standard shipping takes 3-5 business days."

    # Verify that retrieval was called with the expected settings.
    assert len(retrieval_calls) == 1
    assert retrieval_calls[0]["query"] == (
        "How long does standard shipping take?"
    )
    assert retrieval_calls[0]["directory"] == "fake_documents"
    assert retrieval_calls[0]["top_k"] == 5

    # Verify that the retrieved context reached the generator.
    assert len(generator.prompts) == 1
    assert "Standard shipping takes 3-5 business days." in (
        generator.prompts[0]
    )
    assert "How long does standard shipping take?" in (
        generator.prompts[0]
    )


def test_answer_question_returns_no_context_response(monkeypatch):
    def fake_search_documents(**kwargs):
        return []

    monkeypatch.setattr(
        rag_pipeline,
        "search_documents",
        fake_search_documents,
    )

    generator = FakeTextGenerator()

    answer = rag_pipeline.answer_question(
        question="Does the mouse support Bluetooth 5.3?",
        directory="fake_documents",
        generator=generator,
    )

    assert answer == (
        "I could not find relevant information in the available "
        "documents to answer your question."
    )

    # No context means no need to call the text generator.
    assert generator.prompts == []


def test_answer_question_passes_embedding_provider_to_retrieval(
    monkeypatch,
):
    fake_provider = FakeEmbeddingProvider()
    retrieval_calls = []

    def fake_search_documents(**kwargs):
        retrieval_calls.append(kwargs)
        return []

    monkeypatch.setattr(
        rag_pipeline,
        "search_documents",
        fake_search_documents,
    )

    rag_pipeline.answer_question(
        question="What is the return policy?",
        directory="fake_documents",
        generator=FakeTextGenerator(),
        embedding_provider=fake_provider,
    )

    assert retrieval_calls[0]["embedding_provider"] is fake_provider
