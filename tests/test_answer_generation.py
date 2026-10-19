
"""Tests for answer prompt construction and generation."""

import pytest

from src.answer_generation import (
    NO_CONTEXT_RESPONSE,
    build_answer_prompt,
    generate_answer,
)


class FakeTextGenerator:
    """A deterministic generator for tests; it makes no API calls."""

    def __init__(self, response: str = "Standard shipping takes 3-5 days."):
        self.response = response
        self.prompts: list[str] = []

    def generate_text(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_prompt_includes_question_and_retrieved_context():
    chunks = [
        {
            "chunk_id": "shipping_policy.txt:2",
            "content": "Standard shipping takes 3-5 business days.",
        }
    ]

    prompt = build_answer_prompt(
        "How long does standard shipping take?",
        chunks,
    )

    assert "How long does standard shipping take?" in prompt
    assert "Standard shipping takes 3-5 business days." in prompt
    assert "[shipping_policy.txt:2]" in prompt


def test_prompt_includes_all_retrieved_chunks():
    chunks = [
        {
            "chunk_id": "shipping_policy.txt:3",
            "content": "Express shipping costs $15.",
        },
        {
            "chunk_id": "shipping_policy.txt:4",
            "content": "Express shipping takes 1-2 business days.",
        },
    ]

    prompt = build_answer_prompt(
        "What is the cost and delivery time for express shipping?",
        chunks,
    )

    assert "Express shipping costs $15." in prompt
    assert "Express shipping takes 1-2 business days." in prompt


def test_prompt_instructs_model_not_to_invent_facts():
    prompt = build_answer_prompt(
        "Does the mouse support Bluetooth 5.3?",
        [
            {
                "chunk_id": "product_catalog.txt:0",
                "content": "Wireless mouse, priced at $49.99.",
            }
        ],
    )

    assert "Do not invent facts" in prompt
    assert "say so clearly" in prompt
    assert "using only the information" in prompt


def test_prompt_handles_no_retrieved_chunks():
    prompt = build_answer_prompt(
        "What is the shipping cost?",
        [],
    )

    assert "No relevant document chunks were retrieved." in prompt
    assert "Do not invent facts" in prompt
    assert "What is the shipping cost?" in prompt


def test_prompt_rejects_empty_question():
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_answer_prompt("   ", [])


def test_prompt_rejects_chunk_without_string_content():
    chunks = [{"chunk_id": "shipping_policy.txt:2"}]

    with pytest.raises(
        ValueError,
        match="must contain string content",
    ):
        build_answer_prompt("How long does shipping take?", chunks)


def test_generate_answer_returns_generator_response():
    generator = FakeTextGenerator(
        "Standard shipping takes 3-5 business days."
    )
    chunks = [
        {
            "chunk_id": "shipping_policy.txt:2",
            "content": "Standard shipping takes 3-5 business days.",
        }
    ]

    answer = generate_answer(
        "How long does standard shipping take?",
        chunks,
        generator,
    )

    assert answer == "Standard shipping takes 3-5 business days."
    assert len(generator.prompts) == 1
    assert "How long does standard shipping take?" in generator.prompts[0]
    assert "Standard shipping takes 3-5 business days." in generator.prompts[0]


def test_generate_answer_does_not_call_generator_without_context():
    generator = FakeTextGenerator()

    answer = generate_answer(
        "Does the mouse support Bluetooth 5.3?",
        [],
        generator,
    )

    assert answer == NO_CONTEXT_RESPONSE
    assert generator.prompts == []


def test_generate_answer_rejects_empty_question():
    generator = FakeTextGenerator()

    with pytest.raises(ValueError, match="Question cannot be empty"):
        generate_answer("   ", [], generator)

    assert generator.prompts == []
