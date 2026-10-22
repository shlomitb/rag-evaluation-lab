
"""Tests for the Gemini text generator."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.gemini_generator import DEFAULT_MODEL, GeminiTextGenerator


@pytest.fixture
def mock_gemini_client(monkeypatch):
    """Replace the Gemini client so tests never call the real API."""
    client = Mock()
    monkeypatch.setattr(
        "src.gemini_generator.genai.Client",
        Mock(return_value=client),
    )
    return client


def test_generate_text_returns_generated_text(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value = (
        SimpleNamespace(text="  The answer is $5.99.  ")
    )
    generator = GeminiTextGenerator()

    result = generator.generate_text("What is standard shipping?")

    assert result == "The answer is $5.99."
    mock_gemini_client.models.generate_content.assert_called_once_with(
        model=DEFAULT_MODEL,
        contents="What is standard shipping?",
    )


def test_custom_model_is_used(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value = (
        SimpleNamespace(text="Answer")
    )
    generator = GeminiTextGenerator(model="test-model")

    generator.generate_text("Test question")

    mock_gemini_client.models.generate_content.assert_called_once_with(
        model="test-model",
        contents="Test question",
    )


def test_empty_prompt_is_rejected_without_api_call(mock_gemini_client):
    generator = GeminiTextGenerator()

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        generator.generate_text("   ")

    mock_gemini_client.models.generate_content.assert_not_called()


def test_missing_response_text_raises_error(mock_gemini_client):
    mock_gemini_client.models.generate_content.return_value = (
        SimpleNamespace(text=None)
    )
    generator = GeminiTextGenerator()

    with pytest.raises(ValueError, match="Gemini returned no text response"):
        generator.generate_text("Test question")

    mock_gemini_client.models.generate_content.assert_called_once()
