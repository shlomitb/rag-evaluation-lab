
"""Gemini implementation of the TextGenerator interface."""

from google import genai
from dotenv import load_dotenv

from src.answer_generation import TextGenerator


DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiTextGenerator:
    """Generate text using the Gemini API."""

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        """Initialize the Gemini client and select a model."""
        load_dotenv()
        self.client = genai.Client()
        self.model = model

    def generate_text(self, prompt: str) -> str:
        """Generate text from a prompt using Gemini."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        if response.text is None:
            raise ValueError("Gemini returned no text response.")

        return response.text.strip()
