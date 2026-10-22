
"""
Make one real API call to verify Gemini connectivity.
Run with: python -m examples.try_gemini
"""

from src.gemini_generator import GeminiTextGenerator


def main() -> None:
    generator = GeminiTextGenerator()

    answer = generator.generate_text(
        "Answer in one sentence: What is standard shipping?"
    )

    print("Gemini connection successful.")
    print(f"Model: {generator.model}")
    print(f"Response: {answer}")


if __name__ == "__main__":
    main()
