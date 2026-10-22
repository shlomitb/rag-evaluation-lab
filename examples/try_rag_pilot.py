
"""Run a small real-LLM RAG evaluation pilot."""

from pathlib import Path

from src.gemini_generator import GeminiTextGenerator
from src.rag_evaluation import evaluate_rag_question


def main() -> None:
    documents_directory = Path("data/sample_documents")
    generator = GeminiTextGenerator()

    questions = [
        "How much does standard shipping cost, and how long does it take?",
        "How much does the wireless mouse cost, and how long does its battery last?",
        "How long do customers have to return a defective item?",
    ]

    for index, question in enumerate(questions, start=1):
        print("\n" + "=" * 70)
        print(f"QUESTION {index}: {question}")

        result = evaluate_rag_question(
            question=question,
            directory=documents_directory,
            generator=generator,
            top_k=3,
        )

        print("\nRETRIEVED EVIDENCE:")

        for chunk in result["retrieved_chunks"]:
            print(f"\nSource: {chunk.get('source', 'Unknown')}")
            print(f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}")
            print(f"Content: {chunk.get('content', '')}")

        print("\nGENERATED ANSWER:")
        print(result["answer"])


if __name__ == "__main__":
    main()
