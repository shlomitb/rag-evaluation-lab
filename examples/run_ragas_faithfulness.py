"""
Verify that the new evaluator works asynchronously and independently of the rest of your RAG pipeline
RAGAS judged the answer as unsupported by the retrieved context.

Faithfulness is checking whether the answer is supported by the retrieved context—
not whether the answer sounds plausible or whether it contains the expected wording from your deterministic

Run with: python -m examples.run_ragas_faithfulness
"""

import asyncio

from src.ragas_evaluation import RagasFaithfulnessEvaluator


async def main() -> None:
    evaluator = RagasFaithfulnessEvaluator()

    try:
        question = "How long does standard shipping take?"

        retrieved_contexts = [
            "Standard shipping takes 3–5 business days."
        ]

        #answer = "Standard shipping takes 1–2 business days." #Faithfulness result: 0.0
        answer = "Standard shipping takes 3–5 business days." #Faithfulness result: 1.0

        result = await evaluator.evaluate(
            question=question,
            answer=answer,
            retrieved_contexts=retrieved_contexts,
        )

        print("Faithfulness result:", result)

    finally:
        await evaluator.close()


if __name__ == "__main__":
    asyncio.run(main())