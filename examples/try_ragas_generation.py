
from dotenv import load_dotenv
import inspect
import os

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness


def main():
    load_dotenv()

    client = AsyncOpenAI(
        api_key=os.environ["GEMINI_API_KEY"],
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    llm = llm_factory(
        "gemini-3.5-flash-lite",
        provider="openai",
        client=client,
        adapter="instructor",
    )

    metric = Faithfulness(llm=llm)
    print("Score signature:", inspect.signature(metric.score))
    print("Ascore signature:", inspect.signature(metric.ascore))
    result = metric.score(
        user_input="How long does standard shipping take?",
        response="Standard shipping takes 1–2 business days.",
        retrieved_contexts=[
            "Standard shipping takes 3–5 business days."
        ],
    )

    print("Faithfulness result:", result)


if __name__ == "__main__":
    main()
