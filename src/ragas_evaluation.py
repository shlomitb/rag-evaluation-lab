import asyncio
import re

from openai import (
    APIConnectionError,
    APIStatusError,
    RateLimitError,
)
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from typing import Any

load_dotenv()

def get_retry_delay(
    error: Exception,
    attempt: int,
) -> float:
    """
    Determine how long to wait before retrying.

    Prefer the delay suggested by the API.
    Otherwise, use exponential backoff.
    """
    # First, check for a Retry-After response header.
    response = getattr(error, "response", None)

    if response is not None:
        retry_after = response.headers.get("retry-after")

        if retry_after is not None:
            try:
                return float(retry_after)
            except ValueError:
                pass

    # Gemini may include its retry delay in the error message.
    match = re.search(
        r"retry in ([\d.]+)s",
        str(error),
        re.IGNORECASE,
    )

    if match:
        return float(match.group(1))

    # Fallback: exponential backoff.
    return 2 ** attempt

class RagasFaithfulnessEvaluator:
    """
    Evaluate whether an answer is supported by its retrieved contexts
    using the RAGAS Faithfulness metric.
    """

    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=(
                "https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
        )

        self.llm = llm_factory(
            "gemini-3.5-flash-lite",
            provider="openai",
            client=self.client,
            adapter="instructor",
        )

        self.metric = Faithfulness(llm=self.llm)

    async def evaluate(
            self,
            question: str,
            answer: str,
            retrieved_contexts: list[str],
            max_retries: int = 3,
    ) -> float:
        """
        Evaluate answer faithfulness with retry handling.

        max_retries is the number of retries after the
        initial attempt.
        """

        for attempt in range(max_retries + 1):
            try:
                result = await self.metric.ascore(
                    user_input=question,
                    response=answer,
                    retrieved_contexts=retrieved_contexts,
                )

                return result.value

            except (
                    RateLimitError,
                    APIConnectionError,
                    APIStatusError,
            ) as error:

                # Do not retry permanent HTTP errors.
                if (
                        isinstance(error, APIStatusError)
                        and error.status_code < 500
                        and error.status_code != 429
                ):
                    raise

                # No retries remain.
                if attempt == max_retries:
                    raise

                delay = get_retry_delay(
                    error=error,
                    attempt=attempt + 1,
                )

                print(
                    f"RAGAS evaluation failed: {error.__class__.__name__}. "
                    f"Retrying in {delay:.1f} seconds "
                    f"(retry {attempt + 1}/{max_retries})...",
                    flush=True,
                )

                await asyncio.sleep(delay)

        # This line should never be reached.
        raise RuntimeError("Faithfulness evaluation ended unexpectedly.")

    async def close(self) -> None:
        """Close the underlying asynchronous API client."""
        await self.client.close()