from src.ragas_evaluation import get_retry_delay


def test_exponential_backoff_fallback():
    error = Exception("Temporary failure")

    assert get_retry_delay(error, attempt=1) == 2
    assert get_retry_delay(error, attempt=2) == 4
    assert get_retry_delay(error, attempt=3) == 8


def test_retry_delay_from_error_message():
    error = Exception(
        "Please retry in 55.268949147s."
    )

    assert get_retry_delay(error, attempt=1) == 55.268949147