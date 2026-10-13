from pathlib import Path

from gpu_my_ocr.models import InferenceRequest, InferenceResult


def test_request_defaults() -> None:
    request = InferenceRequest("a", Path("a.png"), "Read this")
    assert request.max_output_tokens == 512


def test_result_optional_metrics() -> None:
    result = InferenceResult("a", "text", 20.0)
    assert result.time_to_first_token_ms is None
