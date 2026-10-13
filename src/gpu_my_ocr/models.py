"""Portable benchmark request and result records."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceRequest:
    """Inputs for a single image request."""

    page_id: str
    image_path: Path
    prompt: str
    max_output_tokens: int = 512


@dataclass(frozen=True)
class InferenceResult:
    """Portable inference result; missing provider timings remain None."""

    page_id: str
    text: str
    elapsed_ms: float
    output_tokens: int | None = None
    time_to_first_token_ms: float | None = None
