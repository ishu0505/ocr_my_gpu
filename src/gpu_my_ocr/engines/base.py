"""Minimal engine interface; concrete adapters arrive in phases 1 and 2."""

from typing import Protocol
from gpu_my_ocr.models import InferenceRequest, InferenceResult


class InferenceEngine(Protocol):
    """Asynchronous inference contract shared by supported engines."""

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Return model output and request-level timing for an input page."""
        ...
