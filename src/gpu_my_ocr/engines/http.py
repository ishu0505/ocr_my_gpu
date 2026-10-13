"""Local streaming OpenAI-compatible adapters with per-request immutable diagnostics."""

import asyncio
import base64
import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from gpu_my_ocr.models import InferenceRequest, InferenceResult


@dataclass(frozen=True)
class ServingResult(InferenceResult):
    """Portable HTTP response plus streaming usage and finish reason, never inferred tokens."""

    input_tokens: int | None = None
    finish_reason: str | None = None
    truncated: bool = False
    stream_chunks: int = 0


class RequestFailure(RuntimeError):
    """Failed transport retaining any generated partial text and original error."""

    def __init__(self, message: str, partial_text: str = "") -> None:
        super().__init__(message)
        self.partial_text = partial_text


def consume_stream(lines: Any, started: float, clock: Any = time.perf_counter) -> dict:
    """Collect SSE content/usage; TTFT is first nonempty text delta, not first HTTP byte."""
    text = ""
    usage: dict = {}
    finish = None
    first_token = None
    chunks = 0
    done = False
    try:
        for line in lines:
            decoded = line.decode().strip() if isinstance(line, bytes) else line.strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                done = True
                break
            payload = json.loads(data)
            if "error" in payload:
                raise RequestFailure(json.dumps(payload["error"]), text)
            if payload.get("usage"):
                usage = payload["usage"]
            for choice in payload.get("choices", []):
                content = choice.get("delta", {}).get("content") or ""
                if content:
                    first_token = clock() if first_token is None else first_token
                    text += content
                    chunks += 1
                finish = choice.get("finish_reason") or finish
    except (OSError, ValueError) as error:
        raise RequestFailure(str(error), text) from error
    if not done or finish is None:
        raise RequestFailure("Incomplete SSE response (missing DONE or finish reason)", text)
    return {
        "text": text,
        "output_tokens": usage.get("completion_tokens"),
        "input_tokens": usage.get("prompt_tokens"),
        "finish_reason": finish,
        "time_to_first_token_ms": (first_token - started) * 1000
        if first_token is not None
        else None,
        "truncated": finish == "length",
        "stream_chunks": chunks,
    }


class OpenAIImageEngine:
    """Serve image requests to a loopback vLLM/SGLang endpoint, using the existing Protocol."""

    def __init__(
        self,
        base_url: str,
        model: str,
        engine: str,
        timeout: float = 180,
        schema: dict | None = None,
    ) -> None:
        """Configure the local endpoint and optional separately labelled JSON schema experiment."""
        self.base_url, self.model, self.engine = base_url.rstrip("/"), model, engine
        self.timeout, self.schema = timeout, schema
        if engine not in {"vllm", "sglang"}:
            raise ValueError("Unknown serving engine")
        if not self.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("Benchmarks must use the local loopback server")

    def payload(self, request: InferenceRequest) -> dict:
        """Encode image bytes and unchanged prompt; equivalent greedy settings are explicit."""
        encoded = base64.b64encode(request.image_path.read_bytes()).decode()
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64," + encoded},
                        },
                        {"type": "text", "text": request.prompt},
                    ],
                }
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": 0,
            "top_p": 1,
            "top_k": -1,
            "repetition_penalty": 1,
            "seed": 42,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "financial_page", "strict": True, "schema": self.schema},
            }
        return body

    def _infer_sync(self, request: InferenceRequest) -> ServingResult:
        started = time.perf_counter()
        payload = json.dumps(self.payload(request)).encode()
        http_request = urllib.request.Request(
            self.base_url + "/v1/chat/completions", payload, {"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                result = consume_stream(response, started)
        except urllib.error.HTTPError as error:
            raise RequestFailure(f"HTTP {error.code}: {error.read().decode()}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RequestFailure(str(error)) from error
        return ServingResult(
            page_id=request.page_id, elapsed_ms=(time.perf_counter() - started) * 1000, **result
        )

    async def infer(self, request: InferenceRequest) -> ServingResult:
        """Return streaming generation results without blocking the bounded async load generator."""
        return await asyncio.to_thread(self._infer_sync, request)
