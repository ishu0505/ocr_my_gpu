"""Single-process CUDA BF16 Qwen3-VL adapter, imported only in engine runtime."""

import time
from dataclasses import asdict
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from gpu_my_ocr.models import InferenceRequest, InferenceResult

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


class TransformersEngine:
    """Load one pinned model on CUDA; infer image requests without quantization/offload."""

    def __init__(self, revision: str, max_pixels: int = 1280 * 32 * 32) -> None:
        """Load BF16 weights and processor at revision; cap processor image pixels."""
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native CUDA BF16 support is required; no precision fallback")
        started = time.perf_counter()
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, revision=revision, min_pixels=256 * 32 * 32, max_pixels=max_pixels
        )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            revision=revision,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ).eval()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started
        if any(p.device.type != "cuda" for p in self.model.parameters()):
            raise RuntimeError("CPU/offloaded weights invalidate this baseline")
        if any(p.dtype != torch.bfloat16 for p in self.model.parameters()):
            raise RuntimeError("Non-BF16 model parameter")
        self.details: dict[str, Any] = {}

    async def infer(self, request: InferenceRequest) -> InferenceResult:
        """Return raw generated text/timing; retain tensor, token and memory diagnostics."""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        with Image.open(request.image_path) as source:
            image = source.convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        input_tokens = inputs.input_ids.shape[1]
        torch.cuda.synchronize()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=request.max_output_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                repetition_penalty=1.0,
                use_cache=True,
            )
        torch.cuda.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000
        tokens = output[0, input_tokens:]
        text = self.processor.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.details = {
            "generation_ms": generation_ms,
            "input_tokens": input_tokens,
            "image_grid_thw": inputs.image_grid_thw.tolist(),
            "input_device": str(inputs.input_ids.device),
            "pixel_device": str(inputs.pixel_values.device),
            "truncated": len(tokens) >= request.max_output_tokens
            and int(tokens[-1]) not in eos_ids,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        result = InferenceResult(
            request.page_id, text, (time.perf_counter() - started) * 1000, len(tokens)
        )
        self.details.update(asdict(result))
        return result
