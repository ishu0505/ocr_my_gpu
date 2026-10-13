"""Reproducible sequential baseline runner in the isolated Transformers interpreter."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import threading
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.extraction import PROMPT_VERSION, extraction_prompt, render_page
from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset
from gpu_my_ocr.models import InferenceRequest

FROZEN_HASH = "e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"


def save_json(path: Path, value: Any) -> None:
    """Write a new JSON artifact, rejecting overwrite."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)


class MemorySampler:
    """Sample device-wide NVML memory at 20 ms intervals; separate from torch peaks."""

    def __init__(self) -> None:
        """Initialize NVML and an idle sampling thread."""
        import pynvml

        pynvml.nvmlInit()
        self.nvml = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self.error: str | None = None
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def sample(self) -> None:
        """Capture memory until stopped; report sampling errors explicitly."""
        try:
            while not self.stopped.is_set():
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                self.peak = max(self.peak, used)
                self.stopped.wait(0.02)
        except Exception:  # noqa: BLE001 - persist sampler failures from vendor API
            self.error = traceback.format_exc()

    def close(self) -> None:
        """Join the sampler and release NVML, leaving its observations available."""
        self.stopped.set()
        self.thread.join()
        self.nvml.nvmlShutdown()


def selected_records(manifest: dict, pilot: bool) -> list[dict]:
    """Select IDs 001/010 (both layouts) per category, retaining paired variants."""
    return [
        r
        for r in manifest["records"]
        if r["document_id"].endswith(("-001", "-010"))
        and (not pilot or (r["variant"] == "digital" and r["document_id"].endswith("-010")))
    ]


async def run_request(
    engine: Any, record: dict, args: argparse.Namespace, repeat: int, output: Path
) -> dict:
    """Render, infer and strictly score one request; preserve errors and raw text."""
    page_id = f"{record['document_id']}--{record['variant']}--r{repeat}"
    truth = json.loads((args.dataset / record["ground_truth_path"]).read_text())
    prompt = extraction_prompt(truth)
    save_json(output / f"{page_id}.prompt.json", {"version": PROMPT_VERSION, "text": prompt})
    started = time.perf_counter()
    image_path = output / f"{page_id}.png"
    result = {"page_id": page_id, "record": record, "repeat": repeat}
    try:
        render_page(args.dataset / record["pdf_path"], image_path)
        prediction = await engine.infer(
            InferenceRequest(page_id, image_path, prompt, args.max_tokens)
        )
        (output / f"{page_id}.txt").write_text(prediction.text)
        result.update(asdict(prediction))
        result.update(engine.details)
        result["status"] = "ok"
    except Exception as error:  # noqa: BLE001 - persist each failed inference without repair
        result.update(
            status="oom" if "out of memory" in str(error).lower() else "failed",
            error=traceback.format_exc(),
            text="",
            output_tokens=0,
        )
    result["end_to_end_ms"] = (time.perf_counter() - started) * 1000
    result["score"] = evaluate_prediction(truth, result["text"])
    save_json(output / f"{page_id}.json", result)
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("page_id", "status", "output_tokens", "truncated", "end_to_end_ms")
            }
        ),
        flush=True,
    )
    return result


def configuration(args: argparse.Namespace, integrity: dict) -> dict:
    """Capture immutable settings and the runtime environment before loading weights."""
    from gpu_my_ocr.engines.transformers import MODEL_ID

    return {
        "model": MODEL_ID,
        "revision": REVISION,
        "precision": "bfloat16",
        "attention": "sdpa",
        "device": "cuda:0",
        "quantization": None,
        "manifest": integrity,
        "prompt_version": PROMPT_VERSION,
        "max_new_tokens": args.max_tokens,
        "max_pixels": 1280 * 32 * 32,
        "render_dpi": 150,
        "render_size": [1275, 1650],
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "repetition_penalty": 1.0,
        "seed": 42,
        "warmup": 1,
        "repeats": args.repeats,
        "pilot": args.pilot,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "libraries": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "torchvision",
                "transformers",
                "accelerate",
                "pillow",
                "pymupdf",
                "nvidia-ml-py",
            )
        },
        "gpu": subprocess.check_output(["nvidia-smi"], text=True),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/gpu_my_ocr").rglob("*.py")
        },
        "engine_lock_sha256": hashlib.sha256(
            Path("environments/transformers/uv.lock").read_bytes()
        ).hexdigest(),
    }


async def run(args: argparse.Namespace) -> None:
    """Validate frozen data, load once, warm up, then persist repeated measured requests."""
    from gpu_my_ocr.engines.transformers import TransformersEngine

    args.output.mkdir(parents=True, exist_ok=False)
    integrity = validate_dataset(args.dataset, FROZEN_HASH)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    save_json(args.output / "config.json", configuration(args, integrity))
    sampler = MemorySampler()
    sampler.thread.start()
    try:
        import torch

        torch.manual_seed(42)
        engine = TransformersEngine(REVISION)
        save_json(
            args.output / "load.json",
            {
                "seconds": engine.load_seconds,
                "context_limit": engine.model.config.text_config.max_position_embeddings,
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "gpu": torch.cuda.get_device_name(),
                "parameters": sum(p.numel() for p in engine.model.parameters()),
            },
        )
        records = selected_records(manifest, args.pilot)
        warmup = args.output / "warmup"
        warmup.mkdir()
        await run_request(engine, records[0], args, -1, warmup)
        results = []
        started = time.perf_counter()
        for repeat in range(args.repeats):
            for record in records:
                results.append(await run_request(engine, record, args, repeat, args.output))
        elapsed = time.perf_counter() - started
        save_json(
            args.output / "summary.json",
            {
                "requests": len(results),
                "wall_seconds": elapsed,
                "pages_per_second": len(results) / elapsed,
                "successful_inferences": sum(r["status"] == "ok" for r in results),
                "post_integrity": validate_dataset(args.dataset, FROZEN_HASH),
            },
        )
    except Exception:
        save_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise
    finally:
        sampler.close()
        save_json(
            args.output / "memory.json",
            {"nvml_peak_bytes": sampler.peak, "sampling_interval_ms": 20, "error": sampler.error},
        )


def main() -> None:
    """Run an explicit baseline/pilot CLI in an isolated runtime; output must be new."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.repeats < 1:
        parser.error("positive token limit and repeats required")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
