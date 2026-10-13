"""Controlled Transformers extraction diagnostics, separate from the immutable baseline."""

import argparse
import asyncio
import importlib.metadata
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from gpu_my_ocr.benchmark.accuracy.diagnostics import diagnose, field_diagnostics, parse_view
from gpu_my_ocr.benchmark.accuracy.metrics import ocr_rates
from gpu_my_ocr.benchmark.baseline import FROZEN_HASH, REVISION, save_json, selected_records
from gpu_my_ocr.benchmark.diagnostic_prompts import CRITICAL_FIELDS, VERSIONS, diagnostic_prompt
from gpu_my_ocr.benchmark.extraction import render_page
from gpu_my_ocr.benchmark.gpu_telemetry import GpuTelemetry, parameter_evidence
from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset
from gpu_my_ocr.models import InferenceRequest


def mode_score(truth: dict, text: str, mode: str) -> dict:
    """Score each contract independently; reduced-field and OCR scores are not full-schema scores."""
    if mode == "full":
        return diagnose(truth, text)
    if mode == "ocr":
        return {"transcription_rates": ocr_rates(truth["transcription"], text)}
    parsed = parse_view(text)
    prediction = parsed["prediction"] if isinstance(parsed["prediction"], dict) else {}
    names = CRITICAL_FIELDS[truth["category"]]
    schema_valid = (
        parsed["raw_json_valid"]
        and set(prediction) == {"fields"}
        and isinstance(prediction.get("fields"), dict)
        and set(prediction["fields"]) == set(names)
        and all(isinstance(v, str) for v in prediction["fields"].values())
    )
    return {
        **parsed,
        "reduced_schema_valid": schema_valid,
        "field_diagnostics": field_diagnostics(truth, prediction, names),
    }


async def request(
    engine: object, record: dict, root: Path, out: Path, mode: str, repeat: int, cap: int = 1024
) -> dict:
    """Render and infer one page; save raw outputs, explicit failures and mode-specific scores."""
    truth = json.loads((root / record["ground_truth_path"]).read_text())
    page_id = f"{record['document_id']}--{record['variant']}--r{repeat}"
    prompt = diagnostic_prompt(truth, mode)
    save_json(
        out / f"{page_id}.prompt.json", {"mode": mode, "version": VERSIONS[mode], "text": prompt}
    )
    result = {"page_id": page_id, "record": record, "mode": mode, "repeat": repeat}
    started = time.perf_counter()
    try:
        image = out / f"{page_id}.png"
        render_page(root / record["pdf_path"], image)
        response = await engine.infer(InferenceRequest(page_id, image, prompt, cap))
        result.update(asdict(response), **engine.details, status="ok")
    except Exception as error:  # noqa: BLE001 - preserve per-request runtime failures
        result.update(
            status="oom" if "out of memory" in str(error).lower() else "failed",
            text="",
            output_tokens=0,
            error=traceback.format_exc(),
        )
    result["end_to_end_ms"] = (time.perf_counter() - started) * 1000
    (out / f"{page_id}.txt").write_text(result["text"])
    result["score"] = mode_score(truth, result["text"], mode)
    save_json(out / f"{page_id}.json", result)
    print(
        json.dumps(
            {k: result.get(k) for k in ("mode", "page_id", "status", "output_tokens", "truncated")}
        ),
        flush=True,
    )
    return result


async def profile_request(engine: object, record: dict, root: Path, out: Path) -> dict:
    """Profile a separate 32-token request; its intentionally capped output is not benchmarked."""
    import torch

    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profiler:
            await request(engine, record, root, out, "full", -2, cap=32)
        profiler.export_chrome_trace(str(out / "trace.json"))
        cuda_events = [e for e in profiler.events() if str(e.device_type).endswith("CUDA")]
        return {
            "status": "observed" if cuda_events else "no_cuda_events",
            "cuda_event_count": len(cuda_events),
            "cuda_event_names": sorted({e.name for e in cuda_events})[:50],
            "scope": "separate full-v2 32-token inference; profile excluded from timing",
        }
    except Exception:  # noqa: BLE001 - profiler support may differ; retain failure evidence
        return {"status": "failed", "error": traceback.format_exc()}


async def run(args: argparse.Namespace) -> None:
    """Load the unchanged BF16 adapter once; run separately versioned modes and GPU evidence."""
    import torch

    from gpu_my_ocr.engines.transformers import TransformersEngine

    args.output.mkdir(parents=True, exist_ok=False)
    integrity = validate_dataset(args.dataset, FROZEN_HASH)
    manifest = json.loads((args.dataset / "manifest.json").read_text())
    records = selected_records(manifest, True)
    records += [
        r
        for r in manifest["records"]
        if r["document_id"] == "portfolio-42-010" and r["variant"] == "scan_degraded"
    ]
    telemetry = GpuTelemetry()
    telemetry.thread.start()
    try:
        torch.manual_seed(42)
        engine = TransformersEngine(REVISION)
        save_json(
            args.output / "gpu.json",
            {
                **parameter_evidence(engine.model),
                "compute_capability": list(torch.cuda.get_device_capability()),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "load_seconds": engine.load_seconds,
            },
        )
        for mode in ("full", "ocr", "minimal"):
            out = args.output / mode
            out.mkdir()
            config = {
                "mode": mode,
                "prompt_version": VERSIONS[mode],
                "revision": REVISION,
                "dataset": integrity,
                "precision": "bfloat16",
                "attention": "sdpa",
                "max_output_tokens": 1024,
                "max_pixels": 1310720,
                "min_pixels": 262144,
                "render_dpi": 150,
                "seed": 42,
                "do_sample": False,
                "warmup": 1,
                "repeats": 2,
                "libraries": {
                    p: importlib.metadata.version(p)
                    for p in ("torch", "transformers", "torchvision")
                },
            }
            save_json(out / "config.json", config)
            warmup = out / "warmup"
            warmup.mkdir()
            await request(engine, records[0], args.dataset, warmup, mode, -1)
            started = time.perf_counter()
            results = [
                await request(engine, r, args.dataset, out, mode, repeat)
                for repeat in range(2)
                for r in records
            ]
            elapsed = time.perf_counter() - started
            save_json(
                out / "summary.json",
                {
                    "requests": len(results),
                    "wall_seconds": elapsed,
                    "pages_per_second": len(results) / elapsed,
                },
            )
        profile = args.output / "profile"
        profile.mkdir()
        save_json(
            profile / "evidence.json",
            await profile_request(engine, records[0], args.dataset, profile),
        )
        save_json(args.output / "post_integrity.json", validate_dataset(args.dataset, FROZEN_HASH))
    except Exception:
        save_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise
    finally:
        save_json(args.output / "telemetry.json", telemetry.close())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    asyncio.run(run(parser.parse_args()))
