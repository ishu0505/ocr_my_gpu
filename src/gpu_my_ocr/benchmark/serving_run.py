"""Sequential engine ownership with separately recorded repeated concurrency levels."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import time
import traceback
from pathlib import Path

from gpu_my_ocr.benchmark.baseline import FROZEN_HASH, REVISION, save_json
from gpu_my_ocr.benchmark.concurrency import bounded_requests, concurrency_metrics
from gpu_my_ocr.benchmark.gpu_telemetry import GpuTelemetry, summarize_samples
from gpu_my_ocr.benchmark.server_process import SERVED_MODEL, LocalServer, fetch
from gpu_my_ocr.benchmark.serving_requests import persist_requests, prepare_pages, serving_request
from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset
from gpu_my_ocr.engines.http import OpenAIImageEngine


def save_endpoint(base_url: str, endpoint: str, path: Path) -> None:
    """Retain server metrics/info or an explicit unsupported/error record."""
    try:
        path.write_text(fetch(base_url + endpoint))
    except (OSError, ValueError) as error:
        save_json(path.with_suffix(".error.json"), {"endpoint": endpoint, "error": str(error)})


async def measurement(
    engine: object,
    pages: list[dict],
    concurrency: int,
    output: Path,
    telemetry: GpuTelemetry,
    base_url: str,
) -> dict:
    """Warm up separately, then measure two fixed ordered workload repeats with bounded clients."""
    output.mkdir()
    warmup = await serving_request(engine, pages[0], -1)
    save_json(output / "warmup.json", warmup)
    if warmup["status"] != "ok":
        raise RuntimeError(f"Warmup failed: {warmup.get('error')}")
    await asyncio.to_thread(save_endpoint, base_url, "/metrics", output / "metrics-before.txt")
    sample_start = len(telemetry.samples)
    items = [(page, repeat) for repeat in range(2) for page in pages]

    aborted = False

    async def operation(item: tuple) -> dict:
        nonlocal aborted
        page, repeat = item
        if aborted:
            return {
                "page_id": f"{page['page_id']}--r{repeat}",
                "record": page["record"],
                "repeat": repeat,
                "status": "skipped",
                "text": "",
                "output_tokens": None,
                "reason": "earlier request failure; configuration stopped",
                "score": {"schema_valid": False, "page_success": False},
            }
        result = await serving_request(engine, page, repeat)
        if result["status"] != "ok":
            aborted = True
        return result

    results, elapsed = await bounded_requests(items, concurrency, operation)
    memory = summarize_samples(telemetry.samples[sample_start:])
    persist_requests(output, results)
    summary = {
        **concurrency_metrics(results, elapsed),
        "client_concurrency": concurrency,
        "engine_max_sequences": 8,
        "repeats": 2,
        "warmup": 1,
        "gpu": memory,
        "latency_scope": "prepared PNG through HTTP/SSE; excludes PDF rendering/persistence",
    }
    save_json(output / "summary.json", summary)
    await asyncio.to_thread(save_endpoint, base_url, "/metrics", output / "metrics-after.txt")
    print(json.dumps(summary), flush=True)
    return summary


def configuration(args: argparse.Namespace, server: LocalServer, integrity: dict) -> dict:
    """Record model, launch command, isolated runtime, code hashes and timing differences."""
    return {
        "engine": args.engine,
        "revision": REVISION,
        "precision": "bfloat16",
        "command": server.command,
        "environment_overrides": server.environment_overrides,
        "dataset": integrity,
        "prompt_version": "financial-extraction-v2",
        "max_output_tokens": 1024,
        "render_dpi": 150,
        "min_pixels": 262144,
        "max_pixels": 1310720,
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "repetition_penalty": 1,
        "seed": 42,
        "concurrency_levels": args.concurrency,
        "pilot": args.pilot,
        "constrained": args.constrained,
        "libraries": {
            p: importlib.metadata.version(p)
            for p in (args.engine, "torch", "transformers", "torchvision", "pillow", "pymupdf")
        },
        "engine_lock_sha256": hashlib.sha256(
            Path(f"environments/{args.engine}/uv.lock").read_bytes()
        ).hexdigest(),
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/gpu_my_ocr").rglob("*.py")
        },
        "comparison_limits": [
            "native engine attention and Transformer library differ",
            "prepared PNG timing excludes Phase 1 PDF rendering/raw file writing",
            "tokens/EOS accounting may differ",
            "CUDA graphs and prefix caching disabled",
        ],
    }


async def run(args: argparse.Namespace) -> None:
    """Start one server, gate sweeps on image smoke, retain failures, always clean up resources."""
    from gpu_my_ocr.benchmark.accuracy.schema import prediction_schema

    args.output.mkdir(parents=True, exist_ok=False)
    integrity = validate_dataset(args.dataset, FROZEN_HASH)
    inputs = args.output / "inputs"
    inputs.mkdir()
    pages = prepare_pages(args.dataset, inputs, args.pilot)
    server = LocalServer(args.engine, args.port, args.output)
    save_json(args.output / "config.json", configuration(args, server, integrity))
    telemetry = GpuTelemetry()
    telemetry.thread.start()
    try:
        await server.start()
        save_json(args.output / "startup.json", {"seconds": server.startup_seconds})
        await asyncio.to_thread(
            save_endpoint, server.base_url, "/get_server_info", args.output / "server-info.json"
        )
        engine = OpenAIImageEngine(server.base_url, SERVED_MODEL, args.engine)
        smoke = await serving_request(engine, pages[0], -2)
        save_json(args.output / "smoke.json", smoke)
        if smoke["status"] != "ok":
            raise RuntimeError(f"Image smoke failed: {smoke.get('error')}")
        summaries = []
        for concurrency in args.concurrency:
            out = args.output / f"c{concurrency}"
            summary = await measurement(engine, pages, concurrency, out, telemetry, server.base_url)
            summaries.append(summary)
            if summary["failures"]:
                save_json(
                    args.output / "stopped.json",
                    {
                        "reason": "request failure/OOM",
                        "after_concurrency": concurrency,
                        "skipped_concurrency": [c for c in args.concurrency if c > concurrency],
                    },
                )
                break
        if args.constrained and not any(s["failures"] for s in summaries):
            out = args.output / "structured"
            out.mkdir()
            representatives = [
                p
                for p in pages
                if p["record"]["variant"] == "digital"
                and p["record"]["document_id"].endswith("-010")
            ]
            warmup_engine = OpenAIImageEngine(
                server.base_url,
                SERVED_MODEL,
                args.engine,
                schema=prediction_schema(representatives[0]["truth"]),
            )
            save_json(
                out / "warmup.json", await serving_request(warmup_engine, representatives[0], -1)
            )
            results = []
            sample_start = len(telemetry.samples)
            started = time.perf_counter()
            for repeat in range(2):
                for page in representatives:
                    constrained = OpenAIImageEngine(
                        server.base_url,
                        SERVED_MODEL,
                        args.engine,
                        schema=prediction_schema(page["truth"]),
                    )
                    results.append(await serving_request(constrained, page, repeat))
            elapsed = time.perf_counter() - started
            persist_requests(out, results)
            save_json(
                out / "summary.json",
                {
                    **concurrency_metrics(results, elapsed),
                    "gpu": summarize_samples(telemetry.samples[sample_start:]),
                },
            )
            save_json(
                out / "config.json",
                {
                    "mode": "json-schema-constrained-v1",
                    "requests": len(results),
                    "repeats": 2,
                    "warmup": 1,
                    "client_concurrency": 1,
                    "unconstrained_comparator": "same page records in c1",
                    "no_score_repair": True,
                },
            )
        save_json(args.output / "post_integrity.json", validate_dataset(args.dataset, FROZEN_HASH))
    except Exception:
        save_json(args.output / "failure.json", {"error": traceback.format_exc()})
        raise
    finally:
        await server.close()
        save_json(args.output / "telemetry.json", telemetry.close())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--constrained", action="store_true")
    args = parser.parse_args()
    if (
        not args.concurrency
        or args.concurrency[0] != 1
        or any(c not in [1, 2, 4, 8] for c in args.concurrency)
    ):
        parser.error("sweeps must start at 1 and use 1/2/4/8")
    asyncio.run(run(args))
