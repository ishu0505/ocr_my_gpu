"""Bound client requests; compute throughput from wall time rather than summed latencies."""

import asyncio
import math
import statistics
import time
from collections.abc import Awaitable, Callable
from typing import Any


async def bounded_requests(
    items: list[Any], concurrency: int, operation: Callable[[Any], Awaitable[dict]]
) -> tuple[list[dict], float]:
    """Submit at most concurrency requests; return results and total measured wall seconds."""
    if concurrency < 1:
        raise ValueError("Client concurrency must be positive")
    semaphore = asyncio.Semaphore(concurrency)

    async def submit(item: Any) -> dict:
        queued = time.perf_counter()
        async with semaphore:
            dispatched = time.perf_counter()
            result = await operation(item)
            result["client_queue_ms"] = (dispatched - queued) * 1000
            result["client_total_ms"] = (time.perf_counter() - queued) * 1000
            return result

    started = time.perf_counter()
    results = await asyncio.gather(*(submit(item) for item in items))
    return results, time.perf_counter() - started


def percentile(values: list[float], fraction: float) -> float | None:
    """Return nearest-rank descriptive percentile, or None for absent observations."""
    if not values:
        return None
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def concurrency_metrics(results: list[dict], elapsed: float) -> dict:
    """Report successful/correct rates, latencies and available streamed token/TTFT counts."""
    if elapsed <= 0:
        raise ValueError("Measured wall time must be positive")
    attempted = [r for r in results if r["status"] != "skipped"]
    successful = [r for r in attempted if r["status"] == "ok"]
    latencies = [r["end_to_end_ms"] / 1000 for r in successful]
    ttft = [
        r["time_to_first_token_ms"] / 1000
        for r in successful
        if r.get("time_to_first_token_ms") is not None
    ]
    tokens = [r["output_tokens"] for r in successful if r.get("output_tokens") is not None]
    return {
        "requests": len(results),
        "successful_requests": len(successful),
        "attempted_requests": len(attempted),
        "skipped_requests": len(results) - len(attempted),
        "wall_seconds": elapsed,
        "successful_pages_per_second": len(successful) / elapsed,
        "correct_pages_per_second": sum(r["score"]["page_success"] for r in successful) / elapsed,
        "failures": len(attempted) - len(successful),
        "failure_rate": (len(attempted) - len(successful)) / len(attempted) if attempted else None,
        "ooms": sum(r["status"] == "oom" for r in results),
        "truncations": sum(r.get("truncated", False) for r in results),
        "mean_latency_seconds": statistics.mean(latencies) if latencies else None,
        "p50_latency_seconds": percentile(latencies, 0.5),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "ttft_sample_count": len(ttft),
        "ttft_mean_seconds": statistics.mean(ttft) if ttft else None,
        "output_token_sample_count": len(tokens),
        "output_tokens_per_second": sum(tokens) / elapsed
        if len(tokens) == len(successful) and tokens
        else None,
        "schema_valid": sum(r["score"]["schema_valid"] for r in results),
        "page_success": sum(r["score"]["page_success"] for r in results),
        "server_queue_metrics": "see saved /metrics snapshots if supported; client queue is separate",
        "latency_scope": "active client request (excludes bounded-client waiting)",
    }
