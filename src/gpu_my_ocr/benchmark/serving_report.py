"""Aggregate strict serving results, diagnostics and supported engine metric deltas."""

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from gpu_my_ocr.benchmark.accuracy.diagnostics import diagnose
from gpu_my_ocr.benchmark.diagnostic_report import aggregate


def metric_totals(text: str) -> dict[str, float]:
    """Sum labelled scalar Prometheus samples by name; ignore comments/non-numeric lines."""
    totals: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([-+0-9.eE]+)(?:\s|$)", line)
        if match:
            name, value = match.groups()
            totals[name] = totals.get(name, 0) + float(value)
    return totals


def engine_metric_delta(before: str, after: str) -> dict[str, Any]:
    """Report measured server queue/preemption metrics where exposed; unknowns stay None."""
    start, end = metric_totals(before), metric_totals(after)
    prefix = "vllm:" if any(k.startswith("vllm:") for k in end) else "sglang:"
    sums = [k for k in end if "queue" in k and k.endswith("_sum")]
    queue = {}
    for key in sums:
        count_key = key[:-4] + "_count"
        count = end.get(count_key, 0) - start.get(count_key, 0)
        queue[key] = ((end[key] - start.get(key, 0)) / count) if count > 0 else None
    stage_start = metric_totals(
        "\n".join(line for line in before.splitlines() if 'stage="prefill_waiting"' in line)
    )
    stage_end = metric_totals(
        "\n".join(line for line in after.splitlines() if 'stage="prefill_waiting"' in line)
    )
    stage_key = "sglang:per_stage_req_latency_seconds_sum"
    stage_count_key = "sglang:per_stage_req_latency_seconds_count"
    stage_count = stage_end.get(stage_count_key, 0) - stage_start.get(stage_count_key, 0)
    prefill_waiting = (
        (stage_end[stage_key] - stage_start.get(stage_key, 0)) / stage_count
        if stage_count > 0 and stage_key in stage_end
        else None
    )
    retract_key = "sglang:num_retractions_sum"
    preemption_key = prefix + "num_preemptions_total"
    return {
        "queue_mean_seconds_by_metric": queue or None,
        "preemptions": end[preemption_key] - start.get(preemption_key, 0)
        if preemption_key in end
        else None,
        "prefill_waiting_mean_seconds": prefill_waiting,
        "prefill_waiting_sample_count": stage_count,
        "sglang_retractions": (
            end[retract_key] - start.get(retract_key, 0) if retract_key in end else None
        ),
        "scope": "server-side histogram/counter deltas, not client semaphore waiting",
    }


def quality_report(records: list[dict], dataset: Path) -> dict:
    """Strict all-request usable credit plus labelled diagnostic views without changing scores."""
    fields_total = numeric_total = cells_total = fields_correct = numeric_correct = (
        cells_correct
    ) = 0
    diagnostics = []
    for record in records:
        truth = json.loads((dataset / record["record"]["ground_truth_path"]).read_text())
        names = [name for name, mode in truth["field_modes"].items() if mode == "numeric"]
        fields_total += len(truth["fields"])
        numeric_total += len(names)
        cells_total += sum(len(t["rows"]) * len(t["columns"]) for t in truth["tables"])
        score = record["score"]
        if score["schema_valid"]:
            metrics = score["metrics"]
            fields_correct += metrics["critical_fields_correct"]
            numeric_correct += sum(metrics["fields"][name]["match"] for name in names)
            cells_correct += metrics["tables"]["correct"]
        if record["status"] != "skipped":
            diagnostic = diagnose(truth, record["text"])
            assert diagnostic["strict"] == score, "Saved strict score mismatch"
            diagnostics.append(diagnostic)
    return {
        "field_correct": fields_correct,
        "field_total": fields_total,
        "usable_field_accuracy": fields_correct / fields_total if fields_total else None,
        "numeric_correct": numeric_correct,
        "numeric_total": numeric_total,
        "usable_numeric_accuracy": numeric_correct / numeric_total if numeric_total else None,
        "table_correct": cells_correct,
        "table_total": cells_total,
        "usable_table_accuracy": cells_correct / cells_total if cells_total else None,
        "diagnostics": aggregate(diagnostics),
    }


def load_requests(path: Path) -> list[dict]:
    """Read completed request files only, excluding config/warmup and prompt records."""
    return [
        json.loads(p.read_text())
        for p in sorted(path.glob("*--r*.json"))
        if not p.name.endswith(".prompt.json")
    ]


def report(run: Path, dataset: Path) -> dict:
    """Report each measured concurrency separately; match optional structured cohort to c1."""
    outputs = {}
    baseline = None
    for path in sorted(run.glob("c[1248]")):
        if not (path / "summary.json").exists():
            continue
        records = load_requests(path)
        metrics = {}
        before, after = path / "metrics-before.txt", path / "metrics-after.txt"
        if before.exists() and after.exists():
            metrics = engine_metric_delta(before.read_text(), after.read_text())
        outputs[path.name] = {
            "performance": json.loads((path / "summary.json").read_text()),
            "quality": quality_report(records, dataset),
            "engine_metrics": metrics,
            "input_tokens_range": [
                min(r["input_tokens"] for r in records if r.get("input_tokens") is not None),
                max(r["input_tokens"] for r in records if r.get("input_tokens") is not None),
            ]
            if any(r.get("input_tokens") is not None for r in records)
            else None,
        }
        if path.name == "c1":
            baseline = records
    structured = run / "structured"
    if (structured / "summary.json").exists():
        records = load_requests(structured)
        ids = {r["page_id"] for r in records}
        matched = [r for r in baseline if r["page_id"] in ids] if baseline else []
        outputs["structured"] = {
            "performance": json.loads((structured / "summary.json").read_text()),
            "quality": quality_report(records, dataset),
            "matched_unconstrained_quality": quality_report(matched, dataset),
            "matched_unconstrained_mean_latency_seconds": statistics.mean(
                r["end_to_end_ms"] for r in matched
            )
            / 1000
            if matched
            else None,
        }
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.dataset), indent=2))
