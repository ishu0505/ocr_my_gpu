"""Aggregate raw baseline evidence without repairing or rescoring malformed predictions."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def summarize(records: list[dict]) -> dict[str, Any]:
    """Return measured latency/quality; schema failures stay in all-page denominators."""
    scores = [r["score"] for r in records]
    valid = [s["metrics"] for s in scores if s["schema_valid"]]
    fields = [f for m in valid for f in m["fields"].values()]
    latencies = [r["end_to_end_ms"] / 1000 for r in records]
    cells_total = sum(m["tables"]["total"] for m in valid)
    # Expected denominators are loaded independently; invalid predictions get no credit.
    field_total = sum(r["expected_fields"] for r in records)
    numeric_total = sum(r["expected_numeric"] for r in records)
    table_total = sum(r["expected_cells"] for r in records)
    return {
        "requests": len(records),
        "underlying_documents": len({r["record"]["document_id"] for r in records}),
        "schema_valid": sum(s["schema_valid"] for s in scores),
        "page_success": sum(s["page_success"] for s in scores),
        "failures": sum(r["status"] != "ok" for r in records),
        "truncations": sum(r.get("truncated", False) for r in records),
        "field_correct": sum(f["match"] for f in fields),
        "field_total": field_total,
        "field_accuracy": sum(f["match"] for f in fields) / field_total,
        "numeric_correct": sum(r["numeric_correct"] for r in records),
        "numeric_total": numeric_total,
        "numeric_accuracy": sum(r["numeric_correct"] for r in records) / numeric_total,
        "table_correct": sum(m["tables"]["correct"] for m in valid),
        "table_total": table_total,
        "table_accuracy": sum(m["tables"]["correct"] for m in valid) / table_total,
        "schema_valid_field_accuracy": statistics.mean(f["match"] for f in fields)
        if fields
        else None,
        "schema_valid_table_accuracy": sum(m["tables"]["correct"] for m in valid) / cells_total
        if cells_total
        else None,
        "cer_valid_mean": statistics.mean(m["cer"] for m in valid) if valid else None,
        "wer_valid_mean": statistics.mean(m["wer"] for m in valid) if valid else None,
        "latency_mean_seconds": statistics.mean(latencies),
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": sorted(latencies)[max(0, int(0.95 * len(latencies) + 0.999) - 1)],
        "output_tokens_min": min(r["output_tokens"] for r in records),
        "output_tokens_max": max(r["output_tokens"] for r in records),
        "output_tokens_mean": statistics.mean(r["output_tokens"] for r in records),
        "peak_allocated_bytes": max(r.get("peak_allocated_bytes", 0) for r in records),
        "peak_reserved_bytes": max(r.get("peak_reserved_bytes", 0) for r in records),
    }


def report(run_path: Path, dataset: Path) -> dict:
    """Load request files and frozen truth denominators; summarize groups and repeats."""
    records = []
    for path in sorted(run_path.glob("*--r*.json")):
        if path.name.endswith(".prompt.json"):
            continue
        record = json.loads(path.read_text())
        truth = json.loads((dataset / record["record"]["ground_truth_path"]).read_text())
        numeric = [k for k, mode in truth["field_modes"].items() if mode == "numeric"]
        metrics = record["score"]["metrics"]
        record.update(
            expected_fields=len(truth["fields"]),
            expected_numeric=len(numeric),
            expected_cells=sum(len(t["rows"]) * len(t["columns"]) for t in truth["tables"]),
            numeric_correct=sum(metrics["fields"][k]["match"] for k in numeric) if metrics else 0,
        )
        records.append(record)
    result = {"all": summarize(records)}
    for group in ("category", "variant"):
        result[group] = {
            v: summarize([r for r in records if r["record"][group] == v])
            for v in sorted({r["record"][group] for r in records})
        }
    result["repeats"] = {
        str(v): summarize([r for r in records if r["repeat"] == v])
        for v in sorted({r["repeat"] for r in records})
    }
    result["timing"] = json.loads((run_path / "summary.json").read_text())
    result["memory"] = json.loads((run_path / "memory.json").read_text())
    result["load"] = json.loads((run_path / "load.json").read_text())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    args = parser.parse_args()
    print(json.dumps(report(args.run, args.dataset), indent=2))
