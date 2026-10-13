"""Summarize accuracy views separately; never write into protected Phase 1 results."""

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from gpu_my_ocr.benchmark.accuracy.diagnostics import diagnose


def aggregate(scores: list[dict]) -> dict[str, Any]:
    """Aggregate raw/view/strict metrics with explicit denominators and unavailable OCR."""
    fields = [s["field_diagnostics"] for s in scores if "field_diagnostics" in s]
    rates = [s["transcription_rates"] for s in scores if s.get("transcription_rates") is not None]
    field_total = sum(f["total"] for f in fields)
    numeric_total = sum(f["numeric_total"] for f in fields)
    strict = [s["strict"] for s in scores if "strict" in s]
    conditional = [s["metrics"] for s in strict if s["schema_valid"]]
    tables = [s["table_diagnostics"] for s in scores if "table_diagnostics" in s]
    table_total = sum(t["total"] for t in tables)
    return {
        "requests": len(scores),
        "raw_json_valid": sum(s.get("raw_json_valid", False) for s in scores),
        "markdown_fences": sum(s.get("markdown_fence", False) for s in scores),
        "extra_text": sum(s.get("extra_text", False) for s in scores),
        "diagnostic_object_available": sum(isinstance(s.get("prediction"), dict) for s in scores),
        "strict_schema_valid": sum(s["schema_valid"] for s in strict) if strict else None,
        "strict_page_success": sum(s["page_success"] for s in strict) if strict else None,
        "reduced_schema_valid": sum(s.get("reduced_schema_valid", False) for s in scores),
        "diagnostic_view_schema_valid": sum(
            s["view_schema"]["schema_valid"] for s in scores if "view_schema" in s
        ),
        "field_total": field_total,
        "fields_present": sum(f["present"] for f in fields),
        "field_coverage": sum(f["present"] for f in fields) / field_total if field_total else None,
        "field_exact_correct": sum(f["correct"] for f in fields),
        "field_exact_accuracy": sum(f["correct"] for f in fields) / field_total
        if field_total
        else None,
        "numeric_total": numeric_total,
        "numeric_correct": sum(f["numeric_correct"] for f in fields),
        "numeric_accuracy": sum(f["numeric_correct"] for f in fields) / numeric_total
        if numeric_total
        else None,
        "wrong_numeric_fields": sum(len(f["incorrect_numeric"]) for f in fields),
        "wrong_digit_fields": sum(len(f["incorrect_digits"]) for f in fields),
        "conditional_field_accuracy": sum(m["critical_fields_correct"] for m in conditional)
        / sum(m["critical_fields_total"] for m in conditional)
        if conditional
        else None,
        "table_total": table_total,
        "table_correct": sum(t["correct"] for t in tables),
        "table_accuracy": sum(t["correct"] for t in tables) / table_total if table_total else None,
        "comparable_transcriptions": len(rates),
        "cer_mean": statistics.mean(r["cer"] for r in rates) if rates else None,
        "wer_mean": statistics.mean(r["wer"] for r in rates) if rates else None,
    }


def baseline_report(run: Path, dataset: Path) -> dict:
    """Diagnose saved predictions as new evidence, leaving original raw scores untouched."""
    results = []
    for path in sorted(run.glob("*--r*.json")):
        if path.name.endswith(".prompt.json"):
            continue
        request = json.loads(path.read_text())
        truth = json.loads((dataset / request["record"]["ground_truth_path"]).read_text())
        scores = diagnose(truth, request["text"])
        assert scores["strict"] == request["score"], "Strict evaluator results changed"
        results.append(
            {"page_id": request["page_id"], "record": request["record"], "scores": scores}
        )
    return {"summary": aggregate([r["scores"] for r in results]), "requests": results}


def mode_reports(root: Path) -> dict:
    """Return separate A/B/C summaries; mode output contracts are never pooled."""
    summaries = {}
    for mode in ("full", "ocr", "minimal"):
        records = [
            json.loads(p.read_text())
            for p in sorted((root / mode).glob("*--r*.json"))
            if not p.name.endswith(".prompt.json")
        ]
        summaries[mode] = {
            **aggregate([r["score"] for r in records]),
            "mean_latency_seconds": statistics.mean(r["end_to_end_ms"] for r in records) / 1000,
            "output_tokens_min": min(r["output_tokens"] for r in records),
            "output_tokens_max": max(r["output_tokens"] for r in records),
            "failures": sum(r["status"] != "ok" for r in records),
            "truncations": sum(r.get("truncated", False) for r in records),
            "timing": json.loads((root / mode / "summary.json").read_text()),
        }
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            baseline_report(args.run, args.dataset) if args.baseline else mode_reports(args.run),
            indent=2,
        )
    )
