"""Minimal CLI; experiment commands are implemented phase-by-phase."""

import argparse
import json
from pathlib import Path
from typing import Any

from gpu_my_ocr.preflight import collect_manifest, write_manifest


def main() -> None:
    """Run the selected safe command (no GPU model starts automatically)."""
    parser = argparse.ArgumentParser(description="gpu_my_ocr benchmark controller")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Inspect local prerequisites")
    doctor.add_argument("--output", type=Path, help="Save a new JSON environment manifest")
    doctor.add_argument("--torch-python", type=Path, help="Isolated engine Python for CUDA probe")
    generate = commands.add_parser("dataset-generate", help="Create and freeze synthetic PDFs")
    generate.add_argument("--output", type=Path, required=True, help="New dataset directory")
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument(
        "--per-category", type=int, default=10, help="1..10; default full dataset"
    )
    validate = commands.add_parser("dataset-validate", help="Verify frozen artifacts and PDFs")
    validate.add_argument("dataset", type=Path)
    validate.add_argument("--expected-manifest-sha256", help="Require an externally frozen hash")
    evaluate = commands.add_parser("evaluate", help="Score a prediction against saved JSON truth")
    evaluate.add_argument("--ground-truth", type=Path, required=True)
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, help="New score JSON file")
    args = parser.parse_args()
    if args.command == "doctor":
        manifest = collect_manifest(args.torch_python)
        if args.output is not None:
            write_manifest(manifest, args.output)
        print(json.dumps(manifest, indent=2))
        if manifest["status"] == "failed":
            raise SystemExit(1)
    else:
        print(json.dumps(run_dataset_command(args), indent=2))


def run_dataset_command(args: argparse.Namespace) -> dict[str, Any]:
    """Dispatch CPU-only dataset/evaluation commands, loading their dependencies lazily."""
    if args.command == "dataset-generate":
        from gpu_my_ocr.datasets.synthetic_data.generate import generate_dataset

        return generate_dataset(args.output, args.seed, args.per_category)
    if args.command == "dataset-validate":
        from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset

        return validate_dataset(args.dataset, args.expected_manifest_sha256)
    from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_files

    scores = evaluate_files(args.ground_truth, args.prediction)
    if args.output:
        write_manifest(scores, args.output)
    return scores


if __name__ == "__main__":
    main()
