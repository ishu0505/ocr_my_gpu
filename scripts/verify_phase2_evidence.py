"""Verify protected Phase 1/data files and lossless Phase 2 scoring and input contracts."""

import hashlib
import json
from pathlib import Path

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.baseline import FROZEN_HASH, REVISION
from gpu_my_ocr.benchmark.diagnostic_prompts import diagnostic_prompt
from gpu_my_ocr.benchmark.diagnostic_run import mode_score
from gpu_my_ocr.benchmark.extraction import extraction_prompt
from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset


def protected_evidence(root: Path) -> dict:
    """Require exact original bytes for every frozen document and protected Phase 1 artifact."""
    protected = json.loads((root / "protected-evidence.json").read_text())
    for name, digest in protected.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
    return {"status": "passed", "protected_files": len(protected)}


def verify_diagnostics(root: Path, dataset: Path) -> dict:
    """Recompute all three modes and verify unchanged full-v2 prompts and paired inputs."""
    checked = 0
    for mode in ("full", "ocr", "minimal"):
        directory = root / mode
        config = json.loads((directory / "config.json").read_text())
        assert config["revision"] == REVISION and config["max_output_tokens"] == 1024
        assert config["precision"] == "bfloat16" and config["repeats"] == 2
        for path in directory.glob("*--r*.json"):
            if path.name.endswith(".prompt.json"):
                continue
            record = json.loads(path.read_text())
            truth = json.loads((dataset / record["record"]["ground_truth_path"]).read_text())
            assert record["text"] == path.with_suffix(".txt").read_text()
            assert record["score"] == mode_score(truth, record["text"], mode)
            prompt = json.loads(path.with_suffix(".prompt.json").read_text())
            assert prompt["text"] == diagnostic_prompt(truth, mode)
            assert record["status"] == "ok" and not record["truncated"]
            checked += 1
    assert checked == 36
    return {"status": "passed", "scored_requests": checked, "profile_excluded": True}


def verify_serving(run: Path, dataset: Path) -> dict:
    """Verify BF16/full-v2 serving records, immutable raw outputs and identical PNG bytes."""
    config = json.loads((run / "config.json").read_text())
    assert config["revision"] == REVISION and config["precision"] == "bfloat16"
    assert (
        config["max_output_tokens"] == 1024
        and config["prompt_version"] == "financial-extraction-v2"
    )
    overlay = run / "processor-model"
    if overlay.exists():
        provenance = json.loads((overlay / "overlay-provenance.json").read_text())
        assert provenance["revision"] == REVISION
        assert Path(provenance["source_snapshot"]).name == REVISION
        assert json.loads((overlay / "preprocessor_config.json").read_text())["size"] == {
            "shortest_edge": 262144,
            "longest_edge": 1310720,
        }
        assert all(
            (overlay / name).is_symlink() and str((overlay / name).resolve()) == target
            for name, target in provenance["linked_assets"].items()
        )
    checked = 0
    levels = {}
    for directory in sorted(run.glob("c[1248]")) + (
        [run / "structured"] if (run / "structured").exists() else []
    ):
        identities = {}
        for path in directory.glob("*--r*.json"):
            if path.name.endswith(".prompt.json"):
                continue
            result = json.loads(path.read_text())
            if result["status"] == "skipped":
                continue
            truth = json.loads((dataset / result["record"]["ground_truth_path"]).read_text())
            assert result["text"] == path.with_suffix(".txt").read_text()
            assert result["score"] == evaluate_prediction(truth, result["text"])
            page_id = f"{result['record']['document_id']}--{result['record']['variant']}"
            prompt = json.loads((run / "inputs" / f"{page_id}.prompt.json").read_text())
            assert prompt["text"] == extraction_prompt(truth)
            old = Path("results/phase1/baseline-v2-1024") / f"{page_id}--r0.png"
            current = run / "inputs" / f"{page_id}.png"
            assert (
                hashlib.sha256(old.read_bytes()).digest()
                == hashlib.sha256(current.read_bytes()).digest()
            )
            identities.setdefault(page_id, []).append(result["repeat"])
            checked += 1
        levels[directory.name] = {
            "scored_requests": sum(len(v) for v in identities.values()),
            "distinct_pages": len(identities),
        }
        if (
            not config["pilot"]
            and directory.name != "structured"
            and not (run / "stopped.json").exists()
        ):
            assert len(identities) == 30 and all(sorted(v) == [0, 1] for v in identities.values())
    return {
        "status": "passed",
        "requests": checked,
        "levels": levels,
        "image_bytes_identical_to_phase1": True,
        "original_scores_recomputed": True,
    }


if __name__ == "__main__":
    root = Path("results/phase2")
    dataset = Path("datasets/generated/financial-v1-seed42")
    result = {
        "protected": protected_evidence(root),
        "dataset": validate_dataset(dataset, FROZEN_HASH),
        "diagnostics": verify_diagnostics(root / "transformers-diagnostics-v1", dataset),
    }
    result["serving"] = {
        str(run): verify_serving(run, dataset)
        for run in sorted(root.glob("*primary*"))
        if (run / "config.json").exists()
    }
    result["artifact_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()
    }
    print(json.dumps(result, indent=2))
