"""Audit saved primary predictions, paired identities and immutable visual input evidence."""

import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.baseline import FROZEN_HASH
from gpu_my_ocr.benchmark.extraction import extraction_prompt
from gpu_my_ocr.datasets.synthetic_data.integrity import validate_dataset


def verify_request(path: Path, dataset: Path) -> dict:
    """Recompute strict scores and schema-only prompt from persisted canonical truth."""
    record = json.loads(path.read_text())
    truth = json.loads((dataset / record["record"]["ground_truth_path"]).read_text())
    raw = path.with_suffix(".txt").read_text()
    assert raw == record["text"], f"Raw prediction mismatch: {path}"
    assert evaluate_prediction(truth, raw) == record["score"], f"Score mismatch: {path}"
    prompt = json.loads(path.with_suffix(".prompt.json").read_text())
    assert prompt["text"] == extraction_prompt(truth), f"Prompt mismatch: {path}"
    assert record["input_device"] == record["pixel_device"] == "cuda:0"
    assert not record["truncated"] and record["status"] == "ok"
    image_path = path.with_suffix(".png")
    with Image.open(image_path) as image:
        assert image.size == (1275, 1650) and image.mode == "RGB"
    altered = {
        **truth,
        "fields": dict.fromkeys(truth["fields"], "REDACTED"),
        "transcription": "REDACTED",
        "tables": [
            {**t, "rows": [["REDACTED"] * len(t["columns"]) for _ in t["rows"]]}
            for t in truth["tables"]
        ],
    }
    assert extraction_prompt(altered) == prompt["text"], f"Value leakage: {path}"
    try:
        record["parsed_prediction"] = json.loads(raw)
        record["json_valid"] = True
    except json.JSONDecodeError:
        record["json_valid"] = False
        record["parsed_prediction"] = None
    return record


def verify_run(run_path: Path, dataset: Path) -> dict:
    """Require complete cohort/repeats and unchanged code/data, returning audit evidence."""
    integrity = validate_dataset(dataset, FROZEN_HASH)
    config = json.loads((run_path / "config.json").read_text())
    for source, expected in config["source_hashes"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == expected, source
    assert config["max_new_tokens"] == 1024 and config["precision"] == "bfloat16"
    assert config["repeats"] == 2 and config["warmup"] == 1
    paths = sorted(p for p in run_path.glob("*--r*.json") if not p.name.endswith(".prompt.json"))
    records = [verify_request(p, dataset) for p in paths]
    assert len(records) == 60
    identities = Counter((r["record"]["document_id"], r["record"]["variant"]) for r in records)
    assert len(identities) == 30 and set(identities.values()) == {2}
    assert len({r["record"]["document_id"] for r in records}) == 10
    categories = Counter(r["record"]["category"] for r in records)
    assert len(categories) == 5 and set(categories.values()) == {12}
    assert {r["repeat"] for r in records} == {0, 1}
    repeated = {}
    for r in records:
        key = (r["record"]["document_id"], r["record"]["variant"])
        repeated.setdefault(key, []).append(r["text"])
    artifacts = {
        str(p.relative_to(run_path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run_path.rglob("*"))
        if p.is_file()
    }
    return {
        "status": "passed",
        "integrity": integrity,
        "requests": 60,
        "underlying_documents": 10,
        "distinct_pages": 30,
        "json_valid": sum(r["json_valid"] for r in records),
        "identical_repeat_pairs": sum(texts[0] == texts[1] for texts in repeated.values()),
        "checks": [
            "strict score recomputation",
            "raw text identity",
            "prompt invariance to values",
            "CUDA tensor diagnostics",
            "RGB visual inputs",
            "paired variants and repeats",
            "frozen dataset",
            "source hashes unchanged",
            "no truncation/runtime failure",
        ],
        "structured_predictions": {r["page_id"]: r["parsed_prediction"] for r in records},
        "artifact_sha256": artifacts,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument(
        "--dataset", type=Path, default=Path("datasets/generated/financial-v1-seed42")
    )
    args = parser.parse_args()
    print(json.dumps(verify_run(args.run, args.dataset), indent=2))
