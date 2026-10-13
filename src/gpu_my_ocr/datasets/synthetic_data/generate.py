"""Generate new local datasets and freeze their manifest after PDF validation."""

import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .content import CATEGORIES, VERSION, make_document, stable_seed
from .integrity import VARIANTS, sha256_file, validate_dataset, validate_pdf
from .rendering import PAGE_SIZE, render_digital
from .scans import render_scans, transformations


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Save deterministic JSON to a new file, preserving intentional reading order."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True)
        stream.write("\n")


def generate_document(root: Path, seed: int, category: str, index: int) -> list[dict[str, Any]]:
    """Write one JSON truth first, then its validated PDFs; return manifest records."""
    truth = make_document(seed, category, index)
    document_id = truth["document_id"]
    truth_path = root / "ground_truth" / f"{document_id}.json"
    write_json(truth_path, truth)
    # Rendering reads the persisted source, so truth exists independently of PDF output.
    truth = json.loads(truth_path.read_text())
    pdfs = {variant: root / "pdfs" / f"{document_id}--{variant}.pdf" for variant in VARIANTS}
    render_digital(truth, pdfs["digital"])
    noise_seed = stable_seed(seed, category + ":degradation", index)
    render_scans(pdfs["digital"], pdfs["scan_high"], pdfs["scan_degraded"], noise_seed)
    records = []
    for variant, pdf in pdfs.items():
        validate_pdf(pdf, truth, variant)
        records.append(
            {
                name: truth[name]
                for name in ("dataset_version", "seed", "document_id", "category", "layout")
            }
        )
        records[-1].update(
            {
                "variant": variant,
                "page_count": 1,
                "page_size_points": list(PAGE_SIZE),
                "pdf_path": str(pdf.relative_to(root)),
                "pdf_sha256": sha256_file(pdf),
                "ground_truth_path": str(truth_path.relative_to(root)),
                "ground_truth_sha256": sha256_file(truth_path),
                "transformations": transformations(variant, noise_seed),
            }
        )
    return records


def generate_dataset(output: Path, seed: int = 42, per_category: int = 10) -> dict[str, Any]:
    """Create a new dataset (1..10 documents/category), freeze hashes and validate it.

    Existing output directories are rejected, even if empty. A failed generation
    remains for diagnosis; without a valid manifest/hash it is not a golden dataset.
    """
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    if type(per_category) is not int or not 1 <= per_category <= 10:
        raise ValueError("per_category must be an integer in [1, 10]")
    output.mkdir(parents=True, exist_ok=False)
    for directory in ("ground_truth", "pdfs"):
        (output / directory).mkdir()
    records = [
        record
        for category in CATEGORIES
        for index in range(1, per_category + 1)
        for record in generate_document(output, seed, category, index)
    ]
    manifest = {
        "dataset_version": VERSION,
        "seed": seed,
        "per_category": per_category,
        "underlying_documents": len(CATEGORIES) * per_category,
        "pdf_count": len(records),
        "python": platform.python_version(),
        "libraries": {name: version(name) for name in ("reportlab", "pymupdf", "pillow")},
        "records": records,
    }
    write_json(output / "manifest.json", manifest)
    with (output / "manifest.sha256").open("x") as stream:
        stream.write(sha256_file(output / "manifest.json") + "\n")
    return validate_dataset(output)
