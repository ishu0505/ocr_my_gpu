"""Hash and verify frozen datasets, including PDF text and page geometry."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pymupdf

from .content import CATEGORIES, VERSION, stable_seed
from .rendering import PAGE_SIZE
from .scans import transformations

VARIANTS = ("digital", "scan_high", "scan_degraded")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a local artifact's exact bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pdf(path: Path, truth: dict[str, Any], variant: str) -> None:
    """Require one matching page, expected digital text or raster-only scanned content."""
    with pymupdf.open(path) as pdf:
        if len(pdf) != 1 or tuple(pdf[0].rect)[2:] != PAGE_SIZE:
            raise ValueError(f"Unexpected page geometry: {path}")
        page = pdf[0]
        text = " ".join(page.get_text().split())
        if variant == "digital":
            if text != " ".join(truth["transcription"].split()):
                raise ValueError(f"Digital text differs from canonical truth: {path}")
        elif text or len(page.get_images()) != 1 or page.get_drawings():
            raise ValueError(f"Scan must contain only a raster image: {path}")


def artifact_path(root: Path, relative: str) -> Path:
    """Resolve a manifest artifact, rejecting absolute paths or escape from the dataset."""
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact escapes dataset: {relative}")
    return path


def validate_record_metadata(record: dict[str, Any], truth: dict[str, Any], manifest: dict) -> None:
    """Check IDs, layout, page metadata and transforms against versioned generation rules."""
    for name in ("document_id", "category", "dataset_version", "seed", "layout"):
        if record[name] != truth[name] or (name in manifest and truth[name] != manifest[name]):
            raise ValueError(f"Ground truth metadata mismatch: {name}")
    index = int(truth["document_id"].split("-")[-1])
    expected_id = f"{truth['category']}-{manifest['seed']}-{index:03d}"
    if truth["document_id"] != expected_id or truth["category"] not in CATEGORIES:
        raise ValueError("Document ID/category mismatch")
    if record["page_count"] != 1 or record["page_size_points"] != list(PAGE_SIZE):
        raise ValueError("Manifest page metadata mismatch")
    expected_seed = stable_seed(manifest["seed"], truth["category"] + ":degradation", index)
    if record["transformations"] != transformations(record["variant"], expected_seed):
        raise ValueError("Manifest transformation mismatch")


def validate_dataset(root: Path, expected_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Verify frozen manifest/hash, groups, artifact digests and all PDF text contracts."""
    manifest_path = root / "manifest.json"
    if (
        expected_manifest_sha256 is not None
        and sha256_file(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("Frozen experiment manifest hash mismatch")
    if sha256_file(manifest_path) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    expected_ids = expected_document_ids(manifest)
    groups: dict[str, list[dict[str, Any]]] = {}
    paths = set()
    for record in manifest["records"]:
        path = artifact_path(root, record["pdf_path"])
        truth_path = artifact_path(root, record["ground_truth_path"])
        if path in paths or sha256_file(path) != record["pdf_sha256"]:
            raise ValueError("Duplicate PDF path or PDF hash mismatch")
        paths.add(path)
        if sha256_file(truth_path) != record["ground_truth_sha256"]:
            raise ValueError("Ground truth hash mismatch")
        truth = json.loads(truth_path.read_text())
        validate_record_metadata(record, truth, manifest)
        validate_pdf(path, truth, record["variant"])
        groups.setdefault(record["document_id"], []).append(record)
    if set(groups) != expected_ids:
        raise ValueError("Missing or unexpected underlying document IDs")
    for records in groups.values():
        if sorted(r["variant"] for r in records) != sorted(VARIANTS):
            raise ValueError("Each document requires exactly three variants")
        if len({(r["ground_truth_path"], r["ground_truth_sha256"]) for r in records}) != 1:
            raise ValueError("Variants do not share canonical ground truth")
    return {
        "status": "passed",
        "documents": len(groups),
        "pdfs": len(paths),
        "manifest_sha256": sha256_file(manifest_path),
    }


def expected_document_ids(manifest: dict[str, Any]) -> set[str]:
    """Validate version/count/seed metadata and return the exact expected ID set."""
    if manifest["dataset_version"] != VERSION:
        raise ValueError("Unsupported dataset version")
    count, seed = manifest["per_category"], manifest["seed"]
    if type(count) is not int or not 1 <= count <= 10:
        raise ValueError("Invalid manifest document count")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("Invalid manifest seed")
    if (
        manifest["underlying_documents"] != len(CATEGORIES) * count
        or manifest["pdf_count"] != 3 * len(CATEGORIES) * count
    ):
        raise ValueError("Manifest counts mismatch")
    return {
        f"{category}-{seed}-{index:03d}" for category in CATEGORIES for index in range(1, count + 1)
    }
