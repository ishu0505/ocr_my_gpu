"""End-to-end PDF reproducibility, text contracts, shared truth and integrity checks."""

import io
import json

import pymupdf
import pytest
from PIL import Image

from gpu_my_ocr.datasets.synthetic_data.generate import generate_dataset
from gpu_my_ocr.datasets.synthetic_data.integrity import sha256_file, validate_dataset
from gpu_my_ocr.datasets.synthetic_data.scans import degrade_image


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("datasets") / "first"
    summary = generate_dataset(root, per_category=2)
    assert summary["documents"] == 10 and summary["pdfs"] == 30
    return root


def test_deterministic_regeneration(dataset, tmp_path):
    second = tmp_path / "second"
    generate_dataset(second, per_category=2)
    assert (dataset / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    first_files = {
        str(path.relative_to(dataset)): sha256_file(path)
        for path in dataset.rglob("*")
        if path.is_file()
    }
    second_files = {
        str(path.relative_to(second)): sha256_file(path)
        for path in second.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


def test_pdf_text_and_shared_truth(dataset):
    manifest = json.loads((dataset / "manifest.json").read_text())
    groups = {}
    for record in manifest["records"]:
        truth_path = dataset / record["ground_truth_path"]
        truth = json.loads(truth_path.read_text())
        groups.setdefault(record["document_id"], set()).add(record["ground_truth_sha256"])
        with pymupdf.open(dataset / record["pdf_path"]) as pdf:
            assert len(pdf) == 1
            assert pdf[0].rect.width == 612 and pdf[0].rect.height == 792
            if record["variant"] == "digital":
                assert " ".join(pdf[0].get_text().split()) == " ".join(
                    truth["transcription"].split()
                )
            else:
                assert pdf[0].get_text() == ""
                image = pdf[0].get_images()[0]
                assert image[2:4] == (1275, 1650)
    assert all(len(hashes) == 1 for hashes in groups.values())


def test_scan_pixels_match_source_and_recorded_transform(dataset):
    digital = dataset / "pdfs/invoice-42-001--digital.pdf"
    with pymupdf.open(digital) as pdf:
        pixmap = pdf[0].get_pixmap(dpi=150, colorspace=pymupdf.csGRAY, alpha=False)
        source = Image.frombytes("L", (pixmap.width, pixmap.height), pixmap.samples)
    manifest = json.loads((dataset / "manifest.json").read_text())
    for record in manifest["records"]:
        if record["document_id"] != "invoice-42-001" or record["variant"] == "digital":
            continue
        with pymupdf.open(dataset / record["pdf_path"]) as pdf:
            stream = pdf.extract_image(pdf[0].get_images()[0][0])["image"]
        # MuPDF exports ICC grayscale PDF images as RGB PNGs with equal channels.
        pixels = Image.open(io.BytesIO(stream)).convert("L")
        expected = (
            source
            if record["variant"] == "scan_high"
            else degrade_image(source, record["transformations"]["noise_seed"])
        )
        assert pixels.tobytes() == expected.tobytes()


def test_external_frozen_hash(dataset):
    digest = sha256_file(dataset / "manifest.json")
    assert validate_dataset(dataset, digest)["status"] == "passed"
    with pytest.raises(ValueError, match="Frozen experiment"):
        validate_dataset(dataset, "0" * 64)


def test_degradation_is_deterministic_and_changes_pixels():
    image = Image.new("L", (64, 64), 128)
    first = degrade_image(image, 42)
    assert first.tobytes() == degrade_image(image, 42).tobytes()
    assert first.tobytes() != degrade_image(image, 43).tobytes()
    assert first.tobytes() != image.tobytes()
    assert min(first.tobytes()) >= 120 and max(first.tobytes()) <= 136


def test_overwrite_protection(dataset, tmp_path):
    before = sha256_file(dataset / "manifest.json")
    with pytest.raises(FileExistsError):
        generate_dataset(dataset)
    assert before == sha256_file(dataset / "manifest.json")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileExistsError):
        generate_dataset(empty)


@pytest.mark.parametrize("artifact", ["manifest", "truth", "pdf"])
def test_hash_tampering_detected(dataset, tmp_path, artifact):
    import shutil

    root = tmp_path / "corrupted"
    shutil.copytree(dataset, root)
    manifest = json.loads((root / "manifest.json").read_text())
    target = (
        root
        / {
            "manifest": "manifest.json",
            "truth": manifest["records"][0]["ground_truth_path"],
            "pdf": manifest["records"][0]["pdf_path"],
        }[artifact]
    )
    with target.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_dataset(root)


@pytest.mark.parametrize("count", [0, 11, True])
def test_invalid_count_leaves_no_output(tmp_path, count):
    root = tmp_path / "invalid"
    with pytest.raises(ValueError):
        generate_dataset(root, per_category=count)
    assert not root.exists()
