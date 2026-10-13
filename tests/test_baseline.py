"""Protect blind visual inference and paired selection independently of GPU packages."""

import copy
import json
from pathlib import Path

import pymupdf
from PIL import Image

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.baseline import selected_records
from gpu_my_ocr.benchmark.extraction import extraction_prompt, render_page


def reference():
    return {
        "document_id": "test-id",
        "fields": {"account_id": "SECRET-00001111", "total": "9845.12"},
        "field_modes": {"account_id": "exact", "total": "numeric"},
        "tables": [
            {
                "columns": ["Label", "Amount"],
                "rows": [["PRIVATE", "123.45"]],
                "modes": ["exact", "numeric"],
            }
        ],
        "transcription": "PRIVATE TRANSCRIPTION",
    }


def test_prompt_invariant_to_reference_values():
    truth = reference()
    changed = copy.deepcopy(truth)
    changed["fields"] = dict.fromkeys(truth["fields"], "different")
    changed["tables"][0]["rows"] = [["different", "different"]]
    changed["transcription"] = "different"
    assert extraction_prompt(truth) == extraction_prompt(changed)
    for value in [*truth["fields"].values(), "PRIVATE", "123.45", truth["transcription"]]:
        assert value not in extraction_prompt(truth)


def test_selection_keeps_ids_layouts_and_variants():
    records = [
        {"document_id": f"{c}-42-{i}", "variant": v}
        for c in ("invoice", "receipt")
        for i in ("001", "002", "010")
        for v in ("digital", "scan_high", "scan_degraded")
    ]
    chosen = selected_records({"records": records}, False)
    assert len(chosen) == 12
    assert all(not r["document_id"].endswith("002") for r in chosen)
    assert len(selected_records({"records": records}, True)) == 2


def test_render_visual_input(tmp_path: Path):
    pdf_path, png = tmp_path / "page.pdf", tmp_path / "page.png"
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=612, height=792)
        page.insert_text((40, 40), "Visual only")
        pdf.save(pdf_path)
    render_page(pdf_path, png)
    with Image.open(png) as image:
        assert image.size == (1275, 1650)
        assert image.mode == "RGB"


def test_malformed_outputs_are_not_repaired():
    truth = reference()
    prediction = {
        "fields": truth["fields"],
        "tables": [[truth["tables"][0]["rows"][0]]],
        "transcription": truth["transcription"],
    }
    assert evaluate_prediction(truth, json.dumps(prediction))["page_success"]
    assert not evaluate_prediction(truth, "```json\n" + json.dumps(prediction) + "\n```")[
        "schema_valid"
    ]
    assert not evaluate_prediction(truth, json.dumps(prediction)[:-1])["schema_valid"]


def test_report_invalid_pages_remain_in_denominators():
    from gpu_my_ocr.benchmark.baseline_report import summarize

    truth = reference()
    prediction = {
        "fields": truth["fields"],
        "tables": [truth["tables"][0]["rows"]],
        "transcription": truth["transcription"],
    }
    common = {
        "record": {"document_id": "test-id"},
        "status": "ok",
        "end_to_end_ms": 1000,
        "output_tokens": 100,
        "expected_fields": 2,
        "expected_numeric": 1,
        "expected_cells": 2,
    }
    valid = {**common, "score": evaluate_prediction(truth, prediction), "numeric_correct": 1}
    invalid = {**common, "score": evaluate_prediction(truth, "broken"), "numeric_correct": 0}
    scores = summarize([valid, invalid])
    assert scores["field_accuracy"] == 0.5
    assert scores["numeric_accuracy"] == 0.5
    assert scores["table_accuracy"] == 0.5
    assert scores["page_success"] == 1
    assert scores["cer_valid_mean"] == 0
