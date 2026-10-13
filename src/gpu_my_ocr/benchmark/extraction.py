"""Value-free extraction instructions and uniform visual PDF inputs."""

import json
from pathlib import Path
from typing import Any

import pymupdf

from gpu_my_ocr.benchmark.accuracy.schema import prediction_schema

PROMPT_VERSION = "financial-extraction-v2"


def extraction_prompt(truth: dict[str, Any]) -> str:
    """Build instructions from schema names/shapes only; never include reference values."""
    schema = prediction_schema(truth)
    columns = [table["columns"] for table in truth["tables"]]
    skeleton = {
        "fields": dict.fromkeys(truth["fields"], ""),
        "tables": [[["" for _ in t["columns"]] for _ in t["rows"]] for t in truth["tables"]],
        "transcription": "",
    }
    return (
        f"Extraction contract {PROMPT_VERSION}. Read the supplied document image. "
        "Return only one JSON object, no markdown fences or explanation. "
        "Extract every field as a string, preserving leading zeros, repetitive digits, "
        "case and punctuation. Numeric amounts and ratios must also be strings. "
        "Use printed field labels; snake_case keys correspond to those labels. "
        "Tables contain data rows only, in printed order, excluding headers and titles. "
        "IMPORTANT: tables is an array of TABLES, each table is an array of ROWS, "
        "each row is an array of string CELLS. Keep all three levels of nesting. "
        "Transcription must include ALL visible text in reading order: title, synthetic "
        "notice, labels and values, table headers and cells, narrative and footer. "
        "Separate text segments with whitespace. Do not invent or calculate missing text. "
        f"Fill this exact empty output skeleton with image-derived values: {json.dumps(skeleton)}. "
        f"Table column order: {json.dumps(columns)}. "
        f"Required JSON schema: {json.dumps(schema, separators=(',', ':'))}"
    )


def render_page(pdf_path: Path, image_path: Path) -> None:
    """Render a one-page PDF to RGB PNG at 150 DPI, without reading its text layer."""
    with pymupdf.open(pdf_path) as pdf:
        if len(pdf) != 1:
            raise ValueError("Baseline requires a single page")
        pdf[0].get_pixmap(dpi=150, colorspace=pymupdf.csRGB, alpha=False).save(image_path)
