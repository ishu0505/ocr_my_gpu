"""Score predictions against persisted JSON truth, never against generated model labels."""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .metrics import exact_match, ocr_rates
from .schema import prediction_schema

POLICY = {
    "version": "financial-page-v1",
    "max_cer": 0.01,
    "max_wer": 0.02,
    "all_critical_fields": True,
    "all_table_cells": True,
}


def compare_tables(truth: dict[str, Any], tables: list) -> dict[str, Any]:
    """Compare every ordered table cell by its reference mode; schema ensures shape."""
    comparisons = []
    for table_index, (expected_table, predicted_rows) in enumerate(
        zip(truth["tables"], tables, strict=True)
    ):
        for row_index, (expected_row, predicted_row) in enumerate(
            zip(expected_table["rows"], predicted_rows, strict=True)
        ):
            for column, (expected, predicted, mode) in enumerate(
                zip(expected_row, predicted_row, expected_table["modes"], strict=True)
            ):
                comparisons.append(
                    {
                        "table": table_index,
                        "row": row_index,
                        "column": column,
                        "match": exact_match(expected, predicted, mode),
                        "literal_match": expected == predicted,
                    }
                )
    correct = sum(cell["match"] for cell in comparisons)
    return {
        "correct": correct,
        "total": len(comparisons),
        "cells": comparisons,
        "accuracy": correct / len(comparisons) if comparisons else 1.0,
    }


def evaluate_prediction(truth: dict[str, Any], prediction: object) -> dict[str, Any]:
    """Return schema, critical-field, table and OCR scores against supplied stored truth.

    Malformed JSON/schema failures are failed pages with unavailable metrics, never
    silently coerced to valid predictions. All fields in v1 truth are critical.
    """
    result = {"document_id": truth["document_id"], "policy": POLICY.copy(), "page_success": False}
    if isinstance(prediction, str):
        try:
            prediction = json.loads(prediction)
        except json.JSONDecodeError as error:
            return {**result, "schema_valid": False, "errors": [str(error)], "metrics": None}
    errors = [
        error.message
        for error in Draft202012Validator(prediction_schema(truth)).iter_errors(prediction)
    ]
    if errors:
        return {**result, "schema_valid": False, "errors": errors, "metrics": None}
    fields = {
        name: {
            "match": exact_match(value, prediction["fields"][name], truth["field_modes"][name]),
            "literal_match": value == prediction["fields"][name],
        }
        for name, value in truth["fields"].items()
    }
    table_scores = compare_tables(truth, prediction["tables"])
    rates = ocr_rates(truth["transcription"], prediction["transcription"])
    correct = sum(field["match"] for field in fields.values())
    success = (
        correct == len(fields)
        and table_scores["correct"] == table_scores["total"]
        and rates["cer"] <= POLICY["max_cer"]
        and rates["wer"] <= POLICY["max_wer"]
    )
    return {
        **result,
        "schema_valid": True,
        "errors": [],
        "page_success": success,
        "metrics": {
            **rates,
            "fields": fields,
            "critical_fields_correct": correct,
            "critical_fields_total": len(fields),
            "tables": table_scores,
        },
    }


def evaluate_files(ground_truth: Path, prediction: Path) -> dict[str, Any]:
    """Load canonical JSON truth and raw prediction text, returning independent scores."""
    return evaluate_prediction(json.loads(ground_truth.read_text()), prediction.read_text())
