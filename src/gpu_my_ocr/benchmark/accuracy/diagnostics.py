"""Non-repairing diagnostic views; never substitute these for strict baseline scores."""

import json
from typing import Any

from .evaluate import evaluate_prediction
from .metrics import exact_match, ocr_rates


def parse_view(text: str) -> dict[str, Any]:
    """Parse raw JSON and explicitly label wrapper/extra-text views without changing raw text."""
    try:
        return {
            "raw_json_valid": True,
            "view": "raw",
            "prediction": json.loads(text),
            "parse_error": None,
            "extra_text": False,
            "markdown_fence": False,
        }
    except json.JSONDecodeError as error:
        result = {
            "raw_json_valid": False,
            "parse_error": str(error),
            "prediction": None,
            "view": "unavailable",
            "extra_text": False,
            "markdown_fence": "```" in text,
        }
    # Decode a single existing object; no edits, coercion, dropped cells or inferred values.
    positions = [i for i, char in enumerate(text) if char == "{"]
    for start in positions:
        try:
            prediction, end = json.JSONDecoder().raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(prediction, dict) and (
            "fields" in prediction or "transcription" in prediction
        ):
            prefix, suffix = text[:start].strip(), text[start + end :].strip()
            result.update(
                prediction=prediction,
                view="embedded_object_diagnostic_only",
                extra_text=bool(prefix or suffix),
            )
            return result
    return result


def field_diagnostics(truth: dict, prediction: dict, names: list[str] | None = None) -> dict:
    """Score existing fields independently of schema; missing/type errors receive no credit."""
    names = list(truth["fields"]) if names is None else names
    fields = prediction.get("fields", {})
    fields = fields if isinstance(fields, dict) else {}
    comparisons = {
        name: {
            "present": name in fields,
            "match": exact_match(
                truth["fields"][name], fields.get(name), truth["field_modes"][name]
            ),
            "mode": truth["field_modes"][name],
        }
        for name in names
    }
    correct = sum(c["match"] for c in comparisons.values())
    numeric = [c for c in comparisons.values() if c["mode"] == "numeric"]
    return {
        "fields": comparisons,
        "total": len(names),
        "correct": correct,
        "present": sum(c["present"] for c in comparisons.values()),
        "missing": [name for name, c in comparisons.items() if not c["present"]],
        "incorrect_numeric": [
            name
            for name, c in comparisons.items()
            if c["mode"] == "numeric" and c["present"] and not c["match"]
        ],
        "incorrect_digits": [
            name
            for name, c in comparisons.items()
            if c["present"]
            and not c["match"]
            and (name.endswith("_id") or name == "reference_digits")
        ],
        "numeric_total": len(numeric),
        "numeric_correct": sum(c["match"] for c in numeric),
    }


def table_diagnostics(truth: dict, prediction: dict) -> dict:
    """Compare original positional cells even with invalid structure; never remove headers."""
    tables = prediction.get("tables", [])
    tables = tables if isinstance(tables, list) else []
    correct = present = total = 0
    shapes = []
    for index, table in enumerate(truth["tables"]):
        rows = tables[index] if index < len(tables) and isinstance(tables[index], list) else []
        shapes.append(
            {
                "expected_rows": len(table["rows"]),
                "actual_rows": len(rows),
                "expected_columns": len(table["columns"]),
            }
        )
        for row_index, row in enumerate(table["rows"]):
            cells = (
                rows[row_index]
                if row_index < len(rows) and isinstance(rows[row_index], list)
                else []
            )
            for column, value in enumerate(row):
                total += 1
                actual = cells[column] if column < len(cells) else None
                present += isinstance(actual, str)
                correct += exact_match(value, actual, table["modes"][column])
    return {
        "correct": correct,
        "present": present,
        "total": total,
        "shapes": shapes,
        "actual_tables": len(tables),
        "expected_tables": len(truth["tables"]),
    }


def diagnose(truth: dict, text: str) -> dict[str, Any]:
    """Return strict scores alongside labelled independent views of raw model extraction."""
    parsed = parse_view(text)
    prediction = parsed["prediction"] if isinstance(parsed["prediction"], dict) else {}
    transcription = prediction.get("transcription")
    return {
        **parsed,
        "strict": evaluate_prediction(truth, text),
        "view_schema": evaluate_prediction(truth, prediction),
        "field_diagnostics": field_diagnostics(truth, prediction),
        "table_diagnostics": table_diagnostics(truth, prediction),
        "transcription_rates": ocr_rates(truth["transcription"], transcription)
        if isinstance(transcription, str)
        else None,
    }
