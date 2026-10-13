"""Strict prediction schema derived from stored reference field/table names."""

from typing import Any


def prediction_schema(truth: dict[str, Any]) -> dict[str, Any]:
    """Return a Draft 2020-12 schema requiring string values and ordered table shapes."""
    tables = []
    for table in truth["tables"]:
        row = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": len(table["columns"]),
            "maxItems": len(table["columns"]),
        }
        tables.append(
            {
                "type": "array",
                "items": row,
                "minItems": len(table["rows"]),
                "maxItems": len(table["rows"]),
            }
        )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["fields", "tables", "transcription"],
        "additionalProperties": False,
        "properties": {
            "fields": {
                "type": "object",
                "additionalProperties": False,
                "required": list(truth["fields"]),
                "properties": {name: {"type": "string"} for name in truth["fields"]},
            },
            "tables": {
                "type": "array",
                "prefixItems": tables,
                "items": False,
                "minItems": len(tables),
                "maxItems": len(tables),
            },
            "transcription": {"type": "string"},
        },
    }
