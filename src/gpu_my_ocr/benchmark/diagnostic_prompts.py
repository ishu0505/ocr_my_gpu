"""Separate frozen full-schema, plain OCR and reduced-field prompt contracts."""

import json
from typing import Any

from gpu_my_ocr.benchmark.extraction import extraction_prompt

CRITICAL_FIELDS = {
    "portfolio": ["account_id", "total", "reference_digits"],
    "receipt": ["transaction_id", "subtotal", "total", "reference_digits"],
    "invoice": ["invoice_number", "subtotal", "tax", "total", "reference_digits"],
    "auditor_report": ["report_id", "net_income", "equity", "reference_digits"],
    "insurance_claim": ["claim_id", "requested_total", "approved_total", "reference_digits"],
}
VERSIONS = {
    "full": "financial-extraction-v2",
    "ocr": "visual-transcription-v1",
    "minimal": "critical-fields-v1",
}


def diagnostic_prompt(truth: dict[str, Any], mode: str) -> str:
    """Return mode-specific image instructions from names only; no reference values."""
    if mode == "full":
        return extraction_prompt(truth)
    if mode == "ocr":
        return (
            "Visual transcription contract visual-transcription-v1. Transcribe ALL visible "
            "text from the image in reading order, including title, notice, labels, values, "
            "table title, headers and cells, narrative and footer. Return plain text only, "
            "without markdown or explanation. Preserve case, punctuation, leading zeros "
            "and every repetitive digit exactly. Separate segments with whitespace."
        )
    if mode != "minimal":
        raise ValueError(f"Unknown diagnostic mode: {mode}")
    names = CRITICAL_FIELDS[truth["category"]]
    return (
        "Critical extraction contract critical-fields-v1. Read the supplied document image. "
        "Return only JSON, no markdown or extra text. Extract these printed fields exactly "
        "as strings including numeric amounts, leading zeros and repetitive digits. "
        "Do not calculate or invent values. Fill this exact object from the image: "
        + json.dumps({"fields": dict.fromkeys(names, "")})
    )
