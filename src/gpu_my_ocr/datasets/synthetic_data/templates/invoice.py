"""Fictitious business invoices with payment terms and totals."""

import random
from typing import Any

from ..content import priced_rows, totals


def build(rng: random.Random, index: int) -> dict[str, Any]:
    """Return invoice fields and service items for a local RNG and document index."""
    rows = priced_rows(rng, ["Test consulting hours", "Sample support units", "Demo license seats"])
    return {
        "title": "Commercial Invoice",
        "fields": {
            "invoice_number": f"INV-0000{index:04d}",
            "issue_date": f"2026-09-{index:02d}",
            "due_date": f"2026-10-{index:02d}",
            "payment_terms": "Net 30 days",
            "supplier": "Fictional Northstar Services",
            "customer": f"Synthetic Client {index:02d}",
            "currency": "USD",
            **totals(rows, "0.09"),
        },
        "numeric_fields": ["subtotal", "tax_rate", "tax", "total"],
        "tables": [
            {
                "title": "Invoice details",
                "columns": ["Description", "Quantity", "Unit price", "Amount"],
                "modes": ["exact", "numeric", "numeric", "numeric"],
                "rows": rows,
            }
        ],
        "narrative": ["Payment instructions: TEST ONLY. Do not remit funds."],
    }
