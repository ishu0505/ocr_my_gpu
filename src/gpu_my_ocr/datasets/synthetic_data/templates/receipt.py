"""Fictitious retail receipts with cent-rounded sales tax."""

import random
from typing import Any

from ..content import priced_rows, totals


def build(rng: random.Random, index: int) -> dict[str, Any]:
    """Return receipt fields and priced items for a local RNG and document index."""
    rows = priced_rows(rng, ["Notebook pack", "Desk lamp", "Archive folder"])
    return {
        "title": "Retail Receipt",
        "fields": {
            "merchant": f"Fictional Willow Store {index:02d}",
            "transaction_id": f"00008888{index:06d}",
            "date": f"2026-09-{index:02d}",
            "payment_method": "TEST CARD 0000",
            "currency": "USD",
            **totals(rows),
        },
        "numeric_fields": ["subtotal", "tax_rate", "tax", "total"],
        "tables": [
            {
                "title": "Purchased items",
                "columns": ["Item", "Quantity", "Unit price", "Amount"],
                "modes": ["exact", "numeric", "numeric", "numeric"],
                "rows": rows,
            }
        ],
        "narrative": ["Thank you. This receipt cannot be used as proof of payment."],
    }
