"""Fictitious investment holdings and account valuations."""

import random
from decimal import Decimal
from typing import Any

from ..content import money, priced_rows, totals


def build(rng: random.Random, index: int) -> dict[str, Any]:
    """Return a portfolio's structured fields/table for a local RNG and document index."""
    rows = priced_rows(rng, ["Aster Growth Fund", "Lumen Bond Fund", "Cedar Index Fund"])
    valuation = totals(rows, "0")["total"]
    cash = money(rng.randint(100, 5000))
    return {
        "title": "Investment Portfolio Statement",
        "fields": {
            "account_id": f"000071{index:06d}",
            "holder": f"Fictional Investor {index:02d}",
            "statement_date": f"2026-09-{index:02d}",
            "currency": "USD",
            "holdings_value": valuation,
            "cash_balance": cash,
            "total": money(Decimal(valuation) + Decimal(cash)),
        },
        "numeric_fields": ["holdings_value", "cash_balance", "total"],
        "tables": [
            {
                "title": "Holdings and valuation",
                "columns": ["Holding", "Quantity", "Unit price", "Value"],
                "modes": ["exact", "numeric", "numeric", "numeric"],
                "rows": rows,
            }
        ],
        "narrative": ["Valuations are synthetic test prices, not market quotations."],
    }
