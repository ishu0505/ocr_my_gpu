"""Clearly synthetic audit summaries and opinions."""

import random
from decimal import Decimal
from typing import Any

from ..content import money


def build(rng: random.Random, index: int) -> dict[str, Any]:
    """Return financial summaries/opinion labels for a local RNG and document index."""
    revenue, expenses = Decimal(rng.randint(100000, 900000)), Decimal(rng.randint(10000, 99999))
    assets, liabilities = Decimal(rng.randint(200000, 950000)), Decimal(rng.randint(10000, 199999))
    rows = [
        ["Revenue", money(revenue)],
        ["Expenses", money(expenses)],
        ["Net income", money(revenue - expenses)],
        ["Assets", money(assets)],
        ["Liabilities", money(liabilities)],
        ["Equity", money(assets - liabilities)],
    ]
    return {
        "title": "Independent Auditor Test Report",
        "fields": {
            "report_id": f"AUD-0000{index:04d}",
            "entity": f"Fictional Cedar Entity {index:02d}",
            "period_end": "2025-12-31",
            "report_date": f"2026-09-{index:02d}",
            "audit_firm": "Synthetic Example Audit LLP",
            "currency": "USD",
            "opinion": ["Unqualified", "Qualified", "Adverse", "Disclaimer"][(index - 1) % 4],
            "net_income": money(revenue - expenses),
            "equity": money(assets - liabilities),
        },
        "numeric_fields": ["net_income", "equity"],
        "tables": [
            {
                "title": "Financial summary",
                "columns": ["Measure", "Amount"],
                "modes": ["exact", "numeric"],
                "rows": rows,
            }
        ],
        "narrative": [
            "Opinion label is a synthetic test category, not an actual audit opinion.",
            "SYNTHETIC SIGNATURE GRAPHIC - NOT A REAL SIGNATURE",
        ],
    }
