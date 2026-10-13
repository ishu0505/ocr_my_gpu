"""Fictitious insurance claims with approved amounts and status labels."""

import random
from decimal import Decimal
from typing import Any

from ..content import money


def build(rng: random.Random, index: int) -> dict[str, Any]:
    """Return a claim with requested/approved line amounts for RNG and index."""
    rows = []
    for item in ["Test inspection", "Sample repair", "Demo replacement"]:
        requested = Decimal(rng.randint(10000, 250000)) / 100
        approved = Decimal(money(requested * Decimal("0.80")))
        rows.append([item, money(requested), money(approved)])
    requested_total = sum((Decimal(row[1]) for row in rows), Decimal(0))
    approved_total = sum((Decimal(row[2]) for row in rows), Decimal(0))
    return {
        "title": "Insurance Claim Assessment",
        "fields": {
            "policy_number": f"00003333{index:06d}",
            "claim_id": f"CLM-0000{index:04d}",
            "claimant": f"Fictional Claimant {index:02d}",
            "incident_date": f"2026-08-{index:02d}",
            "assessment_date": f"2026-09-{index:02d}",
            "currency": "USD",
            "status": "Partially approved",
            "requested_total": money(requested_total),
            "approved_total": money(approved_total),
        },
        "numeric_fields": ["requested_total", "approved_total"],
        "tables": [
            {
                "title": "Claim items",
                "columns": ["Claim item", "Requested", "Approved"],
                "modes": ["exact", "numeric", "numeric"],
                "rows": rows,
            }
        ],
        "narrative": ["Approval amounts are fictitious and create no insurance entitlement."],
    }
