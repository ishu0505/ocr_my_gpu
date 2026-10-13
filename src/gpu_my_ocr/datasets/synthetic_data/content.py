"""Shared financial arithmetic and document construction, without PDF dependencies."""

import hashlib
import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Any

VERSION = "financial-synthetic-v1"
NOTICE = "SYNTHETIC TEST DOCUMENT - NOT VALID FOR FINANCIAL USE"
FOOTER = "All names, identifiers and amounts are fictitious. Page 1 of 1."
CATEGORIES = ("portfolio", "receipt", "invoice", "auditor_report", "insurance_claim")


def money(value: Decimal | str | int) -> str:
    """Return a two-decimal financial string using explicit half-up cent rounding."""
    return str(Decimal(value).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def stable_seed(seed: int, category: str, index: int) -> int:
    """Derive a stable local RNG seed independent of generation order or Python hash."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{category}:{index}".encode()).digest()[:8])


def priced_rows(rng: random.Random, names: list[str]) -> list[list[str]]:
    """Return item, integer quantity, unit price and rounded amount rows for names."""
    rows = []
    for name in names:
        quantity = rng.randint(1, 25)
        price = Decimal(rng.randint(125, 25000)) / 100
        rows.append([name, str(quantity), money(price), money(quantity * price)])
    return rows


def totals(rows: list[list[str]], rate: str = "0.07") -> dict[str, str]:
    """Return subtotal, tax rate, cent-rounded tax and total from priced rows."""
    subtotal = sum((Decimal(row[-1]) for row in rows), Decimal(0))
    tax = Decimal(money(subtotal * Decimal(rate)))
    return {
        "subtotal": money(subtotal),
        "tax_rate": rate,
        "tax": money(tax),
        "total": money(subtotal + tax),
    }


def make_document(seed: int, category: str, index: int) -> dict[str, Any]:
    """Build pre-render truth for one category/index using its independently seeded RNG."""
    from gpu_my_ocr.datasets.synthetic_data.templates import (
        auditor_report,
        insurance_claim,
        invoice,
        portfolio,
        receipt,
    )

    builders = dict(
        zip(
            CATEGORIES,
            (
                portfolio.build,
                receipt.build,
                invoice.build,
                auditor_report.build,
                insurance_claim.build,
            ),
            strict=True,
        )
    )
    document = builders[category](random.Random(stable_seed(seed, category, index)), index)
    document_id = f"{category}-{seed}-{index:03d}"
    document.update(
        {
            "dataset_version": VERSION,
            "seed": seed,
            "document_id": document_id,
            "category": category,
            "layout": "ledger" if index % 2 else "form",
        }
    )
    document["fields"]["reference_digits"] = f"0000{index:04d}" + "11112222333344445555"
    document["field_modes"] = field_modes(document)
    document["transcription"] = "\n".join(text_lines(document))
    return document


def field_modes(document: dict[str, Any]) -> dict[str, str]:
    """Declare numeric fields; all remaining fields are critical literal strings."""
    numeric = document.pop("numeric_fields")
    return {name: "numeric" if name in numeric else "exact" for name in document["fields"]}


def text_lines(document: dict[str, Any]) -> list[str]:
    """Return canonical visible text in renderer reading order, before any rendering."""
    lines = [document["title"], NOTICE, f"Document ID: {document['document_id']}"]
    lines += [
        f"{name.replace('_', ' ').title()}: {value}" for name, value in document["fields"].items()
    ]
    for table in document["tables"]:
        lines += [table["title"], *table["columns"]]
        lines += [cell for row in table["rows"] for cell in row]
    lines += document["narrative"]
    return lines + [FOOTER]
