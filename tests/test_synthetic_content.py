"""Check source truth and financial arithmetic without rendering dependencies."""

from decimal import Decimal

import pytest

from gpu_my_ocr.datasets.synthetic_data.content import CATEGORIES, make_document, money


def test_ids_layouts_and_source_determinism():
    documents = [
        make_document(42, category, index) for category in CATEGORIES for index in range(1, 11)
    ]
    assert len({document["document_id"] for document in documents}) == 50
    for document in documents:
        index = int(document["document_id"].split("-")[-1])
        assert document == make_document(42, document["category"], index)
        assert document["fields"]["reference_digits"].startswith("0000")
    for category in CATEGORIES:
        assert {doc["layout"] for doc in documents if doc["category"] == category} == {
            "form",
            "ledger",
        }
    assert make_document(43, "invoice", 1)["fields"] != make_document(42, "invoice", 1)["fields"]


@pytest.mark.parametrize("category", CATEGORIES)
def test_financial_calculations(category):
    for index in range(1, 11):
        document = make_document(42, category, index)
        fields, rows = document["fields"], document["tables"][0]["rows"]
        if category in ("receipt", "invoice", "portfolio"):
            for _, quantity, unit_price, amount in rows:
                assert money(Decimal(quantity) * Decimal(unit_price)) == amount
            subtotal = sum((Decimal(row[-1]) for row in rows), Decimal(0))
            if category == "portfolio":
                assert subtotal == Decimal(fields["holdings_value"])
                assert subtotal + Decimal(fields["cash_balance"]) == Decimal(fields["total"])
            else:
                assert subtotal == Decimal(fields["subtotal"])
                assert money(subtotal * Decimal(fields["tax_rate"])) == fields["tax"]
                assert subtotal + Decimal(fields["tax"]) == Decimal(fields["total"])
        elif category == "auditor_report":
            values = {name: Decimal(amount) for name, amount in rows}
            assert values["Revenue"] - values["Expenses"] == Decimal(fields["net_income"])
            assert values["Assets"] - values["Liabilities"] == Decimal(fields["equity"])
        else:
            for _, requested, approved in rows:
                assert money(Decimal(requested) * Decimal("0.80")) == approved
            for column, field in [(1, "requested_total"), (2, "approved_total")]:
                assert sum((Decimal(row[column]) for row in rows), Decimal(0)) == Decimal(
                    fields[field]
                )


def test_cent_rounding():
    assert money("1.005") == "1.01"
    assert money("0.0049") == "0.00"
