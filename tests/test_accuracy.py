"""Independent golden scoring, especially exact digit and financial behavior."""

import copy
import json

import pytest

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.accuracy.metrics import exact_match, numeric_value, ocr_rates


@pytest.fixture
def reference():
    # Handwritten oracle deliberately does not import the synthetic generator.
    return {
        "document_id": "handwritten",
        "fields": {"id": "000011112222", "total": "1234.50"},
        "field_modes": {"id": "exact", "total": "numeric"},
        "tables": [
            {
                "columns": ["Item", "Amount"],
                "modes": ["exact", "numeric"],
                "rows": [["Test", "1234.50"]],
            }
        ],
        "transcription": "Test 000011112222 1234.50",
    }


@pytest.fixture
def prediction(reference):
    return {
        "fields": copy.deepcopy(reference["fields"]),
        "tables": [copy.deepcopy(table["rows"]) for table in reference["tables"]],
        "transcription": reference["transcription"],
    }


def test_perfect_stored_truth_score(reference, prediction):
    scores = evaluate_prediction(reference, json.dumps(prediction))
    assert scores["schema_valid"] and scores["page_success"]
    assert scores["metrics"]["cer"] == scores["metrics"]["wer"] == 0
    assert scores["metrics"]["tables"]["accuracy"] == 1


def test_decimal_normalization_retains_literal_diagnostic(reference, prediction):
    prediction["fields"]["total"] = "USD 1,234.500"
    scores = evaluate_prediction(reference, prediction)
    assert scores["page_success"]
    assert scores["metrics"]["fields"]["total"] == {"match": True, "literal_match": False}


@pytest.mark.parametrize("value", ["NaN", "Infinity", "1e3", "1,23.00", "1.234,50", 1234.5, True])
def test_invalid_numeric_values(value):
    with pytest.raises((TypeError, ValueError)):
        numeric_value(value)
    assert not exact_match("1234.50", value, "numeric")


def test_long_repetitive_digits():
    digits = "0000" + "1111222233334444" * 32
    changed = digits[:200] + "9" + digits[201:]
    assert not exact_match(digits, digits.lstrip("0"))
    assert not exact_match(digits, changed)
    assert ocr_rates(digits, changed) == {"cer": 1 / len(digits), "wer": 1.0}


def test_ocr_rates_and_empty_text():
    assert ocr_rates("a b", "a c") == {"cer": 1 / 3, "wer": 1 / 2}
    assert ocr_rates("a", "aaa") == {"cer": 2, "wer": 1}
    assert ocr_rates("", "") == {"cer": 0, "wer": 0}
    assert ocr_rates("", "extra") == {"cer": 1, "wer": 1}
    assert ocr_rates("a b", " a\n b ") == {"cer": 0, "wer": 0}
    assert ocr_rates("A", "a")["cer"] == 1


@pytest.mark.parametrize("mutation", ["id", "amount", "cell", "text"])
def test_critical_page_failure(reference, prediction, mutation):
    if mutation == "id":
        prediction["fields"]["id"] = "11112222"
    elif mutation == "amount":
        prediction["fields"]["total"] = "1234.51"
    elif mutation == "cell":
        prediction["tables"][0][0][1] = "1234.51"
    else:
        prediction["transcription"] = "Wrong text"
    assert not evaluate_prediction(reference, prediction)["page_success"]


@pytest.mark.parametrize("mutation", ["missing", "extra", "float", "table_shape", "invalid_json"])
def test_schema_failure(reference, prediction, mutation):
    if mutation == "missing":
        del prediction["fields"]["id"]
    elif mutation == "extra":
        prediction["extra"] = "value"
    elif mutation == "float":
        prediction["fields"]["total"] = 1234.5
    elif mutation == "table_shape":
        prediction["tables"][0].append(["extra", "0.00"])
    else:
        prediction = "{invalid"
    scores = evaluate_prediction(reference, prediction)
    assert not scores["schema_valid"] and not scores["page_success"]
    assert scores["metrics"] is None and scores["errors"]


def test_predefined_page_thresholds(reference, prediction):
    reference["transcription"] = " ".join(["abcdefgh"] * 100)
    words = reference["transcription"].split()
    words[0] = words[1] = "abcdefgx"
    prediction["transcription"] = " ".join(words)
    assert evaluate_prediction(reference, prediction)["page_success"]
    words[2] = "abcdefgx"
    prediction["transcription"] = " ".join(words)
    scores = evaluate_prediction(reference, prediction)
    assert scores["metrics"]["cer"] < 0.01 and scores["metrics"]["wer"] == 0.03
    assert not scores["page_success"]
    words = reference["transcription"].split()
    words[0] = "z" * 40
    prediction["transcription"] = " ".join(words)
    scores = evaluate_prediction(reference, prediction)
    assert scores["metrics"]["cer"] > 0.01 and scores["metrics"]["wer"] == 0.01
    assert not scores["page_success"]
