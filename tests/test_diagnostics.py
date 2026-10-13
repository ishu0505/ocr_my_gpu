"""Diagnostics remain independent from strict baseline correctness."""

import copy
import json

from gpu_my_ocr.benchmark.accuracy.diagnostics import diagnose, parse_view
from gpu_my_ocr.benchmark.diagnostic_prompts import diagnostic_prompt
from gpu_my_ocr.benchmark.gpu_telemetry import summarize_samples


def truth():
    return {
        "category": "portfolio",
        "document_id": "test",
        "fields": {"account_id": "00001111", "total": "1234.50", "reference_digits": "11112222"},
        "field_modes": {"account_id": "exact", "total": "numeric", "reference_digits": "exact"},
        "tables": [
            {
                "columns": ["Name", "Amount"],
                "rows": [["Example", "12.50"]],
                "modes": ["exact", "numeric"],
            }
        ],
        "transcription": "Example 12.50",
    }


def prediction():
    return {
        "fields": truth()["fields"],
        "tables": [[["Example", "12.50"]]],
        "transcription": truth()["transcription"],
    }


def test_wrappers_are_labelled_never_repaired_for_strict_scores():
    text = "```json\n" + json.dumps(prediction()) + "\n```"
    scores = diagnose(truth(), text)
    assert not scores["raw_json_valid"]
    assert scores["markdown_fence"] and scores["extra_text"]
    assert scores["view"] == "embedded_object_diagnostic_only"
    assert not scores["strict"]["schema_valid"]
    assert scores["field_diagnostics"]["correct"] == 3
    assert scores["view_schema"]["page_success"]


def test_schema_failure_does_not_hide_transcription_or_fields():
    pred = prediction()
    pred["tables"][0].insert(0, ["Name", "Amount"])
    scores = diagnose(truth(), json.dumps(pred))
    assert scores["raw_json_valid"]
    assert not scores["strict"]["schema_valid"]
    assert scores["field_diagnostics"]["correct"] == 3
    assert scores["transcription_rates"] == {"cer": 0, "wer": 0}
    assert scores["table_diagnostics"]["correct"] == 0


def test_missing_wrong_numeric_digits_and_equivalent_amounts():
    pred = prediction()
    del pred["fields"]["account_id"]
    pred["fields"]["total"] = "USD 1,234.500"
    pred["fields"]["reference_digits"] = "1112222"
    scores = diagnose(truth(), json.dumps(pred))["field_diagnostics"]
    assert scores["missing"] == ["account_id"]
    assert scores["numeric_correct"] == 1
    assert scores["incorrect_digits"] == ["reference_digits"]
    pred["fields"]["total"] = "1234.51"
    assert diagnose(truth(), json.dumps(pred))["field_diagnostics"]["incorrect_numeric"] == [
        "total"
    ]


def test_no_missing_values_are_invented():
    assert parse_view('{"fields":')["prediction"] is None
    assert diagnose(truth(), '{"fields":')["field_diagnostics"]["correct"] == 0
    assert parse_view("a preface " + json.dumps(prediction()) + " a suffix")["extra_text"]


def test_prompts_are_value_free_and_full_mode_is_identical():
    from gpu_my_ocr.benchmark.extraction import extraction_prompt

    original = truth()
    changed = copy.deepcopy(original)
    changed["fields"] = dict.fromkeys(original["fields"], "REDACTED")
    changed["tables"][0]["rows"] = [["REDACTED", "REDACTED"]]
    changed["transcription"] = "REDACTED"
    for mode in ("full", "ocr", "minimal"):
        assert diagnostic_prompt(original, mode) == diagnostic_prompt(changed, mode)
    assert diagnostic_prompt(original, "full") == extraction_prompt(original)


def test_gpu_sample_summary_distinguishes_absence():
    assert summarize_samples([])["peak_device_bytes"] is None
    summary = summarize_samples(
        [
            {"used_bytes": 100, "gpu_utilization_percent": 10},
            {"used_bytes": 200, "gpu_utilization_percent": 50},
        ]
    )
    assert summary["peak_device_bytes"] == 200
    assert summary["mean_gpu_utilization_percent"] == 30
