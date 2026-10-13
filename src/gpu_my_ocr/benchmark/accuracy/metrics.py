"""Exact financial comparison and conventional reference-normalized OCR edit rates."""

import re
from collections.abc import Sequence
from decimal import Decimal, InvalidOperation

NUMBER = re.compile(r"[+-]?(?:[0-9]+|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?\Z")


def numeric_value(value: str) -> Decimal:
    """Parse exact decimal text, allowing USD/$, grouped commas and outer whitespace.

    For example 'USD 1,200.00' equals '1200'; identifiers must never use this parser.
    Exponents, percentages, nonfinite values, floats and malformed grouping are rejected.
    """
    if not isinstance(value, str):
        raise TypeError("Financial values must be JSON strings")
    text = value.strip()
    if text.startswith("USD "):
        text = text[4:]
    elif text.startswith("$"):
        text = text[1:]
    if not NUMBER.fullmatch(text):
        raise ValueError(f"Invalid decimal string: {value}")
    return Decimal(text.replace(",", ""))


def exact_match(expected: str, predicted: str, mode: str = "exact") -> bool:
    """Compare literal strings or normalized Decimal numbers without float tolerances."""
    if mode == "exact":
        return isinstance(predicted, str) and expected == predicted
    if mode != "numeric":
        raise ValueError(f"Unknown comparison mode: {mode}")
    reference = numeric_value(expected)
    try:
        return reference == numeric_value(predicted)
    except (TypeError, ValueError, InvalidOperation):
        return False


def edit_distance(reference: Sequence, prediction: Sequence) -> int:
    """Return Levenshtein distance using one row of memory for characters or words."""
    previous = list(range(len(prediction) + 1))
    for row, expected in enumerate(reference, start=1):
        current = [row]
        for column, actual in enumerate(prediction, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected != actual),
                )
            )
        previous = current
    return previous[-1]


def ocr_rates(reference: str, prediction: str) -> dict[str, float]:
    """Return CER/WER after whitespace collapse only; empty mismatches score 1.

    Case, punctuation, leading zeros and digit repetitions are retained. Insertions
    can produce rates above 1; values are never capped.
    """
    expected, actual = " ".join(reference.split()), " ".join(prediction.split())
    words, predicted_words = expected.split(), actual.split()
    return {
        "cer": edit_distance(expected, actual) / len(expected) if expected else float(bool(actual)),
        "wer": edit_distance(words, predicted_words) / len(words) if words else float(bool(actual)),
    }
