"""Offline reference output-size estimate; never used by the inference prompt builder."""

import json
from pathlib import Path
from typing import Any


def estimate_budget(dataset: Path, tokenizer: Any) -> dict[str, Any]:
    """Tokenize compact/pretty expected outputs offline; return per-document counts."""
    counts = []
    for path in sorted((dataset / "ground_truth").glob("*.json")):
        truth = json.loads(path.read_text())
        expected = {
            "fields": truth["fields"],
            "tables": [t["rows"] for t in truth["tables"]],
            "transcription": truth["transcription"],
        }
        counts.append(
            {
                "document_id": truth["document_id"],
                "category": truth["category"],
                "compact_tokens": len(
                    tokenizer.encode(json.dumps(expected, separators=(",", ":")))
                ),
                "pretty_tokens": len(tokenizer.encode(json.dumps(expected, indent=2))),
            }
        )
    return {
        "purpose": "offline output-size estimate only; not inference inputs",
        "documents": counts,
        "compact_max": max(c["compact_tokens"] for c in counts),
        "pretty_max": max(c["pretty_tokens"] for c in counts),
    }


if __name__ == "__main__":
    from transformers import AutoTokenizer

    from gpu_my_ocr.benchmark.baseline import REVISION
    from gpu_my_ocr.engines.transformers import MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    print(
        json.dumps(
            estimate_budget(Path("datasets/generated/financial-v1-seed42"), tokenizer), indent=2
        )
    )
