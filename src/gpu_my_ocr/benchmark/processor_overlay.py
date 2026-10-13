"""Legacy SGLang processor bounds without changing the pinned model/cache contents."""

import hashlib
import json
from pathlib import Path

from gpu_my_ocr.benchmark.baseline import REVISION, save_json
from gpu_my_ocr.benchmark.server_process import MODEL


def create_processor_overlay(snapshot: Path, output: Path) -> Path:
    """Link immutable checkpoint assets; copy only preprocessing config with Phase 1 pixel bounds."""
    output.mkdir(exist_ok=False)
    original = snapshot / "preprocessor_config.json"
    config = json.loads(original.read_text())
    config["size"] = {"shortest_edge": 262144, "longest_edge": 1310720}
    for source in snapshot.iterdir():
        if source.is_file() and source.name != "preprocessor_config.json":
            (output / source.name).symlink_to(source.resolve())
    save_json(output / "preprocessor_config.json", config)
    save_json(
        output / "overlay-provenance.json",
        {
            "model": MODEL,
            "revision": REVISION,
            "source_snapshot": str(snapshot),
            "changes": "only processor size shortest_edge/longest_edge; model/tokenizer/weights linked unchanged",
            "source_processor_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
            "overlay_processor_sha256": hashlib.sha256(
                (output / "preprocessor_config.json").read_bytes()
            ).hexdigest(),
            "linked_assets": {p.name: str(p.resolve()) for p in output.iterdir() if p.is_symlink()},
            "reason": "SGLang 0.5.6 Qwen processor reads but does not forward mm-process-config image options",
        },
    )
    return output
