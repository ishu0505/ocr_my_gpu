"""Prepare consistent visual workloads and persist raw streaming serving results."""

import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from gpu_my_ocr.benchmark.accuracy.evaluate import evaluate_prediction
from gpu_my_ocr.benchmark.baseline import save_json, selected_records
from gpu_my_ocr.benchmark.extraction import extraction_prompt, render_page
from gpu_my_ocr.models import InferenceRequest


def prepare_pages(dataset: Path, output: Path, pilot: bool) -> list[dict]:
    """Pre-render selected paired PDFs at Phase 1 resolution and construct unchanged prompts."""
    manifest = json.loads((dataset / "manifest.json").read_text())
    pages = []
    for record in selected_records(manifest, pilot):
        truth = json.loads((dataset / record["ground_truth_path"]).read_text())
        page_id = f"{record['document_id']}--{record['variant']}"
        image = output / f"{page_id}.png"
        render_page(dataset / record["pdf_path"], image)
        prompt = extraction_prompt(truth)
        save_json(
            output / f"{page_id}.prompt.json",
            {"version": "financial-extraction-v2", "text": prompt},
        )
        pages.append(
            {"record": record, "truth": truth, "image": image, "prompt": prompt, "page_id": page_id}
        )
    return pages


async def serving_request(engine: object, page: dict, repeat: int) -> dict:
    """Infer and strictly score an already-rendered page; retain partial text on failure."""
    page_id = f"{page['page_id']}--r{repeat}"
    result = {"page_id": page_id, "record": page["record"], "repeat": repeat}
    started = time.perf_counter()
    try:
        response = await engine.infer(
            InferenceRequest(page_id, page["image"], page["prompt"], 1024)
        )
        result.update(asdict(response), status="ok")
    except Exception as error:  # noqa: BLE001 - request failures must stay in benchmark evidence
        result.update(
            status="oom" if "out of memory" in str(error).lower() else "failed",
            text=getattr(error, "partial_text", ""),
            output_tokens=None,
            error=traceback.format_exc(),
        )
    result["end_to_end_ms"] = (time.perf_counter() - started) * 1000
    result["score"] = evaluate_prediction(page["truth"], result["text"])
    return result


def persist_requests(output: Path, results: list[dict]) -> None:
    """Write original text and full request records into a new measurement directory."""
    for result in results:
        page_id = result["page_id"]
        (output / f"{page_id}.txt").write_text(result["text"])
        save_json(output / f"{page_id}.json", result)
