# Architecture

Target architecture for later phases: controller `gpu_my_ocr` uses YAML configuration to select a model/engine/workload combination. A runner validates configuration, launches or connects to exactly one engine, warms it, then dispatches requests with bounded `asyncio` concurrency. Adapters standardize `infer(page_path, prompt, settings)` results. The evaluator compares outputs with separately stored synthetic truth. The profiler samples NVML while a run is active; it does not claim to directly measure achieved FLOPs. Storage writes request-level Parquet and run manifests, with DuckDB for aggregate queries. Dashboard consumes stored results only.

## Logical modules
- `datasets/synthetic_data/`: five category templates separated from shared Decimal
  construction, ReportLab rendering, PyMuPDF rasterization/Pillow degradation and validation.
  Canonical structured JSON is written before PDFs; local versioned manifest/hash freezes
  50 underlying IDs and their 150 one-page variants.
- `benchmark/accuracy/`: independent stored-truth evaluator with strict prediction JSON
  schema, string/numeric exact fields, ordered table cells, CER/WER and fixed page criteria.
  No generator imports, external OCR services or model-generated reference labels.
- `preflight.py`: stdlib host/tool checks, explicit statuses and new JSON manifests (Phase 0).
- `cuda_probe.py`: standalone script run by the explicitly supplied engine interpreter;
  small verified CUDA calculation, with no model loading. Process exit releases its context.
- `config.py`: config parsing/validation (implemented in later phase)
- `models.py`: typed domain records
- `engines/`: Transformers and local HTTP engine implementations
- `benchmark/`: load generation, timing, accuracy, GPU sampling
- `datasets/`: synthetic generation and benchmark IO
- `storage/`: results and manifests
- `reporting/`: aggregate queries and charts

**Dependency direction:** CLI -> runner -> adapters/benchmark/storage. Adapters should never import reporting/dashboard. Evaluators should never import CUDA/engine code.

Current implemented CLI: `doctor --output PATH --torch-python PATH`. The interpreter
argument is optional; omitting it leaves CUDA unverified. JSON includes command outputs,
errors, timestamps and a separate `phase_0_ready` flag. Failed checks exit nonzero after
saving evidence. Existing manifest files cannot be overwritten. Results remain local/ignored.

Phase 1 also implements `dataset-generate --output NEW_DIRECTORY --seed 42
--per-category 10`, `dataset-validate DIRECTORY [--expected-manifest-sha256 HASH]`
and `evaluate --ground-truth PATH --prediction PATH [--output NEW_FILE]`.
Dataset content is generated with independently seeded category templates and Decimal;
JSON is persisted/reloaded before ReportLab rendering. Shared scans rasterize at 150 DPI
and deterministically degrade pixels with Pillow. Validation checks hashes, group identity,
metadata/transforms, geometry and text-layer contracts. A detached manifest hash freezes
the dataset; future inference records must retain that hash separately.

The evaluator lives in `benchmark/accuracy` and imports no synthetic generator or CUDA
code. It accepts prediction fields, ordered tables and transcription against stored truth;
strict schema failures produce failed pages with unavailable metrics. Phase 1 baseline now adds a sequential isolated-runtime runner and adapter.

## Design constraints
- Internal API uses a small `InferenceRequest` / `InferenceResult` pair; model-native preprocessing may require adapter-specific logic.
- Official PaddleOCR pipeline may not be an OpenAI endpoint; use dedicated adapter.
- vLLM/SGLang engine projects are separate uv environments to avoid conflicting CUDA/PyTorch dependency graphs.
- Quantization and VLM support matrix is measured/documented; no unverified claims.
- No unnecessary Docker, REST service, worker queue, cache, orchestration cluster or cloud database.


## Phase 1 baseline implementation
`benchmark/extraction.py` constructs value-free versioned instructions and renders every
PDF variant through the same visual path. `engines/transformers.py` implements the existing
async engine Protocol (sequential blocking CUDA generation intentionally; no concurrent
GPU requests). It loads once, asserts BF16/CUDA parameters, uses the official processor,
and records synchronized generation and request timings, token/grid/truncation diagnostics,
PyTorch memory. `benchmark/baseline.py` validates the dataset, records configuration/source
hashes, warms up, measures paired repeated requests, strictly scores raw text and samples
NVML in one thread. Failures persist as explicit records; no server or framework introduced.
`benchmark/baseline_report.py` aggregates all-page and valid-only metrics, categories,
variants and repeats. `benchmark/token_budget.py` is an offline expected-length estimator;
its reference values never flow into model inference. Results remain local/ignored.
The adapter inference method and runner setup exceed the 50-line target to keep the timed
CUDA lifecycle and run cleanup visibly contiguous; modules remain below 250 lines.

## Phase 2 additions
`accuracy/diagnostics.py` adds labelled object/field/table/transcription diagnostic views
alongside immutable strict scores. `diagnostic_prompts.py` and `diagnostic_run.py` isolate
A/B/C contracts and preserve original adapter/prompt paths. `gpu_telemetry.py` samples
NVML and verifies actual parameter placement; a separate profiler records CUDA events.
`engines/http.py` implements the existing engine Protocol via streaming loopback image
requests and immutable per-request results, including missing usage/TTFT/truncation and
partial transport failures. It never shares mutable last-request details across clients.
`concurrency.py` bounds active client requests and aggregates wall-time metrics.
`server_process.py` owns/cleans up one local engine process group; `serving_requests.py`
prepares exact visual inputs and strictly scores raw outputs; `serving_run.py` gates
concurrency sweeps on image smoke and records per-level results/failures/metrics.
No controller GPU imports, HTTP service of our own, framework or database was introduced.
Server launch argument lists and contiguous lifecycle/cleanup functions can exceed 50
lines to keep the experimental settings and failure handling reviewable; modules stay
at or below 250 lines. Future refactors must preserve saved experimental evidence.

Legacy SGLang 0.5.6 requires a local processor-config overlay because its image pixel
CLI options are not applied on the Qwen path. `processor_overlay.py` copies only
preprocessor_config.json size bounds, links other immutable checkpoint assets and
records provenance. No model weights/cache or installed Python package is modified.
The launcher selects reviewed version-specific flags and installed CUDA runtime/compiler
roots, recording controlled environment overrides. Code snapshots match historical run
hashes even when compatibility helpers evolve; original Phase 1 source files stay unchanged.
