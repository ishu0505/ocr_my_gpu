# gpu_my_ocr

Hardware-aware OCR/VLM inference benchmarks on a single RTX 4060 (8GB) running Fedora Linux.

## Scope

Compare Transformers, vLLM, and SGLang serving of Qwen3-VL-2B, Qwen2.5-VL-3B, PaddleOCR-VL, with optional olmOCR-2-7B. Measure quality and throughput under controlled, reproducible workloads. Unsupported engine/model/precision pairs are explicitly recorded, not assumed to work.

## Start

1. Inspect `AGENTS.md`, `PLAN.md`, `PROGRESS.md`, `docs/SETUP_FEDORA.md`.
2. Verify GPU driver with `nvidia-smi` before installing GPU frameworks.
3. `uv sync` then `uv run gpu-my-ocr doctor` and `uv run pytest`.
4. Use `configs/models.yaml` as a target list; don't assume all configs are already implemented.
5. Work phase-by-phase under `plans/`. Update progress after tests.

Phase 0 is verified: the `doctor` CLI captures local host/tool/GPU evidence and can
validate the isolated PyTorch runtime. See [Fedora setup](docs/SETUP_FEDORA.md) for exact
locked-environment commands and compatibility sources. Phase 1 is now verified complete:
a preliminary CUDA BF16 baseline ran on 10 underlying documents/30 pages/two repeats.
It fits memory but extraction quality is poor (10% schema validity, 0% page success).
Results and model downloads are excluded from git; see the findings below.

## Synthetic dataset and scoring

Phase 1's dataset/evaluator slice is verified: 50 unique fictitious JSON documents,
10 in each of five categories, with digital/high-quality scan/degraded scan PDFs
(150 one-page PDFs). Ground truth is saved before rendering. The full seed-42 dataset
is byte-reproducible in the pinned environment; no external OCR/API is used.

```bash
uv sync --frozen --python 3.12
uv run --frozen gpu-my-ocr dataset-generate --output datasets/generated/new-preview --per-category 1
uv run --frozen gpu-my-ocr dataset-generate --output datasets/generated/financial-v1-seed42 --seed 42
uv run --frozen gpu-my-ocr dataset-validate datasets/generated/financial-v1-seed42 \
  --expected-manifest-sha256 e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976
```

Output directories must be new; existing datasets are never overwritten. See
[dataset design](learning_docs/06_synthetic_dataset_design.md) for representative PDFs,
scan parameters and limitations, and [accuracy notes](learning_docs/02_ocr_quality_and_ground_truth.md)
for exact-field/Decimal/table/schema/CER/WER scoring and the `evaluate` command.
Phase 1 is verified complete: the preliminary Transformers baseline and strict
results are documented in [baseline findings](docs/PHASE1_BASELINE_FINDINGS.md).

## Run the Phase 1 Transformers baseline
Use the isolated GPU interpreter from the repository root:
```bash
PYTHONPATH=src uv run --frozen --project environments/transformers \
  python -m gpu_my_ocr.benchmark.baseline --max-tokens 1024 --repeats 2 \
  --output results/phase1/new-baseline
uv run --frozen python -m gpu_my_ocr.benchmark.baseline_report results/phase1/new-baseline
uv run --frozen python scripts/verify_phase1_baseline.py results/phase1/new-baseline
```
The output directory must be new. This uses 10 underlying documents, all three rendered
variants, one separate warmup, two repeats, CUDA BF16 SDPA and the pinned checkpoint.
See [baseline findings](docs/PHASE1_BASELINE_FINDINGS.md) and
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md) for measurements, strict schema
failures, environment versions and timing/memory definitions. Never run two GPU engines
at once. This preliminary synthetic experiment does not establish real-document accuracy.


## Phase 2 verified serving and accuracy diagnostics
Phase 2 is complete; Phase 3 has not started. Both BF16 engines were measured at
1/2/4/8 clients on the original frozen paired workload. Financial page success remains
zero; JSON/table compliance and identifier fidelity are the main observed bottlenecks.
All Phase 1 artifacts and measurements are preserved. See [Phase 2 findings](docs/PHASE2_FINDINGS.md)
for full results, official sources, runtime incompatibilities and timing limitations.
[Isolated vLLM commands](environments/vllm/README.md) and
[isolated SGLang commands](environments/sglang/README.md) pin the verified runtimes.

```bash
PYTHONPATH=src uv run --frozen --project environments/transformers \
  python -m gpu_my_ocr.benchmark.diagnostic_run --output results/phase2/new-diagnostics
uv run --frozen python -m gpu_my_ocr.benchmark.serving_report results/phase2/vllm-primary-v1
uv run --frozen python -m gpu_my_ocr.benchmark.serving_report results/phase2/sglang-0.5.6-primary-v1
uv run --frozen python scripts/verify_phase2_evidence.py
```
Use new output directories for new experiments and only one GPU engine at a time.
Evidence remains local under ignored `results/`; retain it with environment locks/source
snapshots. Recommended future controls: BF16 one client for quality, four for balanced
serving. Eight clients increases queueing and approaches the device/cache limits.
