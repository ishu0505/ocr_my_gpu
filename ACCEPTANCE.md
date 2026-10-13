# Acceptance criteria

Check only with concrete verification evidence.

- [x] Fedora NVIDIA driver functional (`nvidia-smi` captured in private/local manifest).
- [x] PyTorch CUDA functional for baseline environment.
- [x] 50 unique synthetic JSON documents (10/category), 150 one-page PDFs across three variants.
- [x] Digital expected text verified; scans image-only; content/page geometry preserved.
- [x] Frozen versioned manifest/hash, deterministic regeneration and overwrite protection verified.
- [x] Independent schema/critical-field/Decimal/table/CER/WER/page evaluator edge cases verified.
- [x] Transformers baseline: 10 underlying documents/30 pages/two repeats; real time,
  strict accuracy, tokens, failures and memory saved.
- [x] vLLM 0.31.0 serves pinned Qwen3-VL BF16 and completes five-category image pilot.
- [x] SGLang 0.5.6 (isolated cuDNN/processor controls) completes real BF16 image requests.
- [x] Controller captures concurrency 1/2/4/8 with defined timing boundaries.
- [x] Baseline model revision/BF16 metadata, failures and hardware manifests persisted.
- [x] Valid fair model/engine comparison with repeated measurements.
- [x] Dashboard OR static reproducible comparison report.
- [x] README includes exact commands, limitations and cited upstream methods.

## Phase 0 evidence — 2026-10-13
`results/preflight/phase0-host-verified.json` (ignored local evidence) records Fedora 44,
driver 615.71.09, RTX 4060 Laptop GPU, SM 8.9, Python 3.12.15 and
PyTorch 2.13.0+cu126. CUDA allocation, FP32 matrix multiplication, synchronization
and CPU-result verification passed; `phase_0_ready=true`. See phase plan for hashes.
Ruff lint/format and all 11 tests passed. No model support or benchmark acceptance claimed.

## Phase 1 dataset/evaluator evidence — 2026-10-13
- `datasets/generated/financial-v1-seed42`: 50 truth JSON and 150 PDFs, each one page.
  Five categories, two layouts each, three variants sharing each underlying ID/truth.
- Frozen SHA256: `e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.
  CLI validation passed with this externally recorded hash. Full regeneration into a
  separate directory produced byte-identical copies of all 202 files.
- All 150 PDFs passed digital expected-text or scan empty-text/image-only checks;
  representative Poppler PNGs inspected for all categories/layouts/scan types.
- 51 tests passed, covering arithmetic, IDs, hashes, pixels, overwrite protection,
  leading/repeated digits, schema, numeric equality, CER/WER and fixed page thresholds.
  Ruff lint/format, wheel/sdist build and archive-content checks passed.
- At dataset-slice completion the Transformers baseline remained unchecked; verified below.


## Phase 1 baseline evidence — 2026-10-13
Primary `results/phase1/baseline-v2-1024`; complete artifact audit
`results/phase1/baseline-v2-1024.audit.json` passed. 60 requests, 30 distinct rendered
pages, 10 paired underlying IDs (two per category/both layouts), two repeats and one
excluded warmup. Frozen dataset and root uv lock unchanged; all repeat pairs identical.
CUDA BF16 official Qwen3-VL-2B revision/SDPA fit, no quantization/offload/OOM/truncation.
Measured 0.0582065 pages/s, p50/p95 17.1116/18.3815 s, cached load 5.3949 s,
NVML peak 4.7901 GiB, torch allocated/reserved 4.2973/4.3535 GiB.
Strict schema 6/60; usable fields 54/600, numeric fields 12/180, tables 0/684,
page success 0/60. This acceptance verifies a reproducible baseline, not good accuracy.
56 tests and lint/format/package/installed-wheel smoke checks passed; independent audit
recomputed every score and verified blind visual inputs/source hashes/frozen artifacts.
Post-run preserved Phase 0 probe passes; GPU memory returned to 12 MiB.
See [findings](docs/PHASE1_BASELINE_FINDINGS.md) for evidence hashes and metric definitions.
Phase 1 complete; Phase 2 and comparative serving acceptance remain unchecked.


## Phase 2 acceptance tracking — 2026-10-13
- [x] Protected original Phase 1 artifacts/measurements and golden dataset by SHA256.
- [x] Separate accuracy diagnostics preserve strict scores and expose raw/view/conditional
  coverage, field/numeric/digit/table/OCR outcomes without silently repaired predictions.
- [x] Controlled three-mode Transformers BF16 inference with separate prompts/configs.
- [x] Actual BF16/CUDA parameter placement and named CUDA profiler kernels, NVML telemetry.
- [x] Streaming HTTP adapter/concurrency metrics/error tests and vLLM image pilot.
- [x] vLLM repeated paired-cohort concurrency 1/2/4/8 evidence or measured unsafe skips.
- [x] SGLang real model/image compatibility and eligible repeated sweeps, or verified failure.
- [x] Final tests/package/GPU/protected-evidence audit and Phase 2 findings/handoff.
Phase 2 verified complete — 2026-10-14. Final independent audit passes: 520 original
files unchanged, 36 diagnostics and 490 serving/structured requests recomputed against
frozen truth; all original input PNGs/prompts and paired IDs verified. Both primary
engines have 240 successful requests; vLLM c8 has one truncation, no OOM/runtime failures.
76 tests, Ruff lint/format, wheel/sdist and installed-wheel smoke pass. Post-run Phase 0
CUDA probe passes and GPU returns to 12 MiB. See [findings](docs/PHASE2_FINDINGS.md).
This closes research acceptance, not financial extraction readiness. Phase 3 not started.
