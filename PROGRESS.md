# Progress journal (append-only)

## Current state
- **Completed:** Phases 0, 1 and 2. Phase 2 verified 2026-10-14 Asia/Singapore.
- **Last verified:** 76 tests, Ruff, packaging/installed-wheel smoke; final evidence audit
  preserves 520 original files and independently recomputes all diagnostic/serving scores.
- **Limitations:** Financial page success remains zero; tiny paired synthetic sample,
  engine numerical differences, one vLLM c8 truncation and SGLang near-device-cap memory.
- **Next action:** Phase 3 planning only when requested; retain BF16 c1 quality control
  and four-client balanced serving settings. No quantization/Phase 3 work started.

## Session log
### 2026-10-13 — repository scaffold
- **Implemented:** Docs, shared agent symlink, Python packaging skeleton, smoke checks, phase plans, learning document stubs.
- **Verified:** Packaging/source-file generation and Python syntax where applicable; no GPU checks or benchmarks.
- **Decisions:** Native Fedora, uv-only Python environments, isolated GPU engine projects, no Docker in MVP.
- **Next:** Validate machine prerequisites, then implement dataset and baseline.

### Handoff template (append new entry; never erase history)
- Date / agent:
- Phase and task:
- Files changed:
- Commands and results:
- Benchmark run IDs (if any):
- Unexpected behavior / blockers:
- Documentation updated:
- Next exact task:

### 2026-10-13 08:25 UTC — Codex, Phase 0 readiness verified
- **Scope/completed:** Read all required repository docs, phase plans, code/tests and git
  status; preserved existing user formatting edits in `engines/base.py`, `test_models.py`
  and the existing root `uv.lock`. Inspected Fedora/CPU/RAM/disk/driver/GPU/tooling before
  installing a GPU framework. Added stdlib preflight JSON evidence and isolated CUDA probe.
- **Changes:** CLI, new `preflight.py`/`cuda_probe.py`, nine preflight regression cases,
  isolated torch pin/index/lock; updated phase plan, master plan, acceptance, ADR-005,
  architecture, setup, runtime README, project README and memory learning notes.
- **Compatibility:** Official PyTorch 2.13.0/cu126 wheel listing, Linux glibc prerequisites
  and NVIDIA backward compatibility reviewed (linked in phase plan/setup).
  Fedora 44/glibc 2.43, driver 615.71.09, SM 8.9; uv 0.13.0 and Python 3.12.15.
- **Commands/results:** `uv add --project environments/transformers --python 3.12
  'torch==2.13.0' --index pytorch=https://download.pytorch.org/whl/cu126`, then explicit
  index + `uv sync --project environments/transformers --python 3.12`: success.
  `uv run --frozen gpu-my-ocr doctor --torch-python environments/transformers/.venv/bin/python
  --output results/preflight/phase0-host-verified.json`: exit 0, ready true, deterministic
  CUDA allocation/FP32 matrix calculation/synchronization/CPU exact check passed.
  Post-exit GPU usage returned to observed 12 MiB.
- **Tests:** `uv run --frozen ruff check .`, `uv run --frozen ruff format --check .`,
  `uv run --frozen pytest`: pass (11 tests); agent symlink script passed.
- **Failures preserved:** Sandbox `nvidia-smi` failed due to hidden device nodes; approved
  host execution worked. First saved probe failed because resolving the venv symlink
  bypassed installed torch; fixed, regression-tested, failed manifest retained.
  Optional NumPy warning retained; no NumPy needed in this smoke calculation.
- **Evidence:** Ignored local verified/failed manifests in `results/preflight/`; SHA256
  hashes and base revision recorded in phase plan. No benchmark run IDs or measurements
  of inference speed/accuracy. No driver changes, model downloads or serving installations.
- **Risks/next:** Phase 1 model/BF16 memory fit and preprocessing compatibility remain
  unverified. Start with deterministic synthetic data + evaluator, then review upstream
  Qwen3-VL baseline requirements before adding Transformers dependencies. Stop here.

### 2026-10-13 09:17 UTC — Codex, Phase 1 synthetic dataset/evaluator slice verified
- **Scope:** Recovered all requested context and dirty worktree; retained completed Phase 0
  code/docs/evidence and prior user formatting edits. Updated planning before implementation.
  No model inference, GPU dependency changes or Phase 2 serving work.
- **Implemented:** Dedicated `datasets/synthetic_data` package: five content templates,
  shared Decimal source construction, two layouts, ReportLab digital renderer, PyMuPDF
  image-only scans, seeded Pillow blur/noise, manifest/hash/metadata/PDF validation.
  JSON is saved/reloaded before rendering. Separate `benchmark/accuracy` evaluator with
  exact IDs, normalized Decimal values/literal diagnostics, strict schema, ordered cells,
  CER/WER and versioned fixed page thresholds. New CLI generate/validate/evaluate commands.
- **Dependencies:** `uv add` installed CPU document/schema libraries; after validation
  pinned ReportLab 5.0.1, PyMuPDF 1.28.2, Pillow 12.3.0, jsonschema 4.26.0. Root lock
  updated intentionally; isolated Phase 0 torch project/lock unchanged. Official API
  documentation reviewed and linked in phase plan/learning notes.
- **Generation:** Initial 5-document/15-PDF preview validated/visually inspected; second
  10-document/30-PDF preview covered both layouts. Full seed-42 generation under
  `datasets/generated/financial-v1-seed42`: 50 JSON truths, 150 one-page PDFs, 49M disk.
  Second full generation in a new directory reproduced all 202 files byte for byte.
- **Frozen evidence:** manifest SHA256
  `e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.
  CLI validation passed with externally pinned hash; all digital PDFs match expected text,
  scans have no extractable text and matching page geometry. Poppler final representative
  PNGs inspected across all categories/layouts/scan types; artifacts stay local/ignored.
- **Tests/checks:** `uv run pytest`: 51 passed; Ruff lint and format-check passed (58 files).
  Tests cover deterministic source/full rendering, unique/stable IDs, financial arithmetic,
  shared truth, scans/pixels, hash tampering, overwrite protection, exact digits, schema,
  Decimal normalization, CER/WER and fixed success thresholds. Source size constraints pass.
  `uv build --out-dir artifacts/package-check`: wheel/sdist success; archive inspection
  confirms packages included and generated dataset excluded. Isolated installed-wheel
  CLI smoke, learning-document links and agent-instruction symlink passed.
  `git diff --check` passed.
- **Failures/resolution:** Lint issues fixed. Pixel regression first compared MuPDF RGB
  PNG bytes with grayscale source bytes; matching color-mode comparison then confirmed
  exact image samples. Noise's small modulo bias documented. No invalid result substituted.
- **Documentation:** Updated PLAN, Phase 1 plan, architecture/protocol, acceptance, ADR-006,
  README, accuracy learning notes and new synthetic-design learning notes with concrete
  JSON/arithmetic and representative PDFs. Preserved all Phase 0 journal/history/hashes.
- **Unresolved/next:** Phase 1 is partial: Transformers baseline remains unimplemented,
  model/runtime/prompt/image-policy compatibility unverified. The existing 512-token cap
  may truncate the expanded extraction contract; validate a fixed budget before runs.
  Next is a separately scoped upstream compatibility review and controlled >=10-page
  Transformers baseline with real timing/memory/accuracy. No inference run IDs or claimed
  model results. Stop within Phase 1; do not begin Phase 2.


### 2026-10-13 13:22 UTC — Codex, Phase 1 BF16 baseline verified complete
- **Scope/completed:** Recovered required docs/code/dirty status, preserved prior user
  work and Phase 0 artifacts/pins; planned before implementation. Official Qwen/model
  integration and Torchvision CUDA wheel compatibility reviewed. No driver modifications,
  quantization, CPU offload, serving installation or Phase 2 work.
- **Implementation:** value-free versioned prompt, uniform 150 DPI visual rasterization,
  CUDA-only BF16 official processor/model adapter, sequential warmup/repeated runner,
  synchronized generation/adapter/end-to-end timing, output tokens/truncation, NVML
  sampler and torch peaks, raw prediction/scoring/config/source/lock/environment evidence;
  aggregate reporter, offline token estimator and independent saved-result audit.
- **Runtime:** Qwen revision 89644892e4d85e24eaac8bacfd4f463576704203,
  Transformers 4.57.6, torch 2.13.0+cu126 unchanged, torchvision 0.28.0+cu126,
  Accelerate 1.15.0. Dependencies added/pinned through isolated uv project and lock.
  Root uv lock unchanged (6505b88081090e0a6d4a13d514644feeeef302757ece4e64a721fc33cd596412).
- **Budget/pilots:** expected output max 518 compact/609 pretty tokens across all 50
  truths, tokenized offline only. v1 pilot cap 2048: 546..626 actual tokens, 0/5 schema;
  v2 pilot cap 1024: 532..576, 1/5 schema; no OOM/truncation. Freeze v2 and 1024 for primary.
- **Primary run:** results/phase1/baseline-v2-1024; 10 underlying IDs (001/010 per category,
  both layouts), 30 paired variants × two repeats=60 requests; separate one warmup;
  60 runtime successes, 0 failures/OOM/truncations; 445..604 tokens. All 30 raw repeat
  pairs identical. BF16 parameters and CUDA input assertions passed, checkpoint context
  limit 262144, actual max context 2368, effective image grid [1,80,62]/992×1280.
- **Measured:** usable exact-field 54/600=9%, numeric 12/180=6.67%, table 0/684,
  JSON 38/60, schema 6/60, page success 0/60; valid-only CER/WER
  0.00278358/0.01041667. Wall 1030.8133 s, 0.0582065 pages/s, latency
  mean/p50/p95 17.1666/17.1116/18.3815 s, cached load 5.3949 s separately.
  NVML sampled peak 4.7901 GiB; torch allocated/reserved 4.2973/4.3535 GiB.
- **Validation:** 56 tests, Ruff lint/format, wheel/sdist build, isolated installed-wheel
  baseline imports and controller CLI smoke passed. Independent audit recomputes all
  strict scores, prompt invariance to values, raw/RGB/CUDA diagnostics, paired IDs,
  unchanged source hashes and all 202 frozen dataset artifacts. Post-run Phase 0 probe
  still passes; GPU returned to 12 MiB after model process exit.
- **Failures:** missing ReportLab import before runner startup; then AutoVideoProcessor
  required Torchvision despite image-only inputs. Saved setup-failures.json and failed
  pilot-2048 artifacts. Schema failures/Markdown fences/headers/empty table cells/digit
  corruption retained without repair. Initial lint findings fixed before primary run.
- **Evidence:** dataset hash e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976;
  audit SHA256 28270042e6860a2d6473fa6e2604d6325336f9efc7fb355044ebac444a8f3b4b;
  aggregate SHA256 39b698a2135ba25ab62767eb69c82fa56ed7c9ba0b8cf0613439a791f418f2d3.
  Local raw artifacts ignored; tracked findings retain human-readable results/hashes.
- **Docs:** PLAN, phase plan, ACCEPTANCE, DECISIONS ADR-007, PROGRESS, protocol,
  architecture, runtime README, reference YAML, project README and learning docs updated;
  new docs/PHASE1_BASELINE_FINDINGS.md. AGENTS unchanged.
- **Closure/risks/next:** Phase 1 acceptance verified complete despite poor extraction
  quality. Preliminary synthetic cohort only; valid-only OCR rates are not cohort rates;
  no power/thermal time series or serving-concurrency support measured. Next is Phase 2
  engine compatibility at concurrency 1 with same BF16/input/prompt/cap controls. Stop here.


### 2026-10-13 13:59 UTC — Codex, Phase 2 diagnostic subphase verified
- **Context/plan:** Read mandatory docs/code/status; dirty Phase 0/1 work preserved.
  Updated phase plan before coding; saved SHA256 for 520 protected Phase 1/data/runtime
  artifacts. No changes to original scorer/prompt/adapter/Transformers environment or truth.
- **Implemented:** Non-repairing labelled parse/wrapper/schema/coverage/exact financial/
  digit/table/text diagnostics; separate value-free full/OCR/minimal prompt contracts;
  controlled BF16 runner, NVML utilization/memory and parameter placement, small CUDA
  profiler request. New streaming HTTP adapter, bounded concurrency metrics and owned
  server lifecycle/workload runner; current API/source/CLI checks and failure handling.
- **Diagnostics:** Preserved Phase 1 strict scores recomputed identically. Diagnostic object
  view: all 600 fields present, 532 correct, all 180 monetary/numeric fields correct,
  60 ID/reference plus eight policy_number mismatches. 22 fenced raw JSON failures;
  strict/view schemas 6/10 of 60. No scored header removal or structure/value repair.
- **Controlled inference:** transformers-diagnostics-v1, six representative visual pages,
  two repeats/three modes; full-v2 2/12 schema, numeric 36/36, diagnostic fields 102/116;
  plain OCR CER/WER 0.001606/0.003924; minimal 12/12 reduced schemas, fields 36/46,
  numeric 22/22. No failures/truncations. Mode-specific rates are not a shared leaderboard.
- **GPU:** 625 BF16/cuda:0 parameter tensors, SM 8.9/no offload; profiler 54,126 CUDA
  events including BF16 compute kernels. Separate 32-token profiled output explicitly
  truncated/excluded. NVML samples retained. Model process exited before serving launch.
- **Tests:** Six diagnostic and six HTTP/SSE/concurrency/error tests passed, Ruff passes.
- **Serving:** uv-only isolated vLLM 0.31.0/SGLang 0.5.21 installs and locks; baseline
  environment/root lock unchanged. Both CLI import/help checks passed. vLLM image pilot
  10/10 runtime success, 0 OOM/truncation, 0.06768 pages/s, mean 14.735 s, schema 2/10;
  primary sweep running. No SGLang GPU measurements claimed yet.
- **Docs/next:** Phase plan, protocol, architecture, ADR-008/009, provisional findings,
  PLAN/current progress updated. Finish serving experiments/tests/evidence before closure;
  no quantization/Phase 3 work.


### 2026-10-13 15:54 UTC — Codex, serving compatibility and vLLM sweep evidence
- **Completed:** vLLM 0.31.0 paired cohort c1/2/4/8, 60 requests/level, fixed original
  BF16/revision/full-v2/1024/pixels. All 240 API successes; one c8 length truncation,
  no errors/OOM. Throughput 0.06931/0.11993/0.18301/0.20371 pages/s;
  mean 14.4148/16.6525/21.6276/37.2897 s, peaks 6.37..6.97 GiB.
  Correct pages/s/table/page success zero. c8 queue mean 7.393 s, 18 KV preemptions.
- **Separate constraint experiment:** matched five categories × two repeats, c1;
  10/10 strict schemas, fields 90/100, numeric 30/30, tables/page success zero.
  Unconstrained matched schemas 2/10. No repaired predictions or pooled mode leaderboard.
- **Preservation:** Interim independent audit verifies 520 protected original files,
  36 diagnostics and 250 vLLM requests; same PNG bytes/full-v2 prompts/scores.
  Matching source snapshots stored for Phase 1/vLLM and failed serving candidates.
- **SGLang 0.5.21 failures:** missing CUDA_HOME resolved using uv's CUDA 13 toolkit;
  loaded BF16 but first image activation JIT fails on missing host cc1plus.
  Source/config/telemetry/logs and original uv project/lock archived; no OS compiler/driver install.
- **Reviewed fallback:** 0.5.6 uses official prebuilt activation kernels, torch 2.9.1,
  Transformers 4.57.1. CUDA namespace transition needed exact locked reinstall;
  loader needs installed cuda_runtime CUDA_HOME. Default cuDNN 9.10 triggers upstream
  BF16 Conv3d safety guard; official 9.16.0.29 uv override pinned, CUDA exact-output
  Conv3d probe passes. Guard was not suppressed; patched runtime tested on GPU.
- **Pixel policy:** Legacy Qwen path ignores image pixel CLI kwargs; first native-resolution
  pilot (10 successes) used 2557 rather than 1717 input tokens and is excluded from fair
  comparison. Local preprocessor size-only overlay links unchanged pinned checkpoint assets.
  Corrected pilot: all ten counts match Phase 1, 10 successes/no failures/truncation,
  0.08885 pages/s, mean 11.232 s, schema 2/10/page success zero; primary sweep running.
- **Code/tests:** Streaming/per-request metrics, safe stop/skip logic, versioned launcher,
  CUDA path discovery, overlay/provenance and guard tests. 75 tests pass, Ruff passes,
  wheel/sdist builds. Optional DLPack/compilation/shutdown warnings retained.
- **Next:** Finish SGLang eligible sweeps or preserve verified unsafe failure, audit final
  evidence/resource release, update findings/acceptance. Phase 2 not yet complete; no Phase 3.


### 2026-10-13 16:26 UTC / 2026-10-14 Asia/Singapore — Phase 2 verified closure
- **Implemented:** Modular diagnostics, A/B/C prompts, GPU/NVML/profiler evidence,
  streaming serving adapter, bounded concurrency, versioned launchers, processor overlay,
  strict per-request scoring and independent preservation/input/evidence audit.
- **Measured:** SGLang 0.5.6 with explicit cuDNN 9.16 override and original-pixel overlay
  completes 240 requests across clients 1/2/4/8: 0.08525/0.13123/0.20226/0.26550 pages/s,
  mean 11.718/15.214/19.601/29.440 s, peak 7.337..7.454 GiB. No failures/OOM/truncations.
  Earlier vLLM completes 240, peak 6.366..6.974 GiB; one c8 length truncation. Both
  table/page success and correct pages/s zero. Raw prediction/metrics/log archives retained.
- **Verified:** Final audit: 520 protected files, frozen manifest, 36 controlled modes,
  250 vLLM plus 240 SGLang records, exact PNG/prompt preservation and raw strict scores.
  76 pytest tests (18.03 s), Ruff check/83 formatted files, wheel/sdist/installed-wheel
  smoke. Original post-run Phase 0 CUDA probe passes; driver unchanged, GPU 12 MiB.
- **Findings:** Low Phase 1 usable scores arise from contract/table compliance and
  identifier fidelity; independent diagnostics find 88.67% fields/100% numeric, without
  replacing original 9%/6.67%. Constraints fix schema, not table/digit facts. New report
  records successful/failed configurations, native queues, timing confounds and small-N limits.
- **Next:** Phase 2 complete. Recommended future BF16 c1 control and c4 balanced serving;
  c8 increases queueing/memory risk. Phase 3 not started; no quantization/OS changes.

- **Final preservation check:** Original 32 Phase 1 source hashes unchanged; all 42 vLLM
  and 44 SGLang historical source snapshots match configs. Supplementary audit saved to
  `results/phase2-source-preservation.audit.json`; installed-wheel imports and CLI help pass.
