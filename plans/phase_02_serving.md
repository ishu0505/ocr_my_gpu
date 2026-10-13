# Phase 2: Accuracy diagnostics, serving and concurrency

**Status:** Verified complete — 2026-10-14 (Asia/Singapore)

## Goal and protected evidence
Preserve Phase 1 measurements/artifacts/prompt/evaluator and frozen golden dataset.
First separate visual recognition from JSON/schema compliance; then validate pinned
Qwen3-VL-2B BF16 in isolated vLLM/SGLang projects, benchmark eligible engines with
client concurrency 1/2/4/8. No quantization or Phase 3 work, no driver changes.

## Tasks in order
- [x] Record protected Phase 1/data/environment hashes; review raw predictions and scorer.
- [x] Add separate diagnostics: parseability, wrappers/extra text, strict schema, missing
  fields/coverage, conditional and present-field exact correctness, numeric/digit/table
  errors, comparable raw transcription CER/WER. Explicit diagnostic-only wrapper views
  never replace raw predictions or original strict scores.
- [x] Compare three versioned prompt modes on five category representatives plus one
  degraded paired page: original v2, plain OCR, 3–5 critical fields; fixed image/model/
  decoding/token cap, separate warmup and two repeats per mode. Preserve all outputs.
- [x] Capture GPU identity/SM, parameter dtype/device/no-offload, sampled NVML memory/
  utilization and small torch.profiler representative CUDA-kernel evidence.
- [x] Review official model/hardware/version/API compatibility; install isolated uv engine
  environments; try vLLM then SGLang, one GPU engine at a time, retain startup failures.
- [x] Concurrency-1 smoke, then same 30-page paired cohort/two repeats as Phase 1.
- [x] Eligible engine sweeps 1/2/4/8 clients with fixed BF16/model/prompt v2/1024 cap/
  processor policy, warmups/repeats, bounded requests/timeouts, streaming TTFT and token
  counts where supported, raw scores/errors, latency/throughput/memory/utilization.
- [x] Stop unsafe/OOM configurations gracefully; separate client concurrency from engine
  batch capacity; record supported metrics and unavailable queue/batch measurements.
- [x] Optional schema-constrained decoding is a separate versioned comparison, not mixed
  into unconstrained engine leaderboard; only use installed official API.
- [x] Tests, Ruff, package smoke, GPU integration, protected-evidence audit and findings;
  update master/progress/decisions/acceptance/protocol/architecture/learning notes.

## Acceptance
Diagnostics and A/B/C inference evidence collected without changing Phase 1 scoring;
GPU inference proven or profiler limitation explicit. Each engine has successful image
inference/measurements or verified documented incompatibility with retained error logs.
For each compatible engine, concurrency 1/2/4 and 8 if safe have repeated measured
results or explicit measured OOM/unsupported skip reason. New adapters/metrics/telemetry/
error handling tested. Frozen data and protected Phase 1 artifacts unchanged. Findings
report actual sample sizes, conditional diagnostics, confounds and recommended Phase 3
starting controls, never inferred engine support. Partial if required evidence missing.

## Risks and controls
8 GB GPU/16 GB RAM: cap model context at 4096 (>= observed input+1024), GPU allocation
~80%, engine max sequences eight, explicit image count one and pixel caps; no precision
fallback. Compiler/wheel/model support verified locally, not inferred from documentation.
Prefix caches and graph/kernel differences can change performance/numerical outputs;
record flags. Image preprocessing/template differences must be inspected, not assumed.
First collect five-category smoke, broaden only after compatibility. Client timeout and
startup deadline bound hangs. Preserve unsuccessful runs and release process groups.
Percentiles on correlated small cohorts are descriptive. Do not silently repair outputs.

## Evidence
Final evidence: [Phase 2 findings](../docs/PHASE2_FINDINGS.md) and
`results/phase2-final.audit.json`. Entries below retain intermediate observations;
the final closure supersedes their pending status. Phase 1 findings remain unchanged.

### Diagnostics subphase verified — 2026-10-13
- [x] Protected 520 Phase 1/data/runtime artifacts by SHA256; unchanged so far.
- [x] Separate diagnostics and value-free mode prompts, six focused tests; original
  evaluator/prompt/Phase 1 adapter/files untouched.
- [x] Existing 60 predictions: 22 raw parse failures all contain Markdown fences and
  recover an existing object in explicitly labelled diagnostic view; 38 raw JSON,
  six strict schemas, ten view schemas. Coverage 600/600; diagnostic fields 532/600,
  numeric fields 180/180. Sixty ID/reference mismatches plus eight policy_number
  mismatches explain all 68 wrong fields. Positional table cells remain 0/684 because
  headers shift rows or schema-valid answers copy empty cells; no header removal scored.
- [x] Controlled `transformers-diagnostics-v1`: six pages (five category form digitals
  plus paired degraded portfolio), two repeats/mode, fixed BF16/SDPA/revision/pixels/1024.
  Full v2: 2/12 schema, diagnostic fields 102/116, numeric 36/36, CER 0.003654.
  Plain OCR: mean CER/WER 0.001606/0.003924, 257..296 tokens, no failures/truncation.
  Reduced fields: 12/12 reduced schemas, exact 36/46, numeric 22/22, 65..86 tokens.
  Modes are not pooled or treated as equivalent workloads.
- [x] GPU proof: 625 parameter tensors BF16/cuda:0, SM 8.9, no CPU offload;
  separate 32-token profiler observed 54,126 CUDA events including BF16 GEMM/convolution
  kernels, trace retained. Intentionally truncated profiler output excluded from accuracy/timing.
- [x] Server adapter/metrics tests: six additional tests pass; installed CLIs import/help.
  Official vLLM 0.31.0 and SGLang 0.5.21/source APIs reviewed; separate uv locks.
- vLLM pilot passed image smoke and 10 requests: 0.067681 pages/s, 14.7351 s mean,
  TTFT 1.3655 s, ~6.37 GiB device peak, two valid schemas/no correct pages/OOM/truncation.
  Primary paired cohort sweeps underway. SGLang model serving remains unverified.


### Serving evidence so far
- [x] vLLM primary c1/2/4/8: 60 requests/level, all API successes/no OOM; one c8
  truncation retained. Pages/s 0.06931/0.11993/0.18301/0.20371;
  mean latency 14.4148/16.6525/21.6276/37.2897 s. GPU peaks 6.37..6.97 GiB.
  Strict schema 6/8/7/6 of 60, table/page success zero. c8 mean queue 7.393 s,
  18 KV preemptions; eight clients did not sustain eight active full contexts.
- [x] vLLM constrained matched five-category/two-repeat/c1 experiment: 10/10 schemas,
  fields 90/100, numeric 30/30, tables/page success zero. Structural validity != facts.
- [x] Independent interim audit passed: 520 protected files, 36 mode requests,
  250 vLLM primary/structured requests, byte-identical original PNGs/prompts and scores.
- [x] SGLang 0.5.21 incompatibility verified: first missing CUDA_HOME, then mandatory
  JIT activation failed on absent host cc1plus. Archives/logs retained, no OS changes.
- [x] Reviewed 0.5.6 prebuilt candidate; explicit CUDA library root and cuDNN 9.16 uv
  override resolves loader/BF16 Conv3d guard. Probe exact output passes on CUDA/BF16.
- [x] Legacy native-resolution pilot excluded: pixel options not forwarded, portfolio
  input 2557 instead of 1717. Local size-only processor overlay links immutable
  checkpoint/tokenizer assets; corrected pilot counts match Phase 1 on all ten requests.
- [x] Corrected SGLang image pilot: 10/10 runtime successes, no OOM/truncation,
  0.08885 pages/s, mean 11.2320 s, TTFT 1.1932 s, schema 2/10/page success zero.
  Device peak 7,856,652,288 bytes. Primary sweep running; no completion claim yet.


### Final closure — 2026-10-14 (2026-10-13 UTC)
- SGLang primary 1/2/4/8: 60 requests/level, 240 successes, no failure/OOM/truncation.
  Pages/s 0.08525/0.13123/0.20226/0.26550; mean seconds 11.718/15.214/19.601/29.440;
  peak device 7.337/7.337/7.366/7.454 GiB. Strict schemas 6/6/5/5 of 60;
  table/page/correct-pages throughput zero. c8 queue 2.134 s and four retractions.
- Independent final audit passes: 520 protected files, manifest/hash, 36 diagnostics,
  240 SGLang and 250 vLLM unconstrained/structured scores, same original images/prompts,
  paired IDs/repeats and local processor overlay provenance. Raw errors/pilots preserved.
- 76 tests pass; Ruff lint/format, wheel/sdist and installed-wheel smoke pass.
  Original Phase 0 post-run CUDA probe passes; GPU returns to 12 MiB, no server loaded.
- Completion verifies reproducible evidence, not extraction usability. Known limitations:
  tiny paired synthetic cohort, different engine arithmetic/timing boundaries, single-page
  warmups, unisolated thermal/CPU conditions, one vLLM c8 truncation, SGLang version-specific
  compiler/cuDNN/processor workarounds and high memory. Full findings document all tradeoffs.
- Phase 3 recommendation only: BF16 c1 control, four clients balanced serving, fixed
  original contract/pixels/revision. No Phase 3 implementation or quantization performed.
