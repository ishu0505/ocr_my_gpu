# Phase 2 accuracy diagnostics and serving findings

Status: verified complete — 2026-10-14 (Asia/Singapore). Phase 1 measurements, report,
source adapter/prompt/evaluator, Transformers environment/lock and frozen dataset
remain unchanged. Phase 3 has not started. These are preliminary synthetic results,
not evidence of usable financial extraction.

## Why Phase 1 usable accuracy was low
Diagnostic run `results/phase2/phase1-accuracy-diagnostics.json` reads all 60 saved
predictions and asserts the original strict scores are identical. Raw JSON parses in
38/60 requests; 22 are fenced/extra-text answers. Every one contains a recoverable,
existing JSON object, exposed only as `embedded_object_diagnostic_only`. No values,
structure or raw scores are repaired. Only 6/60 pass the original schema; even this
explicit wrapper-only view yields 10/60 schema validity, not 60/60.

All 600 expected fields occur in this diagnostic object view. Independent exact credit
is 532/600 (88.67%); numeric fields 180/180 (100%). These are diagnostic field metrics,
not replacements for the original 9%/6.67% usable-contract metrics. Sixty mismatches
occur in ID/reference fields counted by the classifier; eight more occur in numeric-
looking policy_number strings. All 68 wrong fields are identifier/digit fidelity errors,
not monetary arithmetic errors in this cohort. Leading/repeated digits are critical.

Tables often include an extra header row or are flattened. Shape-valid auditor answers
leave all cells empty. Original positional diagnostic cells remain 0/684; this does not
mean all visible table numbers were unreadable. The contract requires ordered data rows,
not headers, and no automated row dropping or table reconstruction is scored.
Transcription inside all 60 diagnostic objects is comparable: mean CER 0.002978/WER
0.012899. Good text edit rates can coexist with financially invalid structured outputs.

The evaluator already accepts equivalent exact Decimal amounts (`USD 1,234.500` versus
`1234.50`), grouped commas, trailing zeros and collapsed transcription whitespace.
It deliberately rejects numeric JSON floats, nonfinite/exponents, altered identifiers,
wrong table shapes and Markdown-wrapped raw JSON. No unfair Decimal rounding/tolerance
issue was found. Full-document schema gating hides otherwise correct fields in the usable
metric, so conditional/coverage/raw-text diagnostics are necessary for interpretation;
the strict financial-page-v1 policy is unchanged.

## Controlled A/B/C experiment
Run `results/phase2/transformers-diagnostics-v1`: five category form digitals plus one
paired degraded portfolio page (six pages, five underlying IDs), two repeats per mode,
one excluded warmup per mode. Same unchanged Transformers adapter, pinned Qwen revision,
BF16/SDPA, seed/greedy generation, 150 DPI visual render, official min/max pixels and
1024 output cap. Mode A is byte-identical full v2 prompt. B is `visual-transcription-v1`;
C is `critical-fields-v1` with category-specific 3–5 critical keys, never ground-truth values.

| Mode | Contract validity | Exact fields | Numeric fields | Mean CER / WER | Mean latency | Pages/s |
|---|---:|---:|---:|---:|---:|---:|
| A full v2 | 2/12 strict schemas | 102/116 diagnostic | 36/36 diagnostic | 0.003654 / 0.013365 diagnostic text | 16.323 s | 0.06072 |
| B plain OCR | JSON not requested | N/A | N/A | 0.001606 / 0.003924 raw text | 8.640 s | 0.11451 |
| C critical fields | 12/12 reduced schemas | 36/46 | 22/22 | N/A | 3.492 s | 0.28632 |

No runtime failures/truncation. Tokens: A 445..576, B 257..296, C 65..86.
These contracts produce different amounts of text and are not a shared leaderboard.
OCR was exactly correct on four of the six pages (including the degraded portfolio);
receipt and insurance strings had small edit errors. Minimal mode correctly reads the
portfolio/auditor reference digits but still misses invoice/receipt/claim or degraded
reference strings. Reduced schema complexity fixes output formatting, not every fact.

The evidence points to JSON/table contract compliance plus digit fidelity as the main
usable-score bottlenecks. It does not prove a specific image-resolution cause: this
experiment held resolution fixed and did not run a higher-resolution ablation. A mild
scan can alter repetitive-digit generation even when its plain OCR is exact; task and
vision effects are entangled. Tiny paired sample/repeats give no population conclusion.

## Direct GPU evidence
`gpu.json` records all 625 parameter tensors in BF16 on cuda:0, 2,127,532,032 parameters,
SM 8.9, native BF16 support and CPU-offloading=false. NVML memory/utilization samples
are retained. Separate torch.profiler inference (32 tokens, deliberately excluded from
benchmark scores/timing) records 54,126 CUDA events, including named BF16 GEMM and
BF16 convolution kernels. Counts include copies/memsets and are not all compute kernels.
Trace and error/status artifact are under `profile/`. No heavy profiler runs in sweeps.

## Fixed serving controls and timing
Both primary runs use Qwen/Qwen3-VL-2B-Instruct revision
`89644892e4d85e24eaac8bacfd4f463576704203`, BF16, original full-schema prompt v2,
1024 output tokens, context 4096, eight sequence slots and 80% GPU allocation.
No quantization or CPU weight offload. CUDA graphs and prefix caches are disabled.
Greedy settings: temperature 0, top_p 1, top_k -1, repetition_penalty 1, seed 42.
All PDFs, including digital PDFs, use identical 150 DPI RGB visual rendering
(1275×1650). Processor bounds are 262144..1310720 pixels. Audited PNG bytes and
prompt strings match Phase 1; input token ranges match 1717..1764. Equal input counts
are evidence of policy alignment, not proof of bit-identical internal tensors.

Each engine loaded once, then tested clients 1/2/4/8 with 60 requests per level:
30 distinct pages, 10 underlying IDs (two/category), all three paired variants, two
repeats. One single-page warmup per level is excluded. A single warmup does not warm
every concurrency shape. Runtime/attention/library versions differ across engines.
Prepared-PNG HTTP latency excludes PDF rendering and persistence, whereas Phase 1
included them; wall throughput includes scoring overhead. Model/server startup is
separate. TTFT means first nonempty streamed text at the client, not a GPU timestamp.
Token counts come from actual server usage. NVML samples device-wide memory at 100 ms;
brief peaks can be missed. GPU utilization describes the measured level.

## Measured serving sweeps
Each row is 60 requests. All 480 primary requests completed at the API level, no request
failures or OOMs. vLLM c8 has one explicitly retained length-truncated output; SGLang
has none. Table accuracy, strict page success and correct pages/sec are zero throughout.
The Phase 1 reference remains 0.0582065 pages/s, 17.1666 s mean, 4.7901 GiB device peak,
9% usable fields, 6.67% usable numeric and 10% schema validity.

| Engine / clients | Successful pages/s | Mean / p50 / p95 seconds | TTFT s | Output tokens/s | Device peak GiB | GPU util | Schema % | Fields % | Numeric % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vLLM / 1 | 0.06931 | 14.415 / 14.352 / 15.701 | 1.366 | 38.30 | 6.366 | 83.34% | 10.00 | 9.00 | 6.67 |
| vLLM / 2 | 0.11993 | 16.652 / 16.814 / 18.703 | 1.518 | 65.90 | 6.366 | 94.29% | 13.33 | 11.33 | 10.00 |
| vLLM / 4 | 0.18301 | 21.628 / 21.938 / 24.914 | 1.655 | 100.83 | 6.657 | 94.65% | 11.67 | 9.83 | 8.89 |
| vLLM / 8 | 0.20371 | 37.290 / 36.110 / 49.689 | 12.155 | 114.13 | 6.974 | 96.00% | 10.00 | 9.00 | 6.67 |
| SGLang / 1 | 0.08525 | 11.718 / 11.715 / 12.682 | 1.685 | 47.05 | 7.337 | 98.30% | 10.00 | 9.00 | 6.67 |
| SGLang / 2 | 0.13123 | 15.214 / 15.091 / 16.736 | 1.828 | 72.46 | 7.337 | 99.11% | 10.00 | 9.00 | 6.67 |
| SGLang / 4 | 0.20226 | 19.601 / 19.793 / 22.532 | 2.127 | 111.75 | 7.366 | 98.93% | 8.33 | 7.50 | 5.56 |
| SGLang / 8 | 0.26550 | 29.440 / 29.053 / 37.563 | 5.379 | 146.64 | 7.454 | 99.34% | 8.33 | 7.50 | 5.56 |

vLLM endpoint startup: 75.158 s; SGLang: 27.065 s, both cached checkpoint readiness,
not pure weight-load latency. vLLM logs separately report about 7.725 s weight load.
The same model does not generate identical outputs across native implementations:
vLLM c1 matches 38/60 Transformers raw texts, SGLang c1 matches 30/60. Both c1 runs
have identical 30/30 repeat pairs; higher concurrency introduces repeat differences.
Greedy decoding and a fixed seed do not guarantee BF16 cross-engine equivalence.

## Scheduling and memory constraints
vLLM uses native FlashAttention for language/vision. Its logs report 11984 KV tokens
(~1.28 GiB cache). At eight clients, logs often show only five/six running requests,
with others waiting. Mean server queue is 7.393 s and 18 KV preemptions occur, versus
near-zero queue and zero preemptions at 1/2/4. c8 mean TTFT rises to 12.155 s, p95 total
latency to 49.689 s, with a modest throughput gain over c4. The truncated request is
`insurance_claim-42-010--scan_degraded--r0`; its cap remains 1024.

SGLang uses Triton language attention and SDPA vision. At eight clients logs show
six/seven running and occasional queues. Mean server queue is 2.134 s, four retractions,
TTFT 5.379 s and peak 7.454 GiB, leaving little device headroom. At c1/2/4 queue means
are 0.000524/0.000509/0.001729 s and no retractions occur. Separate prefill_waiting
stage histogram means are 0.0378/0.0362/0.1192/2.5649 s, with 60/60/62/79 stage entries;
these are stage visits, not one event per completed request, and must not be combined
with decoding-stage times. Client semaphore wait is separately logged. Eight clients
are not proof of eight simultaneous full-context sequences or a particular kernel batch.

## Compatibility established through actual inference
- **vLLM 0.31.0**: torch 2.13.0, Transformers 5.17.0, torchvision 0.28.0,
  CUDA 13 wheel dependencies. Pinned checkpoint BF16 works on SM 8.9, primary sweeps pass.
- **SGLang 0.5.21**: BF16 weights load after discovering the uv CUDA toolkit, but
  first image inference fails because mandatory activation JIT needs host `cc1plus`.
  No C++ compiler is installed. Original startup/JIT logs and environment are archived;
  this version has no successful inference benchmark, not zero measured throughput.
- **SGLang 0.5.6**: reviewed prebuilt activation implementation avoids that mandatory
  compiler path; torch 2.9.1+cu128, Transformers 4.57.1, torchvision 0.24.1,
  sgl-kernel 0.3.18.post2. Legacy loader requires CUDA_HOME pointing to installed uv
  nvidia/cuda_runtime. Default cuDNN 9.10 triggers the upstream BF16 Conv3d safety
  guard. An explicit project-only uv override pins cuDNN 9.16.0.29. The guard is not
  suppressed; a CUDA BF16 Conv3d probe returns the exact expected value 1536, then
  real multimodal inference/sweeps succeed. No driver, OS compiler or original baseline
  dependency changes. Switching CUDA namespaces required a frozen locked reinstall.
- The old SGLang Qwen processor ignores CLI pixel kwargs. A successful native-resolution
  pilot used 2557 portfolio input tokens, versus 1717 in Phase 1, and is excluded from
  the fair comparison. A local preprocessor-only overlay sets size bounds; every other
  checkpoint/tokenizer/weight asset links the immutable revision. Shared HF cache stays
  untouched; provenance and symlink targets are audited. Corrected pilot and all primary
  input counts match the baseline. Optional DLPack compilation warnings are nonfatal
  and retained. Shutdown warnings remain in logs; post-run GPU has returned to 12 MiB.

These are explicit isolated runtime tradeoffs, not a universal SGLang incompatibility
claim or an assertion that the older version is preferable on a compiler-equipped host.

## Separate schema-constrained decoding
vLLM official `response_format` JSON schema experiment is separate: five form digitals,
two repeats, concurrency 1, same full v2 prompt/cap/BF16. 10/10 valid schemas,
90/100 exact fields, 30/30 numeric, but 0/114 table cells and zero correct pages.
Matched unconstrained subset has 2/10 schemas, 18/100 usable fields and 4/30 usable numeric.
Mean constrained latency 14.051 s and 0.07047 pages/s, versus matched unconstrained
mean 14.464 s. Small sample/compiler warmup prevents a speed claim. Constraints improve
structure and usable score visibility; empty tables and wrong digit facts remain.
SGLang constrained decoding was not tested; this experiment was optional.

## Reproducibility and validation
`results/phase2-final.audit.json` passes: 520 protected original artifacts unchanged,
frozen manifest `e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`,
36 controlled diagnostic requests, 240 SGLang and 250 vLLM primary/structured requests.
Every raw strict score was independently recomputed. Prompts are value-free, PNG bytes
match Phase 1, repeated IDs/category/variant pairing is preserved, and overlay provenance
is checked. Raw texts, per-request scores, configs, full environments, server metrics/logs,
telemetry, source snapshots and unsuccessful pilot artifacts remain local under `results/phase2`.
Results are intentionally git-ignored; retain that directory alongside the repo/locks.
Snapshot copies preserve the source versions recorded in the run configurations.
`results/phase2-source-preservation.audit.json` additionally verifies 32 original
Phase 1 source files are unchanged and all 42 vLLM/44 SGLang snapshot hashes match.

Final checks: 76 pytest tests, Ruff lint/format, wheel/sdist and installed-wheel module/
CLI smoke. Post-benchmark original Phase 0 CUDA probe passes, driver 615.71.09 unchanged.
See `results/preflight/phase2-post-benchmark.json`. No model remains loaded.

Commands from repository root (use new output directories and one engine at a time):
```bash
uv sync --frozen --project environments/vllm --python 3.12
PYTHONPATH=src uv run --frozen --project environments/vllm python -m gpu_my_ocr.benchmark.serving_run --engine vllm --concurrency 1 2 4 8 --constrained --output results/phase2/new-vllm
uv sync --frozen --project environments/sglang --python 3.12
PYTHONPATH=src uv run --frozen --project environments/sglang python -m gpu_my_ocr.benchmark.serving_run --engine sglang --concurrency 1 2 4 8 --output results/phase2/new-sglang
uv run --frozen python scripts/verify_phase2_evidence.py
uv run --frozen pytest -q
uv run --frozen ruff check .
uv run --frozen ruff format --check .
```

## Limitations and Phase 3 starting configuration
This is ten underlying synthetic documents, paired variants and repeated measurements,
not 60 independent samples per row or a full-dataset conclusion. p95 values are descriptive.
No randomized sweep order, thermal/power isolation or exclusively idle CPU is enforced;
development/tests overlapped portions of the long runs. Streaming timings have different
boundaries from Phase 1 and engine token/EOS accounting may differ. No higher-resolution
controlled Transformers ablation, engine kernel-batch trace or confidence interval study.
The separate 32-token profiler proves CUDA execution but is not a performance benchmark.

For Phase 3, preserve **BF16 concurrency 1** as the quality/reference control, original
full v2/1024/context4096/pixels/revision and frozen evaluator. Use **four clients** for a
balanced throughput comparison (vLLM 0.18301, SGLang 0.20226 pages/s); eight clients are
throughput-oriented with substantial queueing, vLLM truncation/preemptions and SGLang
VRAM risk. vLLM leaves more device headroom; the verified older SGLang configuration
is faster here but needs its documented isolated cuDNN/processor workarounds.
Structured output is a separate quality experiment, not a replacement baseline.
Do not infer that quantization will fix schema/table/digit errors. Phase 2 acceptance is
complete because trustworthy diagnostics/compatibility/comparisons exist; financial
extraction quality remains inadequate. Phase 3 has not started.

## Official sources reviewed
- [Qwen3-VL model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models/)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM 0.31.0 chat protocol](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/entrypoints/openai/chat_completion/protocol.py)
- [SGLang 0.5.21 Qwen3-VL](https://github.com/sgl-project/sglang/blob/v0.5.21/python/sglang/srt/models/qwen3_vl.py)
- [SGLang 0.5.6 Qwen3-VL](https://github.com/sgl-project/sglang/blob/v0.5.6/python/sglang/srt/models/qwen3_vl.py)
- [SGLang 0.5.6 processor](https://github.com/sgl-project/sglang/blob/v0.5.6/python/sglang/srt/multimodal/processors/qwen3_vl.py)
- [PyTorch BF16 Conv3d regression](https://github.com/pytorch/pytorch/issues/168167)

## Final artifact checksums
- `results/phase2-final.audit.json`: `94b66090a73defdc09e39d5edfcacd86dd37976d2ed3f3e1a8b661598752ce56`
- `results/phase2/vllm-final-report.json`: `d302919bd31425debb143e55815a58ce31ef9b9ee46dbb10c97766513785b8a9`
- `results/phase2/sglang-0.5.6-primary-v1/aggregate.json`: `f161999a16227b1b6a78299084bff3ae7c73892bbcf7b0784adf8c8fed3955b6`
- `environments/vllm/uv.lock`: `1f118b971507dc2cd8b2bdfb8dca618c65c4a621d7a4631feb8f87b910e22abe`
- `environments/sglang/uv.lock`: `5dd329f909f1ddf52e74202f008b6d3efc48fbbaa6b1cf9eaaeb3f4866889b82`
- `results/phase2-source-preservation.audit.json`: `c32b237a35dca777fd99368a41b38e7774f9e0720b42ce5d27b0dbaad3bf15ed`
