# Experiment protocol

## Primary metrics
- **Request latency:** client monotonic submit-to-response; p50 and p95 (report sample count).
- **Throughput:** completed pages / elapsed steady-state wall time.
- **Correct pages/sec:** pages meeting documented correctness policy / same wall time.
- **Correctness:** exact match of critical numeric fields, CER and WER on comparable transcribed text.
- **Memory:** NVML device memory peak and PyTorch allocated/reserved memory *separately*, where available.
- **Reliability:** OOM, startup fail, request error, timeout, unsupported and skipped classifications.
- **Startup:** model cold-load time in a separate metric, never blended with steady-state throughput.

## Benchmark matrix
Phase 1: Qwen3-VL-2B-Instruct BF16 Transformers; five-category pilot, then
10 underlying IDs × three variants × two repeats (60 requests; preliminary).
Phase 2: compatible vLLM/SGLang engines; concurrency [1,2,4,8].
Phase 3: Qwen2.5-VL-3B and PaddleOCR-VL official runtime, and compatible 4/8-bit checkpoints; optional olmOCR-2-7B.
Phase 4: resolution, max tokens, max sequences, and memory budget, one dimension at a time.

## Control confounds
- Synthetic v1: 50 unique JSON documents, 10 per category, each with digital selectable
  text, high-quality grayscale 150 DPI scan and degraded scan (Gaussian blur 0.45 px,
  bounded seeded approximately uniform integer noise +/-8 gray levels). US Letter, one page each.
- Write canonical truth before rendering. Freeze SHA256 manifest before experiments;
  verify files before use and record manifest hash in every future inference run.
- Pair variants by document ID. Report per-category/per-variant results and use document
  IDs as statistical clusters (50 independent underlying samples, not 150).
- Prediction contract: JSON object with string-valued fields, ordered string table rows
  and transcription. IDs/dates/labels use literal equality including leading zeros.
  Decimal numeric fields accept correctly grouped commas, optional USD/$ and trailing
  decimal zeros; disallow exponent/nonfinite/float input. Record literal equality too.
- CER/WER normalize whitespace only, preserve case/punctuation/digits, use Levenshtein
  distance divided by reference characters/words. Empty reference scores 0 if both empty,
  1 otherwise. Rates may exceed 1 for insertions.
- Page success requires schema valid, all critical fields exact by declared modes,
  all table cells correct with exact table shape, CER <=0.01 and WER <=0.02.
  Fix thresholds before inference; no model-derived truth or LLM-as-judge.
- Same page ID, rendering, ground truth and task instructions where supported; document variations in proprietary preprocessing.
- Record deterministic seeds, versioned checkpoints, decoding sampling parameters, warmup count and repetition count.
- Model output format and tokenization may differ. Use pages/sec and exact correctness as cross-model axes; don't equate tokens/sec across tokenizers without a caveat.
- Use one GPU engine/process per run and wait for prior process to exit.
- Avoid aggressive sweeps that fill SSD/HF cache without available disk checks.

## Frozen synthetic dataset v1
Seed 42; manifest SHA256
`e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.
The 50-document/150-PDF dataset and complete byte regeneration were validated locally;
these are generation results, not inference results. See
`learning_docs/06_synthetic_dataset_design.md` for versions, transformations, commands
and representative documents. Numeric field formatting is retained in source strings
and literal-match diagnostics even when normalized Decimal equality succeeds.

The old 512-token cap was unverified for the full fields/tables/transcription contract.
Phase 1 verified a fixed 1024-token cap; see the measured baseline contract below. Supply images consistently to all variants rather than digital text
extraction to only one variant. Any model failure or truncation remains a recorded failure.


## Phase 1 measured baseline contract
Official [model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and
[Transformers integration](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
reviewed. Pin revision `89644892e4d85e24eaac8bacfd4f463576704203`;
Transformers 4.57.6, PyTorch 2.13.0/cu126, matching Torchvision 0.28.0/cu126,
Accelerate 1.15.0, official processor/chat template, SDPA, CUDA-only BF16 weights.
No quantization, CPU offload or precision fallback. Controller environment unchanged.

Version `financial-extraction-v2` supplies only schema names, dimensions, column names,
empty string skeleton and extraction instructions alongside a rendered page image.
Expected values never enter messages; offline truth tokenization only estimates output
size and is a separate tool. No JSON fence stripping, coercion or prediction repair.
The earlier v1 pilot is retained as a schema-failing experiment.

All PDF types are rendered at 150 DPI RGB (1275 × 1650), including digital PDFs;
processor min_pixels=262144, max_pixels=1310720. Effective grid is recorded per request.
Primary fixed cap: 1024 new tokens, verified by the five-category pilot and 60-request run. Greedy
`do_sample=False`; temperature/top_p/top_k=None (inactive), repetition_penalty=1,
use_cache=True, seed=42; concurrency=1; one separate warmup; two sequential repeats.
IDs 001 and 010 cover both layouts in each category; all three variants share truth.
No primary request changes prompt version, output cap or preprocessing policy.

Model load time excludes checkpoint download and interpreter startup; cache is warm.
Adapter latency covers image open, official preprocessing, H2D transfer, synchronized
CUDA generation and decoding. End-to-end request latency additionally includes PDF
rendering and raw text writing, but excludes truth/prompt loading, scoring and JSON
result persistence. Steady-state wall throughput includes those controller overheads.
Generation-only latency is separately synchronized and recorded. No TTFT measurement.
NVML device-wide peak sampled every 20 ms includes loading/warmup/requests; PyTorch
allocated/reserved peaks reset per request. They are distinct metrics, not summed.

Invalid schemas receive no exact-field/numeric/table credit in all-request aggregate
rates; CER/WER remain unavailable for invalid predictions, with valid-only means clearly
labeled. Preserve original strict evaluator results. Repeated requests and paired variants
are correlated: report 10 underlying documents/30 distinct pages, not 60 independent pages.
Raw evidence uses new output directories and independently pinned dataset checks before
and after the run. Source hashes and engine lock hash accompany model/environment versions.

Primary run `baseline-v2-1024` is verified complete: 60/60 runtime successes, 0 truncations,
all 30 repeated pairs identical, 0.0582065 pages/s, p50/p95 17.1116/18.3815 s,
4.7901 GiB sampled device peak. Schema validity 6/60 and page success 0/60 are retained
quality failures, not repaired or excluded. See [final findings](PHASE1_BASELINE_FINDINGS.md)
for exact all-request/valid-only metrics, evidence hashes, limitations and reproduction.

## Phase 2 diagnostics and serving protocol
Phase 1 files/strict evaluator/prompt/environment and golden dataset are protected by
`results/phase2/protected-evidence.json`. New diagnostic outputs never overwrite them.
Raw JSON parseability, wrappers, strict schema and independent field coverage/exact
financial/digit/table/text views are separate. Any single-object wrapper view is explicitly
labelled diagnostic-only; no invented values, header removal, coercion or scored repair.
The original all-request usable score and conditional diagnostics answer different questions.

A/B/C diagnostic modes: original full-v2, visual-transcription-v1, critical-fields-v1;
five category form digital pages plus a degraded portfolio pair, two repeats, one warmup,
fixed original BF16/model revision/SDPA/150 DPI/min-max pixels/1024 cap/seed/greedy decoding.
Compare task-specific quality; never pool their contracts, token volumes or throughput.
Profiler uses one separate 32-token inference excluded from benchmark scores/timing.
NVML reports sampled device memory/utilization; actual dtype/device and CUDA events are
separate evidence. Utilization alone does not establish parameter placement or precision.

Servers: separate uv locks, single active GPU model/process group, loopback endpoint only,
600 s startup deadline, 180 s per-request transport deadline, bounded graceful cleanup.
Pinned original Qwen revision, BF16 without quantization/offload, context 4096, 80% device
budget, eight engine sequence slots, original v2/1024, same rendered input PNGs and
min/max pixel settings. Disable graphs and prefix caches; report actual native attention,
processor/library/CUDA version differences. No silent precision/input adjustments.

Concurrency sweep: 30 paired pages × two repeats per level 1/2/4/8, warmup excluded.
Client semaphore limits active HTTP requests; pending client wait and active latency are
recorded separately. Engine-side queue/cache/scheduling gauges/histograms/logs remain
separate from client concurrency and are unavailable unless actually reported by the engine.
Streaming TTFT is first nonempty content chunk; server-reported usage supplies token counts.
Missing usage is unavailable, never approximated from chunks or words. Interrupted streams
retain partial text and fail explicitly. On first failure/OOM, queued requests are skipped
and later unsafe levels stop, with attempted/failure/skipped counts kept distinct.

Prepared-input serving latency excludes PDF rendering/raw persistence, which Phase 1
included. Do not call this an identical end-to-end boundary. Store wall duration, successful
pages/s, correct pages/s, output tokens/s, successful-request mean/nearest-rank p50/p95,
TTFT, strict schema/accuracy, sampled device memory/utilization, failures and raw outputs.
Models/attention libraries differ; engine changes need not preserve token outputs.
Optional JSON-schema-constrained output keeps the full v2 prompt and is a separate
five-category/two-repeat/concurrency-1 run, with warmup and matching unconstrained subset.
Structural validity is distinct from factual extraction correctness.


### Verified Phase 2 runs and interpretation
Final run IDs: `vllm-primary-v1`, `sglang-0.5.6-primary-v1`, and separately
`transformers-diagnostics-v1`. Both serving levels use one single-page warmup each;
this does not warm all concurrency shapes. Fixed prompt/cap/precision/revision persists
through all requests, including the one vLLM c8 truncation. Each engine scores 240
primary requests; constrained vLLM adds ten distinct experiment requests. Effective
SGLang pixel policy requires a documented local size-only processor overlay because
legacy CLI kwargs are not forwarded; original model assets/cache remain untouched.
Record both queue_time and SGLang prefill_waiting stage histograms separately; stage
visits can exceed requests. Device-wide NVML peaks and counters are samples/deltas,
not exact tensor batch tracing. Development/CPU and thermal conditions are not isolated.
Final audit verifies 520 protected files, 36 diagnostics, 490 serving scores, identical
rendered input bytes/prompts and ID pairing. See [findings](PHASE2_FINDINGS.md).
