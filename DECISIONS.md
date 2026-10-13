# Architecture decision log

## ADR-001: Native Fedora, Docker optional
- **Date:** 2026-10-13
- **Decision:** Start with native uv-managed environments; no Docker requirement.
- **Reason:** Lowest initial complexity for measuring one local GPU. Docker can be introduced for incompatible serving stacks later.
- **Revisit when:** GPU kernel dependency problems repeatedly block vLLM/SGLang.

## ADR-002: One shared agent instruction file
- **Date:** 2026-10-13
- **Decision:** `AGENTS.md` canonical, `CLAUDE.md -> AGENTS.md` symlink.
- **Reason:** Codex/Claude use one source of truth; plans/progress separately tracked.

## ADR-003: Multiple isolated uv projects
- **Date:** 2026-10-13
- **Decision:** Controller root project plus engine-specific `environments/*/pyproject.toml` and independent locks.
- **Reason:** GPU serving dependencies may conflict and change rapidly.

## ADR-004: Truthful measurements
- **Date:** 2026-10-13
- **Decision:** Record incompatible, failed, skipped, and OOM configurations; never impute scores.
- **Reason:** Research credibility and reproducibility.

## ADR-005: Phase 0 baseline runtime and preflight boundary
- **Date:** 2026-10-13
- **Decision:** Keep controller stdlib-only; use existing uv Python 3.12.15 and pinned
  PyTorch 2.13.0 from the official CUDA 12.6 wheel index in `environments/transformers`.
  Keep the PyTorch index explicit and track the independent lock. Transformers/model
  preprocessing dependencies are selected in Phase 1; vLLM/SGLang pins in Phase 2.
- **Reason:** Python 3.14 system default exceeds repository constraints. Official PyTorch
  wheel listing and NVIDIA backward compatibility permit cu126 on driver 615.71.09;
  a real CUDA calculation passed. No system compiler/driver upgrade is needed for this slice.
- **Evidence:** See `plans/phase_00_environment.md` and `docs/SETUP_FEDORA.md` for sources.
- **Decision:** `doctor` runs the chosen engine interpreter in a subprocess, never imports
  torch in the controller, writes new local manifests only and returns nonzero on failed checks.
  Without an interpreter, CUDA is skipped and `phase_0_ready` remains false.
- **Reason:** Preserve failures, release GPU context on process exit, and distinguish host
  checks from verified CUDA readiness. Preserve venv symlink path when launching Python.
- **Revisit when:** A Phase 1 model requires a different PyTorch version, or Phase 2 engines
  require different wheels. Their support is not inferred from this smoke check.

## ADR-006: Local synthetic truth and paired PDF variants
- **Date:** 2026-10-13
- **Decision:** Write deterministic structured JSON before PDF rendering. Generate 10
  documents/category across portfolios, receipts, invoices, auditor reports and insurance
  claims; digital, grayscale 150 DPI high-quality scan and seeded mildly degraded scan
  for each ID. Use ReportLab, PyMuPDF and Pillow locally; Decimal monetary arithmetic.
- **Reason:** Known source values remove reliance on external OCR, paid APIs or LLM judges.
  Shared underlying IDs allow paired quality comparisons across 150 PDF variants.
- **Decision:** Freeze SHA256 artifact manifest plus detached manifest hash; no overwrite
  switch. Use pinned rendering dependencies for byte reproducibility. Root gains CPU-only
  document/schema libraries; GPU engine isolation and Phase 0 preflight remain unchanged.
- **Decision:** Independent evaluator requires string JSON for fields/cells, strict schema,
  exact IDs and Decimal numeric equality; retains literal-match diagnostics and OCR rates.
  Page success policy fixed at all fields/cells correct, CER <=0.01, WER <=0.02.
- **Limit:** Synthetic layouts/content cannot establish accuracy on real financial documents.
  Treat the three variants as paired views, not independent underlying samples.
- **Verified:** Full seed-42 generation/regeneration yielded 50 JSON truths and 150 PDFs
  with identical bytes for all 202 files. Pinned CPU packages: ReportLab 5.0.1, PyMuPDF
  1.28.2, Pillow 12.3.0 and jsonschema 4.26.0. GPU engine lock unchanged.
- **Detail:** v1 noise uses seeded bytes modulo 17, creating a small distribution bias;
  record this exact algorithm rather than claiming perfect uniformity. Numeric-mode
  normalization never applies to account/transaction/policy IDs or reference digit strings.
- **Integrity:** Validation can require the independently retained manifest hash in addition
  to the local sidecar. This detects deliberate dataset/sidecar replacement between runs.

## ADR-007: Reproducible visual BF16 baseline and strict schema failures
- **Date:** 2026-10-13
- **Decision:** Pin Qwen3-VL-2B-Instruct revision
  `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers 4.57.6,
  Torchvision 0.28.0 CUDA 12.6 matching the preserved PyTorch 2.13.0 pin;
  SDPA, all parameters CUDA BF16, no quantization/offload. Add dependencies only to
  isolated Transformers project; preserve driver and root/Phase 0 runtime pins.
- **Evidence:** Official model/integration docs reviewed; BF16/CUDA parameter assertions,
  GPU generation, five-category pilot passed without OOM. AutoProcessor initially
  failed without Torchvision (video processor dependency even for still images).
  ReportLab also needed by the existing dataset validator. Failed pilot artifacts retained.
- **Decision:** Render all variants at 150 DPI RGB; cap official processor at 1,310,720
  pixels. Greedy generation instead of the model card's sampling recipe gives controlled
  repeat measurements; temperature is inactive/None, not a hidden sampling fallback.
- **Decision:** `financial-extraction-v2` adds an empty output skeleton to specify table
  nesting. This is shape-only metadata, never reference values. Freeze prompt before
  primary run. Preserve malformed JSON, empty cells, headers and wrong digits verbatim.
- **Decision:** Fixed primary cap 1024: all 50 offline expected JSON lengths <=518 compact
  or <=609 pretty tokens; five-category v2 pilot generated 532..576 tokens without
  truncation. A cap is a limit, not a preallocated static KV cache. Keep dataset/evaluator
  unchanged; no reduced contract was necessary for GPU memory fit.
- **Decision:** Preliminary cohort is IDs 001/010 per category, both layouts; 30 paired
  pages, two sequential repeats, one excluded warmup. No phase-2 infrastructure.
- **Decision:** Report exact-field/numeric/table credit over all expected items, with zero
  credit for schema failures; also valid-only diagnostics. CER/WER are valid-only and
  unavailable for schema failures. No filtering bad predictions out of page denominators.
- **Tradeoff:** Unconstrained greedy JSON is a reproducible baseline even with poor schema
  adherence. Future prompt/decoder experiments need new versions/runs; no post-hoc repair.

## ADR-008: Phase 2 diagnostics preserve usable baseline scoring
- **Date:** 2026-10-13
- **Decision:** Keep financial-page-v1, full-v2 and all Phase 1 artifacts/runtime files
  unchanged. Record protected hashes. Add labelled independent field/coverage/text views,
  including explicitly identified wrapper-only objects, never repaired benchmark outputs.
- **Reason:** Phase 1 usable-contract metrics conflate recognition with schema failures;
  the diagnostic field view observes 532/600 correct fields/180 correct numeric fields,
  while exact digits and table/JSON contracts still fail. These views are not a new baseline.
- **Decision:** Compare full-v2, plain OCR and 3–5 critical fields as separate contracts;
  keep model/precision/images/cap/decoding fixed, do not pool throughput/quality rankings.
- **Decision:** One small separate CUDA profiler request plus actual parameter placement
  and NVML utilization/memory evidence, not heavy profiling during performance sweeps.

## ADR-009: Conservative native serving and explicit measurement boundaries
- **Date:** 2026-10-13
- **Decision:** Isolated pinned vLLM 0.31.0/SGLang 0.5.21 environments and uv locks;
  preserve torch/Transformers baseline environment. Native BF16, max context 4096,
  80% device allocation, eight engine sequence slots; disable graphs and prefix caches.
- **Reason:** Original context is <=2788 with its cap; 4096 allows headroom on 8 GB.
  Eager execution bounds graph memory; disabled caches avoid repeated paired-page gains.
  Installed CLI/source support is checked before launch; support still requires GPU tests.
- **Decision:** Streaming OpenAI-compatible adapter with bounded async clients, durable
  raw records, explicit usage/TTFT/finish/error evidence. No tokens inferred from chunks.
  Stop queued requests and higher levels after failures; preserve failed/skipped counts.
- **Tradeoff:** Native attention, CUDA wheels and newer Transformers are engine dependency
  requirements and confound pure engine-only attribution. Prepared-PNG HTTP timing excludes
  Phase 1 PDF rendering/raw persistence; disclose this instead of claiming identical latency.
- **Decision:** Optional constrained JSON uses installed standard response_format schema
  API and is a separate versioned experiment, not blended with unconstrained comparison.

## ADR-010: SGLang host compatibility and legacy processor configuration
- **Date:** 2026-10-13
- **Observed:** 0.5.21 requires CUDA_HOME through eager DeepEP import; the existing
  uv-installed CUDA 13 toolkit resolves that. Its first BF16 image then fails because
  mandatory activation JIT needs host cc1plus, absent on this Fedora machine.
- **Decision:** Retain both failures and 0.5.21 uv project/lock/source snapshots. Review
  official 0.5.6 Qwen3-VL/prebuilt activation implementation as a native fallback rather
  than modifying system compiler/drivers. Precision/model/cap/full prompt remain fixed.
- **Decision:** Pin 0.5.6's torch 2.9.1/Transformers 4.57.1 and documented cuDNN
  9.16.0.29 workaround using an explicit uv dependency override. Default cuDNN 9.10
  triggered the official BF16 Conv3d safety guard; never bypass it. CUDA BF16 Conv3d
  probe verifies finite exact all-ones output with loaded cuDNN 91600, CUDA 12.8.
- **Observed:** Transitioning NVIDIA CUDA distributions can delete shared namespace
  library files during uninstall; exact locked reinstall restored consistent dependencies.
  Legacy sgl-kernel needs CUDA_HOME pointing to its installed cuda_runtime library root.
- **Observed:** 0.5.6 Qwen processor reads image config but does not pass it to HF; native
  pilot had 2557 portfolio input tokens instead of 1717 and is not a fair primary comparator.
- **Decision:** Local processor overlay changes only preprocessor size to Phase 1's
  262144/1310720 pixel bounds; all model/tokenizer/checkpoint assets link to immutable
  original HF snapshot. Record provenance and hashes; never modify the shared model cache.
  Corrected smoke observes 1717 tokens. Keep failed/native-resolution pilots distinct.
- **Tradeoff:** Legacy CLI lacks newer CPU preprocess-cache controls; radix/embedding
  cache defaults/flags and library/backend differences are recorded, not assumed identical.
  Optional DLPack JIT warnings remain; no manual pip or installed-package source patching.


## ADR-011: Verified Phase 2 controls for future comparisons
- **Date:** 2026-10-14 (Asia/Singapore)
- **Decision:** Keep BF16 concurrency 1 as the future quality control; use four clients
  as the balanced serving comparison, with original revision/full-v2/1024/pixel policy.
- **Evidence:** c4 vLLM/SGLang measured 0.18301/0.20226 successful pages/s. Eight clients
  raise queues/cache pressure; vLLM has 18 preemptions/one truncation, SGLang four
  retractions/7.454 GiB peak. No primary runtime failures/OOMs; correct pages/s is zero.
- **Tradeoff:** SGLang is faster here but the verified 0.5.6 runtime needs explicit
  cuDNN/processor controls. vLLM leaves more headroom. Engine library/native arithmetic
  differences are unavoidable documented confounds, not precision/input changes.
- **Decision:** Schema constraints stay separate; 100% schema validity does not solve
  zero table/page accuracy. Phase 2 research acceptance is complete without claiming
  financial extraction usability. No Phase 3 or quantization work starts in this session.
