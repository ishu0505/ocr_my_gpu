# Phase 1 Transformers BF16 baseline findings

Primary run: `results/phase1/baseline-v2-1024` (local, ignored raw evidence).
Status: **Phase 1 verified complete**, 2026-10-13. The complete primary run and
independent audit passed. Extraction quality is poor; phase closure verifies a measured
baseline, not suitability for financial use. Phase 2 has not started.

## Question and controlled setup
Can Qwen3-VL-2B-Instruct execute a complete visual extraction contract in BF16 on an
8 GB RTX 4060 Laptop GPU, and how faithfully does its raw JSON match frozen synthetic truth?
The baseline uses the official Hugging Face processor/chat template and Transformers,
not a text-layer shortcut. Full financial-page-v1 evaluation remains unchanged.

Revision `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers 4.57.6,
PyTorch 2.13.0+cu126, Torchvision 0.28.0+cu126, Accelerate 1.15.0; SDPA;
all 2,127,532,032 parameters on CUDA in BF16; no quantization or offload.
A CUDA-only parameter assertion and successful generation with image/text input tensors
on cuda:0 establish GPU execution. Native BF16 capability check passed. CPU preprocessing
is expected and included in adapter timing. No drivers or root controller pins changed.

Ten underlying documents: IDs 001 and 010 in each of five categories (ledger/form layouts).
Digital/high/degraded variants give 30 distinct pages; two sequential repeats give 60
measured requests, not 60 independent samples. One excluded warmup; seed 42;
concurrency 1; do_sample=False; sampling temperature/top_p/top_k inactive (None),
repetition_penalty=1, use_cache=True. The deliberately greedy baseline differs from the
model card's sampling recipe. No settings change between primary requests.

Every page is rasterized at 150 DPI RGB, 1275 × 1650; official image processor uses
min_pixels=262144/max_pixels=1310720. Actual grid and input token counts are retained.
Checkpoint context limit is 262144 tokens, not a claim that this context fits 8 GB.
Full truth values stay out of messages; only schema names/shapes, table columns,
instructions and an empty output skeleton accompany the image.

## Token-budget experiment and retained failures
Offline tokenizer sizing across all 50 canonical JSON outputs: max 518 tokens compact,
609 pretty. This computation is separate from inference and never supplies values to it.
First GPU pilot, extraction-v1, cap 2048: five measured category representatives,
546–626 tokens, no truncations/OOM. All five failed schema (flattened tables/header rows).
Second pilot, extraction-v2 with explicit nesting and an empty skeleton, cap 1024:
532–576 tokens, no truncations/OOM, 1/5 schema valid, 0/5 page success. The schema-valid
auditor output copied empty table cells and corrupted repeated digits despite good text
CER: 0.00279, WER: 0.01042. A schema-valid answer is not a correct extraction.

Primary cap 1024 allows observed output plus headroom above all pretty expected sizes;
512 was insufficient as an unverified default. No reduced contract or precision fallback
was necessary. The decoder stops on EOS and does not reserve 1024 actual output tokens.
Truncation is explicitly detected when the cap is reached without EOS.

Before these measured pilots: a launch failed on the validator's missing ReportLab import;
then AutoProcessor failed because AutoVideoProcessor required Torchvision even for images.
The latter failure/config/memory are retained in `results/phase1/pilot-2048`. Dependencies
were added through uv in the isolated environment. Matching official cu126 Torchvision
was validated; PyTorch stayed pinned. These are setup failures, not BF16/OOM failures.

## Measurement definitions and interpretation
Model loading excludes download/interpreter startup; weights are already cached.
End-to-end request latency includes PDF rendering, image preprocessing, H2D transfer,
synchronized generation, decoding and raw text writing; excludes reference/prompt loading,
scoring and score-JSON persistence. Wall throughput includes the sequential runner's
additional scoring/persistence overhead. Generation-only and adapter latencies are saved.
NVML device-wide peak (20 ms sampling) includes model load and warmup; torch allocated/
reserved peaks are recorded per request. Device memory can include non-model consumers.

Strict schema failures get zero credit in all-request exact-field/numeric/table rates;
individual evaluator metrics stay unavailable for those failures. CER/WER means use
schema-valid requests only and must be read with the validity count. Repeated pages and
variants are correlated; category and variant results are descriptive, preliminary,
not a full-dataset accuracy estimate or evidence on real financial documents.

## Official sources
- [Qwen checkpoint/model card](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)
- [Transformers Qwen3-VL integration](https://huggingface.co/docs/transformers/model_doc/qwen3_vl)
- [Official CUDA 12.6 Torchvision wheels](https://download.pytorch.org/whl/cu126/torchvision/)
- [Torchvision installation compatibility](https://github.com/pytorch/vision#installation)


## Primary measured results

| Metric | Observed result |
|---|---:|
| Underlying documents / distinct pages / measured requests | 10 / 30 / 60 |
| Runtime successes; OOM/errors; truncations | 60/60; 0; 0 |
| Parseable raw JSON | 38/60 (63.33%) |
| Strict schema validity | 6/60 (10%) |
| Usable exact-field credit, including invalid pages | 54/600 (9%) |
| Critical numeric field exact-match credit, including invalid pages | 12/180 (6.67%) |
| Ordered table cell credit, including invalid pages | 0/684 (0%) |
| Strict page success | 0/60 (0%) |
| Valid-only field accuracy | 90% across 6 requests / 3 distinct pages |
| Valid-only table accuracy | 0% |
| Valid-only mean CER / WER | 0.00278358 / 0.01041667 |
| Request mean / p50 / p95 latency | 17.1666 / 17.1116 / 18.3815 s |
| Steady-state wall duration / throughput | 1030.8133 s / 0.0582065 pages/s |
| Correct pages/s | 0 |
| Output tokens min / mean / max (includes generated EOS) | 445 / 552.37 / 604 |
| Cached model/processor load (separate) | 5.3949 s |
| PyTorch peak allocated / reserved | 4.2973 / 4.3535 GiB |
| NVML sampled device peak | 4.7901 GiB (5,143,330,816 bytes) |
| Input tokens / effective model image size | 1717–1764 / 992×1280 pixels |
| Identical repeated raw predictions | 30/30 pairs |

All pages had image_grid_thw=[1,80,62], 1240 merged image tokens. Actual combined
input+output was at most 2368 tokens; input plus the fixed cap is at most 2788.
The 262144 checkpoint context limit was not exercised. BF16 vision/language weights
and the selected image/token budget fit the GPU; this is not a concurrency memory test.

| Category | Schema-valid requests / 12 | Usable field accuracy | p50 latency (s) |
|---|---:|---:|---:|
| auditor_report | 6 | 45.0% | 16.559 |
| insurance_claim | 0 | 0.0% | 17.416 |
| invoice | 0 | 0.0% | 18.307 |
| portfolio | 0 | 0.0% | 16.656 |
| receipt | 0 | 0.0% | 16.655 |

| Variant | Schema-valid requests / 20 | Usable field accuracy | p50 latency (s) |
|---|---:|---:|---:|
| digital | 2 | 9.0% | 17.116 |
| scan_degraded | 0 | 0.0% | 17.377 |
| scan_high | 4 | 18.0% | 16.873 |


Repeat 0/1 mean latency was 17.1536/17.1797 s; exact raw outputs, token counts and
scores matched for every pair. This demonstrates within-run repeatability, not identical
results across GPU kernels/library changes. p95 is descriptive for 60 correlated requests;
no confidence interval or real-world accuracy claim is made.

The six schema-valid requests are auditor-001 high scan, auditor-010 digital/high scan,
each repeated twice. All contain empty table cells and a wrong reference digit string.
For example auditor-010's printed `0000001011112222333344445555` was predicted as
`00000010111222233334445555`. The 90% valid-only field accuracy therefore coexists
with zero page success and zero table accuracy. Valid-only numeric fields are 12/12,
but the all-request usable numeric credit is 12/180; do not present 100% as cohort accuracy.
Thirty-two additional requests parse as JSON but fail schema; 22 are not raw JSON
(e.g. Markdown fences). Invalid schemas receive no usable field/table credit, which
measures end-to-end contract compliance rather than character recognition in isolation.

## Verification and evidence
56 unit tests passed; Ruff lint/format, wheel/sdist build, installed-wheel baseline imports,
controller CLI smoke and git whitespace check passed. Real GPU pilots and primary run
are integration tests. `scripts/verify_phase1_baseline.py` independently recomputed all
60 scores, checked raw-text identity, prompt invariance to every reference value,
RGB image dimensions, CUDA input diagnostics, all paired IDs/repeats, unchanged source
hashes and all 202 frozen dataset files. Losslessly parsed model JSON is also saved in
the audit (null for malformed raw JSON), without fence stripping or structure repair.

Dataset manifest remains
`e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.
The post-benchmark `results/preflight/phase1-post-baseline.json` reports phase_0_ready=true
and the preserved torch 2.13.0+cu126 CUDA arithmetic check passes. After model process
exit, nvidia-smi observed 12 MiB used; one GPU engine was active at a time.
Raw predictions, RGB images, scoring, value-free prompts, versions/config/source hashes,
model-load and memory records are in the primary run directory; they are intentionally
local/ignored. The audit includes SHA256 for every run artifact, including warmup.

- `results/phase1/baseline-v2-1024.audit.json` SHA256: `28270042e6860a2d6473fa6e2604d6325336f9efc7fb355044ebac444a8f3b4b`.

- `results/phase1/baseline-v2-1024/aggregate.json` SHA256: `39b698a2135ba25ab62767eb69c82fa56ed7c9ba0b8cf0613439a791f418f2d3`.

- `environments/transformers/uv.lock` SHA256: `53a9cfcf6fd8b75e520a4de0b71e85593150ff73cb426f8989eae79bf8196d3f`.


## Lessons, limits and Phase 2 starting point
The full model fits; increasing available memory would not by itself repair these
JSON/table/digit failures. Table shape instructions alone are unreliable with this
checkpoint/greedy prompt, and an empty skeleton can be copied rather than filled.
Preserve this raw baseline before testing a better prompt or constrained decoding.
Do not weaken frozen schema or use repaired predictions as baseline evidence.

Ten clustered documents, two synthetic layouts, English/USD, short tables and mild
scan degradation cannot establish full-dataset or real financial extraction accuracy.
No laptop power/thermal time series, NVML compute-utilization trace, TTFT, energy,
CPU peak-memory or cross-session determinism was measured. NVML is a sampled peak.
Poor validity prevents a broad OCR-quality conclusion from the valid-only CER/WER.

Recommended initial Phase 2 control: the same pinned Qwen checkpoint, CUDA BF16,
concurrency 1, 150 DPI visual inputs, max_pixels=1310720, prompt v2, fixed 1024 output
cap, greedy decoding, one warmup/two repeats and identical paired IDs. Verify each
serving engine's Qwen3-VL integration and effective preprocessing independently before
raising concurrency. Preserve strict scoring and distinguish throughput from correct
pages/s. No serving engine has been installed or claimed compatible by this phase.
Phase 1 is complete because baseline/runtime/reproducibility acceptance is verified;
financially usable extraction remains an explicitly measured unmet quality goal.
