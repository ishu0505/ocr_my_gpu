# Measured results and conclusions

## Phase 1 question and setup
Can a full Qwen3-VL-2B visual JSON extraction baseline fit RTX 4060 8 GB in native BF16?
Yes for this pinned single-request configuration; usable extraction quality remains poor.
[Full findings](../docs/PHASE1_BASELINE_FINDINGS.md) give exact controls, versions,
evidence hashes, timing/memory definitions and category/variant results.

## Measurements — 2026-10-13
Run `results/phase1/baseline-v2-1024`: 10 underlying synthetic documents, 30 paired
visual pages, two repeats, 60 requests; warmup/model loading separate. BF16 SDPA,
1024 output cap, effective 992×1280 images. 0 OOM/failures/truncations, 445–604 tokens.
Measured 0.0582065 pages/s, mean/p50/p95 latency 17.1666/17.1116/18.3815 s;
cached load 5.3949 s; device NVML peak 4.7901 GiB; torch allocated/reserved
4.2973/4.3535 GiB. Raw predictions match exactly in 30/30 repeated pairs.

Strict schema: 6/60; page success: 0/60. All-request usable fields 54/600 (9%),
numeric fields 12/180 (6.67%), tables 0/684. Invalid schemas get no usable credit;
this is end-to-end contract accuracy, not a claim that all their recognized text is wrong.
Valid-only fields 90%, CER 0.00278358/WER 0.01041667, based on only three distinct
schema-valid auditor pages repeated twice. Their empty table cells and repetitive digit
errors demonstrate why transcription edit rate alone is not sufficient.

## Validation and interpretation
56 tests, Ruff, package/installed-wheel smoke checks and independent saved-result audit
passed. All scores recomputed against unchanged frozen truth; prompts invariant to truth
values; all types processed as rendered RGB images; source hashes unchanged. Phase 0
probe still passes and GPU usage returns to 12 MiB after process exit.

Phase 1 acceptance is complete: a reproducible measured baseline exists. This does not
mean the model is ready for financial extraction. English synthetic templates and 10
clustered documents cannot establish full-dataset/real-world accuracy. Hardware power,
thermal dynamics, energy, TTFT and engine concurrency were not measured.
Recommended Phase 2 control: same pinned Qwen BF16, concurrency 1, visual policy,
prompt v2, 1024 cap and paired corpus/repeats; verify each engine independently before
changing concurrency. Preserve this baseline before any new prompt/decoder quality run.
No Phase 2 work has begun.


## Phase 2 completed observation — 2026-10-14
This supersedes the Phase 1 handoff above; Phase 1 measurements remain unchanged.
Offline diagnostics separate 88.67% independently exact fields/100% monetary numeric
from 9%/6.67% strict usable scores. JSON/schema/table failures and identifier digit errors
explain the gap; a good OCR CER is insufficient for critical financial correctness.
Actual vLLM and corrected SGLang BF16 sweeps completed all 480 primary requests.
At four clients: vLLM 0.18301 pages/s, 21.628 s mean, 6.657 GiB peak; SGLang 0.20226,
19.601 s and 7.366 GiB. Eight increases throughput but also queueing/memory pressure.
Strict correct pages/sec remains zero. Separate constrained decoding fixes schema validity
but leaves table/factual errors. These are preliminary ten-document results with paired
variants/repeats and differing engine arithmetic, not a production-quality conclusion.
76 tests and the independent final protection/scoring audit pass. Phase 2 complete;
Phase 3 not started. Future controls: BF16 c1 for quality, c4 for balanced serving.
See [findings](../docs/PHASE2_FINDINGS.md) for versions, failures and raw evidence paths.
