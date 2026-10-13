# Serving, client concurrency and continuous batching

## Question and controls
Does more than one client improve useful OCR throughput on this 8 GB GPU?
Client concurrency is the number of active HTTP requests. Engine capacity is a separate
limit: Phase 2 sets eight sequence slots, but that does not guarantee eight requests fit
at their maximum context or that every CUDA invocation contains eight sequences.
Continuous batching lets an engine insert/remove requests while others are decoding.
Observe actual scheduler/cache/queue metrics; do not infer batch size from client settings.

Phase 2 preserves Qwen BF16/revision/full-v2/1024 and the exact Phase 1 rendered PNGs.
Servers use 4096 context, 80% device budget, no offload/quantization, graphs/prefix caches
disabled. Native attention/CUDA/Transformers versions differ from the original SDPA runtime.
Separate client pending time from request latency and engine queue time.

## First observed serving result — 2026-10-13
Run `results/phase2/vllm-primary-v1/c1`: 60 requests/30 pages/10 underlying IDs,
two repeats, one excluded warmup. All requests succeeded; no OOM/truncation.
Observed 0.069307 pages/s, mean/p50/p95 14.4148/14.3520/15.7006 s,
mean first nonempty streamed-text TTFT 1.3663 s, 38.3039 reported output tokens/s.
Peak device memory 6.3665 GiB; mean sampled utilization 83.34%, maximum 100%.
Server queue histogram mean 0.0000381 s, zero preemptions. These are serving measurements,
not guaranteed numbers for a fresh machine or a compiled/graph-enabled configuration.

Schema 6/60 and zero correct pages: more throughput does not make extraction usable.
Prepared-PNG HTTP timing excludes Phase 1 PDF rendering/raw writing, and completion
usage can count EOS differently; do not assign the whole speed change to engine kernels.
38/60 raw outputs were identical to Transformers; all 30 repeated pairs were identical
within this c1 run. Equal checkpoint/precision does not guarantee cross-engine outputs.

## Procedure and interpretation
`benchmark/concurrency.py` uses an async semaphore over HTTP requests offloaded to
threads. Pending client time is recorded; successful-page throughput divides completion
count by measured wall time, not by sum of overlapping request latencies. Token throughput
requires actual server usage; absent token/TTFT metrics remain unavailable.
The server saves `/metrics` before/after for histogram/counter deltas and scheduler logs.
Concurrency 2 logs show two running requests; stronger batch-shape claims need profiling.
All 1/2/4/8 levels are complete; final observations are in
[Phase 2 findings](../docs/PHASE2_FINDINGS.md).

On an OOM/failure, pending requests are explicitly skipped and higher unsafe levels stop;
attempted failure rate excludes unattempted skips. Bound transport/startup deadlines and
terminate only the process group owned by the run. Never load both serving models together.
Percentiles from 60 paired/repeated requests are descriptive, not 60 independent document
samples. Correct pages/s remains the practical quality-aware throughput axis.


## Completed preliminary curves — 2026-10-14
vLLM pages/s at 1/2/4/8: 0.06931/0.11993/0.18301/0.20371; mean request seconds
14.415/16.652/21.628/37.290. SGLang: 0.08525/0.13123/0.20226/0.26550 and
11.718/15.214/19.601/29.440. Each level has 60 requests/30 pages/10 paired IDs,
not 60 independent documents. At c8, vLLM mean queue 7.393 s, 18 KV preemptions,
one output truncation; SGLang mean queue 2.134 s and four retractions. No runtime
failure/OOM. Both correct pages/s remain zero. Scheduler logs show fewer than eight
running full-context requests despite eight clients. SGLang prefill_waiting stage counts
can exceed request counts; stage histogram means are not total request queue time.
Four clients is a balanced preliminary setting; one client remains the quality control.
