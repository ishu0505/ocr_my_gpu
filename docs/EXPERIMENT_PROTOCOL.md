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
Phase 1: Qwen3-VL-2B (BF16) Transformers; 10-page smoke then 50 pages.
Phase 2: compatible vLLM/SGLang engines; concurrency [1,2,4,8].
Phase 3: Qwen2.5-VL-3B and PaddleOCR-VL official runtime, and compatible 4/8-bit checkpoints; optional olmOCR-2-7B.
Phase 4: resolution, max tokens, max sequences, and memory budget, one dimension at a time.

## Control confounds
- Same page ID, rendering, ground truth and task instructions where supported; document variations in proprietary preprocessing.
- Record deterministic seeds, versioned checkpoints, decoding sampling parameters, warmup count and repetition count.
- Model output format and tokenization may differ. Use pages/sec and exact correctness as cross-model axes; don't equate tokens/sec across tokenizers without a caveat.
- Use one GPU engine/process per run and wait for prior process to exit.
- Avoid aggressive sweeps that fill SSD/HF cache without available disk checks.
