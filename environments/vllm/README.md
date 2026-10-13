# Isolated vLLM runtime

Phase 2 pins vLLM 0.31.0, independent uv lock/Python 3.12.15. Official installed
Qwen3VL architecture and SM>=7.5 support were reviewed; real BF16 image pilot and
primary concurrency 1/2/4/8 passed on SM 8.9. One c8 output hit the fixed token cap;
no runtime failure/OOM. A separate JSON-schema experiment also completed.
Do not use the Transformers environment's torch index/pins in this runtime.
Full installed versions/CUDA dependencies are in its lock and per-run config.

From repository root, with no other model on the GPU:
```bash
uv sync --frozen --project environments/vllm --python 3.12
PYTHONPATH=src uv run --frozen --project environments/vllm \
  python -m gpu_my_ocr.benchmark.serving_run --engine vllm --pilot \
  --concurrency 1 --output results/phase2/new-vllm-pilot
PYTHONPATH=src uv run --frozen --project environments/vllm \
  python -m gpu_my_ocr.benchmark.serving_run --engine vllm \
  --concurrency 1 2 4 8 --constrained --output results/phase2/new-vllm-primary
```
The runner owns one loopback process group, bounds startup/transport time, saves logs,
streams image responses, strictly scores predictions and shuts the server down.
Fixed BF16/pinned Qwen revision/full-v2/1024, same 150 DPI rendered pages, max pixels
1310720, context 4096, eight sequence slots, 80% allocation, no offload/quantization.
Eager execution/no prefix caching; native FlashAttention selected in the observed run.
The constrained experiment is separate; it does not alter the original benchmark.
See [Phase 2 findings](../../docs/PHASE2_FINDINGS.md) for actual results/limits.
