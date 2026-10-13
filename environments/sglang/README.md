# Isolated SGLang runtime

Verified Phase 2 configuration: SGLang 0.5.6, torch 2.9.1+cu128, Transformers 4.57.1,
torchvision 0.24.1, sgl-kernel 0.3.18.post2 and explicit project-only cuDNN 9.16.0.29
uv override. Separate lock/Python 3.12.15; never sync these packages into Phase 0/1.
240 real BF16 image requests pass clients 1/2/4/8 with no failures/OOM/truncation.
Strict financial page success remains zero. See [findings](../../docs/PHASE2_FINDINGS.md).

From repository root, after the previous engine releases GPU resources:
```bash
uv sync --frozen --project environments/sglang --python 3.12
PYTHONPATH=src uv run --frozen --project environments/sglang \
  python -m gpu_my_ocr.benchmark.serving_run --engine sglang --pilot \
  --concurrency 1 --output results/phase2/new-sglang-pilot
PYTHONPATH=src uv run --frozen --project environments/sglang \
  python -m gpu_my_ocr.benchmark.serving_run --engine sglang \
  --concurrency 1 2 4 8 --output results/phase2/new-sglang-primary
```
The runner owns one loopback process group and records startup, errors, logs, telemetry
and native metrics. Original pinned Qwen revision/BF16/full-v2/1024; rendered visual
inputs, context 4096/eight slots/80% memory, Triton language/SDPA vision attention,
disabled CUDA graphs/radix cache. No offload, quantization or OS driver/compiler changes.
CUDA_HOME is discovered from the venv nvidia/cuda_runtime for the legacy loader.

SGLang 0.5.21 loaded weights but failed mandatory activation JIT on absent host cc1plus;
its environment/logs are archived, with no successful inference claim. The reviewed
0.5.6 prebuilt path succeeds. Default cuDNN 9.10 hits an upstream BF16 Conv3d guard;
9.16 is explicitly pinned and a CUDA BF16 exact-output probe passes. Guard not disabled.
When transitioning NVIDIA shared namespaces, `uv sync --frozen --reinstall --project
environments/sglang` restored exact locked files after stale shared files were removed.

Legacy Qwen does not forward CLI pixel settings. The runner makes a local processor
size-only overlay enforcing 262144..1310720 pixels, linking immutable checkpoint assets
without modifying the HF cache. Overlay provenance/input tokens/PNG bytes are audited.
The initial native-resolution pilot is excluded from fair comparison. Optional DLPack
compiler warnings use a working fallback and remain in logs. c8 peak is 7.454 GiB;
four clients is the balanced recommendation, c1 is the quality control.
