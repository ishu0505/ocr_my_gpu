# transformers isolated runtime

Phase 0 verified Python 3.12.15 and PyTorch 2.13.0+cu126 on the host RTX 4060 Laptop GPU.
The official CUDA 12.6 wheel index is explicit for torch; dependencies are locked.
Phase 1 installs the pinned Transformers/model preprocessing packages listed below.

From repository root:
```bash
uv sync --frozen --project environments/transformers --python 3.12
uv run --frozen gpu-my-ocr doctor \
  --torch-python environments/transformers/.venv/bin/python \
  --output results/preflight/new-check.json
```

See `docs/SETUP_FEDORA.md` for upstream sources, observed versions and sandbox limitations.
The torch-only calculation passed with an optional NumPy warning preserved in the manifest.
Phase 1 pilots verified Qwen3-VL BF16 memory fit; strict extraction quality is measured
separately. Do not share this environment with the controller or serving runtimes.

## Phase 1 baseline
Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`;
Transformers 4.57.6, Torchvision 0.28.0+cu126, Accelerate 1.15.0, native BF16 SDPA.
PyTorch remains 2.13.0+cu126. Official processor needs Torchvision even for still images.
ReportLab/PyMuPDF/Pillow/jsonschema support the same frozen validator and visual rendering;
nvidia-ml-py records device-wide memory. Exact transitive versions are in the uv lock.

From repository root (GPU execution requires host device access):
```bash
PYTHONPATH=src uv run --frozen --project environments/transformers \
  python -m gpu_my_ocr.benchmark.baseline --pilot --repeats 1 \
  --max-tokens 1024 --output results/phase1/my-new-pilot
PYTHONPATH=src uv run --frozen --project environments/transformers \
  python -m gpu_my_ocr.benchmark.baseline --repeats 2 \
  --max-tokens 1024 --output results/phase1/my-new-baseline
uv run --frozen python -m gpu_my_ocr.benchmark.baseline_report \
  results/phase1/my-new-baseline
```
Output directories must be new. Model loads once; warmup separate; raw text, scores,
images, value-free prompts, source/lock/dataset hashes, load time and memory are retained.
See the protocol for timing boundaries and all-page versus valid-only score definitions.
Do not run another GPU engine concurrently.
