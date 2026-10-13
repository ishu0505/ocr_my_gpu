# gpu_my_ocr

Hardware-aware OCR/VLM inference benchmarks on a single RTX 4060 (8GB) running Fedora Linux.

## Scope

Compare Transformers, vLLM, and SGLang serving of Qwen3-VL-2B, Qwen2.5-VL-3B, PaddleOCR-VL, with optional olmOCR-2-7B. Measure quality and throughput under controlled, reproducible workloads. Unsupported engine/model/precision pairs are explicitly recorded, not assumed to work.

## Start

1. Inspect `AGENTS.md`, `PLAN.md`, `PROGRESS.md`, `docs/SETUP_FEDORA.md`.
2. Verify GPU driver with `nvidia-smi` before installing GPU frameworks.
3. `uv sync` then `uv run gpu-my-ocr doctor` and `uv run pytest`.
4. Use `configs/models.yaml` as a target list; don't assume all configs are already implemented.
5. Work phase-by-phase under `plans/`. Update progress after tests.

The repository starts as a *scaffold*, not an already implemented inference benchmark. Results and model downloads are excluded from git.
