# Master plan

**Goal:** reproducible model × engine × precision × concurrency benchmarks on Fedora / RTX 4060 8 GB.
**Status:** Scaffold prepared; no model installed, GPU checked, or actual benchmarks run.

## Phase order
| Phase | Target day | Outcome | State |
|---|---|---|---|
| 0 | Day 1 | Fedora/NVIDIA readiness; baseline engine environment compatibility | Not started |
| 1 | Day 1 | Synthetic datasets and scoring; single-model Transformers baseline | Not started |
| 2 | Day 2 | vLLM and SGLang adapters; controlled concurrency benchmark | Not started |
| 3 | Day 3 | Model variants, compatible quantization and accuracy sweeps | Not started |
| 4 | Day 4 | GPU profiling, optimization, repetition, DuckDB + dashboard | Not started |
| 5 | Day 5 | End-to-end reproducibility, documentation, published findings | Not started |

## Constraints
Single RTX 4060, 8 GB VRAM, i7, 16 GB host RAM, SSD. Native Fedora and uv. No Docker by default. One serving engine running at a time. Models: Qwen3-VL-2B-Instruct, Qwen2.5-VL-3B-Instruct, PaddleOCR-VL (version verify), optional olmOCR 7B quantized. Model support must be established per-engine/per-version; no blind Cartesian sweep.

## Science rules
- Ground truth generated separately from model outputs; evaluate blind.
- Use fixed image resolutions and record effective preprocessing per model.
- Report startup separately from warmed throughput; store request-level records.
- Record p50, p95 only for reasonable sample sizes; note small-N uncertainty.
- Sample NVML usage; distinguish device-wide memory from PyTorch allocations.
- Define page correctness explicitly: exact critical fields plus CER threshold where applicable.
- Quantized checkpoint provenance and quantization coverage are mandatory.
- Experimental failures and unavailable configurations must remain in results.

## Primary deliverable
CLI to run/configure benchmark; tested synthetic ground truth + quality evaluator; serialized results/manifest; engine adapters; one comparative report. Dashboard is optional if time constrained.
