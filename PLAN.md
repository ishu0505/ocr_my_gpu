# Master plan

**Goal:** reproducible model × engine × precision × concurrency benchmarks on Fedora / RTX 4060 8 GB.
**Status:** Phase 0 and Phase 1 verified complete (2026-10-13). Frozen 50-document/
150-PDF dataset and strict evaluator preserved. Qwen3-VL-2B-Instruct CUDA BF16 baseline:
10 underlying documents, 30 pages, 60 repeated requests; 0.0582 pages/s, p50 17.11 s,
4.79 GiB peak device memory, 10% schema validity and 0% page success. This closes
benchmark acceptance, not financial extraction usability. Phase 2 verified complete (2026-10-14): diagnostics, GPU proof, both engines and repeated
1/2/4/8-client sweeps audited. Phase 3 not started.

## Phase order
| Phase | Target day | Outcome | State |
|---|---|---|---|
| 0 | Day 1 | Fedora/NVIDIA readiness; baseline engine environment compatibility | Verified complete (2026-10-13) |
| 1 | Day 1 | Synthetic datasets and scoring; single-model Transformers baseline | Verified complete (2026-10-13); preliminary BF16 baseline measured |
| 2 | Day 2 | Accuracy diagnostics; vLLM/SGLang and controlled concurrency | Verified complete (2026-10-14); preliminary comparisons audited |
| 3 | Day 3 | Model variants, compatible quantization and accuracy sweeps | Not started |
| 4 | Day 4 | GPU profiling, optimization, repetition, DuckDB + dashboard | Not started |
| 5 | Day 5 | End-to-end reproducibility, documentation, published findings | Not started |

## Constraints
Single RTX 4060, 8 GB VRAM, i7, 16 GB host RAM, SSD. Native Fedora and uv. No Docker by default. One serving engine running at a time. Models: Qwen3-VL-2B-Instruct, Qwen2.5-VL-3B-Instruct, PaddleOCR-VL (version verify), optional olmOCR 7B quantized. Model support must be established per-engine/per-version; no blind Cartesian sweep.

## Science rules
- Local synthetic JSON precedes PDF rendering: 10 documents in each of five categories
  (portfolios, receipts, invoices, auditor reports, insurance claims), two layouts each.
  Each has digital, high-quality image-only and mildly degraded image-only variants:
  50 underlying documents, 150 one-page PDFs. No external OCR/API or LLM ground truth.
- Freeze version/seed/IDs/category/variant/transforms/artifact hashes/truth paths in the
  manifest before inference; paired variants are not independent statistical samples.
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
