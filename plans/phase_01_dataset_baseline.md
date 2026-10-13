# Phase 1: Synthetic data and naive baseline

**Status:** Verified complete — 2026-10-13; preliminary baseline measured, extraction quality poor

## Goal
Generate 50 unique one-page financial documents (10 each: portfolios, receipts,
invoices, auditor reports, insurance claims), each in digital, high-quality scan and
mildly degraded scan variants: 150 PDFs. Structured JSON is saved before rendering
and is the canonical truth. Implement an independent deterministic evaluator first.
Transformers adapter and repeated baseline are now verified; see final evidence below.

## Scope, interfaces and dependencies
- `datasets/synthetic_data/`: separate category templates, shared rendering, scan
  transforms, manifest generation and validation. CLI generate/validate commands.
- `benchmark/accuracy/`: JSON schema, critical exact fields, Decimal numeric equality,
  CER/WER, ordered table cells, predefined page success; no generator or GPU imports.
- ReportLab, PyMuPDF, Pillow and jsonschema via `uv add`, pinned after local validation.
- Versioned JSON + SHA256 manifest and detached manifest hash, relative artifact paths,
  stable document IDs and recorded seed/layout/transformations/library versions.
- No overwrite/regeneration into existing directories. Generate previews in separate
  new directories; failed generations remain visibly incomplete without a frozen manifest.

## Tasks
- [x] Update dataset planning/architecture/protocol/decisions before implementation.
- [x] Implement five templates with two layouts, Decimal calculations and synthetic labels.
- [x] Generate structured truth before rendering; image-only scans at 150 DPI,
  deterministic blur radius 0.45 px and bounded approximately uniform grayscale noise +/-8.
- [x] Implement independent evaluator and focused regression tests.
- [x] Generate/inspect preview, then validate full 50-document/150-PDF dataset.
- [x] Freeze manifest/hash, update learning docs and record real evidence.
- [x] Review official model compatibility, implement Transformers adapter and measure
  10 underlying documents/30 pages/two repeats (60 requests).
- [x] Inspect upstream support and dependencies applicable to the dataset slice.
- [x] Define dataset/evaluator interfaces and verify them with focused tests.
- [x] Implement one small vertical slice; measure or validate.
- [x] Handle error and unsupported states explicitly.
- [x] Update `PROGRESS.md`, `ACCEPTANCE.md`, `DECISIONS.md` and learning docs as relevant.

## Acceptance
Dataset slice: 50 unique documents, 150 validated PDFs, 50 pre-render JSON truths,
two layouts per category, consistent page geometry/content/order, digital expected
text and scanned text absence, stable IDs, reproducible regeneration into new directories,
integrity/overwrite protections and correct metrics including repetitive digits.
Phase completion additionally requires measured Transformers baseline; dataset code
alone cannot complete Phase 1.

## Risks
OOM, incompatible kernels, measurement confounds, unexpected setup effort; record explicit mitigation.
Synthetic simplification limits real-world generalization. Pin rendering versions for byte
reproducibility; keep IDs as strings to preserve zeros. Group variants by underlying ID,
never count 150 variants as 150 independent samples. Store generated data locally/ignored.

## Evidence and results
- Software: Python 3.12.15, uv 0.13.0; `uv add reportlab pymupdf pillow jsonschema`
  succeeded; after tests, `uv add 'reportlab==5.0.1' 'pymupdf==1.28.2'
  'pillow==12.3.0' 'jsonschema==4.26.0'` pinned validated versions.
- Initial preview `datasets/generated/preview-v1`, seed 42, per-category 1:
  5 underlying documents/15 PDFs validated. Second preview `preview-layouts-v1`,
  per-category 2: 10 documents/30 PDFs validated; both layouts visually inspected.
- `uv run --frozen gpu-my-ocr dataset-generate --output
  datasets/generated/financial-v1-seed42 --seed 42`: passed, 50 documents/150 PDFs.
- Independent full regeneration in `financial-v1-seed42-regenerated`: same manifest
  and byte-identical SHA256 for all 202 files, including 50 truth JSON and 150 PDFs.
- Frozen manifest SHA256: `e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.
  `dataset-validate` passed with `--expected-manifest-sha256` set to this value.
- Full dataset disk usage: `du -sh` reports 49M. All PDFs have matching Letter geometry;
  digital extraction matches canonical text and both scans have no text layer.
- Poppler `pdftoppm -r 85 -singlefile -png` previews inspected for all categories,
  both layouts and scan types; final PNGs in `artifacts/phase1-preview/final-*.png`.
- `uv run pytest`: 51 passed (16.96 s observed); Ruff lint/format: pass, 58 files
  formatted. All source modules <=250 lines/functions <=50 lines. `git diff --check`: pass.
- `uv build --out-dir artifacts/package-check`: wheel/sdist built; archive inspection
  confirms generator/templates/evaluator/preflight packaged and generated data excluded.
  Isolated installed-wheel `gpu-my-ocr --help` smoke passed. Learning-document links,
  exact category/variant counts and agent-instruction symlink verified.
- New regression initially failed due to MuPDF RGB PNG export of ICC grayscale images;
  grayscale comparison verified exact source/transform pixels. No production fallback
  or content change was needed. Initial lint failures fixed before full generation.
- Root lock SHA256: `6505b88081090e0a6d4a13d514644feeeef302757ece4e64a721fc33cd596412`.
  Phase 0 engine lock and verified host manifest hashes remain unchanged.
- Dataset-slice handoff: at 09:17 UTC no inference run existed and the output budget
  was unverified. This historical handoff is superseded by the baseline evidence below.

## Upstream sources reviewed
- [ReportLab invariant output test](https://hg.reportlab.com/hg-public/reportlab/file/b0ba1e17431a/tests/test_invariant.py)
- [PyMuPDF rendering](https://pymupdf.readthedocs.io/en/latest/recipes-images.html) and
  [page/image/text API](https://pymupdf.readthedocs.io/en/latest/page.html)
- [Pillow GaussianBlur](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html)
- [jsonschema validators](https://python-jsonschema.readthedocs.io/en/stable/api/jsonschema/validators/)

## Baseline implementation plan — 2026-10-13
Goal: pinned Qwen3-VL-2B-Instruct CUDA BF16, no quantization/offload, full v1 contract.
Tasks: official compatibility review; isolated dependencies; value-free versioned prompt;
150 DPI visual rendering; adapter; per-category output-budget pilot; 10 IDs (two layouts
per category) × three variants × two repeats, one warmup; strict scoring and findings.
Acceptance: observed CUDA BF16 generation, no truncation at fixed primary cap, immutable
manifest before/after, raw outputs and scoring, load/latency/tokens/PyTorch + NVML memory,
unit/integration checks, Ruff, pytest, package smoke. Poor accuracy is measured rather
than repaired. Risks: GPU sandbox requires host execution; model download, OOM, schema
failure, long digit sequences; retain failed runs and do not silently fall back precision.
Sources: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct and
https://huggingface.co/docs/transformers/model_doc/qwen3_vl (official processor/SDPA).


## Final baseline evidence — 2026-10-13
- [x] Official Qwen model/integration review; revision
  `89644892e4d85e24eaac8bacfd4f463576704203`, Transformers 4.57.6,
  torch 2.13.0+cu126 preserved, matching torchvision 0.28.0+cu126, BF16 SDPA.
- [x] CUDA/BF16 parameter assertions and image/text input diagnostics, successful GPU
  generation, complete checkpoint vision/language fit; no precision fallback/offload.
- [x] Prompt v2 contains only schema metadata/empty skeleton/instructions; tests and
  independent audit verify invariance when all truth values change. Full evaluator unchanged.
- [x] Offline max expected compact/pretty JSON 518/609 tokens; v1 cap-2048 and v2 cap-1024
  category pilots retained. Fixed primary 1024 cap: actual 445..604 tokens, no truncations.
- [x] Primary `results/phase1/baseline-v2-1024`: IDs 001/010 in all categories,
  all three visual variants, separate warmup, two repeats, 60/60 runtime successes.
  No OOM/errors; all 30 repeat pairs have byte-identical raw predictions.
- [x] Strict results: JSON 38/60, schema 6/60, page success 0/60;
  usable exact-field 54/600 (9%), numeric 12/180 (6.67%), tables 0/684.
  Valid-only CER/WER 0.00278358/0.01041667 (six correlated requests).
- [x] Wall 1030.8133 s; 0.0582065 pages/s; mean/p50/p95 17.1666/17.1116/18.3815 s;
  cached model load separately 5.3949 s. NVML peak 5,143,330,816 bytes (4.7901 GiB),
  torch allocated/reserved 4.2973/4.3535 GiB. Input tokens 1717..1764, grid [1,80,62].
- [x] Independent audit recomputes all scores, validates raw outputs/prompts/images/IDs,
  source hashes and all 202 unchanged frozen files; audit SHA256
  `28270042e6860a2d6473fa6e2604d6325336f9efc7fb355044ebac444a8f3b4b`.
  Aggregate SHA256 `39b698a2135ba25ab62767eb69c82fa56ed7c9ba0b8cf0613439a791f418f2d3`.
- [x] 56 tests, Ruff lint/format, package build/installed-wheel imports and CLI smoke;
  post-run Phase 0 CUDA probe passed; model process exit returned NVML usage to 12 MiB.
- [x] Findings/protocol/architecture/decisions/master plan/acceptance/learning docs updated.
  [Human-readable findings](../docs/PHASE1_BASELINE_FINDINGS.md) contain settings,
  actual counts, category/variant results, definitions, raw evidence hashes and limits.
- **Failures retained:** ReportLab import setup failure; official processor's missing
  Torchvision video dependency; v1/v2 schema failures including fences/headers/empty cells.
- **Closure:** Acceptance requires an observed reproducible baseline, not minimum accuracy.
  All Phase 1 acceptance verified. Financial usability fails the measured quality policy.
  Phase 2 remains unstarted. Next: verify each serving engine at concurrency 1 using these
  fixed controls; do not infer compatibility, memory/concurrency fit or accuracy improvement.
