# Deterministic synthetic financial dataset

## Question and design

Can we build a reliable local accuracy test without sending financial documents to an
external service? Yes: generate fictitious structured records first, save their JSON,
and render exactly those records. This establishes a known reference for controlled
benchmarks. It does not establish any model's accuracy.

Version `financial-synthetic-v1` contains 10 portfolios, 10 receipts, 10 invoices,
10 auditor reports and 10 insurance claims. Each category has a compact ledger layout
and a shaded form layout, both with a structured table. Every page visibly identifies
itself as a synthetic test document; the auditor's zigzag graphic is explicitly labeled
as a synthetic signature. All names, accounts, claims and amounts are fictitious.

## From structured JSON to selectable PDF

A local SHA256-derived RNG seed depends on generation seed, category and index. This
makes each document independent of generation order. IDs such as `invoice-42-001`
remain stable. A different seed changes content and IDs. Dates are fixed artificial
2026 dates, with no wall-clock timestamp affecting output.

The source stores field values, comparison modes, ordered table columns/rows,
narrative and a canonical visible transcription. Monetary arithmetic uses Decimal,
never binary float. Multiplication gives line amounts; subtotal sums the line amounts;
tax is rounded once to cents using ROUND_HALF_UP; total adds subtotal and rounded tax.

Concrete excerpt from the seed-42 invoice (full source contains more fields):

```json
{
  "document_id": "invoice-42-001",
  "category": "invoice",
  "fields": {
    "invoice_number": "INV-00000001",
    "subtotal": "6126.31",
    "tax_rate": "0.09",
    "tax": "551.37",
    "total": "6677.68"
  },
  "tables": [{
    "columns": ["Description", "Quantity", "Unit price", "Amount"],
    "rows": [
      ["Test consulting hours", "2", "237.24", "474.48"],
      ["Sample support units", "25", "222.17", "5554.25"],
      ["Demo license seats", "1", "97.58", "97.58"]
    ]
  }]
}
```

Here 474.48 + 5554.25 + 97.58 = 6126.31. Multiplying by 0.09 gives
551.3679, rounded to 551.37, making total 6677.68. This arithmetic is known before
rendering, so an OCR or VLM output cannot influence the correct answer.

ReportLab places source strings on a one-page US Letter canvas (612 x 792 points).
Built-in Helvetica avoids machine-dependent font discovery. Its invariant output mode
removes varying PDF timestamps/IDs; see the
[upstream invariant test](https://hg.reportlab.com/hg-public/reportlab/file/b0ba1e17431a/tests/test_invariant.py).
The renderer rejects overflowing text/cells and content exceeding one page.
PyMuPDF then checks that extracted digital text exactly equals the whitespace-normalized
source transcription. Visual inspection is also required: text extraction cannot detect
every bad alignment or clipping problem.

## Making the two scans

The digital page is rasterized at 150 DPI to 1275 x 1650 grayscale pixels. This
resolution is the documented v1 high-quality control, not a claim about all scanners.
See [PyMuPDF rasterization guidance](https://pymupdf.readthedocs.io/en/latest/recipes-images.html).
The raster is stored losslessly as a PNG image inside a fresh PDF with identical page
dimensions. No digital page objects or hidden OCR text layer are copied across.

For the degraded variant, Pillow applies Gaussian blur with radius 0.45 pixels,
then bounded noise of -8 to +8 gray levels with clipping to 0..255. See the
[GaussianBlur API](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html).
Noise bytes come from a separate per-document seeded Random generator and are mapped
modulo 17. This is approximately uniform: 256 is not divisible by 17, so one bin is
slightly more common. The exact algorithm, noise seed, blur and amplitude are recorded.
No rotation, cropping, JPEG loss or random change to content occurs in v1.

Blur spreads sharp edges, making thin strokes and adjacent digits harder to separate.
Noise perturbs light/dark pixels, reducing clean edge contrast. These are controlled
visual stresses; whether they hurt a particular model must be measured later.
Both scanned PDFs must have empty extracted text, exactly one raster image and no
vector drawings. A regression test verifies embedded grayscale pixels against the
digital raster and recorded degradation. MuPDF exports ICC grayscale images as RGB
PNGs; the test compares their grayscale samples.

## Freeze and reproduce

```bash
uv sync --frozen --python 3.12
uv run --frozen gpu-my-ocr dataset-generate \
  --output datasets/generated/my-preview --seed 42 --per-category 1
uv run --frozen gpu-my-ocr dataset-generate \
  --output datasets/generated/financial-v1-seed42 --seed 42
uv run --frozen gpu-my-ocr dataset-validate datasets/generated/financial-v1-seed42 \
  --expected-manifest-sha256 e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976
```

Every output directory must be new, including previews. Regenerate into another new
directory to compare bytes; there is no overwrite switch. A failed generation remains
for diagnosis. A usable dataset must pass validation and have both manifest and hash.
The command can generate 1..10 documents per category; default 10 produces the full set.

The manifest records version, seed, dependency/Python versions, document ID/category,
layout, variant, page size/count, relative PDF/truth paths, SHA256 hashes and scan
transformations. `manifest.sha256` freezes the exact manifest bytes. Store its hash
outside the dataset in future run records and pass it to validation: a local sidecar
alone cannot detect someone deliberately updating both files. This is integrity and
overwrite protection, not a filesystem-level immutable or cryptographically signed store.

Observed software: Python 3.12.15, ReportLab 5.0.1, PyMuPDF 1.28.2, Pillow 12.3.0,
jsonschema 4.26.0. Versions are pinned and the controller lock is tracked as project
work. Byte reproducibility is verified within this environment; changing interpreter,
renderer, architecture or compression implementation may change bytes.

## Evidence and representative pages

2026-10-13: preview of 5 documents/15 PDFs validated first; a second preview covered
both layouts with 10 documents/30 PDFs. Poppler PNGs were visually inspected for all
five categories, both layouts and scan types. After that, the full 50-document/150-PDF
dataset validated and independently regenerated. All 202 files were byte-identical
(50 truth JSON + 150 PDFs + manifest + sidecar). Disk usage: 49 MiB as reported by `du -sh`.
Frozen manifest SHA256:
`e42e5491bceda9ba5b2891c3a8b715654651f763c6c0dc59e30b862022f88976`.

Representative local artifacts (generated files are intentionally ignored in git):

| Category | Digital PDF | High-quality scan | Degraded scan |
|---|---|---|---|
| Portfolio | [Digital](../datasets/generated/financial-v1-seed42/pdfs/portfolio-42-001--digital.pdf) | [High](../datasets/generated/financial-v1-seed42/pdfs/portfolio-42-001--scan_high.pdf) | [Degraded](../datasets/generated/financial-v1-seed42/pdfs/portfolio-42-001--scan_degraded.pdf) |
| Receipt | [Digital](../datasets/generated/financial-v1-seed42/pdfs/receipt-42-001--digital.pdf) | [High](../datasets/generated/financial-v1-seed42/pdfs/receipt-42-001--scan_high.pdf) | [Degraded](../datasets/generated/financial-v1-seed42/pdfs/receipt-42-001--scan_degraded.pdf) |
| Invoice | [Digital](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-001--digital.pdf) | [High](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-001--scan_high.pdf) | [Degraded](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-001--scan_degraded.pdf) |
| Auditor report | [Digital](../datasets/generated/financial-v1-seed42/pdfs/auditor_report-42-001--digital.pdf) | [High](../datasets/generated/financial-v1-seed42/pdfs/auditor_report-42-001--scan_high.pdf) | [Degraded](../datasets/generated/financial-v1-seed42/pdfs/auditor_report-42-001--scan_degraded.pdf) |
| Insurance claim | [Digital](../datasets/generated/financial-v1-seed42/pdfs/insurance_claim-42-001--digital.pdf) | [High](../datasets/generated/financial-v1-seed42/pdfs/insurance_claim-42-001--scan_high.pdf) | [Degraded](../datasets/generated/financial-v1-seed42/pdfs/insurance_claim-42-001--scan_degraded.pdf) |

For a form example, open [invoice-42-010](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-010--digital.pdf).
Final preview PNGs are under `artifacts/phase1-preview/final-*.png`.

## Limits and next experiment

The dataset has English ASCII text, USD amounts, one font family, two shared layout
styles, short tables and one page per document. Insurance claims use a fixed 80%
partial-approval rule. It omits handwriting, camera perspective, severe damage,
multilingual text, real signatures, multipage continuity and varied real scanner optics.
Clean templates can overestimate real-world performance. Fifty underlying documents
also give limited statistical precision; three variants are correlated, not three new
independent examples. Split and resample by underlying document ID.

See [accuracy notes](02_ocr_quality_and_ground_truth.md) for schema and page-success
criteria. At dataset freeze no model results existed. The subsequent Phase 1 BF16 baseline below
verified a visual input policy/output cap and measured 30 pages with two repeats.

## Visual baseline cohort and output sizing
Phase 1 baseline uses document indices 001 and 010 in each category, covering ledger and
form layouts. All three PDF variants are rendered through one 150 DPI RGB path; digital
text is never passed to the VLM. The full frozen dataset is validated before and after.
These 10 underlying documents/30 pages are a preliminary cohort, with two correlated
measurement repeats, not a full 50-document accuracy conclusion.
Offline canonical-output tokenization (never model input) found maxima 518 compact or
609 pretty JSON tokens across all 50 documents. Five-category CUDA pilots verified that
a fixed 1024-token cap covers observed full-contract output lengths with headroom.
Values are withheld from messages; category schema field names, table column names and
row counts are allowed contract metadata. Tests change every value without changing prompts.
See [findings](../docs/PHASE1_BASELINE_FINDINGS.md); baseline images/predictions stay in
ignored result directories, not in the frozen dataset. Scientific results preserve failures.
