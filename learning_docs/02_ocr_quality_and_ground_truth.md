# OCR accuracy and ground truth

## Concepts to understand
Ground truth is the answer we know before inference. Here a seeded program creates
structured financial records and saves them as JSON before any PDF exists. A renderer
places those values on the page. A model prediction is compared with the saved JSON,
never with another model's interpretation. No OCR API or LLM judge creates labels.

This makes controlled error analysis possible: if an invoice total is `"6677.68"`,
we know that `"6677.88"` is wrong even if the rest of the document looks plausible.
It does not prove that the artificial page resembles every real invoice.

The evaluator has three complementary views:

- **Fields:** all v1 fields are critical. IDs, dates, currency, names and labels must
  match literally. `"INV-00000001"` cannot become `"INV-1"`. Quantities, prices,
  taxes and amounts use exact Decimal equality after a limited normalization.
- **Tables:** rows and columns retain their order. Every cell uses its declared
  literal or numeric mode. Extra/missing tables, rows or cells fail the JSON schema.
- **Text:** character error rate (CER) and word error rate (WER) count edits needed
  to turn a prediction into the reference transcription. Rates divide by reference
  length; insertions can exceed 100%. We collapse whitespace but keep case,
  punctuation, leading zeros and repeated digits.

For `"a b"` versus `"a c"`, CER is 1/3 and WER is 1/2. Empty versus empty scores zero;
an empty reference with nonempty prediction scores one. Neither metric replaces
financial-field checks: one wrong digit can be serious despite a low page-wide CER.

Amounts must arrive as JSON strings. `"USD 1,234.500"`, `"$1,234.50"` and `"1234.50"`
represent the same Decimal amount. Only outer whitespace, a USD/$ prefix, valid
thousands commas and trailing fractional zeros are normalized. Float JSON values,
exponents, NaN, malformed separators, percentages and European decimal formatting
are rejected. Currency is a separate exact field. Literal-match diagnostics remain
available, so a normalized amount does not hide a formatting difference.

The [`financial-page-v1` policy](../src/gpu_my_ocr/benchmark/accuracy/evaluate.py) requires
valid JSON schema, every critical field correct, every table cell correct,
CER <=0.01 and WER <=0.02. Schema validation uses the
[Draft 2020-12 validator](https://python-jsonschema.readthedocs.io/en/stable/api/jsonschema/validators/).
Malformed predictions fail with errors and unavailable metrics; they are not silently repaired.

## Hypothesis
Known values and paired scans allow visual degradation to be studied separately
from document content. Whether a VLM's financial accuracy decreases on degraded scans
is a future inference hypothesis; it has not been tested yet.

## Experimental setup
Dataset `financial-synthetic-v1`, seed 42, 50 underlying IDs in five categories,
three variants per ID. The prediction schema is derived from stored field names and
table dimensions; values are never supplied to the model as reference answers.
Use the same prompt/decoding/resolution across paired variants in future experiments.

Example source excerpt from `invoice-42-001`:

```json
{
  "invoice_number": "INV-00000001",
  "currency": "USD",
  "subtotal": "6126.31",
  "tax_rate": "0.09",
  "tax": "551.37",
  "total": "6677.68",
  "reference_digits": "0000000111112222333344445555"
}
```

The full prediction contract is `{"fields": {...}, "tables": [[[...]]],
"transcription": "..."}`. Include every reference field and all ordered table cells
as strings. The excerpt above is explanatory, not a complete valid prediction.

```bash
uv run --frozen gpu-my-ocr evaluate \
  --ground-truth datasets/generated/financial-v1-seed42/ground_truth/invoice-42-001.json \
  --prediction results/predictions/invoice-42-001.json \
  --output results/scores/invoice-42-001.json
```

Prediction files in this command must come from a future inference run. The command
does not run a model. An evaluation command returning normally means scoring completed;
consult `page_success` and `schema_valid` for the accuracy outcome.

## Measurements (cite run IDs)
2026-10-13: evaluator tests used independently handwritten oracle values, not model
outputs. Verified perfect controls, wrong IDs/amounts/cells, schema failures, malformed
JSON, whitespace and empty text, rates over one, page thresholds and a 516-character
digit string with a single substitution. Full project suite: 51 passing tests.
At dataset-slice completion no inference measurements existed; subsequent Phase 1
pilots and the measured primary baseline are described below and in the findings.

Representative [digital invoice](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-001--digital.pdf)
and its [degraded scan](../datasets/generated/financial-v1-seed42/pdfs/invoice-42-001--scan_degraded.pdf)
share the same [canonical truth](../datasets/generated/financial-v1-seed42/ground_truth/invoice-42-001.json).
Generated artifacts are local/ignored; regenerate them using the dataset design notes.

## Interpretation and limitations
The scoring contract is now testable without a GPU. A result can distinguish faithful
transcription, structurally valid extraction and financially correct extraction.
Schema validity alone is not accuracy, and a perfect control test is not a model result.
Strict names/dates are deliberate for v1; future policies must be versioned before runs.

For paired analysis, count 50 underlying document IDs rather than 150 independent
samples. Report digital/high/degraded accuracy within each ID/category; any bootstrap
or train/test split should group all variants of an ID together. Broad real-world claims
need independently labeled real documents and additional layout/language diversity.

## First Transformers pilots — 2026-10-13
Five-category CUDA BF16 pilot `pilot-2048-ready`, prompt v1: 0/5 valid schemas,
546–626 output tokens, no truncation. Explicit nesting plus an empty output skeleton in
`financial-extraction-v2` improved validity to 1/5 in `pilot-v2-1024`, but page success
remained 0/5. Examples: Markdown fences; extra table headers; a schema-valid auditor
answer with all table cells left empty; repeated-digit corruption. Do not strip fences,
drop headers or fill empty cells after inference to claim better raw extraction quality.
Version prompt experiments separately and freeze them for each benchmark.

The schema-valid auditor pilot had CER 0.00279/WER 0.01042 while getting 0/12 table cells
correct. Good transcription edit rates alone do not establish financially usable JSON.
The strict evaluator remains unchanged. All-request exact-field/numeric/table aggregates
assign zero credit to invalid schemas; valid-only CER/WER are labeled with their sample
count and are not whole-cohort OCR accuracy. No real-document generalization follows.
See [baseline findings](../docs/PHASE1_BASELINE_FINDINGS.md) for primary measured results.

## Phase 2 diagnosis: contract failure versus reading fidelity
New diagnostic artifacts preserve every Phase 1 strict score. A labelled wrapper-only
object view covers all 600 fields; exact diagnostic fields 532/600 and numerical fields
180/180 are much higher than usable strict credit. This is not a repaired benchmark.
22 raw outputs were fenced; only 6 raw/10 diagnostic objects satisfied the full schema.
Tables added headers/flattened rows or copied empty skeleton cells; digits/IDs/policies
accounted for all 68 incorrect fields. No missing expected fields were observed.

A six-page, five-ID controlled experiment kept model/images/precision/cap/decoding fixed.
Plain OCR produced mean CER 0.001606/WER 0.003924; reduced extraction yielded 12/12
reduced schemas but only 36/46 exact critical fields, with numeric 22/22. Removing JSON
complexity improved formatting and reduced output length, not every repetitive-digit fact.
Original full-v2 mode remains separately scored; different contracts are not a leaderboard.
Decimal-equivalent formatting was already accepted and tests preserve strict digit identity.
Higher-resolution causality was not tested; keep visual-reading limitations as hypotheses.
See [Phase 2 findings](../docs/PHASE2_FINDINGS.md) for actual sample sizes and raw evidence.


## Structured decoding observation — 2026-10-14
The separate vLLM five-category/two-repeat experiment reaches 10/10 strict schemas,
90/100 fields and 30/30 monetary numeric fields, yet 0/114 positional table cells and
zero correct pages. The matched unconstrained subset has 2/10 valid schemas. Structure
constraints make otherwise usable facts accessible; they do not force truthful identifiers
or complete table cells. Preserve both original raw predictions and strict baseline scores.
See [final Phase 2 findings](../docs/PHASE2_FINDINGS.md) for complete counts/confounds.
