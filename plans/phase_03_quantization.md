# Phase 3: Model and precision sweeps

**Status:** Not started

## Goal
Add Qwen 3B, PaddleOCR-VL official adapter and compatible INT4/INT8 variants; optionally olmOCR; log failed and unsupported cells.

## Tasks
- [ ] Inspect upstream support and dependencies applicable to this phase.
- [ ] Identify interfaces and write minimal tests first where sensible.
- [ ] Implement one small vertical slice; measure or validate.
- [ ] Handle error and unsupported states explicitly.
- [ ] Update `PROGRESS.md`, `ACCEPTANCE.md`, `DECISIONS.md` and learning docs as relevant.

## Acceptance
At least one quantization comparison with documented checkpoint and scored numeric extraction.

## Risks
OOM, incompatible kernels, measurement confounds, unexpected setup effort; record explicit mitigation.

## Evidence and results
_Not yet collected. Include commands, revisions, and saved run IDs._
