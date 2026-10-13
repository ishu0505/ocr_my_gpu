# Phase 2: vLLM + SGLang serving

**Status:** Not started

## Goal
Create isolated uv engine projects, validate model compatibility, common HTTP adapters, bounded load generator, warmup and concurrency 1/2/4/8.

## Tasks
- [ ] Inspect upstream support and dependencies applicable to this phase.
- [ ] Identify interfaces and write minimal tests first where sensible.
- [ ] Implement one small vertical slice; measure or validate.
- [ ] Handle error and unsupported states explicitly.
- [ ] Update `PROGRESS.md`, `ACCEPTANCE.md`, `DECISIONS.md` and learning docs as relevant.

## Acceptance
At least one model each on both supported servers, or documented verified incompatibility; request-level timing saved.

## Risks
OOM, incompatible kernels, measurement confounds, unexpected setup effort; record explicit mitigation.

## Evidence and results
_Not yet collected. Include commands, revisions, and saved run IDs._
