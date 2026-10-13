# Phase 0: Machine readiness

**Status:** Not started

## Goal
Document Fedora/kernel/driver/GPU/uv versions; test NVIDIA/CUDA with minimal PyTorch; decide compatible framework versions; record environment manifest.

## Tasks
- [ ] Inspect upstream support and dependencies applicable to this phase.
- [ ] Identify interfaces and write minimal tests first where sensible.
- [ ] Implement one small vertical slice; measure or validate.
- [ ] Handle error and unsupported states explicitly.
- [ ] Update `PROGRESS.md`, `ACCEPTANCE.md`, `DECISIONS.md` and learning docs as relevant.

## Acceptance
nvidia-smi works; cuda available in baseline; no OS driver changes without approval.

## Risks
OOM, incompatible kernels, measurement confounds, unexpected setup effort; record explicit mitigation.

## Evidence and results
_Not yet collected. Include commands, revisions, and saved run IDs._
