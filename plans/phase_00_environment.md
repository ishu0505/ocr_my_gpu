# Phase 0: Machine readiness

**Status:** Verified complete — 2026-10-13; stopped at phase boundary

## Goal
Document Fedora/kernel/driver/GPU/uv versions; test NVIDIA/CUDA with minimal PyTorch; decide compatible framework versions; record environment manifest.

## Tasks
- Scope: stdlib-only `doctor` CLI with JSON manifest and explicit command statuses;
  isolated Python 3.12/PyTorch wheel runtime with a small deterministic CUDA calculation.
- Interfaces: `doctor --output PATH --torch-python PATH`; failures return nonzero,
  an omitted runtime is recorded as skipped. No model loading or serving adapters.
- Dependencies: controller stays GPU-free; select PyTorch 2.13.0/cu126 from official
  wheel instructions after checking driver compatibility. Serving pins remain Phase 2 work.
- [x] Inspect upstream support and dependencies applicable to this phase.
- [x] Identify interfaces and write minimal tests first where sensible.
- [x] Implement one small vertical slice; measure or validate.
- [x] Handle error and unsupported states explicitly.
- [x] Update `PROGRESS.md`, `ACCEPTANCE.md`, `DECISIONS.md` and learning docs as relevant.

## Acceptance
nvidia-smi works; cuda available in baseline; no OS driver changes without approval.

## Risks
OOM, incompatible kernels, measurement confounds, unexpected setup effort; record explicit mitigation.
Sandbox hides `/dev/nvidia*`: validate GPU checks outside the sandbox, retaining failure evidence.
System Python 3.14 is outside repository constraints: use existing uv Python 3.12.15.
Driver-reported CUDA ceiling is not the installed toolkit or wheel runtime; record separately.

## Evidence and results
- Base revision: `3c9642f6d8e904a3616bc476f43894c5569ed2bb`; worktree changes uncommitted.
- OS: Fedora 44, kernel `7.2.8-200.fc44.x86_64`, glibc 2.43.
- CPU: i7-13700HX, 24 logical CPUs; RAM total 16,438,509,568 bytes;
  available at verified check 9,074,311,168 bytes. Free SSD space initially 932 GiB (`df -h .`).
- GPU: RTX 4060 **Laptop** GPU, SM 8.9; driver 615.71.09; driver CUDA UMD 13.4.
  NVML snapshot: total 8,188 MiB, free 7,877 MiB, used 12 MiB.
  Free memory is observed, not inferred by subtracting used from total.
- uv 0.13.0, controller/runtime Python 3.12.15; system Python 3.14.8 intentionally unused.
- `uv add --project environments/transformers --python 3.12 'torch==2.13.0'
  --index pytorch=https://download.pytorch.org/whl/cu126`; then explicit index configuration
  and `uv sync --project environments/transformers --python 3.12` succeeded.
- `uv run --frozen gpu-my-ocr doctor --torch-python environments/transformers/.venv/bin/python
  --output results/preflight/phase0-host-verified.json`: exit 0, `phase_0_ready=true`;
  torch 2.13.0+cu126, wheel runtime 12.6, allocation + 32x32 FP32 matrix multiply +
  synchronization + CPU exact-result check passed. Post-exit `nvidia-smi` again shows 12 MiB used.
- Optional NumPy absent warning preserved in probe stderr; not required for this calculation.
  `nvcc` absent on PATH; no system toolkit or drivers installed.
- Initial sandbox `nvidia-smi` failed (device nodes hidden); host check passed after escalation.
  First manifest failed because symlink resolution bypassed the venv; fixed with `absolute()`
  and regression test. Failed manifest retained as `results/preflight/phase0-host.json`.
- `uv run --frozen ruff check .`, `uv run --frozen ruff format --check .`,
  `uv run --frozen pytest`: all pass, 11 tests. `bash scripts/check_agent_links.sh`: pass.
- SHA256 verified manifest: `278fa731c9308511b22ab58dfd66b707fea8cec87d61dfac45502778a8ef2a8d`.
- SHA256 failed manifest: `d4b27498b8d32bbe1f7614cd8ab8dbec0167bdb1890605603d72f286c97b022b`.
- SHA256 controller lock (existing, preserved): `fa2bb85840071d31d4feed9c379be2100b0869ff871b6c747da747f254704864`.
- SHA256 engine lock: `acf4978e0e41159d184b6593a7a03ae67340662c185d8e4fdc3933a916f5ef02`.
- No benchmark run IDs: this was environment validation only. Phase 1 next: seeded synthetic
  generator and independent ground truth/evaluator, followed by upstream model/runtime review.

## Upstream compatibility sources (reviewed 2026-10-13)
- [PyTorch Linux prerequisites](https://pytorch.org/get-started/locally/): glibc >=2.28.
- [PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/): 2.13.0/cu126 wheel listed.
- [NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html):
  newer drivers support CUDA 12.x via backward compatibility. This guided wheel selection;
  the actual calculation establishes local readiness.
