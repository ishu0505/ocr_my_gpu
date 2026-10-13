# Fedora host setup and preflight

## 1. Verify host first (do not install drivers blindly)
```bash
cat /etc/fedora-release
uname -r
lspci | grep -Ei 'vga|3d|nvidia'
nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
command -v uv && uv --version
```

If `nvidia-smi` fails, diagnose proprietary NVIDIA driver, Secure Boot signing and kernel module status before proceeding. Fedora uses `dnf` for OS-level packages; `uv` manages *Python* dependencies only. Do not use `uv` to install kernel drivers.

## 2. Controller
```bash
uv sync
uv run gpu-my-ocr doctor
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## 3. GPU engines
The `environments/` directories are isolated uv projects. Phase 0 pinned PyTorch only
in the baseline project after host compatibility inspection. Reproduce the verified setup
from the checked-in locks (from repository root):
```bash
uv sync --frozen --python 3.12
uv sync --frozen --project environments/transformers --python 3.12
uv run --frozen gpu-my-ocr doctor \
  --torch-python environments/transformers/.venv/bin/python \
  --output results/preflight/my-host.json
```
Use a new output filename each time; manifests reject overwrites. `doctor` without
`--torch-python` checks the host but records CUDA as skipped and `phase_0_ready=false`.
Failed commands preserve stderr and return nonzero. GPU checks need access to `/dev/nvidia*`;
the agent sandbox hides these nodes, so this session used approved host execution.

Verified 2026-10-13: Fedora 44, glibc 2.43, kernel 7.2.8-200.fc44.x86_64,
driver 615.71.09, RTX 4060 Laptop GPU (SM 8.9), Python 3.12.15, uv 0.13.0,
PyTorch 2.13.0+cu126. NVML reported 8,188 MiB total and 7,877 MiB free;
these are snapshots, not future workload guarantees. CUDA calculation passed.
System Python 3.14.8 is outside this project's declared range; leave it unchanged.

[PyTorch's Linux prerequisites](https://pytorch.org/get-started/locally/) require glibc >=2.28.
[Official wheel instructions](https://pytorch.org/get-started/previous-versions/) list
PyTorch 2.13.0/cu126. [NVIDIA compatibility guidance](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
allows older CUDA runtimes on newer drivers. Driver-reported CUDA UMD 13.4 is distinct
from wheel runtime 12.6. `nvcc` is absent on PATH; the wheel calculation works without
a system compiler. Custom kernel compilation needs a separate compatibility review.
The torch-only probe logs an optional NumPy-missing warning; no NumPy operations were used.

Transformers/preprocessing dependencies are Phase 1 work. For vLLM/SGLang, consult
pinned framework installation instructions and model support in Phase 2 before installing.
Do not install engines in the controller; keep independent locks and log actual versions.

## 4. Benchmark hygiene
Run just one GPU inference process. Close browser GPU-heavy tasks. Capture temperatures, power and driver metadata. Keep workload identical across models when meaningful. Record warmup separately.

## 5. Docker
Not necessary for Phase 0–5. It is a contingency if native SGLang/vLLM compatibility is difficult. Containers still use the host NVIDIA driver.
