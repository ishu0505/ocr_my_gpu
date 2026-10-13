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
```

## 3. GPU engines
The `environments/` directories are intentional isolated uv projects, with dependencies added only *after* CUDA/driver compatibility is checked using upstream docs. For example:
```bash
cd environments/transformers
uv add torch transformers accelerate pillow
uv sync
```
Do not install every engine into the root environment. For vLLM/SGLang, consult pinned framework installation instructions, compatible PyTorch/CUDA wheel and NVIDIA driver versions first. Keep separate `uv.lock` files and log actual versions.

## 4. Benchmark hygiene
Run just one GPU inference process. Close browser GPU-heavy tasks. Capture temperatures, power and driver metadata. Keep workload identical across models when meaningful. Record warmup separately.

## 5. Docker
Not necessary for Phase 0–5. It is a contingency if native SGLang/vLLM compatibility is difficult. Containers still use the host NVIDIA driver.
