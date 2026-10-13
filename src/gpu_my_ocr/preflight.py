"""Capture local readiness without importing GPU frameworks into the controller."""

import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def run_check(command: list[str], timeout: int = 30) -> dict[str, Any]:
    """Run a command with bounded time; return status, command, output and errors."""
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "status": "failed", "error": str(error)}
    return {
        "command": command,
        "status": "passed" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def collect_manifest(torch_python: Path | None = None) -> dict[str, Any]:
    """Return host/tool/GPU evidence, optionally probing an isolated torch interpreter."""
    gpu_fields = "name,driver_version,memory.total,memory.free,memory.used,compute_cap"
    checks = {
        "nvidia_smi": run_check(["nvidia-smi"]),
        "gpu": run_check(["nvidia-smi", f"--query-gpu={gpu_fields}", "--format=csv"]),
        "cpu": run_check(["lscpu"]),
        "ram": run_check(["free", "-b"]),
        "uv": run_check(["uv", "--version"]),
    }
    checks["nvcc"] = (
        run_check(["nvcc", "--version"])
        if shutil.which("nvcc")
        else {"status": "skipped", "reason": "No toolkit compiler on PATH; not needed for wheels"}
    )
    # Resolving a venv's Python symlink would bypass its installed packages.
    checks["torch"] = (
        run_check(
            [str(torch_python.absolute()), str(Path(__file__).with_name("cuda_probe.py"))], 120
        )
        if torch_python is not None
        else {"status": "skipped", "reason": "No --torch-python supplied; CUDA remains unverified"}
    )
    required = ["nvidia_smi", "gpu", "cpu", "ram", "uv"]
    if torch_python is not None:
        required.append("torch")
    ready = all(checks[name]["status"] == "passed" for name in required)
    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed" if ready else "failed",
        "phase_0_ready": ready and checks["torch"]["status"] == "passed",
        "host": {
            "os": platform.freedesktop_os_release(),
            "kernel": platform.release(),
            "architecture": platform.machine(),
            "libc": platform.libc_ver(),
            "controller_python": platform.python_version(),
        },
        "checks": checks,
    }


def write_manifest(manifest: dict[str, Any], output: Path) -> None:
    """Write JSON evidence to a new file; reject overwrites of previous observations."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
