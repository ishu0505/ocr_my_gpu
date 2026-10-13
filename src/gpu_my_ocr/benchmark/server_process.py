"""Bounded local server lifecycle; retain startup logs and terminate only owned processes."""

import asyncio
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import sysconfig
import time
import urllib.error
import urllib.request
from pathlib import Path

from gpu_my_ocr.benchmark.baseline import REVISION

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SERVED_MODEL = "qwen3-vl-baseline"


def server_command(engine: str, port: int, version: str | None = None) -> list[str]:
    """Return fixed BF16/eager single-GPU server arguments for the installed engine."""
    pixels = {"min_pixels": 262144, "max_pixels": 1310720}
    if engine == "vllm":
        return [
            sys.executable,
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--served-model-name",
            SERVED_MODEL,
            "--revision",
            REVISION,
            "--tokenizer-revision",
            REVISION,
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.8",
            "--max-num-batched-tokens",
            "4096",
            "--enforce-eager",
            "--no-enable-prefix-caching",
            "--mm-processor-cache-gb",
            "0",
            "--mm-processor-kwargs",
            json.dumps(pixels),
            "--limit-mm-per-prompt",
            '{"image":1,"video":0}',
            "--cpu-offload-gb",
            "0",
            "--generation-config",
            "vllm",
            "--seed",
            "42",
            "--load-format",
            "safetensors",
        ]
    if engine != "sglang":
        raise ValueError("Unknown engine")
    command = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        SERVED_MODEL,
        "--revision",
        REVISION,
        "--dtype",
        "bfloat16",
        "--context-length",
        "4096",
        "--mem-fraction-static",
        "0.8",
        "--max-running-requests",
        "8",
        "--disable-radix-cache",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--attention-backend",
        "triton",
        "--mm-process-config",
        json.dumps({"image": pixels}),
        "--cpu-offload-gb",
        "0",
        "--random-seed",
        "42",
        "--enable-metrics",
        "--mm-preprocess-cache-size-mb",
        "0",
        "--mm-attention-backend",
        "sdpa",
        "--image-processor-backend",
        "torchvision",
    ]

    if version == "0.5.6":
        # Installed legacy API uses one graph flag and lacks the newer CPU cache/backend flags.
        for flag in (
            "--cuda-graph-backend-decode",
            "--cuda-graph-backend-prefill",
            "--mm-preprocess-cache-size-mb",
            "--image-processor-backend",
        ):
            index = command.index(flag)
            del command[index : index + 2]
        command.append("--disable-cuda-graph")
    return command


def server_environment(engine: str, toolkit: Path | None = None) -> dict[str, str]:
    """Use controlled overrides and discover uv-installed CUDA runtime/compiler roots."""
    overrides = {"VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
    if engine == "sglang":
        if toolkit is None:
            nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
            toolkit = nvidia / "cu13"
            if not (toolkit / "bin" / "nvcc").is_file():
                toolkit = nvidia / "cuda_runtime"
        if (toolkit / "bin" / "nvcc").is_file() or (toolkit / "lib" / "libcudart.so.12").is_file():
            overrides["CUDA_HOME"] = str(toolkit.resolve())
    return overrides


def fetch(url: str, timeout: float = 2) -> str:
    """Read a loopback server endpoint without swallowing HTTP or transport errors."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


class LocalServer:
    """Own a subprocess group and log file, with a finite readiness deadline."""

    def __init__(self, engine: str, port: int, output: Path) -> None:
        """Build a process command; bind only loopback and refuse an already-live endpoint."""
        self.command = server_command(engine, port, importlib.metadata.version(engine))
        if engine == "sglang" and importlib.metadata.version(engine) == "0.5.6":
            from huggingface_hub import snapshot_download

            from gpu_my_ocr.benchmark.processor_overlay import create_processor_overlay

            snapshot = Path(snapshot_download(MODEL, revision=REVISION, local_files_only=True))
            overlay = create_processor_overlay(snapshot, output / "processor-model")
            index = self.command.index("--model-path")
            self.command[index + 1] = str(overlay.resolve())
        self.environment_overrides = server_environment(engine)
        self.base_url = f"http://127.0.0.1:{port}"
        self.output = output
        self.process: subprocess.Popen | None = None
        self.log = None
        self.startup_seconds: float | None = None

    async def start(self, timeout: float = 600) -> None:
        """Launch and wait for model readiness, retaining every startup failure in the log."""
        try:
            await asyncio.to_thread(fetch, self.base_url + "/v1/models")
        except (urllib.error.URLError, OSError):
            pass  # Connection refusal is the required idle-port state, not a server failure.
        else:
            raise RuntimeError("Refusing to use an existing server outside this run")
        self.log = (self.output / "server.log").open("x")
        environment = {**os.environ, **self.environment_overrides}
        started = time.perf_counter()
        self.process = await asyncio.to_thread(
            subprocess.Popen,
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        while time.perf_counter() - started < timeout:
            if self.process.poll() is not None:
                raise RuntimeError(f"Server exited {self.process.returncode}; see server.log")
            try:
                await asyncio.to_thread(fetch, self.base_url + "/v1/models")
                self.startup_seconds = time.perf_counter() - started
                return
            except (urllib.error.URLError, OSError):
                await asyncio.sleep(1)
        raise TimeoutError("Server startup deadline exceeded; see server.log")

    async def close(self) -> None:
        """Gracefully stop the owned process group; bound cleanup and preserve exit status."""
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # Already-exited owned process has no resources to terminate.
            try:
                await asyncio.to_thread(self.process.wait, timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                await asyncio.to_thread(self.process.wait, timeout=10)
        if self.log is not None:
            self.log.close()
