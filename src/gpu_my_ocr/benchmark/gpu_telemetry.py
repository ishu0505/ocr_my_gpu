"""Sample NVML memory/utilization; optional profiler supplies direct CUDA event evidence."""

import statistics
import threading
import time
import traceback
from typing import Any


def summarize_samples(samples: list[dict]) -> dict[str, Any]:
    """Return device-wide sampled peaks/means; absent samples remain unavailable."""
    utilizations = [
        s["gpu_utilization_percent"] for s in samples if s["gpu_utilization_percent"] is not None
    ]
    return {
        "samples": len(samples),
        "peak_device_bytes": max((s["used_bytes"] for s in samples), default=None),
        "mean_gpu_utilization_percent": statistics.mean(utilizations) if utilizations else None,
        "max_gpu_utilization_percent": max(utilizations, default=None),
    }


class GpuTelemetry:
    """Collect bounded-frequency device observations; never infer offload from utilization."""

    def __init__(self, interval: float = 0.1) -> None:
        """Initialize GPU zero's NVML handle and retain errors rather than suppress them."""
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.interval = interval
        self.samples: list[dict] = []
        self.errors: list[str] = []
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.started = time.perf_counter()
        name = pynvml.nvmlDeviceGetName(self.handle)
        self.identity = {
            "name": name.decode() if isinstance(name, bytes) else name,
            "driver": pynvml.nvmlSystemGetDriverVersion(),
            "total_bytes": pynvml.nvmlDeviceGetMemoryInfo(self.handle).total,
        }

    def sample(self) -> None:
        """Append memory/utilization snapshots until stopped, reporting vendor errors."""
        try:
            while not self.stopped.is_set():
                memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
                utilization = self.nvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.samples.append(
                    {
                        "elapsed_seconds": time.perf_counter() - self.started,
                        "used_bytes": memory.used,
                        "free_bytes": memory.free,
                        "gpu_utilization_percent": utilization.gpu,
                    }
                )
                self.stopped.wait(self.interval)
        except self.nvml.NVMLError:
            self.errors.append(traceback.format_exc())

    def close(self) -> dict:
        """Stop sampling and return all observations plus explicitly scoped summary."""
        self.stopped.set()
        self.thread.join()
        self.nvml.nvmlShutdown()
        return {
            "identity": self.identity,
            "interval_seconds": self.interval,
            "scope": "device-wide, includes load/warmup/measurement/profile where used",
            "errors": self.errors,
            "summary": summarize_samples(self.samples),
            "observations": self.samples,
        }


def parameter_evidence(model: Any) -> dict[str, Any]:
    """Return actual parameter device/dtype counts and CPU offload status for a torch model."""
    from collections import Counter

    parameters = list(model.parameters())
    devices = Counter(str(p.device) for p in parameters)
    return {
        "parameter_devices": dict(devices),
        "parameter_dtypes": dict(Counter(str(p.dtype) for p in parameters)),
        "cpu_offloading": any(p.device.type != "cuda" for p in parameters),
        "parameter_count": sum(p.numel() for p in parameters),
    }
