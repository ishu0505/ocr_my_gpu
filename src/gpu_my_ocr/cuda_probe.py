"""Standalone probe executed only by the explicitly selected engine interpreter."""

import json
import platform


def probe_cuda() -> dict[str, object]:
    """Verify CUDA allocation, matrix multiplication and CPU transfer; return evidence.

    Raises import/runtime errors if torch or CUDA is unavailable. No CPU fallback.
    Allocations are small, and the subprocess exit releases its CUDA context.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is False")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    with torch.inference_mode():
        inputs = torch.ones((32, 32), device="cuda", dtype=torch.float32)
        result = inputs @ inputs
        torch.cuda.synchronize()
        if not torch.equal(result.cpu(), torch.full((32, 32), 32.0)):
            raise RuntimeError("CUDA matrix multiplication returned incorrect values")
    return {
        "status": "passed",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "wheel_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": properties.name,
        "compute_capability": [properties.major, properties.minor],
        "device_count": torch.cuda.device_count(),
        "free_bytes_before_calculation": free_bytes,
        "total_bytes": total_bytes,
        "operation": "32x32 FP32 ones matrix multiplication; all outputs equal 32",
    }


if __name__ == "__main__":
    print(json.dumps(probe_cuda(), indent=2))
