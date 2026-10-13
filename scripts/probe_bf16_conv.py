"""Verify the isolated legacy runtime's required BF16 Conv3d cuDNN workaround."""

import json
import time

import torch


def main() -> None:
    """Run a small CUDA vision-patch projection and verify exact all-ones outputs."""
    layer = torch.nn.Conv3d(3, 1152, (2, 16, 16), stride=(2, 16, 16), bias=False)
    layer = layer.to(device="cuda", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        layer.weight.fill_(1)
    inputs = torch.ones((128, 3, 2, 16, 16), dtype=torch.bfloat16, device="cuda")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output = layer(inputs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    assert output.dtype == torch.bfloat16 and output.device.type == "cuda"
    assert bool(torch.all(output == 1536)), "Vision projection result mismatch"
    print(
        json.dumps(
            {
                "status": "passed",
                "scope": "small CUDA BF16 Conv3d compatibility probe, not model OCR",
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "cudnn_version": torch.backends.cudnn.version(),
                "seconds": elapsed,
                "output_shape": list(output.shape),
                "exact_expected_output": 1536,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
