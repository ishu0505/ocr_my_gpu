# GPU memory and inference

## Concepts to understand
Driver compatibility, wheel runtime and toolkit compiler are separate facts. Here
`nvidia-smi` reports CUDA UMD 13.4, torch reports CUDA runtime 12.6, and no `nvcc`
is on PATH. A CUDA wheel calculation works without installing a system compiler.
See [NVIDIA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and [PyTorch binary installation](https://pytorch.org/get-started/locally/).

## Hypothesis
The existing driver can execute an official cu126 PyTorch wheel without OS changes.
This hypothesis addresses runtime readiness only, not VLM memory fit or throughput.

## Experimental setup
2026-10-13 Phase 0: isolated uv Python 3.12.15, torch 2.13.0+cu126, one RTX 4060
Laptop GPU, driver 615.71.09. A subprocess allocated a 32x32 FP32 ones tensor,
multiplied it by itself, synchronized CUDA, transferred results to CPU and checked
every output equals 32. No model, warmup, repeats or performance timing was involved.

## Measurements (cite run IDs)
Evidence: local `results/preflight/phase0-host-verified.json` at 08:18:43 UTC;
manifest hash in `plans/phase_00_environment.md`. Operation passed, SM 8.9, one device.
NVML before probe: total 8,188 MiB, free 7,877 MiB, used 12 MiB. CUDA API after
context initialization: total 8,272,019,456 bytes, free 8,152,612,864 bytes.
Host RAM: 16,438,509,568 bytes total, 9,074,311,168 available at that snapshot.
After subprocess exit, NVML again showed 12 MiB used. No benchmark run ID or speed result.

## Interpretation and limitations
The runtime hypothesis passed. Memory API values were taken at different times and
scopes; they are not interchangeable with PyTorch tensor allocations. In particular,
do not infer available VRAM by subtracting NVML used from total; query free directly.
These readings do not establish a model's weights + vision activations + KV cache fit.
At Phase 0, BF16, attention kernels, model loading and engine support were unverified.
The subsequent Phase 1 evidence below verifies the selected Qwen BF16 baseline only.

Two failures informed the implementation: sandbox-hidden GPU device nodes caused
`nvidia-smi` failure despite a functioning host driver; resolving a venv Python symlink
bypassed its packages. Approved host execution and retaining the venv path fixed them.
The failed manifest is retained. The optional NumPy warning remains recorded, with
no fallback or fabricated result.

## Phase 1 BF16 pilot observations — 2026-10-13
Qwen3-VL-2B-Instruct has 2,127,532,032 parameters: BF16 weight storage alone is about
3.96 GiB (parameter-count estimate, not a device measurement). Measured v2 pilot
PyTorch peak allocated: 4,614,203,392 bytes; reserved: 4,674,551,808 bytes;
NVML device-wide sampled peak: 5,143,330,816 bytes. Do not add these overlapping measures.
NVML includes CUDA/runtime and any other device consumers, plus load/warmup peaks.
The complete vision+language checkpoint fit this GPU without quantization or offload.
No conclusion about another model, longer context, higher resolution or concurrency follows.
Native CUDA BF16 capability and all-weight dtype/device assertions passed; image/text
input tensors on cuda:0 and successful generation confirm actual GPU inference.
Generation outputs stopped before a 1024-token limit; a dynamic cache grows with actual
tokens, so the cap does not mean every request generates or allocates all 1024 tokens.
Official preprocessing reduced the rendered 1275×1650 image to a recorded patch grid;
use image_grid_thw rather than treating raw PDF pixels as identical to model resolution.
See [baseline findings](../docs/PHASE1_BASELINE_FINDINGS.md) for finalized primary evidence.

## Phase 2 direct execution proof and eager serving memory
The controlled Transformers diagnostic run records 625 parameter tensors on cuda:0 in
BF16, SM 8.9 and no CPU weight offload. A separate 32-token torch.profiler request
observed 54,126 CUDA events, including BF16 GEMM/convolution kernels. Event counts
include memory copies/memsets, so do not call every event a compute kernel or infer FLOPs.
Profiler timing/intentional token truncation is excluded from accuracy/performance runs.
NVML sampled peak was 5,155,913,728 bytes, mean utilization 88.19% over the full
load/warmup/three-mode/profile window. Device utilization is not a precision measurement.

vLLM c1 reserves a larger engine cache than Transformers: sampled peak ~6.37 GiB,
versus the original 4.79 GiB dynamic-inference peak. Both fit, but reserved KV capacity,
contexts and graph allocations affect concurrent requests. Phase 2 caps engine context
at 4096 and device fraction at 0.8, disables graphs and prefix caches, and records native
backend differences. Neither an eight-slot setting nor remaining VRAM proves eight
simultaneous maximum-length requests fit. Sweep and observe actual queues/preemptions.
See [findings](../docs/PHASE2_FINDINGS.md); no driver/precision fallback was made.


## Final Phase 2 memory observation — 2026-10-14
SGLang 0.5.6 with patched isolated cuDNN and original-pixel processor overlay peaks
at 7.337/7.337/7.366/7.454 GiB at 1/2/4/8 clients. vLLM peaks 6.366/6.366/6.657/6.974.
These are sampled device-wide measurements, not weight sizes or PyTorch allocations.
A fitting BF16 model does not imply that eight full contexts fit in cache; measured
preemptions/retractions and queueing rise at eight clients. No OOM occurred. The older
SGLang native-resolution pilot is excluded because it used more visual tokens. Neither
precision nor the frozen dataset was changed to make the fair sweep fit.
