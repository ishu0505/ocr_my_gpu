# Architecture

Controller `gpu_my_ocr` uses YAML configuration to select a model/engine/workload combination. A runner validates configuration, launches or connects to exactly one engine, warms it, then dispatches requests with bounded `asyncio` concurrency. Adapters standardize `infer(page_path, prompt, settings)` results. The evaluator compares outputs with separately stored synthetic truth. The profiler samples NVML while a run is active; it does not claim to directly measure achieved FLOPs. Storage writes request-level Parquet and run manifests, with DuckDB for aggregate queries. Dashboard consumes stored results only.

## Logical modules
- `config.py`: config parsing/validation (implemented in later phase)
- `models.py`: typed domain records
- `engines/`: Transformers and local HTTP engine implementations
- `benchmark/`: load generation, timing, accuracy, GPU sampling
- `datasets/`: synthetic generation and benchmark IO
- `storage/`: results and manifests
- `reporting/`: aggregate queries and charts

**Dependency direction:** CLI -> runner -> adapters/benchmark/storage. Adapters should never import reporting/dashboard. Evaluators should never import CUDA/engine code.

## Design constraints
- Internal API uses a small `InferenceRequest` / `InferenceResult` pair; model-native preprocessing may require adapter-specific logic.
- Official PaddleOCR pipeline may not be an OpenAI endpoint; use dedicated adapter.
- vLLM/SGLang engine projects are separate uv environments to avoid conflicting CUDA/PyTorch dependency graphs.
- Quantization and VLM support matrix is measured/documented; no unverified claims.
- No unnecessary Docker, REST service, worker queue, cache, orchestration cluster or cloud database.
