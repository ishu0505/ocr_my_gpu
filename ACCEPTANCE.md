# Acceptance criteria

Check only with concrete verification evidence.

- [ ] Fedora NVIDIA driver functional (`nvidia-smi` captured in private/local manifest).
- [ ] PyTorch CUDA functional for baseline environment.
- [ ] 50-page synthetic dataset and immutable ground truth generated with deterministic seed.
- [ ] Evaluators tested for digit exact match and CER with edge cases.
- [ ] Transformers baseline runs at least 10 pages; time + accuracy + memory saved.
- [ ] vLLM serves compatible VLM endpoint and completes image requests.
- [ ] SGLang serves compatible VLM endpoint and completes image requests.
- [ ] Controller captures concurrency 1/2/4/8 with defined timing boundaries.
- [ ] Model precision metadata, failures and hardware manifests persisted.
- [ ] Valid fair model/engine comparison with repeated measurements.
- [ ] Dashboard OR static reproducible comparison report.
- [ ] README includes exact commands, limitations and cited upstream methods.
