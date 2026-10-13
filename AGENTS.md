# Shared agent instructions (Claude Code + Codex)

You are working on `gpu_my_ocr`, a 3–5 day research MVP for hardware-aware VLM/OCR inference benchmarking. Fedora, RTX 4060 8 GB, system RAM 16 GB, Python with uv.

## Session start — mandatory
1. Read `PLAN.md`, `PROGRESS.md`, `DECISIONS.md`, `ACCEPTANCE.md`.
2. Read the active `plans/phase_*.md` and `docs/ARCHITECTURE.md`.
3. Inspect git status, existing code and tests; don't overwrite user changes.
4. State the exact step you will implement and acceptance criteria, then implement that step only.
5. If relevant compatibility facts are uncertain, consult upstream official docs before modifying dependencies.

## During work
- KISS: choose the simplest accurate solution, avoid speculative abstractions and unnecessary services.
- Idiomatic Python, readable descriptive snake_case names, type annotations on public APIs.
- Small, cohesive modules; aim for <=250 lines per Python file and <=50 lines per function, justify exceptions.
- Docstrings for public functions/classes stating purpose, arguments, and returns; add examples for non-obvious logic. Comments explain *why*, not what the code says.
- Prefer dataclasses/Protocols where simple; don't introduce a framework merely to solve an internal design problem.
- Never silently swallow exceptions, retry OOM forever, or substitute invalid results.
- Never fabricate hardware measurements or benchmark successes; distinguish observed, estimated, unsupported, skipped, failed.
- Keep fair comparisons: same inputs, same prompt, same token cap, warmup, repeats, sampling settings; log exceptions.
- One GPU engine at a time. Ensure resources are released before switching models.
- No unsafe destructive commands; seek approval before deleting tracked files or altering OS GPU drivers.
- Python dependencies: `uv add`, `uv add --dev`, `uv sync`, `uv run`; no pip/poetry/conda or untracked manual package installation. Engine projects under `environments/` use their own uv projects/locks.
- `ruff` lint/format and `pytest` before marking a task complete. Pin reproducible versions after compatibility validation.
- Keep user/company documents and secrets out of the benchmark; synthetic data by default. No raw financial data in git.

## Documentation workflow — mandatory
- BEFORE coding a step: ensure active `plans/phase_XX.md` has goal, tasks, risks, acceptance criteria.
- AFTER coding a step: test it, update checkboxes and evidence in that phase plan; append a timestamped entry to `PROGRESS.md` with changes/tests/results/blockers/next action.
- Update `DECISIONS.md` for tradeoffs, `ACCEPTANCE.md` for verified outcomes, `docs/ARCHITECTURE.md` for architecture changes, and `learning_docs/` for scientific observations.
- Only modify `AGENTS.md` when *long-lived agent policy* changes. Never put session notes here.
- Don't mark a phase complete based only on code existing; require verified acceptance criteria.
- On session end include a concise handoff: completed, tested, risks, next actionable step.

## Priority
Correctness + reproducibility > performance claims > dashboard polish. Work on one phase at a time; avoid overengineering. If a configuration fails, record why.
