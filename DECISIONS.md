# Architecture decision log

## ADR-001: Native Fedora, Docker optional
- **Date:** 2026-10-13
- **Decision:** Start with native uv-managed environments; no Docker requirement.
- **Reason:** Lowest initial complexity for measuring one local GPU. Docker can be introduced for incompatible serving stacks later.
- **Revisit when:** GPU kernel dependency problems repeatedly block vLLM/SGLang.

## ADR-002: One shared agent instruction file
- **Date:** 2026-10-13
- **Decision:** `AGENTS.md` canonical, `CLAUDE.md -> AGENTS.md` symlink.
- **Reason:** Codex/Claude use one source of truth; plans/progress separately tracked.

## ADR-003: Multiple isolated uv projects
- **Date:** 2026-10-13
- **Decision:** Controller root project plus engine-specific `environments/*/pyproject.toml` and independent locks.
- **Reason:** GPU serving dependencies may conflict and change rapidly.

## ADR-004: Truthful measurements
- **Date:** 2026-10-13
- **Decision:** Record incompatible, failed, skipped, and OOM configurations; never impute scores.
- **Reason:** Research credibility and reproducibility.
