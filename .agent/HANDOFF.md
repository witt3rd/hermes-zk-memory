# HANDOFF — hermes-zk-memory

**Last updated:** 2026-08-19 — judge LLM is now config-driven (`memory.zk_judge.provider/model`); baked anthropic default removed.

## State

- Primary clone on `master` at `origin/master` tip (`a7b9751`); clean (only pre-existing untracked `uv.lock`).
- `AGENTS.md` is now the single source of truth; `README.md` is a symlink to it. `skills/caretaker/` added.
- **43 tests pass** (`.venv/bin/python -m pytest`). Uses real hermes-agent at `/home/dt/.hermes/hermes-agent`; sibling `zk-memory` on `sys.path` via conftest.

## What this is

A **thin `MemoryProvider` adapter** over `witt3rd/zk-memory` (host-agnostic zettelkasten library).
Same split shape as `hermes-prospecta`/`prospecta`.

## Current surface

- `__init__.py` — `ZkMemoryProvider` constructs `Memory`, delegates every op; keeps tool text,
  threading, root/config resolution, `register_auxiliary_task`. Sibling import is relative
  `from . import llm` with a flat-import fallback (`import llm`), so it loads under both hermes'
  package loader and pytest's flat import.
- `llm.py` — the `StructuredLLM` adapter: `TASK_KEY` (`zk_memory_judge`),
  `build_structured_llm(provider, model)` (the bound callable the plugin passes to `Memory`),
  `hermes_structured_llm(messages, *, schema, name)` (config-reading backward-compat wrapper),
  `_resolve_client(provider=None, model=None)`, `_forced_tool_call`. No baked provider/model —
  the being's `memory.zk_judge` config supplies them explicitly.
- `__init__.py` `initialize()` reads `memory.zk_judge.provider/model` from the profile
  config.yaml; missing/incomplete config disables retain (llm=None) with a clear log, never
  falls back to a hermes default model.
- Deps: `pyproject.toml` (`[lancedb]`) + `plugin.yaml` both pinned to `zk-memory @ ...@v0.4.0`.

## Version-pin history

The plugin follows zk-memory tags: `@v0.1.0` → `@v0.2.0` → `@v0.3.0` → `@v0.4.0`. **Keep both
`pyproject.toml` and `plugin.yaml` in sync** — a past release left `plugin.yaml` stale.

## Gotchas

- **`tests/conftest.py` sibling path is hardcoded** (`/home/dt/src/witt3rd/zk-memory`), not derived from `REPO_ROOT.parent` (worktree lives under `hermes-zk-memory.wt/`).
- **Operator install** needs `zk-memory` importable; symlink-only installs `ImportError`.
- **Import shape is load-bearing** — bare `import llm` fails under hermes' loader (silently); keep the relative + fallback shape.
- Backticks in `git commit -m` get shell-substituted — use `-F file`.

## Next

1. Optional: register `skills/caretaker/` in fleet-ops (`~/.agents/skills/hermes-zk-memory` → repo `skills/`).
2. If the fleet shared-corpus deployment lands, wire `Memory.tend_writes` to a caretaker host/agent and document the `zk-memory` pip dep in being-plugin / fleet installers.
