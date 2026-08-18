# HANDOFF — hermes-zk-memory

**Last updated:** 2026-08-18 — pinned to `zk-memory@v0.4.0` (tend_writes); docs + caretaker skill added.

## State

- Primary clone on `master` at `origin/master` tip (`a7b9751`); clean (only pre-existing untracked `uv.lock`).
- `AGENTS.md` is now the single source of truth; `README.md` is a symlink to it. `skills/caretaker/` added.
- **39 tests pass** (`.venv/bin/python -m pytest`). Uses real hermes-agent at `/home/dt/.hermes/hermes-agent`; sibling `zk-memory` on `sys.path` via conftest.

## What this is

A **thin `MemoryProvider` adapter** over `witt3rd/zk-memory` (host-agnostic zettelkasten library).
Same split shape as `hermes-prospecta`/`prospecta`.

## Current surface

- `__init__.py` — `ZkMemoryProvider` constructs `Memory`, delegates every op; keeps tool text,
  threading, root/config resolution, `register_auxiliary_task`. Sibling import is relative
  `from . import llm` with a flat-import fallback (`import llm`), so it loads under both hermes'
  package loader and pytest's flat import.
- `llm.py` — the `StructuredLLM` adapter: `TASK_KEY` (`zk_memory_judge`), `_DEFAULT_PROVIDER`
  (`anthropic`), `_DEFAULT_MODEL` (`claude-sonnet-5`), `_resolve_client`, `_forced_tool_call`,
  `hermes_structured_llm(messages, *, schema, name)`.
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
