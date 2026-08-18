# HANDOFF — hermes-zk-memory

**Last updated:** 2026-08-17 — split complete (PR #2 merged), dep pinned to `zk-memory@v0.1.0` (PR #3).

## State

- Primary clone on `master` at `origin/master` (`688358d`, PR #2 merged); clean (only pre-existing
  untracked `uv.lock`).
- Working branch **`feat/pin-zk-memory`** in worktree `hermes-zk-memory.wt/feat--pin-zk-memory`,
  pushed, **PR #3** open against `master` — pins `zk-memory` git dep to `@v0.1.0`.
- **39 tests pass** in the worktree (`.venv/bin/python -m pytest`). Uses the real hermes-agent at
  `/home/dt/.hermes/hermes-agent` for `agent.*` imports; sibling `zk-memory` on `sys.path` via conftest.

## What this is now

A **thin `MemoryProvider` adapter** over `witt3rd/zk-memory` (the host-agnostic zettelkasten
library). Same split shape as `hermes-prospecta`/`prospecta`.

## What changed (this split, now complete)

- **PR #2 merged** → `__init__.py` constructs `Memory(root, llm=_llm.hermes_structured_llm,
  tracer=_trace)`, delegates every substantive op; keeps only the Hermes-shaped surface (tool text,
  threading, root/config resolution, `register_auxiliary_task`). `_process_candidate` and the
  distill/merge orchestration moved to `zk_memory.retain` / `zk_memory.judge`.
- **`llm.py`** — the `StructuredLLM` adapter: `TASK_KEY`, `_DEFAULT_PROVIDER`, `_DEFAULT_MODEL`,
  `_resolve_client`, `_forced_tool_call`, `hermes_structured_llm(messages, *, schema, name)` (builds
  the tool dict from `zk_memory.judge.TOOL_DESCRIPTIONS` + schema, routes the existing
  forced-tool-call path). Live retain behavior unchanged.
- **Deleted** `zk.py`, `lancedb_fts.py`, `probe.py`, `tests/test_zk_*`, `tests/test_llm_judge.py`,
  `tests/test_probe.py` — moved to `zk-memory`.
- **`pyproject.toml` / `plugin.yaml`** — depend on `zk-memory @ git+...@v0.1.0` (and
  `pip_dependencies`); `@v0.1.0` pin via PR #3.
- **Tests** — provider + register only; mock `Memory.retain_turn` / `search` or inject a stub
  `StructuredLLM`; `conftest.py` puts the sibling `zk-memory` on `sys.path`.

## Where I left off

PR #3 (the `@v0.1.0` pin) is the only open thread. `zk-memory` is tagged `v0.1.0` and pushed; the
pin is verified to resolve (`zk-memory==0.1.0` at the tag commit). **39 plugin + 59 library tests
green.**

## Gotchas

- **Sibling path in `conftest.py` is hardcoded** (`/home/dt/src/witt3rd/zk-memory`), NOT derived
  from `REPO_ROOT.parent` — the worktree lives under `hermes-zk-memory.wt/`, so the naive derivation
  points at a nonexistent `hermes-zk-memory.wt/zk-memory`. Hardcode like the `_REAL_HERMES_CANDIDATES`.
- **Operator install**: `zk-memory` is now a runtime dep. A symlink-only install without
  `pip_dependencies` applied will `ImportError` on `from zk_memory import Memory`. Declared in
  `plugin.yaml` + README.
- **Plugin name / tool names / receipt / corpus format / auxiliary key / default root unchanged** —
  do not "clean up" these to match the new split; they're the compatibility contract.
- **`hermes_structured_llm`** extracts `system`/`user` from the messages list the library builds —
  order-independent, handles missing roles gracefully.
- **A `.venv` in the worktree** is gitignored; running tests needs `uv venv .venv` + `uv pip install
  pytest` per fresh worktree.

## Next

1. Merge PR #3 (the `@v0.1.0` pin; suite already green).
2. After merge, `git wt-rm feat/pin-zk-memory` to clean the worktree, then delete the merged remote
   branch (`git push origin --delete feat/pin-zk-memory`).
3. If an install break appears later, document the new pip dep in the being-plugin / fleet installers.