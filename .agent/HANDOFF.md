# HANDOFF — hermes-zk-memory

**Last updated:** 2026-08-17 — split into host-agnostic `zk-memory` + thin adapter (PR #2).

## State

- Primary clone on `master` at `origin/master` (clean; only untracked `uv.lock`).
- Working branch **`feat/wrap-zk-memory`** in worktree `hermes-zk-memory.wt/feat--wrap-zk-memory`,
  pushed, **PR #2** open against `master`. Not yet merged (by design — see below).
- **39 tests pass** in the worktree (`.venv/bin/python -m pytest`). Uses the real hermes-agent at
  `/home/dt/.hermes/hermes-agent` for `agent.*` imports; sibling `zk-memory` on `sys.path` via conftest.

## What this is now

A **thin `MemoryProvider` adapter** over `witt3rd/zk-memory` (the host-agnostic zettelkasten
library). Same split shape as `hermes-prospecta`/`prospecta`.

## What changed

- **`__init__.py`** — constructs `Memory(root, llm=_llm.hermes_structured_llm, tracer=_trace)`,
  delegates every substantive op; keeps only the Hermes-shaped surface: tool text formatting,
  threading (`sync_turn` daemon thread + bounded-join), root/config resolution (`HERMES_HOME/zk`,
  `memory.zk_corpus_root`), and `register_auxiliary_task`. `_process_candidate` and the
  distill/merge orchestration **left the plugin** → `zk_memory.retain` / `zk_memory.judge`.
- **`llm.py`** — reduced to the `StructuredLLM` adapter: `TASK_KEY`, `_DEFAULT_PROVIDER`,
  `_DEFAULT_MODEL`, `_resolve_client`, `_forced_tool_call`, and `hermes_structured_llm(messages, *,
  schema, name)` (builds the tool dict from `zk_memory.judge.TOOL_DESCRIPTIONS` + schema, routes the
  existing forced-tool-call path). Live retain behavior unchanged.
- **Deleted** `zk.py`, `lancedb_fts.py`, `probe.py`, `tests/test_zk_*`, `tests/test_llm_judge.py`,
  `tests/test_probe.py` — moved to `zk-memory`.
- **`pyproject.toml` / `plugin.yaml`** — depend on `zk-memory @ git+...` (and `pip_dependencies`).
- **Tests** — provider + register only; mock `Memory.retain_turn` / `search` or inject a stub
  `StructuredLLM`; `conftest.py` puts the sibling `zk-memory` on `sys.path`.

## Where I left off

PR #2 is the deliverable. **Do not merge until the plugin suite is green against the sibling
library** — it is (39 plugin + 59 library, both green locally). The `zk-memory` repo landed first so
the plugin's git dep resolves.

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

## Next

1. Merge PR #2 (plugin suite green against `zk-memory`).
2. After merge, `git wt-rm feat/wrap-zk-memory` to clean the worktree.
3. Optional: pin the `zk-memory` git dep to `@v0.1.0` once that tag exists.
4. If an install break appears later, document the new pip dep in the being-plugin / fleet installers.