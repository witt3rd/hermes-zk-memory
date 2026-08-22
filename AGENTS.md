# hermes-zk-memory

Zettelkasten memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A curated corpus of atomic notes — one thought per note, own words, plain markdown links —
rather than a raw transcript log. Judgment happens at **write time** (an LLM decides whether a
turn is worth a note and drafts it), not deferred entirely to recall-time ranking.

This repo is a **thin adapter** over the host-agnostic [zk-memory](https://github.com/witt3rd/zk-memory)
library (same shape as the `hermes-prospecta` / `prospecta` split). The corpus operations, the
retain pipeline, the gardener pass, and the write-time prompts/schemas all live in `zk-memory`;
this plugin constructs `zk_memory.Memory`, implements its `StructuredLLM` protocol via the
auxiliary-task forced-tool-call path (`llm.py`), and owns the Hermes-shaped surface (tool text,
threading, root/config resolution, auxiliary task registration).

## Goals

- **Thin adapter, not a fork (P9).** Delegate every substantive op to `zk_memory`. If the plugin
  starts reimplementing merge-or-create or distill orchestration, push it down into the library.
- **Live retain behavior unchanged by the split.** The adapter routes the library's
  `(messages, schema, name)` calls through the existing forced-tool-call path.

## Merits (load-bearing)

- **The same operations serve volitional and automatic motions** — `zk_search`/`zk_read`/
  `zk_write`/`zk_integrate`/`zk_tend` and `prefetch`/`sync_turn`/`on_pre_compress` all drive
  the same `zk_memory` corpus ops; "auto" is an optional convenience, not a parallel code path.
  `zk_integrate` is the careful write (merge-or-create via `Memory.integrate`), distinct from
  the fast append-only `zk_write`.
- **Append-only merge + collision-safe write** (owned by the library): a bad merge can at worst
  add a misplaced fragment, never destroy content.
- **Plugin name, tool names, receipt text, corpus format, auxiliary task key, default root are a
  compatibility contract** — do not "clean up" them to match the split.

## Concepts

- **`sync_turn`** fires off-thread after each turn and runs the library's two-stage judgment
  (`Memory.retain_turn`): distill (concept / entity_update / decision) → per-candidate
  search + `judge_merge` → write or append-only merge.
- **`on_pre_compress`** runs the same pipeline over the about-to-be-dropped batch,
  synchronously, returning a receipt string for the compressor's summary prompt.
- **`Memory.tend_writes`** is the gardener pass (recent-writes-first integration) — driven by a
  caretaker host/agent, not by this plugin's per-turn hooks.
- **The LLM is a `StructuredLLM` adapter** (`llm.py`): `build_structured_llm(provider, model)`
  returns the callable the library's retain pipeline calls — it builds the tool dict from
  `zk_memory.judge.TOOL_DESCRIPTIONS` + schema and routes the plugin's auxiliary-task
  forced-tool-call path (`hermes_structured_llm` remains as a config-reading backward-compat
  wrapper).

## Mechanisms

- **Write-time judge is config-driven, never inherited.** The being owns which LLM runs its
  judgment: `initialize()` reads the whole `auxiliary.zk_memory_judge` block — the config hermes
  already manages for the plugin's auxiliary task — and binds a `build_structured_llm` to those
  explicit values. `provider`/`model`/`base_url`/`api_key` are passed straight to
  `agent.auxiliary_client._resolve_task_provider_model` (which always-wins over config/auto);
  `timeout`/`extra_body` are forwarded to the provider call. So routing rides hermes' transport
  but the model and transport knobs are the being's, not hermes' default. Missing/incomplete
  `provider`/`model` **disables retain** (llm=None, corpus ops still work) with a clear log — we
  never silently fall back to a baked model. Future vector recall would read a sibling
  `auxiliary.zk_memory_embedding` block.
- **Import shape.** `from . import llm as _llm` (relative) — required by Hermes' plugin loader,
  which registers the provider under a synthetic `_hermes_user_memory.<name>` namespace package
  without putting the plugin dir on `sys.path`. The sibling import has a **flat-import fallback**
  (`import llm as _llm`) so the root `__init__.py` also imports under pytest, whose flat import
  has no package parent. Do not revert to a bare `import llm` only — it breaks hermes' loader.
- **Tests.** `pytest` in the repo root. `[tool.pytest.ini_options] testpaths = ["tests"]` keeps
  pytest from importing the root `__init__.py` flat as a package init. `tests/conftest.py`
  reproduces hermes' package load (`_hermes_user_memory.hermes_zk_memory` with sibling
  submodules) and puts the sibling `zk-memory` checkout on `sys.path`. Provider tests mock
  `Memory.retain_turn` / `search` or inject a stub `StructuredLLM`.
- **Pinning.** `pyproject.toml` + `plugin.yaml` depend on
  `zk-memory @ git+...@v<tag>` (both files, kept in sync). Bump by tagging zk-memory, then
  repinning both here.

## Gotchas

- **Sibling path in `conftest.py` is hardcoded** (`/home/dt/src/witt3rd/zk-memory`), NOT derived
  from `REPO_ROOT.parent` — the worktree lives under `hermes-zk-memory.wt/`, so the naive
  derivation points at a nonexistent path.
- **Operator install.** `zk-memory` is a runtime dep; a symlink-only install without
  `pip_dependencies` applied will `ImportError` on `from zk_memory import Memory`.
- **`hermes_structured_llm`** extracts `system`/`user` from the messages the library builds —
  order-independent, handles missing roles gracefully.

## House rules

- **House git (fleet_git, two modes).** **Mode 1 — active iteration:** work on
  `master`, commit small, revertable batches frequently, and never accumulate
  uncommitted work. Push to a branch + PR when a change is ready; sync with
  origin/master before starting and before a PR. **Mode 2 — parallel feature
  work:** use linked worktrees under `hermes-zk-memory.wt/<branch>/`
  (`git wt-new` / `git wt-rm`). **Clean end-state:** no stale worktrees, no
  leftover local branches beyond `master`, mainline at origin tip, primary
  clone clean.
- **AGENTS.md is the single source of truth.** `README.md` is a symlink to this file for GitHub;
  there is no separate human doc.
- **Reversible-first.** Prefer changes that are easy to revert; never leave the repo worse than
  you found it.

## Scope and audience

- **Universal:** the adapter contract, import shape, pinning, house git.
- **Caretaker:** see `skills/caretaker/` for the lived experience.

Last updated: 2026-08-18 (v0.4.0 pin).