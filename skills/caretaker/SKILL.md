---
name: caretaker
description: Lived-experience caretaker skill for the hermes-zk-memory plugin repo. Use for ANY work here — the thin adapter over zk-memory, the StructuredLLM adapter, the Hermes package-load import shape, pinning the zk-memory dep, test discipline, and house git. Triggers: hermes-zk-memory, zk-memory provider, zk_memory_judge, memory provider, repin zk-memory.
metadata:
  home: /home/dt/src/witt3rd/hermes-zk-memory
  scope: repo
---

# hermes-zk-memory — caretaker

Lived experience for stewarding `~/src/witt3rd/hermes-zk-memory`, the Hermes `MemoryProvider`
adapter over the host-agnostic `zk-memory` library. The **charter is `AGENTS.md`** — read it
first. This skill is how the caretaker acts.

## What this is

A **thin adapter** (same shape as `hermes-prospecta`/`prospecta`). All corpus logic, the retain
pipeline, the gardener pass, and the write-time prompts/schemas live in the sibling
`zk-memory` repo (`/home/dt/src/witt3rd/zk-memory`); this plugin owns the Hermes-shaped surface.

## Package map

```
__init__.py   # ZkMemoryProvider: constructs Memory, delegates; tool text, threading,
              #   root/config resolution, register_auxiliary_task
llm.py        # the StructuredLLM adapter: TASK_KEY, _resolve_client, _forced_tool_call,
              #   hermes_structured_llm(messages, *, schema, name)
plugin.yaml   # plugin metadata + pip_dependencies (zk-memory @ git+...@v<tag>)
pyproject.toml# package + the same zk-memory git dep (keep in sync with plugin.yaml)
tests/        # provider + register; conftest loads the plugin the way hermes does
```

## The import shape (non-obvious, load-bearing)

Hermes loads a user-installed memory provider under the synthetic namespace
`_hermes_user_memory.<name>` and registers its siblings as submodules, but does NOT put the
plugin's own dir on `sys.path`. So **bare `import llm` fails under hermes** (silently — no
provider instance, recall/retain never fire; caught by the being-plugin Behavioral rig).

- `__init__.py` must use the **relative** `from . import llm as _llm`.
- It also carries a **flat-import fallback** (`import llm as _llm`) so pytest's flat import of
  the root `__init__.py` (no package parent) works.
- `tests/conftest.py` reproduces hermes' package load (`_hermes_user_memory.hermes_zk_memory`
  + sibling submodules) and `[tool.pytest.ini_options] testpaths=["tests"]` keeps pytest from
  importing the root `__init__.py` flat.

## Pinning / releasing

Version bumps are a three-step dance:
1. Land + tag the library (`zk-memory` `vX.Y.Z`).
2. Bump **both** `pyproject.toml` and `plugin.yaml` to `@vX.Y.Z` (they must stay in sync — a
   past release left `plugin.yaml` stale at `@v0.1.0` when pyproject went to `@v0.2.0`).
3. Verify the dep resolves (`uv pip install --dry-run ...@vX.Y.Z`) and run the suite green, then
   PR + merge + `git wt-rm`.

## Gotchas

- **Sibling path in `tests/conftest.py` is hardcoded** (`/home/dt/src/witt3rd/zk-memory`), not
  derived from `REPO_ROOT.parent` — the worktree lives under `hermes-zk-memory.wt/`.
- **Operator install** needs `zk-memory` importable (symlink-only installs `ImportError`).
- **`hermes_structured_llm`** pulls `system`/`user` out of the library-built messages,
  order-independent.
- Backticks in `git commit -m` get shell-substituted — write the message to a file (`-F`).

## House git + caretaker loop

- Worktrees only (`git wt-new <branch>`); primary clone stays on `master`.
- Orient via `.agent/HANDOFF.md`; leave the repo at the clean end-state (no stale worktrees,
  no leftover branches, master at origin tip); write the HANDOFF on sleep.

## References

- `AGENTS.md` — charter. Sibling: the `zk-memory` caretaker skill + `caretaker` machine skill.
