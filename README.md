# hermes-zk-memory

Zettelkasten memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A curated corpus of atomic notes — one thought per note, own words, plain
markdown links — rather than a raw transcript log. Judgment happens at
**write time** (an LLM decides whether a turn is worth a note, and drafts it),
not deferred entirely to recall-time ranking.

This repo is a **thin adapter** over the host-agnostic
[zk-memory](https://github.com/witt3rd/zk-memory) library (same shape as the
`hermes-prospecta` / `prospecta` split). The corpus operations, the retain
pipeline, and the write-time prompts/schemas all live in `zk-memory`; this
plugin constructs `zk_memory.Memory`, implements its `StructuredLLM` protocol
via the auxiliary-task forced-tool-call path (`llm.py`), and owns the
Hermes-shaped surface (tool text, threading, root/config resolution, auxiliary
task registration).

## Design

The same operations serve both the volitional tool surface and the automatic
recall/retain motions — "auto" is an optional convenience layered on the same
underlying corpus operations, not a separate code path:

| Operation | Volitional tool | Automatic motion |
|---|---|---|
| Search the corpus | `zk_search` | `prefetch()` (auto-recall, before each turn) |
| Write / merge a note | `zk_write` | `sync_turn()` (auto-retain, after each turn) |
| Read one note | `zk_read` | — |
| Tend the garden (repair/check/mint) | `zk_tend` | — |

### Auto-retain: distill, then merge-or-create

`sync_turn` fires off-thread after each turn and runs the library's two-stage
judgment (`zk_memory.retain.retain_turn`) — no queue, no batching, no separate
cron-scheduled integration pass:

1. **Distill** (one LLM call, sees only the raw turn) splits it into zero or
   more candidates, each tagged:
   - `concept` — a self-contained, evergreen idea with enough weight to stand
     alone as a new node.
   - `entity_update` — a temporal or attribute-level fact (e.g. "Judy is
     arriving in two weeks") that would be a useless orphan as its own note;
     it belongs appended to an existing entity/topic note instead.
2. **Merge-or-create**, per candidate: search the candidate's topic (no
   LLM). No hits → straight to create, no LLM call spent. One or more hits →
   fetch their full bodies and make **one** comparison call across all of
   them (`judge_merge`) deciding whether the new information belongs in an
   existing note or is genuinely new. A `merge_target_ref` that doesn't match
   one of the fetched hits is never trusted — falls back to create.
3. **Write**: new note, or `merge` — **append-only**, never a rewrite of
   existing prose, so a wrong merge can at worst add a misplaced fragment,
   never destroy content. `merge` takes a corpus-wide flock around the append
   so concurrent writers can't interleave.

### `on_pre_compress`: promoting what compaction is about to drop

Hermes calls `MemoryProvider.on_pre_compress(messages)` immediately before
context compaction discards a batch of messages. This runs the same
distill-then-merge-or-create judgment as `sync_turn`, scoped to exactly that
batch, **synchronously** (the caller needs the return value before
compaction proceeds) — so compaction gets a targeted chance to promote what
it's about to drop, on top of whatever `sync_turn`'s per-turn cadence already
caught.

## Diagnostics

Every module logs failures via the standard `logging` module (`logger.warning`,
usually with a traceback). For the success path — what actually happened, not
just what broke — every retain/recall decision calls `zk_memory.probe.trace`,
which:

- always logs at INFO (`zk-memory trace: <event> {...fields}`)
- appends one JSON line to `$HERMES_HOME/.zk-memory-trace.jsonl` (sibling to
  the corpus, same convention as the LanceDB index cache — derived/diagnostic
  artifacts live beside `zk/`, never inside it)

Traced events: `registered`, `initialized`, `shutdown`, `tool_call` (every
`zk_search`/`zk_read`/`zk_write`/`zk_tend` invocation), `prefetch` (query +
hit count), `sync_turn_distilled` (candidate count), `candidate_decision`
(merge vs. create vs. skipped, per candidate), `pre_compress` (message +
candidate counts). Tail the trace file to watch retention decisions live:

```bash
tail -f "$HERMES_HOME/.zk-memory-trace.jsonl" | jq .
```

## Install

`zk-memory` is a runtime dependency (declared via `plugin.yaml`
`pip_dependencies`). Install it in the Hermes env:

```bash
pip install 'zk-memory @ git+https://github.com/witt3rd/zk-memory.git'
# optional rich recall:
pip install 'zk-memory[lancedb] @ git+https://github.com/witt3rd/zk-memory.git'
```

Then symlink this plugin into the Hermes install:

```bash
git clone https://github.com/witt3rd/hermes-zk-memory /path/to/hermes-zk-memory
ln -s /path/to/hermes-zk-memory $HERMES_HOME/plugins/memory/zk-memory
```

Activate by setting, in `config.yaml`:

```yaml
memory:
  provider: "zk-memory"
```

This is independent of `memory.memory_enabled` / `memory.user_profile_enabled`,
which only gate Hermes's own built-in MEMORY.md/USER.md store.

## License

MIT. See LICENSE.