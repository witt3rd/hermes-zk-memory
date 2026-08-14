# hermes-zk-memory

Zettelkasten memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A curated corpus of atomic notes — one thought per note, own words, plain
markdown links — rather than a raw transcript log. Judgment happens at
**write time** (an LLM decides whether a turn is worth a note, and drafts it),
not deferred entirely to recall-time ranking.

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

`sync_turn` fires off-thread after each turn and runs a two-stage judgment —
no queue, no batching, no separate cron-scheduled integration pass:

1. **Distill** (one LLM call, sees only the raw turn) splits it into zero or
   more candidates, each tagged:
   - `concept` — a self-contained, evergreen idea with enough weight to stand
     alone as a new node.
   - `entity_update` — a temporal or attribute-level fact (e.g. "Judy is
     arriving in two weeks") that would be a useless orphan as its own note;
     it belongs appended to an existing entity/topic note instead.
2. **Merge-or-create**, per candidate: `zk_search` the candidate's topic (no
   LLM). No hits → straight to `zk.write()`, no LLM call spent. One or more
   hits → fetch their full bodies and make **one** comparison call across all
   of them (`judge_merge`) deciding whether the new information belongs in an
   existing note or is genuinely new. A `merge_target_ref` that doesn't match
   one of the fetched hits is never trusted — falls back to create.
3. **Write**: `zk.write()` (new note) or `zk.merge()` — **append-only**, never
   a rewrite of existing prose, so a wrong merge can at worst add a
   misplaced fragment, never destroy content. `zk.merge()` takes a
   corpus-wide flock around the append so concurrent writers (two
   `sync_turn` calls, or a volitional `zk_write` racing an automatic retain)
   can't interleave.

The distill call sees the full raw turn with no normal-path truncation (only
a very large hard safety cap) — this is why the auxiliary task defaults to a
1M-context model. See `llm.py` for the full rationale.

## Install

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
