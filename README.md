# hermes-zk-memory

Zettelkasten memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent).

A curated corpus of atomic notes — one thought per note, own words, plain
markdown links — rather than a raw transcript log. Judgment happens at
**write time** (an LLM decides whether a turn is worth a note, and drafts it),
not deferred entirely to recall-time ranking.

Status: scaffold. Implementation in progress — see the plugin's `__init__.py`
for the current state of the `MemoryProvider` surface.

## Design

The same operations serve both the volitional tool surface and the automatic
recall/retain motions — "auto" is an optional convenience layered on the same
underlying corpus operations, not a separate code path:

| Operation | Volitional tool | Automatic motion |
|---|---|---|
| Search the corpus | `zk_search` | `prefetch()` (auto-recall, before each turn) |
| Write a new note | `zk_write` | `sync_turn()` (auto-retain, after each turn — gated on an LLM judging the turn worth a note) |
| Read one note | `zk_read` | — |
| Tend the garden (repair/check/mint) | `zk_tend` | — |

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
