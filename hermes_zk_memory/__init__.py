"""hermes-zk-memory — a Hermes MemoryProvider wrapping zk-memory.

Thin adapter (P9): the zettelkasten corpus logic, the retain pipeline,
and the write-time prompts/schemas all live in the host-agnostic
``zk_memory`` library. This plugin constructs ``zk_memory.Memory``,
implements the ``StructuredLLM`` protocol via the auxiliary-task
forced-tool-call path (``llm.py``), and owns the Hermes-shaped surface:
tool text formatting, threading, config/root resolution, and the
auxiliary-task registration.

One corpus of atomic notes. The same operations back both the volitional
tool surface (get_tool_schemas/handle_tool_call) and the automatic
recall/retain motions (prefetch/sync_turn) — "auto" is an optional
convenience over the same underlying corpus operations, not a parallel
code path.

sync_turn's write-time judgment is a forced-tool-call LLM invocation
(zk_memory.judge) routed through the plugin's own registered auxiliary
task ("zk_memory_judge") — see llm.py for why this is the hermes-native
mechanism (not ctx.llm, which memory-provider plugins can't reach).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider

# Lazy-loaded imports — zk_memory library is NOT imported at scope.
# The official hermes memory plugins (hindsight, etc.) work because they
# only import from agent.memory_provider and their own local modules at
# scope. The zk_memory library is imported lazily inside methods to avoid
# the loading-failure that occurs when the module is loaded under hermes'
# synthetic namespace (_user_namespace.<name>) before the library is
# importable. Aligns with hindsight's pattern.
import importlib

def _zk_memory():  # noqa: E302
    """Lazy import of the zk_memory library, cached after first call."""
    try:
        return importlib.import_module("zk_memory")
    except ImportError:
        raise ImportError("zk-memory library not installed; install with: pip install zk-memory[lancedb]")

def _zk_trace():  # noqa: E302
    """Lazy import of zk_memory.probe.trace, cached after first call."""
    return _zk_memory().probe.trace

# Relative (not bare) sibling import: hermes loads a user-installed memory
# provider under the synthetic namespace _hermes_user_memory.<name> and
# registers its siblings as submodules, but does NOT put the provider's own
# directory on sys.path. Bare `import llm` raises ModuleNotFoundError under
# that loader, so no provider instance is created and recall/retain silently
# never fire (caught by the being-plugin Behavioral rig). `from . import llm`
# resolves against the package hermes registers.
#
# A flat import (e.g. pytest importing this root __init__.py directly, with
# no package parent) has no relative sibling to resolve — fall back to the
# absolute sibling import there, so the module stays importable in both
# contexts.
try:
    from . import llm as _llm
except ImportError:
    import llm as _llm  # noqa: E402

logger = logging.getLogger(__name__)


class ZkMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider wrapping zk_memory.Memory."""

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._memory = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "zk-memory"

    def is_available(self) -> bool:
        # No network calls; the corpus is plain files, rg fallback needs
        # no extra dependency. Always available (zk_memory importable).
        try:
            import zk_memory  # noqa: F401
        except ImportError:
            return False
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME")
        if not hermes_home:
            raise RuntimeError("HERMES_HOME not provided; cannot resolve corpus root")
        # Default corpus root: HERMES_HOME/zk (backwards compatible).
        # Being-plugin writes memory.zk_corpus_root to the profile's config.yaml
        # during provisioning. If present, use that path instead.
        self._root = Path(hermes_home) / "zk"
        llm = None
        provider = model = None
        try:
            from hermes_cli.config import load_config_readonly
            cfg = load_config_readonly()
            memory_cfg = cfg.get("memory") or {}
            zk_root_rel = memory_cfg.get("zk_corpus_root")
            if zk_root_rel:
                self._root = Path(hermes_home) / zk_root_rel
            zk_judge = memory_cfg.get("zk_judge") or {}
            provider = str(zk_judge.get("provider", "")).strip() or None
            model = str(zk_judge.get("model", "")).strip() or None
        except Exception:
            pass  # fallback to default root; judge config stays unset
        if provider and model:
            llm = _llm.build_structured_llm(provider, model)
        else:
            # The being owns which LLM runs its write-time judgment; without an
            # explicit provider/model we do NOT inherit hermes' default routing
            # -- retain disables (corpus ops still work).
            logger.warning(
                "zk-memory: memory.zk_judge.provider/model not configured in the "
                "profile config; write-time retain disabled (corpus ops still work)"
            )
        from zk_memory import Memory as _Memory
        self._memory = _Memory(
            root=self._root,
            llm=llm,
            tracer=_zk_trace(),
            # Beings run on a git-backed filesystem (rg backend reads live repo
            # files — single-writer, unsafe to share, races concurrent writes).
            # Default to LanceDB FTS for shared-safe recall; operator can force
            # rg/auto via ZK_MEMORY_BACKEND.
            backend=os.environ.get("ZK_MEMORY_BACKEND", "").strip() or "fts",
        )
        _zk_trace()("initialized", self._root, session_id=session_id)

    def shutdown(self) -> None:
        t = self._sync_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)
        _zk_trace()("shutdown", self._root)

    # ---- volitional tool surface -------------------------------------
    #
    # These four schemas/handlers ARE the automatic recall/retain motions
    # below, just invoked by the agent directly instead of by the
    # per-turn hermes hooks.

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [
                {
                    "name": "zk_search",
                    "description": (
                        "Search the being's zettelkasten — the deep well of recorded "
                        "lived experience (decisions, people, seams, open questions). "
                        "Takes a query and returns ranked note hits (title + snippet). "
                        "Recall first: before answering from assumptions, search what "
                        "the corpus already holds."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "What to find in the corpus."},
                            "limit": {"type": "integer", "description": "Max hits (default 8)."},
                        },
                        "required": ["query"],
                    },
                },
                {
                    "name": "zk_read",
                    "description": (
                        "Read one zettel in full from the being's corpus. Reference by "
                        "uuid (the canonical anchor), slug (filename stem), or path. "
                        "Links to other notes are resolved and returned so you can "
                        "follow the thread."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string", "description": "The note's uuid, slug, or filename."},
                        },
                        "required": ["ref"],
                    },
                },
                {
                    "name": "zk_write",
                    "description": (
                        "Set down a new zettel in the corpus — one atomic thought, own "
                        "words, plain-markdown links to what it connects to. The uuid is "
                        "minted via linlink; naming is YYYYMMDD-slug.md (flat, no "
                        "subdirectories). When a thought lands that isn't yet held, this "
                        "is how the mind grows."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slug": {"type": "string", "description": "Short hyphenated slug for the note (without date prefix)."},
                            "title": {"type": "string", "description": "Human title for the note."},
                            "body": {"type": "string", "description": "The note body: ONE atomic thought, own words. Link to related notes with [label](slug.md)."},
                        },
                        "required": ["slug", "title", "body"],
                    },
                },
                {
                    "name": "zk_tend",
                    "description": (
                        "Tend the zettelkasten garden: heals moved/renamed references "
                        "(repair), checks integrity (check), or mints missing uuids "
                        "(mint). Run check after any reorg; repair when links break."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["repair", "check", "mint"],
                                "description": "The linlink maintenance action to run.",
                            },
                        },
                        "required": ["action"],
                    },
                },
            ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if self._memory is None:
            return "error: provider not initialized"
        args = args or {}
        try:
            if tool_name == "zk_search":
                result = self._search_text(args.get("query", ""), int(args.get("limit", 8)))
            elif tool_name == "zk_read":
                result = self._read_text(args.get("ref", ""))
            elif tool_name == "zk_write":
                result = self._write_text(args.get("slug", ""), args.get("title", ""), args.get("body", ""))
            elif tool_name == "zk_tend":
                result = self._tend_text(args.get("action", ""))
            else:
                _zk_trace()("tool_call", self._root, tool=tool_name, ok=False, reason="unknown_tool")
                return f"error: unknown tool: {tool_name}"
            _zk_trace()("tool_call", self._root, tool=tool_name, ok=not result.startswith("error"))
            return result
        except Exception as e:
            logger.warning("zk-memory tool %s failed: %s", tool_name, e)
            _zk_trace()("tool_call", self._root, tool=tool_name, ok=False, error=str(e))
            return f"error: {e}"

    def _search_text(self, query: str, limit: int) -> str:
        if not query:
            return "error: query is required"
        assert self._memory is not None
        hits = self._memory.search(query, limit=limit)
        if not hits:
            return f"no notes found for {query!r}"
        lines = [f"Found {len(hits)} note(s) for {query!r}:"]
        for h in hits:
            title = h.get("title") or h.get("slug", "?")
            path = h.get("path", "")
            snippet = (h.get("snippet") or "").strip()
            lines.append(f"- {title}  ({path})  [ref: {h.get('uuid') or h.get('slug')}]")
            if snippet:
                lines.append(f"    {snippet[:160]}")
        return "\n".join(lines)

    def _read_text(self, ref: str) -> str:
        if not ref:
            return "error: ref is required (uuid, slug, or path)"
        assert self._memory is not None
        result = self._memory.read(ref)
        if not result["found"]:
            return f"no note found for ref {ref!r}"
        note = result["note"]
        body = note.get("body", "").strip()
        head = f"# {note.get('title', note.get('slug'))}  [{note.get('uuid', '')}]  ({note.get('path')})"
        out = [head, "", body]
        if result["links"]:
            out.append("")
            out.append("links:")
            for l in result["links"]:
                mark = "→" if l["resolved"] else "✗"
                out.append(f"  {mark} {l['label']} -> {l['ref']}" + (f" ({l['title']})" if l["resolved"] else " (missing)"))
        return "\n".join(out)

    def _write_text(self, slug: str, title: str, body: str) -> str:
        slug, title, body = slug.strip(), title.strip(), body.strip()
        if not slug or not title or not body:
            return "error: slug, title, and body are all required"
        assert self._memory is not None
        result = self._memory.write(slug, title, body)
        if result.get("ok"):
            return f"zettel written: {result['path']}  uuid={result.get('uuid', '')}"
        return f"error: {result.get('err', 'write failed')}"

    def _tend_text(self, action: str) -> str:
        if action not in ("repair", "check", "mint"):
            return "error: action must be one of repair, check, mint"
        assert self._memory is not None
        result = self._memory.tend(action)
        head = "ok" if result.get("ok") else "FAILED"
        out = result.get("output") or result.get("err") or ""
        return f"zk_tend {action}: {head}\n{out.strip()[:1000]}"

    # ---- automatic motions ---------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Auto-recall: search the corpus, fence the hits for the model.

        Best-effort: never raises, failures return "".
        """
        if self._memory is None or not query:
            return ""
        try:
            hits = self._memory.search(query, limit=5)
        except Exception as e:
            logger.warning("zk-memory prefetch failed: %s", e)
            _zk_trace()("prefetch", self._root, query=query, ok=False, error=str(e))
            return ""
        _zk_trace()("prefetch", self._root, query=query, ok=True, hits=len(hits))
        if not hits:
            return ""
        lines = [f'<recall query="{query}">']
        for h in hits:
            title = h.get("title", h.get("slug", "?"))
            slug = h.get("slug", "")
            snippet = (h.get("snippet") or "").strip()[:200]
            lines.append(f"  [{slug}] {title}")
            if snippet:
                lines.append(f"    {snippet}")
        lines.append("</recall>")
        return "\n".join(lines)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Auto-retain: off-thread, delegates the two-stage LLM judgment
        to ``self._memory.retain_turn`` (the library's pipeline).

        Fires on a daemon thread and returns immediately; any prior
        in-flight sync is bounded-joined first so writes stay ordered.
        """
        if self._memory is None:
            return

        def _run() -> None:
            try:
                self._memory.retain_turn(user_content, assistant_content, session_id=session_id)
            except Exception:
                logger.warning("zk-memory: sync_turn failed", exc_info=True)
                _zk_trace()("sync_turn_failed", self._root, session_id=session_id)

        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="zk-memory-sync")
        self._sync_thread.start()

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called by hermes immediately before context compression
        discards ``messages`` (the middle window about to be
        summarized away).

        Runs the same distill-then-merge-or-create judgment sync_turn
        uses (via ``self._memory.retain_messages``), scoped to exactly
        this about-to-be-dropped batch. This closes the gap
        docs/memory.md §6.8.2 flagged: without it, nothing promotes what
        compaction is about to drop into the zettelkasten specifically
        at the compaction boundary — retain relied entirely on
        sync_turn's per-turn cadence already having caught up, which a
        burst of turns racing sync_turn's background thread could outrun.

        Runs SYNCHRONOUSLY, unlike sync_turn: the caller needs the
        return value before compress() proceeds, and this fires rarely
        (once per compaction event), so the extra latency is
        proportionate to the cost compaction already pays for its own
        summarization call.

        Known tradeoff: if sync_turn already processed these same turns
        (the common case), this re-distills the same content — the merge
        judge should usually recognize an already-written near-duplicate
        and merge into it rather than create a new note, but a redundant
        append is possible. Accepted as a rare, benign cost rather than
        engineering de-duplication against sync_turn's already-processed
        turns.

        Returns a short receipt threaded into the compressor's summary
        prompt (memory_context) — the retention itself already happened
        as a side effect by the time this returns; the receipt is a bonus
        hint for the summarizer, not the mechanism. Never raises; returns
        "" on any failure or nothing to retain.
        """
        if self._memory is None or not messages:
            return ""
        try:
            titles = self._memory.retain_messages(messages)
            if not titles:
                return ""
            return (
                "[zk-memory] Retained to the zettelkasten before this "
                "compaction: " + ", ".join(titles)
            )
        except Exception:
            logger.warning("zk-memory: on_pre_compress failed", exc_info=True)
            _zk_trace()("pre_compress_failed", self._root, n_messages=len(messages))
            return ""


def register(ctx) -> None:
    """Called by Hermes memory plugin discovery."""
    # No default provider/model here: the being owns which LLM runs its
    # write-time judgment via config.yaml (memory.zk_judge.provider/model).
    # The auxiliary task is registered for hermes' accounting/routing/UI
    # surface, but routing is explicit per-call from the plugin's config --
    # never inherited from hermes' default model.
    ctx.register_auxiliary_task(
        key=_llm.TASK_KEY,
        display_name="ZK memory write-time judge",
        description="Distills turns into zettel candidates and judges merge-vs-create.",
    )
    ctx.register_memory_provider(ZkMemoryProvider())
    # No corpus root yet at register() time (that's resolved per-session in
    # initialize()) -- log-only, no trace file write.
    _zk_trace()("registered", None, task_key=_llm.TASK_KEY)