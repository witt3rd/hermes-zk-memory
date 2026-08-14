"""hermes-zk-memory — a zettelkasten-backed Hermes MemoryProvider.

One corpus of atomic notes. The same operations back both the volitional
tool surface (get_tool_schemas/handle_tool_call) and the automatic
recall/retain motions (prefetch/sync_turn) — "auto" is an optional
convenience over the same underlying corpus operations, not a parallel
code path.

sync_turn's write-time judgment is one forced-tool-call LLM invocation
(llm.judge_turn) routed through the plugin's own registered auxiliary
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

import zk
import llm as _llm

logger = logging.getLogger(__name__)


class ZkMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider wrapping a zettelkasten corpus."""

    def __init__(self) -> None:
        self._root: Optional[Path] = None
        self._sync_thread: Optional[threading.Thread] = None

    @property
    def name(self) -> str:
        return "zk-memory"

    def is_available(self) -> bool:
        # No network calls; the corpus is plain files, rg fallback needs
        # no extra dependency. Always available.
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        hermes_home = kwargs.get("hermes_home") or os.environ.get("HERMES_HOME")
        if not hermes_home:
            raise RuntimeError("HERMES_HOME not provided; cannot resolve corpus root")
        self._root = Path(hermes_home) / "zk"

    def shutdown(self) -> None:
        t = self._sync_thread
        if t is not None and t.is_alive():
            t.join(timeout=5.0)

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
        if self._root is None:
            return "error: provider not initialized"
        args = args or {}
        try:
            if tool_name == "zk_search":
                return self._search_text(args.get("query", ""), int(args.get("limit", 8)))
            if tool_name == "zk_read":
                return self._read_text(args.get("ref", ""))
            if tool_name == "zk_write":
                return self._write_text(args.get("slug", ""), args.get("title", ""), args.get("body", ""))
            if tool_name == "zk_tend":
                return self._tend_text(args.get("action", ""))
            return f"error: unknown tool: {tool_name}"
        except Exception as e:
            logger.warning("zk-memory tool %s failed: %s", tool_name, e)
            return f"error: {e}"

    def _search_text(self, query: str, limit: int) -> str:
        if not query:
            return "error: query is required"
        hits = zk.search(query, self._root, limit=limit)
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
        result = zk.read(ref, self._root)
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
        result = zk.write(slug, title, body, self._root)
        if result.get("ok"):
            return f"zettel written: {result['path']}  uuid={result.get('uuid', '')}"
        return f"error: {result.get('err', 'write failed')}"

    def _tend_text(self, action: str) -> str:
        if action not in ("repair", "check", "mint"):
            return "error: action must be one of repair, check, mint"
        result = zk.tend(action, self._root)
        head = "ok" if result.get("ok") else "FAILED"
        out = result.get("output") or result.get("err") or ""
        return f"zk_tend {action}: {head}\n{out.strip()[:1000]}"

    # ---- automatic motions ---------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Auto-recall: search the corpus, fence the hits for the model.

        Best-effort: never raises, failures return "".
        """
        if self._root is None or not query:
            return ""
        try:
            hits = zk.search(query, self._root, limit=5)
        except Exception as e:
            logger.warning("zk-memory prefetch failed: %s", e)
            return ""
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
        """Auto-retain: off-thread, two-stage LLM judgment.

        1. distill_turn — no corpus visibility — splits the turn into
           zero or more candidates, each a "concept" (self-contained new
           node) or an "entity_update" (a fact fragment that belongs on
           an existing note, not standalone).
        2. Per candidate: zk.search() its topic (no LLM). If there are
           hits, judge_merge() compares the candidate against ALL fetched
           hit bodies in one call and decides merge-into-existing vs.
           create-new. No hits -> straight to create, no LLM call spent.
        3. Write: zk.merge() (append-only) or zk.write() (new note).

        No queue, no batching, no second cron-scheduled pass: everything
        the old extractor+integrator pipeline did collapses into this
        one off-thread run per turn.

        Fires on a daemon thread and returns immediately; any prior
        in-flight sync is bounded-joined first so writes stay ordered.
        """
        if self._root is None:
            return

        def _run() -> None:
            try:
                candidates = _llm.distill_turn(user_content, assistant_content)
                for candidate in candidates:
                    self._process_candidate(candidate)
            except Exception:
                logger.warning("zk-memory: sync_turn failed", exc_info=True)

        if self._sync_thread is not None and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(target=_run, daemon=True, name="zk-memory-sync")
        self._sync_thread.start()

    def _process_candidate(self, candidate: Dict[str, Any]) -> None:
        """Route one distilled candidate: merge into an existing note if
        the merge judge picks one of the search hits, otherwise create a
        new note. Never raises — failures are logged and skipped so one
        bad candidate doesn't drop the rest of the turn's candidates.
        """
        try:
            topic = (candidate.get("topic") or candidate.get("title") or "").strip()
            hits = zk.search(topic, self._root, limit=3) if topic else []

            target_ref = None
            if hits:
                notes = []
                for h in hits:
                    ref = h.get("uuid") or h.get("slug")
                    if not ref:
                        continue
                    result = zk.read(ref, self._root, resolve_links=False)
                    if result["found"]:
                        notes.append(result["note"])
                if notes:
                    decision = _llm.judge_merge(candidate, notes)
                    if decision and decision.get("action") == "merge":
                        candidate_ref = (decision.get("merge_target_ref") or "").strip()
                        valid_refs = {n.get("uuid") for n in notes if n.get("uuid")}
                        if candidate_ref and candidate_ref in valid_refs:
                            target_ref = candidate_ref
                        elif candidate_ref:
                            logger.warning(
                                "zk-memory: merge_target_ref %r not among fetched hits; falling back to create",
                                candidate_ref,
                            )

            if target_ref:
                content = (candidate.get("content") or "").strip()
                if not content:
                    return
                result = zk.merge(target_ref, content, self._root)
                if not result.get("ok"):
                    logger.warning("zk-memory: merge failed: %s", result.get("err"))
                return

            slug = (candidate.get("slug") or "").strip()
            title = (candidate.get("title") or "").strip()
            content = (candidate.get("content") or "").strip()
            if not slug or not title or not content:
                logger.warning("zk-memory: candidate incomplete (slug/title/content); skipping")
                return
            result = zk.write(slug, title, content, self._root)
            if not result.get("ok"):
                logger.warning("zk-memory: create failed: %s", result.get("err"))
        except Exception:
            logger.warning("zk-memory: candidate processing failed", exc_info=True)

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Called by hermes immediately before context compression
        discards ``messages`` (the middle window about to be
        summarized away).

        Runs the same distill-then-merge-or-create judgment sync_turn
        uses (via _llm.distill_messages / self._process_candidate),
        scoped to exactly this about-to-be-dropped batch. This closes
        the gap docs/memory.md §6.8.2 flagged: without it, nothing
        promotes what compaction is about to drop into the
        zettelkasten specifically at the compaction boundary — retain
        relied entirely on sync_turn's per-turn cadence already having
        caught up, which a burst of turns racing sync_turn's
        background thread could outrun.

        Runs SYNCHRONOUSLY, unlike sync_turn: the caller needs the
        return value before compress() proceeds, and this fires
        rarely (once per compaction event), so the extra latency is
        proportionate to the cost compaction already pays for its own
        summarization call.

        Known tradeoff: if sync_turn already processed these same
        turns (the common case), this re-distills the same content —
        the merge judge should usually recognize an already-written
        near-duplicate and merge into it rather than create a new
        note, but a redundant append is possible. Accepted as a rare,
        benign cost rather than engineering de-duplication against
        sync_turn's already-processed turns.

        Returns a short receipt threaded into the compressor's summary
        prompt (memory_context) — the retention itself already
        happened as a side effect by the time this returns; the
        receipt is a bonus hint for the summarizer, not the mechanism.
        Never raises; returns "" on any failure or nothing to retain.
        """
        if self._root is None or not messages:
            return ""
        try:
            candidates = _llm.distill_messages(messages)
            titles = []
            for candidate in candidates:
                self._process_candidate(candidate)
                label = (candidate.get("title") or candidate.get("topic") or "").strip()
                if label:
                    titles.append(label)
            if not titles:
                return ""
            return (
                "[zk-memory] Retained to the zettelkasten before this "
                "compaction: " + ", ".join(titles)
            )
        except Exception:
            logger.warning("zk-memory: on_pre_compress failed", exc_info=True)
            return ""


def register(ctx) -> None:
    """Called by Hermes memory plugin discovery."""
    ctx.register_auxiliary_task(
        key=_llm.TASK_KEY,
        display_name="ZK memory write-time judge",
        description="Distills turns into zettel candidates and judges merge-vs-create.",
        defaults={"provider": _llm._DEFAULT_PROVIDER, "model": _llm._DEFAULT_MODEL},
    )
    ctx.register_memory_provider(ZkMemoryProvider())
