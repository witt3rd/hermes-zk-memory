"""LLM access for sync_turn's write-time judgment.

Two forced-tool-calls per candidate-bearing turn, both routed through
hermes-agent's own auxiliary-task machinery (agent.auxiliary_client),
under the plugin-owned auxiliary task "zk_memory_judge" (registered in
__init__.py's register()):

  1. distill_turn — sees the raw turn only, no corpus visibility. Splits
     it into zero or more candidates, each tagged:
       - "concept": a self-contained evergreen idea — a new node.
       - "entity_update": a temporal/attribute-level fact (e.g. "Judy
         arriving in two weeks") that would be a useless orphan as its
         own note — it belongs appended to an existing entity note.
     Both kinds flow through the same merge-or-create decision below;
     the kind only shapes what gets drafted.

  2. judge_merge — one call per candidate, given the full body of every
     zk_search hit for that candidate's topic (fetched, not just
     snippets). Decides: does this belong in one of these existing
     notes (merge), or is it genuinely new (create)? One call compares
     across all hits at once rather than one call per hit — cheaper and
     lets the model reason comparatively.

Deliberately NOT ctx.llm: memory-provider plugins never receive a real
PluginContext (register() only forwards register_* calls — see
plugins/memory/__init__.py's _ProviderCollector), so ctx.llm is
unreachable from this plugin category regardless of preference.

Import of agent.auxiliary_client is lazy and guarded so this module
stays importable (and unit-testable) outside a hermes-agent install —
every public function degrades to None/[] rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK_KEY = "zk_memory_judge"

# Default routing for the auxiliary task (register_auxiliary_task
# defaults — see __init__.py). Explicit rather than "auto": this call
# fires on every non-trivial turn, so leaving it to auto-detection risks
# silently resolving to an expensive frontier model for what's usually a
# short yes/no + draft. claude-sonnet-5 is the cheapest Anthropic model
# with a 1M-context option, which matters here: distill_turn sees the
# full raw turn (a large paste, long tool output) with no truncation
# below the hard safety cap. Deployments that expect large turns as a
# matter of course should point auxiliary.zk_memory_judge.model at
# their platform's 1M-context variant explicitly — the exact model-id
# syntax for that is provider/deployment-specific, so it's not baked in
# here as a literal default.
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "claude-sonnet-5"

# Hard safety cap only — NOT a normal-path truncation. At 1M context,
# ordinary turns never hit this; it exists solely so a pathological
# paste can't blow past provider limits or balloon cost unbounded.
_CHARS_PER_TOKEN = 4
_MAX_INPUT_TOKENS = 900_000


def _truncate_to_max_tokens(text: str, max_tokens: int) -> str:
    max_chars = max_tokens * _CHARS_PER_TOKEN
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


# ---------------------------------------------------------------------------
# Stage 1 — distill
# ---------------------------------------------------------------------------

_DISTILL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_candidates",
        "description": (
            "Record zero or more retain candidates extracted from this "
            "turn. Always call this tool exactly once per invocation."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["worth_retaining", "candidates"],
            "properties": {
                "worth_retaining": {
                    "type": "boolean",
                    "description": "True if this turn contains anything worth retaining.",
                },
                "candidates": {
                    "type": "array",
                    "description": (
                        "MUST be empty when worth_retaining is false; MUST "
                        "contain at least one entry when true."
                    ),
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "topic", "title", "slug", "content"],
                        "properties": {
                            "kind": {
                                "type": "string",
                                "enum": ["concept", "entity_update"],
                                "description": (
                                    "'concept': a self-contained evergreen "
                                    "idea, substantial enough to stand alone "
                                    "as a new node. 'entity_update': a "
                                    "temporal or attribute-level fact about "
                                    "an existing entity/topic — would be a "
                                    "useless orphan as its own note."
                                ),
                            },
                            "topic": {
                                "type": "string",
                                "description": (
                                    "What this is about, in a few words — "
                                    "used to search the corpus for a "
                                    "possible existing home (e.g. an entity "
                                    "name, a project, a recurring theme)."
                                ),
                            },
                            "title": {
                                "type": "string",
                                "description": "Title to use IF this becomes a new note.",
                            },
                            "slug": {
                                "type": "string",
                                "description": "Short hyphenated slug to use IF this becomes a new note (no date prefix).",
                            },
                            "content": {
                                "type": "string",
                                "description": (
                                    "For 'concept': the full atomic thought, "
                                    "own words, one idea. For "
                                    "'entity_update': just the fact/update "
                                    "fragment, own words, not a transcript."
                                ),
                            },
                        },
                    },
                },
            },
        },
    },
}

_DISTILL_SYSTEM_PROMPT = """You are the write-time distiller for a zettelkasten memory. \
Given a piece of conversation (a single turn, or a longer excerpt about to \
be dropped by context compaction), extract zero or more retain candidates.

There are two very different kinds of output, and conflating them ruins the \
corpus:

- CONCEPT: a self-contained, evergreen idea. It has enough conceptual weight \
to stand entirely on its own as a new node, ready to be linked to other \
ideas.
- ENTITY_UPDATE: a temporal or attribute-level data point (e.g. "Judy \
arriving in two weeks"). If made its own standalone note, it would be a \
useless orphan. It belongs appended to an existing entity/topic note \
instead.

Most turns yield nothing: routine questions, small talk, tool mechanics, and \
anything already obvious from context are not worth retaining at all — \
worth_retaining=false, empty candidates.

When something IS worth retaining, draft each candidate's content in your \
own words — never a transcript excerpt."""


def _distill_text(text: str) -> list[dict[str, Any]]:
    """Shared distillation call over an already-built transcript string."""
    client, model = _resolve_client()
    if client is None:
        return []
    text = _truncate_to_max_tokens(text, _MAX_INPUT_TOKENS)
    parsed = _forced_tool_call(
        client, model, _DISTILL_SYSTEM_PROMPT, text, _DISTILL_TOOL, "record_candidates"
    )
    if not parsed or not parsed.get("worth_retaining"):
        return []
    candidates = parsed.get("candidates") or []
    return [c for c in candidates if isinstance(c, dict)]


def distill_turn(user_content: str, assistant_content: str) -> list[dict[str, Any]]:
    """Run the write-time distillation call over one turn. Returns a
    (possibly empty) list of candidate dicts. Never raises.
    """
    return _distill_text(f"USER: {user_content}\n\nASSISTANT: {assistant_content}")


def distill_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the write-time distillation call over a batch of messages —
    the shape hermes hands ``MemoryProvider.on_pre_compress`` (the
    turns about to be dropped by context compaction), as opposed to
    ``distill_turn``'s single (user, assistant) pair.

    System messages are skipped (already protected/not being dropped).
    Never raises; returns [] on any failure or if there's nothing to
    distill from.
    """
    lines = []
    for m in messages:
        role = (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) or ""
        if role == "system":
            continue
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if not content or not isinstance(content, str):
            continue
        lines.append(f"{role.upper()}: {content}")
    if not lines:
        return []
    return _distill_text("\n\n".join(lines))


# ---------------------------------------------------------------------------
# Stage 3 — judge merge vs. create (one call per candidate, all hits at once)
# ---------------------------------------------------------------------------

_MERGE_JUDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_merge_decision",
        "description": (
            "Decide whether new information belongs in one of the given "
            "existing notes, or is genuinely new. Always call this tool "
            "exactly once."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["merge", "create"],
                    "description": (
                        "'merge' if one of the existing notes is the right "
                        "home for this information; 'create' if none is."
                    ),
                },
                "merge_target_ref": {
                    "type": "string",
                    "description": (
                        "The uuid of the existing note to merge into. "
                        "Required when action is 'merge'; omit otherwise."
                    ),
                },
            },
        },
    },
}

_MERGE_JUDGE_SYSTEM_PROMPT = """You are the merge judge for a zettelkasten memory. \
Given a piece of new information and a short list of existing notes, decide \
whether the new information belongs appended to one of those existing \
notes, or whether it's genuinely new and deserves its own note.

Merge only when the existing note is truly the same entity/topic — not \
merely related. When in doubt, prefer 'create': a wrong merge pollutes an \
existing note; a missed merge just means slight duplication, which is the \
safer failure."""


def judge_merge(candidate: dict[str, Any], hit_notes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Run the merge-vs-create judgment for one candidate against its
    fetched search hits. Returns the parsed decision dict, or None on
    any failure (callers should treat None as "create"). Never raises.
    """
    if not hit_notes:
        return None
    client, model = _resolve_client()
    if client is None:
        return None

    notes_text = "\n\n".join(
        f"[{n.get('uuid', '')}] {n.get('title', n.get('slug', '?'))}\n{n.get('body', '').strip()}"
        for n in hit_notes
    )
    kind = candidate.get("kind", "concept")
    content = candidate.get("content", "")
    user_text = (
        f"New information (kind={kind}):\n{content}\n\n"
        f"Existing notes found for this topic:\n{notes_text}"
    )
    return _forced_tool_call(
        client, model, _MERGE_JUDGE_SYSTEM_PROMPT, user_text, _MERGE_JUDGE_TOOL, "record_merge_decision"
    )


# ---------------------------------------------------------------------------
# Shared LLM plumbing
# ---------------------------------------------------------------------------

def _resolve_client():
    """Return (client, model) via the plugin's own auxiliary task, or
    (None, None) on any failure — including agent.auxiliary_client not
    being importable (standalone/test context)."""
    try:
        from agent import auxiliary_client as aux
    except ImportError:
        return None, None
    try:
        provider, model, base_url, api_key, api_mode = (
            aux._resolve_task_provider_model(task=TASK_KEY)
        )
        return aux.resolve_provider_client(
            provider,
            model,
            explicit_base_url=base_url,
            explicit_api_key=api_key,
            api_mode=api_mode,
            task=TASK_KEY,
        )
    except Exception:
        logger.warning("zk-memory: LLM client resolution failed", exc_info=True)
        return None, None


def _forced_tool_call(
    client: Any,
    model: Any,
    system_prompt: str,
    user_text: str,
    tool: dict[str, Any],
    tool_name: str,
) -> Optional[dict[str, Any]]:
    """One forced-tool-call completion; returns the parsed arguments
    dict, or None on any failure. Never raises."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            tools=[tool],
            tool_choice="required",
            max_tokens=1500,
        )
        message = response.choices[0].message
        tool_calls = getattr(message, "tool_calls", None) or (
            message.get("tool_calls") if isinstance(message, dict) else None
        )
        if not tool_calls:
            return None
        call = tool_calls[0]
        args_raw = (
            getattr(call.function, "arguments", None)
            if hasattr(call, "function")
            else call.get("function", {}).get("arguments")
        )
        if not args_raw:
            return None
        return json.loads(args_raw)
    except Exception:
        logger.warning("zk-memory: %s call failed", tool_name, exc_info=True)
        return None
