"""LLM access for sync_turn's write-time judgment.

Routes through hermes-agent's own auxiliary-task machinery
(agent.auxiliary_client), registered as the plugin-owned auxiliary task
"zk_memory_judge" (see register() in __init__.py). This is the
hermes-native way for an in-process plugin to get a provider/model
users can pin independently of their main chat model — the same
machinery hermes-observational-memory uses via call_llm(task=...), but
routed through resolve_provider_client() directly because call_llm has
no tool_choice / forced-tool-call support.

Deliberately NOT ctx.llm: memory-provider plugins never receive a real
PluginContext (register() only forwards register_* calls — see
plugins/memory/__init__.py's _ProviderCollector), so ctx.llm is
unreachable from this plugin category regardless of preference.

Import of agent.auxiliary_client is lazy and guarded so this module
stays importable (and unit-testable) outside a hermes-agent install —
_resolve_client() and judge_turn() degrade to (None, None) / None
rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

TASK_KEY = "zk_memory_judge"

_JUDGE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "record_retain",
        "description": (
            "Decide whether this turn is worth a permanent note in the "
            "zettelkasten, and if so, draft it. Always call this tool "
            "exactly once."
        ),
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "required": ["worth_retaining"],
            "properties": {
                "worth_retaining": {
                    "type": "boolean",
                    "description": (
                        "True if this turn contains a durable fact, "
                        "decision, preference, or distinction worth "
                        "recalling later. False for small talk, routine "
                        "tool mechanics, or anything already obvious from "
                        "context."
                    ),
                },
                "slug": {
                    "type": "string",
                    "description": "Short hyphenated slug (no date prefix). Required when worth_retaining is true.",
                },
                "title": {
                    "type": "string",
                    "description": "Human title for the note. Required when worth_retaining is true.",
                },
                "body": {
                    "type": "string",
                    "description": (
                        "ONE atomic thought, in your own words, not a "
                        "transcript. Link related notes with "
                        "[label](slug.md). Required when worth_retaining "
                        "is true."
                    ),
                },
            },
        },
    },
}

_SYSTEM_PROMPT = """You are the write-time judge for a zettelkasten memory. \
Given one conversational turn, decide whether it contains something worth \
a permanent, atomic note — a durable fact, decision, preference, or \
distinction that would be useful to recall in a future conversation.

Most turns are NOT worth a note: routine questions, small talk, tool \
mechanics, and anything that's just restating what's already obvious from \
context should be worth_retaining=false with no other fields.

When something IS worth retaining, draft ONE atomic note: your own words, \
not a transcript excerpt. One thought per note."""


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


def judge_turn(user_content: str, assistant_content: str) -> Optional[dict[str, Any]]:
    """Run the write-time judgment call. Returns the parsed tool-call
    arguments dict, or None on any failure. NEVER raises.
    """
    client, model = _resolve_client()
    if client is None:
        return None
    turn_text = f"USER: {user_content}\n\nASSISTANT: {assistant_content}"
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": turn_text},
            ],
            tools=[_JUDGE_TOOL],
            tool_choice="required",
            max_tokens=1000,
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
        logger.warning("zk-memory: judge_turn call failed", exc_info=True)
        return None
