"""Hermes adapter implementing ``zk_memory``'s StructuredLLM protocol.

Routes structured LLM calls through hermes-agent's own auxiliary-task
machinery (``agent.auxiliary_client``), under the plugin-owned auxiliary
task "zk_memory_judge" (registered in ``__init__.py``'s register()). This
is the hermes-native mechanism for memory-provider plugins: they never
receive a real PluginContext (register() only forwards register_* calls
-- see plugins/memory/__init__.py's _ProviderCollector), so ctx.llm is
unreachable from this plugin category regardless of preference.

All prompts and JSON schemas live in the library (``zk_memory.judge``);
this module is the thin adapter that turns a ``(messages, schema, name)``
call into the forced-tool-call path, unchanged from before the split.

Import of agent.auxiliary_client is lazy and guarded so this module
stays importable (and unit-testable) outside a hermes-agent install --
``_resolve_client`` degrades to None and every call becomes a no-op
rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from zk_memory.judge import TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)

TASK_KEY = "zk_memory_judge"

# Default routing for the auxiliary task (register_auxiliary_task
# defaults -- see __init__.py). Explicit rather than "auto": this call
# fires on every non-trivial turn, so leaving it to auto-detection risks
# silently resolving to an expensive frontier model for what's usually a
# short yes/no + draft. claude-sonnet-5 is the cheapest Anthropic model
# with a 1M-context option, which matters here: distill sees the full raw
# turn (a large paste, long tool output) with no truncation below the
# hard safety cap (owned by the library). Deployments that expect large
# turns as a matter of course should point
# auxiliary.zk_memory_judge.model at their platform's 1M-context variant
# explicitly -- the exact model-id syntax for that is
# provider/deployment-specific, so it's not baked in here as a literal.
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_MODEL = "claude-sonnet-5"


def hermes_structured_llm(
    messages: list[dict[str, str]],
    *,
    schema: dict,
    name: str,
) -> Optional[dict[str, Any]]:
    """StructuredLLM adapter: one forced-tool-call completion.

    Builds the tool dict from ``name`` + the library's tool description +
    ``schema``, and returns the parsed arguments dict, or None on any
    failure. Never raises. Without a resolvable client (no hermes-agent,
    no configured auxiliary task) it degrades to None -- the retain
    pipeline treats None as "nothing to retain".
    """
    client, model = _resolve_client()
    if client is None:
        return None
    system_prompt = ""
    user_text = ""
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_text = content
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": TOOL_DESCRIPTIONS.get(name, name),
            "parameters": schema,
        },
    }
    return _forced_tool_call(client, model, system_prompt, user_text, tool, name)


# ---------------------------------------------------------------------------
# Shared LLM plumbing
# ---------------------------------------------------------------------------

def _resolve_client():
    """Return (client, model) via the plugin's own auxiliary task, or
    (None, None) on any failure -- including agent.auxiliary_client not
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