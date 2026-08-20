"""Hermes adapter implementing ``zk_memory``'s StructuredLLM protocol.

Routes structured LLM calls through hermes-agent's own auxiliary-task
machinery (``agent.auxiliary_client``), under the plugin-owned auxiliary
task "zk_memory_judge" (registered in ``__init__.py``'s register()). This
is the hermes-native mechanism for memory-provider plugins: they never
receive a real PluginContext (register() only forwards register_* calls
-- see plugins/memory/__init__.py's _ProviderCollector), so ctx.llm is
unreachable from this plugin category regardless of preference.

The model/provider are NOT baked here and do NOT default to hermes'
runtime: they are supplied explicitly by the plugin from the profile's
config.yaml (``memory.zk_judge.provider/model``), so the being owns which
LLM runs its write-time judgment. All prompts and JSON schemas live in the
library (``zk_memory.judge``); this module is the thin adapter that turns
a ``(messages, schema, name)`` call into the forced-tool-call path.

Import of agent.auxiliary_client is lazy and guarded so this module
stays importable (and unit-testable) outside a hermes-agent install --
``_resolve_client`` degrades to None and every call becomes a no-op
rather than raising.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Optional

from zk_memory.judge import TOOL_DESCRIPTIONS

logger = logging.getLogger(__name__)

TASK_KEY = "zk_memory_judge"


def _read_judge_config() -> tuple[Optional[str], Optional[str]]:
    """Return (provider, model) from the profile's ``auxiliary.zk_memory_judge``.

    This is the existing config block hermes manages for the plugin's
    auxiliary task (register_auxiliary_task bridges it to
    ``AUXILIARY_ZK_MEMORY_JUDGE_*`` env vars). Both provider and model are
    required for the write-time judge to run. Never raises: returns
    (None, None) when hermes_cli is unreachable or the block is
    absent/incomplete.
    """
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly()
    except Exception:
        return None, None
    aux = cfg.get("auxiliary") or {}
    task_cfg = aux.get(TASK_KEY) or {}
    provider = str(task_cfg.get("provider", "")).strip() or None
    model = str(task_cfg.get("model", "")).strip() or None
    return provider, model


def build_structured_llm(
    provider: Optional[str],
    model: Optional[str],
) -> Callable[..., Optional[dict[str, Any]]]:
    """Return a StructuredLLM callable bound to explicit ``provider`` +
    ``model`` (the profile-config values, threaded through hermes' own
    auxiliary_client transport). A missing provider/model yields a no-op
    callable, so retain degrades gracefully instead of guessing a model."""
    def _adapter(
        messages: list[dict[str, str]],
        *,
        schema: dict,
        name: str,
    ) -> Optional[dict[str, Any]]:
        return _call_judge(provider, model, messages, schema=schema, name=name)
    return _adapter


def hermes_structured_llm(
    messages: list[dict[str, str]],
    *,
    schema: dict,
    name: str,
) -> Optional[dict[str, Any]]:
    """StructuredLLM adapter that resolves provider/model from the profile
    config on each call. Kept for backward compatibility; the plugin's
    ``initialize()`` prefers an explicit ``build_structured_llm`` bound at
    setup time."""
    provider, model = _read_judge_config()
    return _call_judge(provider, model, messages, schema=schema, name=name)


def _call_judge(
    provider: Optional[str],
    model: Optional[str],
    messages: list[dict[str, str]],
    *,
    schema: dict,
    name: str,
) -> Optional[dict[str, Any]]:
    """One forced-tool-call completion, or None on any failure. Never
    raises."""
    if not provider or not model:
        logger.warning(
            "zk-memory: auxiliary.%s.provider/model not configured; "
            "retain disabled (nothing to retain)",
            TASK_KEY,
        )
        return None
    client, resolved_model = _resolve_client(provider=provider, model=model)
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
    return _forced_tool_call(client, resolved_model, system_prompt, user_text, tool, name)


# ---------------------------------------------------------------------------
# Shared LLM plumbing
# ---------------------------------------------------------------------------

def _resolve_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Return (client, model) via the plugin's own auxiliary task, or
    (None, None) on any failure -- including agent.auxiliary_client not
    being importable (standalone/test context). The explicit
    provider/model always win over config/auto (auxiliary_client's
    documented priority), so the caller's ``memory.zk_judge`` values are
    authoritative -- nothing is inherited from hermes' default routing."""
    try:
        from agent import auxiliary_client as aux
    except ImportError:
        return None, None
    try:
        provider, model, base_url, api_key, api_mode = (
            aux._resolve_task_provider_model(
                task=TASK_KEY, provider=provider, model=model,
            )
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