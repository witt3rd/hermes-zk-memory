"""Minimal stand-in for hermes-agent's agent.auxiliary_client.

Only used as a fallback when a real hermes-agent checkout isn't importable
(see tests/conftest.py). Tests always monkeypatch llm._resolve_client (or
this module's functions) directly, so these bodies never need to do
anything real — they exist purely so ``from agent import auxiliary_client``
and ``from agent.auxiliary_client import ...`` succeed at import time.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def _resolve_task_provider_model(*, task: str) -> Tuple[Any, Any, Any, Any, Any]:
    """Stub: real signature returns (provider, model, base_url, api_key, api_mode)."""
    raise RuntimeError("stub agent.auxiliary_client._resolve_task_provider_model called")


def resolve_provider_client(
    provider: Any,
    model: Any,
    *,
    explicit_base_url: Any = None,
    explicit_api_key: Any = None,
    api_mode: Any = None,
    task: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Stub: real signature returns (client, model)."""
    raise RuntimeError("stub agent.auxiliary_client.resolve_provider_client called")
