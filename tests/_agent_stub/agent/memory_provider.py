"""Minimal stand-in for hermes-agent's agent.memory_provider.MemoryProvider.

Only the abstract-ish shape the plugin needs: a base class with the
lifecycle methods it overrides. No abc.ABC enforcement here (keeping this
stub simple) — real hermes-agent's MemoryProvider is what actually enforces
the contract; this fallback only needs to be importable and subclassable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class MemoryProvider:
    """No-op base class mirroring the real MemoryProvider's public shape."""

    @property
    def name(self) -> str:
        return "stub-provider"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        pass

    def shutdown(self) -> None:
        pass

    def system_prompt_block(self) -> str:
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return []

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        pass
