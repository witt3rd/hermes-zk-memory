"""hermes-zk-memory — a zettelkasten-backed Hermes MemoryProvider.

One corpus of atomic notes. The same operations back both the volitional
tool surface (get_tool_schemas/handle_tool_call) and the automatic
recall/retain motions (prefetch/sync_turn) — "auto" is an optional
convenience over the same underlying corpus operations, not a parallel
code path.

Scaffold only — engine (search/read/write/tend) to be ported in.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider


class ZkMemoryProvider(MemoryProvider):
    """Hermes MemoryProvider wrapping a zettelkasten corpus."""

    def __init__(self) -> None:
        self._hermes_home = None

    @property
    def name(self) -> str:
        return "zk-memory"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        raise NotImplementedError

    def shutdown(self) -> None:
        pass

    # ---- volitional tool surface ----

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        raise NotImplementedError

    # ---- automatic motions ----

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        raise NotImplementedError

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        raise NotImplementedError


def register(ctx) -> None:
    """Called by Hermes memory plugin discovery."""
    ctx.register_memory_provider(ZkMemoryProvider())
