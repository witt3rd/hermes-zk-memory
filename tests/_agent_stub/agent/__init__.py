"""Minimal standalone stand-in for hermes-agent's ``agent`` package.

Only used when a real hermes-agent checkout isn't importable on this
machine (see tests/conftest.py). Provides just enough surface for
hermes-zk-memory's plugin module (``from agent.memory_provider import
MemoryProvider`` and ``from agent import auxiliary_client``) to import
cleanly under test, with no network/provider dependencies.
"""
