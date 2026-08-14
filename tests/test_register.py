"""Module-level register(ctx) -- plugin discovery entry point."""

from __future__ import annotations


def test_register_registers_auxiliary_task_and_memory_provider(plugin_module, stub_ctx):
    plugin_module.register(stub_ctx)

    assert len(stub_ctx.auxiliary_tasks) == 1
    task = stub_ctx.auxiliary_tasks[0]
    assert task["key"] == "zk_memory_judge"
    assert task["display_name"]
    assert task["description"]
    # Explicit, not "auto"/blank -- this call fires on every non-trivial
    # turn, so leaving routing to auto-detection risks silently resolving
    # to an expensive frontier model. claude-sonnet-5 is the cheapest
    # Anthropic model with a 1M-context option, which matters because
    # distill_turn sees the full raw turn with no normal-path truncation.
    assert task["defaults"] == {"provider": "anthropic", "model": "claude-sonnet-5"}

    assert len(stub_ctx.memory_providers) == 1
    assert isinstance(stub_ctx.memory_providers[0], plugin_module.ZkMemoryProvider)
