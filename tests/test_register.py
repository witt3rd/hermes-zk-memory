"""Module-level register(ctx) -- plugin discovery entry point."""

from __future__ import annotations


def test_register_registers_auxiliary_task_and_memory_provider(plugin_module, stub_ctx):
    plugin_module.register(stub_ctx)

    assert len(stub_ctx.auxiliary_tasks) == 1
    task = stub_ctx.auxiliary_tasks[0]
    assert task["key"] == "zk_memory_judge"
    assert task["display_name"]
    assert task["description"]
    # No baked provider/model: the being owns the judge via config.yaml
    # (memory.zk_judge.provider/model). Without explicit config we disable
    # retain rather than inherit hermes' default routing model.
    assert task["defaults"] is None

    assert len(stub_ctx.memory_providers) == 1
    assert isinstance(stub_ctx.memory_providers[0], plugin_module.ZkMemoryProvider)
