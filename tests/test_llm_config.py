"""Config-driven write-time judge (auxiliary.zk_memory_judge).

The being owns which LLM runs its write-time judgment: the provider reads
``auxiliary.zk_memory_judge.provider/model`` — the existing config block
hermes manages for the plugin's auxiliary task — and binds a StructuredLLM
to those explicit values (never inheriting hermes' default routing).
Missing/incomplete config disables retain (llm=None), loudly.

hermes_cli isn't importable standalone, so tests inject a fake
``hermes_cli.config`` module whose ``load_config_readonly`` returns a
controlled value.
"""

from __future__ import annotations

import sys
import types

import pytest


def _install_fake_config(monkeypatch, cfg):
    """Register a fake ``hermes_cli.config`` so ``from hermes_cli.config
    import load_config_readonly`` resolves and returns ``cfg``."""
    pkg = types.ModuleType("hermes_cli")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "hermes_cli", pkg)
    mod = types.ModuleType("hermes_cli.config")
    mod.load_config_readonly = lambda: cfg
    monkeypatch.setitem(sys.modules, "hermes_cli.config", mod)
    return mod


def _with_zk_judge(cfg_dict):
    """Config value for ``load_config_readonly`` with the auxiliary block set."""
    base = {
        "auxiliary": {
            "zk_memory_judge": {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"},
        }
    }
    base["auxiliary"]["zk_memory_judge"].update(cfg_dict)
    return base


def test_build_structured_llm_requires_provider_and_model(llm_module):
    adapter = llm_module.build_structured_llm(None, None)
    assert adapter([{"role": "user", "content": "hi"}], schema={}, name="judge") is None


def test_build_structured_llm_threads_provider_model_to_resolution(llm_module, monkeypatch):
    """The explicit provider/model must reach the auxiliary_client resolver
    (which always-wins over config/auto), never a hermes default."""
    captured = {}
    monkeypatch.setattr(
        llm_module, "_resolve_client",
        lambda provider=None, model=None: captured.update(provider=provider, model=model) or (object(), "resolved"),
    )
    monkeypatch.setattr(
        llm_module, "_forced_tool_call",
        lambda *a, **k: {"action": "merge"},
    )
    adapter = llm_module.build_structured_llm("openrouter", "deepseek/deepseek-v4-flash-0731")
    result = adapter([{"role": "user", "content": "u"}], schema={}, name="judge")
    assert result == {"action": "merge"}
    assert captured == {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"}


def test_initialize_wires_judge_from_config(plugin_module, tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, _with_zk_judge({}))
    captured = {}
    marker = object()

    def _build(provider, model):
        captured["provider"] = provider
        captured["model"] = model
        return marker

    # initialize() calls the plugin's OWN llm module (_llm), loaded under the
    # synthetic namespace -- patch that object, not the top-level import.
    monkeypatch.setattr(plugin_module._llm, "build_structured_llm", _build)
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path))

    assert captured == {"provider": "openrouter", "model": "deepseek/deepseek-v4-flash-0731"}
    assert provider._memory._llm is marker


def test_initialize_without_judge_config_disables_retain(plugin_module, tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, {})
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path))

    assert provider._memory._llm is None  # retain disabled, corpus ops still work
    assert provider._root == tmp_path / "zk"