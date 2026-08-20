"""Config-driven write-time judge (auxiliary.zk_memory_judge).

The being owns which LLM runs its write-time judgment: the provider reads
the ``auxiliary.zk_memory_judge`` block — the existing config hermes manages
for the plugin's auxiliary task — and binds a StructuredLLM that threads the
whole block (provider/model/base_url/api_key/timeout/extra_body) explicitly,
never inheriting hermes' default routing. Missing/incomplete config disables
retain (llm=None), loudly.

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


def _full_judge():
    """A full auxiliary.zk_memory_judge block."""
    return {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.example",
        "api_key": "sk-test",
        "timeout": 90,
        "extra_body": {"reasoning": {"effort": "medium"}},
    }


def _with_zk_block(block):
    """Config value for ``load_config_readonly`` with the auxiliary block."""
    return {"auxiliary": {"zk_memory_judge": dict(block)}}


def test_build_structured_llm_requires_provider_and_model(llm_module):
    adapter = llm_module.build_structured_llm({})
    assert adapter([{"role": "user", "content": "hi"}], schema={}, name="judge") is None


def test_build_structured_llm_threads_full_config_to_resolution(llm_module, monkeypatch):
    """The whole block must reach the auxiliary_client resolver and the
    forced tool call -- provider/model/base_url/api_key always-wins over
    config/auto, never a hermes default; timeout/extra_body forward."""
    resolved = {}
    call_kwargs = {}

    def _resolve(provider=None, model=None, base_url=None, api_key=None):
        resolved.update(provider=provider, model=model, base_url=base_url, api_key=api_key)
        return (object(), "resolved-model")

    def _forge(client, model, sp, ut, tool, tool_name, *, timeout=None, extra_body=None):
        call_kwargs.update(timeout=timeout, extra_body=extra_body, model=model)
        return {"action": "merge"}

    monkeypatch.setattr(llm_module, "_resolve_client", _resolve)
    monkeypatch.setattr(llm_module, "_forced_tool_call", _forge)

    adapter = llm_module.build_structured_llm(_full_judge())
    result = adapter([{"role": "user", "content": "u"}], schema={}, name="judge")

    assert result == {"action": "merge"}
    assert resolved == {
        "provider": "openrouter",
        "model": "deepseek/deepseek-v4-flash-0731",
        "base_url": "https://openrouter.example",
        "api_key": "sk-test",
    }
    assert call_kwargs["timeout"] == 90
    assert call_kwargs["extra_body"] == {"reasoning": {"effort": "medium"}}


def test_initialize_wires_full_block_from_config(plugin_module, tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, _with_zk_block(_full_judge()))
    captured = {}
    marker = object()

    def _build(cfg):
        captured["cfg"] = cfg
        return marker

    # initialize() calls the plugin's OWN llm module (_llm), loaded under the
    # synthetic namespace -- patch that object, not the top-level import.
    monkeypatch.setattr(plugin_module._llm, "build_structured_llm", _build)
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path))

    assert captured["cfg"] == _full_judge()
    assert provider._memory._llm is marker


def test_initialize_without_judge_config_disables_retain(plugin_module, tmp_path, monkeypatch):
    _install_fake_config(monkeypatch, {})
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path))

    assert provider._memory._llm is None  # retain disabled, corpus ops still work
    assert provider._root == tmp_path / "zk"