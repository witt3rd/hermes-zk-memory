"""ZkMemoryProvider.initialize() -- root resolution from kwargs / env."""

from __future__ import annotations

import pytest


def test_initialize_resolves_root_from_kwarg(plugin_module, tmp_path):
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(tmp_path))
    assert provider._root == tmp_path / "zk"


def test_initialize_resolves_root_from_env_when_kwarg_absent(plugin_module, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1")
    assert provider._root == tmp_path / "zk"


def test_initialize_kwarg_takes_precedence_over_env(plugin_module, tmp_path, monkeypatch):
    env_home = tmp_path / "env-home"
    kwarg_home = tmp_path / "kwarg-home"
    env_home.mkdir()
    kwarg_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(env_home))

    provider = plugin_module.ZkMemoryProvider()
    provider.initialize(session_id="s1", hermes_home=str(kwarg_home))
    assert provider._root == kwarg_home / "zk"


def test_initialize_raises_when_neither_kwarg_nor_env_present(plugin_module, monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    provider = plugin_module.ZkMemoryProvider()
    with pytest.raises(RuntimeError, match="HERMES_HOME"):
        provider.initialize(session_id="s1")


def test_is_available_always_true(plugin_module):
    provider = plugin_module.ZkMemoryProvider()
    assert provider.is_available() is True


def test_name_property(plugin_module):
    provider = plugin_module.ZkMemoryProvider()
    assert provider.name == "zk-memory"
