"""ZkMemoryProvider.prefetch() -- auto-recall, fenced for the model."""

from __future__ import annotations

import zk


def test_prefetch_no_hits_returns_empty_string(provider, monkeypatch):
    monkeypatch.setattr(zk, "search", lambda query, root, limit=5: [])
    assert provider.prefetch("anything") == ""


def test_prefetch_with_hits_returns_fenced_recall_block(provider, monkeypatch):
    monkeypatch.setattr(
        zk,
        "search",
        lambda query, root, limit=5: [
            {"title": "Some Title", "slug": "some-slug", "snippet": "a relevant snippet here"}
        ],
    )
    out = provider.prefetch("my query")
    assert out.startswith('<recall query="my query">')
    assert out.endswith("</recall>")
    assert "Some Title" in out
    assert "some-slug" in out
    assert "a relevant snippet here" in out


def test_prefetch_empty_query_returns_empty_string(provider, monkeypatch):
    calls = []
    monkeypatch.setattr(zk, "search", lambda query, root, limit=5: calls.append(1) or [])
    assert provider.prefetch("") == ""
    assert calls == []  # never even calls search


def test_prefetch_uninitialized_root_returns_empty_string(plugin_module):
    fresh = plugin_module.ZkMemoryProvider()
    assert fresh.prefetch("query") == ""


def test_prefetch_search_exception_returns_empty_string(provider, monkeypatch):
    def _boom(*a, **kw):
        raise RuntimeError("search blew up")

    monkeypatch.setattr(zk, "search", _boom)
    assert provider.prefetch("query") == ""
