"""ZkMemoryProvider.get_tool_schemas() / handle_tool_call()."""

from __future__ import annotations


def _corpus(provider):
    import zk_memory.corpus as corpus
    return corpus


def test_get_tool_schemas_returns_exactly_four_expected_tools(provider):
    schemas = provider.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"zk_search", "zk_read", "zk_write", "zk_tend"}
    assert len(schemas) == 4
    for s in schemas:
        assert "description" in s and s["description"]
        assert "parameters" in s


def test_handle_tool_call_unknown_tool_name(provider):
    out = provider.handle_tool_call("zk_bogus", {})
    assert out == "error: unknown tool: zk_bogus"


def test_handle_tool_call_uninitialized_provider(plugin_module):
    fresh = plugin_module.ZkMemoryProvider()
    out = fresh.handle_tool_call("zk_search", {"query": "x"})
    assert out == "error: provider not initialized"


def test_handle_tool_call_zk_search_missing_query(provider):
    out = provider.handle_tool_call("zk_search", {})
    assert out == "error: query is required"


def test_handle_tool_call_zk_search_no_hits(provider, monkeypatch):
    monkeypatch.setattr(provider._memory, "search", lambda query, **kw: [])
    out = provider.handle_tool_call("zk_search", {"query": "nothing"})
    assert "no notes found" in out


def test_handle_tool_call_zk_search_with_hits(provider, monkeypatch):
    monkeypatch.setattr(
        provider._memory,
        "search",
        lambda query, **kw: [
            {"title": "T1", "slug": "t1", "path": "20260101-t1.md", "uuid": "u1", "snippet": "some snippet"}
        ],
    )
    out = provider.handle_tool_call("zk_search", {"query": "hi", "limit": 3})
    assert "Found 1 note(s)" in out
    assert "T1" in out
    assert "some snippet" in out


def test_handle_tool_call_zk_read_missing_ref(provider):
    out = provider.handle_tool_call("zk_read", {})
    assert out == "error: ref is required (uuid, slug, or path)"


def test_handle_tool_call_zk_read_not_found(provider):
    out = provider.handle_tool_call("zk_read", {"ref": "nope"})
    assert "no note found" in out


def test_handle_tool_call_zk_read_found_with_links(provider):
    corpus = _corpus(provider)
    corpus.write("t1", "T One", "Body about [t2](t2-slug.md).", provider._root)
    written = corpus.list_notes(provider._root)[0]

    out = provider.handle_tool_call("zk_read", {"ref": written["uuid"] or written["slug"]})
    assert "T One" in out
    assert "links:" in out
    assert "t2-slug" in out


def test_handle_tool_call_zk_write_missing_fields(provider):
    out = provider.handle_tool_call("zk_write", {"slug": "s", "title": "", "body": "b"})
    assert out == "error: slug, title, and body are all required"


def test_handle_tool_call_zk_write_success(provider):
    out = provider.handle_tool_call(
        "zk_write", {"slug": "new-note", "title": "New Note", "body": "Some body text."}
    )
    assert out.startswith("zettel written:")
    assert "new-note" in out


def test_handle_tool_call_zk_write_duplicate_reports_error(provider):
    provider.handle_tool_call("zk_write", {"slug": "dup", "title": "Dup", "body": "b1"})
    out = provider.handle_tool_call("zk_write", {"slug": "dup", "title": "Dup2", "body": "b2"})
    assert out.startswith("error:")
    assert "already exists" in out


def test_handle_tool_call_zk_tend_invalid_action(provider):
    out = provider.handle_tool_call("zk_tend", {"action": "explode"})
    assert out == "error: action must be one of repair, check, mint"


def test_handle_tool_call_zk_tend_missing_linlink(provider, monkeypatch):
    corpus = _corpus(provider)
    monkeypatch.setattr(corpus.shutil, "which", lambda name: None)
    out = provider.handle_tool_call("zk_tend", {"action": "check"})
    assert "zk_tend check: FAILED" in out
    assert "linlink not on PATH" in out


def test_handle_tool_call_exception_path_returns_error_string(provider, monkeypatch):
    def _boom(*a, **kw):
        raise ValueError("kaboom")

    monkeypatch.setattr(provider._memory, "search", _boom)
    out = provider.handle_tool_call("zk_search", {"query": "x"})
    assert out == "error: kaboom"