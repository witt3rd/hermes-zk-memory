"""ZkMemoryProvider.on_pre_compress() -- the promote-before-drop hook
hermes calls immediately before context compaction discards a batch of
messages. Synchronous (unlike sync_turn): the caller needs the return
value before compress() proceeds.

The retain itself (distill -> merge-or-create) is delegated to
``self._memory.retain_messages`` and covered by zk-memory's test_retain;
here we test the provider's synchronous orchestration + receipt."""

from __future__ import annotations


def _md_files(provider):
    if not provider._root.exists():
        return []
    return list(provider._root.glob("*.md"))


def test_on_pre_compress_uninitialized_root_is_a_noop(plugin_module):
    fresh = plugin_module.ZkMemoryProvider()
    assert fresh.on_pre_compress([{"role": "user", "content": "hi"}]) == ""


def test_on_pre_compress_empty_messages_is_a_noop(provider, monkeypatch):
    called = []
    monkeypatch.setattr(
        provider._memory, "retain_messages", lambda msgs, **kw: called.append(1) or []
    )
    assert provider.on_pre_compress([]) == ""
    assert called == []


def test_on_pre_compress_no_candidates_returns_empty_string(provider, monkeypatch):
    monkeypatch.setattr(provider._memory, "retain_messages", lambda msgs, **kw: [])
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert provider.on_pre_compress(messages) == ""
    assert _md_files(provider) == []


def test_on_pre_compress_delegates_batch_and_returns_receipt(provider, monkeypatch):
    """Unlike sync_turn, there's no background thread -- retain_messages
    is called synchronously and the receipt is built from its result."""
    captured = {}
    monkeypatch.setattr(
        provider._memory,
        "retain_messages",
        lambda msgs, **kw: captured.update({"msgs": msgs}) or ["Atomic Notes"],
    )

    messages = [
        {"role": "user", "content": "what's a zettelkasten?"},
        {"role": "assistant", "content": "one idea per note"},
    ]
    receipt = provider.on_pre_compress(messages)

    assert captured["msgs"] == messages
    assert "Atomic Notes" in receipt
    assert "zk-memory" in receipt


def test_on_pre_compress_passes_through_system_messages(provider, monkeypatch):
    """System-message filtering is the library's job (retain_messages);
    the provider must forward the whole batch unchanged."""
    captured = {}
    monkeypatch.setattr(
        provider._memory,
        "retain_messages",
        lambda msgs, **kw: captured.update({"msgs": msgs}) or [],
    )
    messages = [
        {"role": "system", "content": "SYSTEM_SECRET_MARKER"},
        {"role": "user", "content": "hello"},
    ]
    provider.on_pre_compress(messages)
    assert captured["msgs"] == messages


def test_on_pre_compress_never_raises_when_retain_messages_raises(provider, monkeypatch):
    def _raise(msgs, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(provider._memory, "retain_messages", _raise)
    messages = [{"role": "user", "content": "u"}]
    assert provider.on_pre_compress(messages) == ""