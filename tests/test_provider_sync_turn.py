"""ZkMemoryProvider.sync_turn() / shutdown() -- off-thread delegation to
``self._memory.retain_turn`` (the library's two-stage write-time judgment;
the pipeline itself is covered by zk-memory's own test_retain). We mock
``retain_turn`` on the provider's Memory instance, never touching a real
LLM provider or the corpus write path."""

from __future__ import annotations

import time


def _join(provider, timeout=5.0):
    if provider._sync_thread is not None:
        provider._sync_thread.join(timeout=timeout)


def _md_files(provider):
    if not provider._root.exists():
        return []
    return list(provider._root.glob("*.md"))


def test_sync_turn_returns_immediately_even_if_retain_is_slow(provider, monkeypatch):
    def _slow_retain(user, assistant, **kw):
        time.sleep(2.0)

    monkeypatch.setattr(provider._memory, "retain_turn", _slow_retain)

    start = time.monotonic()
    provider.sync_turn("hello", "hi there")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"sync_turn blocked for {elapsed:.2f}s"
    _join(provider)


def test_sync_turn_delegates_to_retain_turn_with_the_turn(provider, monkeypatch):
    captured = {}

    def _fake_retain(user, assistant, **kw):
        captured["user"] = user
        captured["assistant"] = assistant
        captured["session_id"] = kw.get("session_id", "")
        return []

    monkeypatch.setattr(provider._memory, "retain_turn", _fake_retain)
    provider.sync_turn("the user turn", "the assistant turn", session_id="sid-1")
    _join(provider)

    assert captured == {
        "user": "the user turn",
        "assistant": "the assistant turn",
        "session_id": "sid-1",
    }
    assert _md_files(provider) == []


def test_sync_turn_never_raises_when_retain_raises(provider, monkeypatch):
    def _boom(user, assistant, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(provider._memory, "retain_turn", _boom)
    provider.sync_turn("u", "a")  # must not raise
    _join(provider)


def test_sync_turn_bounded_joins_prior_inflight_thread(provider, monkeypatch):
    """A second sync_turn() call bounded-joins the first before starting
    its own thread, so writes stay ordered."""
    order = []

    def _retain_first(user, assistant, **kw):
        time.sleep(0.2)
        order.append("first-done")

    def _retain_second(user, assistant, **kw):
        order.append("second-done")

    monkeypatch.setattr(provider._memory, "retain_turn", _retain_first)
    provider.sync_turn("u1", "a1")
    first_thread = provider._sync_thread

    monkeypatch.setattr(provider._memory, "retain_turn", _retain_second)
    provider.sync_turn("u2", "a2")

    _join(provider)
    assert not first_thread.is_alive()
    assert order == ["first-done", "second-done"]


def test_sync_turn_uninitialized_root_is_a_noop(plugin_module):
    fresh = plugin_module.ZkMemoryProvider()
    fresh.sync_turn("u", "a")
    assert fresh._sync_thread is None


def test_shutdown_joins_inflight_sync_thread(provider, monkeypatch):
    finished = []

    def _slow_retain(user, assistant, **kw):
        time.sleep(0.3)
        finished.append(True)

    monkeypatch.setattr(provider._memory, "retain_turn", _slow_retain)
    provider.sync_turn("u", "a")
    provider.shutdown()  # should block until the thread completes (join timeout=5.0)

    assert finished == [True]