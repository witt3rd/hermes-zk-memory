"""ZkMemoryProvider.sync_turn() / shutdown() -- off-thread write-time judge.

llm.judge_turn is referenced inside __init__.py as ``_llm.judge_turn`` --
we monkeypatch it on the ``llm`` module object (the same module object
``_llm`` in __init__.py points to), never touching a real LLM provider.
"""

from __future__ import annotations

import time

import llm as llm_module


def _wait_for_file(path, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def test_sync_turn_returns_immediately_even_if_judge_is_slow(provider, monkeypatch):
    def _slow_judge(user_content, assistant_content):
        time.sleep(2.0)
        return None

    monkeypatch.setattr(llm_module, "judge_turn", _slow_judge)

    start = time.monotonic()
    provider.sync_turn("hello", "hi there")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"sync_turn blocked for {elapsed:.2f}s"
    # clean up the background thread before the test ends
    provider._sync_thread.join(timeout=5.0)


def test_sync_turn_not_worth_retaining_writes_nothing(provider, monkeypatch):
    monkeypatch.setattr(llm_module, "judge_turn", lambda u, a: {"worth_retaining": False})
    provider.sync_turn("hello", "hi there")
    provider._sync_thread.join(timeout=5.0)

    assert list(provider._root.glob("*.md")) == [] if provider._root.exists() else True


def test_sync_turn_judge_returns_none_writes_nothing(provider, monkeypatch):
    monkeypatch.setattr(llm_module, "judge_turn", lambda u, a: None)
    provider.sync_turn("hello", "hi there")
    provider._sync_thread.join(timeout=5.0)

    assert not provider._root.exists() or list(provider._root.glob("*.md")) == []


def test_sync_turn_worth_retaining_writes_note(provider, monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "judge_turn",
        lambda u, a: {
            "worth_retaining": True,
            "slug": "durable-fact",
            "title": "Durable Fact",
            "body": "This is a durable fact worth remembering.",
        },
    )
    provider.sync_turn("what's my favorite color?", "your favorite color is blue")
    provider._sync_thread.join(timeout=5.0)

    files = list(provider._root.glob("*durable-fact.md"))
    assert len(files) == 1
    text = files[0].read_text()
    assert "# Durable Fact" in text
    assert "This is a durable fact worth remembering." in text


def test_sync_turn_worth_retaining_but_incomplete_fields_writes_nothing(provider, monkeypatch):
    monkeypatch.setattr(
        llm_module,
        "judge_turn",
        lambda u, a: {"worth_retaining": True, "slug": "", "title": "T", "body": "B"},
    )
    provider.sync_turn("u", "a")
    provider._sync_thread.join(timeout=5.0)

    assert not provider._root.exists() or list(provider._root.glob("*.md")) == []


def test_sync_turn_bounded_joins_prior_inflight_thread(provider, monkeypatch):
    """A second sync_turn() call bounded-joins the first before starting
    its own thread, so writes stay ordered."""
    order = []

    def _judge_first(u, a):
        time.sleep(0.2)
        order.append("first-done")
        return {"worth_retaining": False}

    def _judge_second(u, a):
        order.append("second-done")
        return {"worth_retaining": False}

    monkeypatch.setattr(llm_module, "judge_turn", _judge_first)
    provider.sync_turn("u1", "a1")
    first_thread = provider._sync_thread

    monkeypatch.setattr(llm_module, "judge_turn", _judge_second)
    provider.sync_turn("u2", "a2")

    provider._sync_thread.join(timeout=5.0)
    assert not first_thread.is_alive()
    assert order == ["first-done", "second-done"]


def test_sync_turn_uninitialized_root_is_a_noop(plugin_module, monkeypatch):
    fresh = plugin_module.ZkMemoryProvider()
    called = []
    monkeypatch.setattr(llm_module, "judge_turn", lambda u, a: called.append(1) or None)
    fresh.sync_turn("u", "a")
    assert fresh._sync_thread is None
    assert called == []


def test_shutdown_joins_inflight_sync_thread(provider, monkeypatch):
    finished = []

    def _slow_judge(u, a):
        time.sleep(0.3)
        finished.append(True)
        return None

    monkeypatch.setattr(llm_module, "judge_turn", _slow_judge)
    provider.sync_turn("u", "a")
    provider.shutdown()  # should block until the thread completes (join timeout=5.0)

    assert finished == [True]
