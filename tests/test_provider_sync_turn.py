"""ZkMemoryProvider.sync_turn() / shutdown() -- off-thread, two-stage
write-time judgment (distill -> per-candidate search+merge-or-create).

llm.distill_turn / llm.judge_merge are referenced inside __init__.py as
``_llm.distill_turn`` / ``_llm.judge_merge`` -- we monkeypatch them on the
``llm`` module object (the same module object ``_llm`` in __init__.py
points to), never touching a real LLM provider. zk.search/zk.read are
monkeypatched where a fixed hit set is needed for determinism; zk.write/
zk.merge are left real so file effects can be asserted directly.
"""

from __future__ import annotations

import time

import llm as llm_module
import zk as zk_module


def _join(provider, timeout=5.0):
    if provider._sync_thread is not None:
        provider._sync_thread.join(timeout=timeout)


def _md_files(provider):
    if not provider._root.exists():
        return []
    return list(provider._root.glob("*.md"))


CONCEPT_CANDIDATE = {
    "kind": "concept",
    "topic": "atomic notes",
    "title": "Atomic Notes",
    "slug": "atomic-notes",
    "content": "One idea per note, own words.",
}


def test_sync_turn_returns_immediately_even_if_distill_is_slow(provider, monkeypatch):
    def _slow_distill(user_content, assistant_content):
        time.sleep(2.0)
        return []

    monkeypatch.setattr(llm_module, "distill_turn", _slow_distill)

    start = time.monotonic()
    provider.sync_turn("hello", "hi there")
    elapsed = time.monotonic() - start

    assert elapsed < 0.5, f"sync_turn blocked for {elapsed:.2f}s"
    _join(provider)


def test_sync_turn_no_candidates_writes_nothing(provider, monkeypatch):
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [])
    provider.sync_turn("hello", "hi there")
    _join(provider)
    assert _md_files(provider) == []


def test_sync_turn_creates_new_note_when_no_search_hits(provider, monkeypatch):
    """No zk_search hits at all -> straight to create, no judge_merge call."""
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [CONCEPT_CANDIDATE])
    monkeypatch.setattr(zk_module, "search", lambda query, root, **kw: [])

    judge_calls = []
    monkeypatch.setattr(
        llm_module, "judge_merge", lambda candidate, hits: judge_calls.append(1) or None
    )

    provider.sync_turn("what's a zettelkasten?", "one idea per note")
    _join(provider)

    assert judge_calls == []  # never called -- no hits to compare against
    files = [f for f in _md_files(provider) if "atomic-notes" in f.name]
    assert len(files) == 1
    text = files[0].read_text()
    assert "# Atomic Notes" in text
    assert "One idea per note, own words." in text


def test_sync_turn_creates_new_note_when_judge_says_create(provider, monkeypatch):
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [CONCEPT_CANDIDATE])
    monkeypatch.setattr(
        zk_module,
        "search",
        lambda query, root, **kw: [{"uuid": "existing-1", "slug": "unrelated"}],
    )
    monkeypatch.setattr(
        zk_module,
        "read",
        lambda ref, root, **kw: {
            "found": True,
            "note": {"uuid": "existing-1", "title": "Unrelated", "body": "not the same thing"},
        },
    )
    monkeypatch.setattr(llm_module, "judge_merge", lambda candidate, hits: {"action": "create"})

    provider.sync_turn("u", "a")
    _join(provider)

    files = [f for f in _md_files(provider) if "atomic-notes" in f.name]
    assert len(files) == 1


def test_sync_turn_merges_when_judge_says_merge_with_valid_ref(provider, monkeypatch):
    # Seed a real existing note to merge into.
    provider._root.mkdir(parents=True, exist_ok=True)
    write_result = zk_module.write("judy", "Judy", "Judy is a colleague.", provider._root)
    assert write_result["ok"]
    target_uuid = write_result["uuid"]

    entity_candidate = {
        "kind": "entity_update",
        "topic": "Judy",
        "title": "",
        "slug": "",
        "content": "Judy is arriving in two weeks.",
    }
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [entity_candidate])
    monkeypatch.setattr(
        zk_module, "search", lambda query, root, **kw: [{"uuid": target_uuid, "slug": "judy"}]
    )
    monkeypatch.setattr(
        llm_module,
        "judge_merge",
        lambda candidate, hits: {"action": "merge", "merge_target_ref": target_uuid},
    )

    provider.sync_turn("when's Judy arriving?", "in two weeks")
    _join(provider)

    # No new note created for the entity_update...
    new_notes = [f for f in _md_files(provider) if "judy" not in f.name]
    assert new_notes == []
    # ...and the existing Judy note was appended to, not replaced.
    judy_file = write_result["path"]
    text = __import__("pathlib").Path(judy_file).read_text()
    assert "Judy is a colleague." in text
    assert "Judy is arriving in two weeks." in text


def test_sync_turn_falls_back_to_create_when_merge_ref_invalid(provider, monkeypatch):
    """judge_merge names a ref that wasn't among the fetched hits --
    never trust an unverified ref; fall back to create instead."""
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [CONCEPT_CANDIDATE])
    monkeypatch.setattr(
        zk_module, "search", lambda query, root, **kw: [{"uuid": "hit-1", "slug": "s"}]
    )
    monkeypatch.setattr(
        zk_module,
        "read",
        lambda ref, root, **kw: {"found": True, "note": {"uuid": "hit-1", "title": "T", "body": "B"}},
    )
    monkeypatch.setattr(
        llm_module,
        "judge_merge",
        lambda candidate, hits: {"action": "merge", "merge_target_ref": "hallucinated-ref"},
    )

    provider.sync_turn("u", "a")
    _join(provider)

    files = [f for f in _md_files(provider) if "atomic-notes" in f.name]
    assert len(files) == 1  # fell back to create, not silently dropped


def test_sync_turn_incomplete_create_candidate_skipped(provider, monkeypatch):
    incomplete = {"kind": "concept", "topic": "x", "title": "", "slug": "", "content": "y"}
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [incomplete])
    monkeypatch.setattr(zk_module, "search", lambda query, root, **kw: [])

    provider.sync_turn("u", "a")
    _join(provider)
    assert _md_files(provider) == []


def test_sync_turn_merge_with_empty_content_skipped(provider, monkeypatch):
    provider._root.mkdir(parents=True, exist_ok=True)
    write_result = zk_module.write("judy", "Judy", "Judy is a colleague.", provider._root)
    target_uuid = write_result["uuid"]

    empty_candidate = {"kind": "entity_update", "topic": "Judy", "title": "", "slug": "", "content": ""}
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [empty_candidate])
    monkeypatch.setattr(
        zk_module, "search", lambda query, root, **kw: [{"uuid": target_uuid, "slug": "judy"}]
    )
    monkeypatch.setattr(
        zk_module,
        "read",
        lambda ref, root, **kw: {"found": True, "note": {"uuid": target_uuid, "title": "Judy", "body": "Judy is a colleague."}},
    )
    monkeypatch.setattr(
        llm_module, "judge_merge", lambda candidate, hits: {"action": "merge", "merge_target_ref": target_uuid}
    )

    provider.sync_turn("u", "a")
    _join(provider)

    text = __import__("pathlib").Path(write_result["path"]).read_text()
    assert "Judy is a colleague." in text
    assert "---\n" not in text.split("Judy is a colleague.", 1)[1]  # nothing appended after


def test_sync_turn_multiple_candidates_processed_independently(provider, monkeypatch):
    second_candidate = {
        "kind": "concept",
        "topic": "second idea",
        "title": "Second Idea",
        "slug": "second-idea",
        "content": "A different atomic thought.",
    }
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: [CONCEPT_CANDIDATE, second_candidate])
    monkeypatch.setattr(zk_module, "search", lambda query, root, **kw: [])
    monkeypatch.setattr(llm_module, "judge_merge", lambda candidate, hits: None)

    provider.sync_turn("u", "a")
    _join(provider)

    names = {f.name for f in _md_files(provider)}
    assert any("atomic-notes" in n for n in names)
    assert any("second-idea" in n for n in names)


def test_sync_turn_bounded_joins_prior_inflight_thread(provider, monkeypatch):
    """A second sync_turn() call bounded-joins the first before starting
    its own thread, so writes stay ordered."""
    order = []

    def _distill_first(u, a):
        time.sleep(0.2)
        order.append("first-done")
        return []

    def _distill_second(u, a):
        order.append("second-done")
        return []

    monkeypatch.setattr(llm_module, "distill_turn", _distill_first)
    provider.sync_turn("u1", "a1")
    first_thread = provider._sync_thread

    monkeypatch.setattr(llm_module, "distill_turn", _distill_second)
    provider.sync_turn("u2", "a2")

    _join(provider)
    assert not first_thread.is_alive()
    assert order == ["first-done", "second-done"]


def test_sync_turn_uninitialized_root_is_a_noop(plugin_module, monkeypatch):
    fresh = plugin_module.ZkMemoryProvider()
    called = []
    monkeypatch.setattr(llm_module, "distill_turn", lambda u, a: called.append(1) or [])
    fresh.sync_turn("u", "a")
    assert fresh._sync_thread is None
    assert called == []


def test_shutdown_joins_inflight_sync_thread(provider, monkeypatch):
    finished = []

    def _slow_distill(u, a):
        time.sleep(0.3)
        finished.append(True)
        return []

    monkeypatch.setattr(llm_module, "distill_turn", _slow_distill)
    provider.sync_turn("u", "a")
    provider.shutdown()  # should block until the thread completes (join timeout=5.0)

    assert finished == [True]
