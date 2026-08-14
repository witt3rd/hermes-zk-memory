"""ZkMemoryProvider.on_pre_compress() -- the promote-before-drop hook
hermes calls immediately before context compaction discards a batch of
messages. Synchronous (unlike sync_turn): the caller needs the return
value before compress() proceeds.
"""

from __future__ import annotations

import llm as llm_module
import zk as zk_module


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


def test_on_pre_compress_uninitialized_root_is_a_noop(plugin_module, monkeypatch):
    fresh = plugin_module.ZkMemoryProvider()
    called = []
    monkeypatch.setattr(llm_module, "distill_messages", lambda msgs: called.append(1) or [])
    result = fresh.on_pre_compress([{"role": "user", "content": "hi"}])
    assert result == ""
    assert called == []


def test_on_pre_compress_empty_messages_is_a_noop(provider, monkeypatch):
    called = []
    monkeypatch.setattr(llm_module, "distill_messages", lambda msgs: called.append(1) or [])
    assert provider.on_pre_compress([]) == ""
    assert called == []


def test_on_pre_compress_no_candidates_returns_empty_string(provider, monkeypatch):
    monkeypatch.setattr(llm_module, "distill_messages", lambda msgs: [])
    messages = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}]
    assert provider.on_pre_compress(messages) == ""
    assert _md_files(provider) == []


def test_on_pre_compress_writes_synchronously_and_returns_receipt(provider, monkeypatch):
    """Unlike sync_turn, there's no background thread -- the write and
    the receipt string must both be done by the time this returns."""
    monkeypatch.setattr(llm_module, "distill_messages", lambda msgs: [CONCEPT_CANDIDATE])
    monkeypatch.setattr(zk_module, "search", lambda query, root, **kw: [])

    messages = [
        {"role": "user", "content": "what's a zettelkasten?"},
        {"role": "assistant", "content": "one idea per note"},
    ]
    receipt = provider.on_pre_compress(messages)

    assert "Atomic Notes" in receipt
    assert "zk-memory" in receipt
    files = [f for f in _md_files(provider) if "atomic-notes" in f.name]
    assert len(files) == 1  # already written -- no thread to join


def test_on_pre_compress_skips_system_messages(provider, monkeypatch):
    # Use the real distill_messages logic (not mocked) to prove system
    # messages are excluded from what's sent to the judge.
    sent_texts = []

    def _fake_forced_tool_call(client, model, system_prompt, user_text, tool, tool_name):
        sent_texts.append(user_text)
        return {"worth_retaining": False, "candidates": []}

    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (object(), "m"))
    monkeypatch.setattr(llm_module, "_forced_tool_call", _fake_forced_tool_call)

    messages = [
        {"role": "system", "content": "SYSTEM_SECRET_MARKER"},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    provider.on_pre_compress(messages)

    assert len(sent_texts) == 1
    assert "SYSTEM_SECRET_MARKER" not in sent_texts[0]
    assert "hello" in sent_texts[0]
    assert "hi" in sent_texts[0]


def test_on_pre_compress_falls_back_to_create_when_merge_ref_invalid(provider, monkeypatch):
    """Same safety check as sync_turn's candidate processing -- a
    hallucinated merge ref must not be trusted here either, since both
    paths share _process_candidate."""
    monkeypatch.setattr(llm_module, "distill_messages", lambda msgs: [CONCEPT_CANDIDATE])
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

    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    receipt = provider.on_pre_compress(messages)

    assert "Atomic Notes" in receipt
    files = [f for f in _md_files(provider) if "atomic-notes" in f.name]
    assert len(files) == 1


def test_on_pre_compress_never_raises_when_distill_messages_raises(provider, monkeypatch):
    def _raise(msgs):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_module, "distill_messages", _raise)
    messages = [{"role": "user", "content": "u"}]
    assert provider.on_pre_compress(messages) == ""
