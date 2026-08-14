"""llm.distill_turn() / llm.judge_merge() / llm._resolve_client() --
the two-stage write-time judgment. No real LLM provider ever contacted;
all client access is monkeypatched."""

from __future__ import annotations

import json
import sys
import types


def _fake_response_object_style(args_json: str):
    """Build a fake response matching the object-attribute access pattern:
    response.choices[0].message.tool_calls[0].function.arguments"""

    function = types.SimpleNamespace(name="record_candidates", arguments=args_json)
    call = types.SimpleNamespace(function=function)
    message = types.SimpleNamespace(tool_calls=[call])
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def _fake_response_dict_style(args_json: str):
    """Build a fake response matching the dict-shaped fallback branch:
    message is a dict, tool_calls entries are dicts, and
    call.get('function', {}).get('arguments') is used."""

    message = {"tool_calls": [{"function": {"arguments": args_json}}]}
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


class _FakeClient:
    def __init__(self, response):
        self._response = response
        self.last_call_kwargs = None
        completions = types.SimpleNamespace(create=self._create)
        self.chat = types.SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return self._response


class _ExplodingClient:
    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                raise RuntimeError("provider is down")


# ---------------------------------------------------------------------------
# distill_turn
# ---------------------------------------------------------------------------


def test_distill_turn_returns_empty_when_client_unresolved(llm_module, monkeypatch):
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (None, None))
    assert llm_module.distill_turn("hi", "hello") == []


def test_distill_turn_returns_empty_when_not_worth_retaining(llm_module, monkeypatch):
    response = _fake_response_object_style(json.dumps({"worth_retaining": False, "candidates": []}))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    assert llm_module.distill_turn("u", "a") == []


def test_distill_turn_parses_candidates_and_forces_tool_choice(llm_module, monkeypatch):
    candidates = [
        {
            "kind": "concept",
            "topic": "atomic notes",
            "title": "Atomic Notes",
            "slug": "atomic-notes",
            "content": "One idea per note.",
        },
        {
            "kind": "entity_update",
            "topic": "Judy",
            "title": "",
            "slug": "",
            "content": "Judy is arriving in two weeks.",
        },
    ]
    payload = {"worth_retaining": True, "candidates": candidates}
    response = _fake_response_object_style(json.dumps(payload))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "some-model"))

    result = llm_module.distill_turn("user said something", "assistant replied")
    assert result == candidates
    assert client.last_call_kwargs["tool_choice"] == "required"
    assert client.last_call_kwargs["tools"][0]["function"]["name"] == "record_candidates"
    assert client.last_call_kwargs["model"] == "some-model"


def test_distill_turn_parses_dict_style_tool_call(llm_module, monkeypatch):
    payload = {"worth_retaining": False, "candidates": []}
    response = _fake_response_dict_style(json.dumps(payload))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    assert llm_module.distill_turn("u", "a") == []


def test_distill_turn_filters_non_dict_candidates(llm_module, monkeypatch):
    payload = {"worth_retaining": True, "candidates": ["not-a-dict", {"kind": "concept"}]}
    response = _fake_response_object_style(json.dumps(payload))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    assert llm_module.distill_turn("u", "a") == [{"kind": "concept"}]


def test_distill_turn_never_raises_on_client_exception(llm_module, monkeypatch):
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (_ExplodingClient(), "m"))
    assert llm_module.distill_turn("u", "a") == []


def test_distill_turn_never_raises_on_malformed_json(llm_module, monkeypatch):
    response = _fake_response_object_style("{not valid json")
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    assert llm_module.distill_turn("u", "a") == []


def test_distill_turn_truncates_pathological_input(llm_module, monkeypatch):
    """The hard safety cap, not normal-path truncation -- only bites on
    an absurdly large input, and the call must still go through."""
    payload = {"worth_retaining": False, "candidates": []}
    response = _fake_response_object_style(json.dumps(payload))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    monkeypatch.setattr(llm_module, "_MAX_INPUT_TOKENS", 10)  # 10 tokens ~= 40 chars

    huge = "x" * 100_000
    llm_module.distill_turn(huge, huge)
    sent_text = client.last_call_kwargs["messages"][1]["content"]
    assert len(sent_text) <= 40 + len("USER: \n\nASSISTANT: ")


# ---------------------------------------------------------------------------
# judge_merge
# ---------------------------------------------------------------------------


def test_judge_merge_returns_none_with_no_hits(llm_module, monkeypatch):
    # Should short-circuit before even resolving a client.
    called = []
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: called.append(1) or (None, None))
    result = llm_module.judge_merge({"kind": "concept", "content": "x"}, [])
    assert result is None
    assert called == []


def test_judge_merge_returns_none_when_client_unresolved(llm_module, monkeypatch):
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (None, None))
    hit_notes = [{"uuid": "abc", "title": "Existing", "body": "..."}]
    assert llm_module.judge_merge({"kind": "concept", "content": "x"}, hit_notes) is None


def test_judge_merge_parses_merge_decision_and_includes_hit_bodies(llm_module, monkeypatch):
    decision = {"action": "merge", "merge_target_ref": "note-uuid-1"}
    response = _fake_response_object_style(json.dumps(decision))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    hit_notes = [
        {"uuid": "note-uuid-1", "title": "Judy", "body": "Judy's travel plans."},
        {"uuid": "note-uuid-2", "title": "Unrelated", "body": "Nothing to do with Judy."},
    ]
    candidate = {"kind": "entity_update", "content": "Judy is arriving in two weeks."}

    result = llm_module.judge_merge(candidate, hit_notes)
    assert result == decision
    assert client.last_call_kwargs["tool_choice"] == "required"
    assert client.last_call_kwargs["tools"][0]["function"]["name"] == "record_merge_decision"
    sent_text = client.last_call_kwargs["messages"][1]["content"]
    # Both hit bodies were fetched and included -- a single comparison
    # call across all hits, not one call per hit.
    assert "note-uuid-1" in sent_text
    assert "note-uuid-2" in sent_text
    assert "Judy's travel plans." in sent_text
    assert "Nothing to do with Judy." in sent_text


def test_judge_merge_parses_create_decision(llm_module, monkeypatch):
    decision = {"action": "create"}
    response = _fake_response_object_style(json.dumps(decision))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    hit_notes = [{"uuid": "note-uuid-1", "title": "Unrelated", "body": "..."}]
    result = llm_module.judge_merge({"kind": "concept", "content": "new idea"}, hit_notes)
    assert result == decision


def test_judge_merge_never_raises_on_client_exception(llm_module, monkeypatch):
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (_ExplodingClient(), "m"))
    hit_notes = [{"uuid": "note-uuid-1", "title": "T", "body": "B"}]
    assert llm_module.judge_merge({"kind": "concept", "content": "x"}, hit_notes) is None


def test_judge_merge_never_raises_on_malformed_json(llm_module, monkeypatch):
    response = _fake_response_object_style("{not valid json")
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))
    hit_notes = [{"uuid": "note-uuid-1", "title": "T", "body": "B"}]
    assert llm_module.judge_merge({"kind": "concept", "content": "x"}, hit_notes) is None


# ---------------------------------------------------------------------------
# _resolve_client (unchanged plumbing, shared by both stages)
# ---------------------------------------------------------------------------


def test_resolve_client_returns_none_none_when_provider_resolution_raises(llm_module, monkeypatch):
    import agent.auxiliary_client as aux

    def _raise(**kwargs):
        raise RuntimeError("no provider configured")

    monkeypatch.setattr(aux, "_resolve_task_provider_model", _raise)
    assert llm_module._resolve_client() == (None, None)


def test_resolve_client_returns_none_none_when_agent_auxiliary_client_unimportable(
    llm_module, monkeypatch
):
    import agent

    had_attr = hasattr(agent, "auxiliary_client")
    monkeypatch.setitem(sys.modules, "agent.auxiliary_client", None)
    if had_attr:
        monkeypatch.delattr(agent, "auxiliary_client", raising=False)

    assert llm_module._resolve_client() == (None, None)


def test_resolve_client_returns_client_and_model_on_success(llm_module, monkeypatch):
    import agent.auxiliary_client as aux

    monkeypatch.setattr(
        aux,
        "_resolve_task_provider_model",
        lambda task: ("openai", "gpt-test", None, "sk-fake", None),
    )
    sentinel_client = object()
    monkeypatch.setattr(
        aux,
        "resolve_provider_client",
        lambda provider, model, **kw: (sentinel_client, model),
    )

    client, model = llm_module._resolve_client()
    assert client is sentinel_client
    assert model == "gpt-test"


# ---------------------------------------------------------------------------
# distill_messages -- the on_pre_compress entry point (batch, not a single turn)
# ---------------------------------------------------------------------------


def test_distill_messages_returns_empty_for_no_messages(llm_module):
    assert llm_module.distill_messages([]) == []


def test_distill_messages_returns_empty_when_only_system_messages(llm_module, monkeypatch):
    called = []
    monkeypatch.setattr(llm_module, "_distill_text", lambda text: called.append(text) or [])
    result = llm_module.distill_messages([{"role": "system", "content": "SOUL.md stuff"}])
    assert result == []
    assert called == []  # never even reaches the LLM call -- nothing to distill


def test_distill_messages_excludes_system_and_non_string_content(llm_module, monkeypatch):
    captured = {}

    def _fake_distill_text(text):
        captured["text"] = text
        return []

    monkeypatch.setattr(llm_module, "_distill_text", _fake_distill_text)
    messages = [
        {"role": "system", "content": "SECRET_SYSTEM_MARKER"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
        {"role": "assistant", "content": [{"type": "tool_use"}]},  # non-string content, skipped
        {"role": "tool", "content": None},  # falsy content, skipped
    ]
    llm_module.distill_messages(messages)

    assert "SECRET_SYSTEM_MARKER" not in captured["text"]
    assert "USER: hello there" in captured["text"]
    assert "ASSISTANT: hi back" in captured["text"]


def test_distill_messages_delegates_to_distill_text(llm_module, monkeypatch):
    payload = {"worth_retaining": True, "candidates": [{"kind": "concept"}]}
    response = _fake_response_object_style(json.dumps(payload))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    messages = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
    result = llm_module.distill_messages(messages)
    assert result == [{"kind": "concept"}]
