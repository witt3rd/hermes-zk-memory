"""llm.judge_turn() / llm._resolve_client() -- write-time judge, no real
LLM provider ever contacted. All client access is monkeypatched."""

from __future__ import annotations

import json
import sys
import types


def test_judge_turn_returns_none_when_client_unresolved(llm_module, monkeypatch):
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (None, None))
    assert llm_module.judge_turn("hi", "hello") is None


def _fake_response_object_style(args_json: str):
    """Build a fake response matching the object-attribute access pattern:
    response.choices[0].message.tool_calls[0].function.arguments"""

    function = types.SimpleNamespace(name="record_retain", arguments=args_json)
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


def test_judge_turn_parses_object_style_tool_call(llm_module, monkeypatch):
    verdict = {
        "worth_retaining": True,
        "slug": "test-slug",
        "title": "Test Title",
        "body": "Test body.",
    }
    response = _fake_response_object_style(json.dumps(verdict))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "some-model"))

    result = llm_module.judge_turn("user said something", "assistant replied")
    assert result == verdict
    # forced tool_choice + the judge tool were actually sent
    assert client.last_call_kwargs["tool_choice"] == "required"
    assert client.last_call_kwargs["tools"][0]["function"]["name"] == "record_retain"
    assert client.last_call_kwargs["model"] == "some-model"


def test_judge_turn_parses_dict_style_tool_call(llm_module, monkeypatch):
    verdict = {"worth_retaining": False}
    response = _fake_response_dict_style(json.dumps(verdict))
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "some-model"))

    result = llm_module.judge_turn("u", "a")
    assert result == verdict


def test_judge_turn_returns_none_when_no_tool_calls(llm_module, monkeypatch):
    message = types.SimpleNamespace(tool_calls=None)
    choice = types.SimpleNamespace(message=message)
    response = types.SimpleNamespace(choices=[choice])
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    assert llm_module.judge_turn("u", "a") is None


def test_judge_turn_returns_none_when_arguments_empty(llm_module, monkeypatch):
    response = _fake_response_object_style("")
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    assert llm_module.judge_turn("u", "a") is None


def test_judge_turn_never_raises_on_client_exception(llm_module, monkeypatch):
    class _ExplodingClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise RuntimeError("provider is down")

    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (_ExplodingClient(), "m"))
    assert llm_module.judge_turn("u", "a") is None


def test_judge_turn_never_raises_on_malformed_json(llm_module, monkeypatch):
    response = _fake_response_object_style("{not valid json")
    client = _FakeClient(response)
    monkeypatch.setattr(llm_module, "_resolve_client", lambda: (client, "m"))

    assert llm_module.judge_turn("u", "a") is None


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
