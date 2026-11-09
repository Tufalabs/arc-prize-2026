"""Duck exposes the existing ullm classifier without replaying budget-cut draws."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest
import ullm.backend
import ullm.conv
import ullm.manager

import platypus.llm
from duck import presets
from duck.agent import DuckAgent
from platypus import LLMMoveContext


def test_training_classification_is_an_explicit_build_override(monkeypatch: pytest.MonkeyPatch) -> None:
    def noop(*args: Any, **kwargs: Any) -> None:
        pass

    monkeypatch.setattr(presets, "configure", noop)
    monkeypatch.setattr(DuckAgent, "prepare_for_backend", noop)
    assert DuckAgent().classify_tool_calls is False
    assert DuckAgent.build(classify_tool_calls=True).classify_tool_calls is True


@pytest.mark.parametrize("classifies", [False, True])
async def test_turn_forwards_classification_and_disables_only_classifying_cap_replay(
    monkeypatch: pytest.MonkeyPatch, classifies: bool
) -> None:
    agent = DuckAgent(classify_tool_calls=classifies)
    tools = agent._turn_tools()

    def turn_tools(*args: Any) -> Any:
        return tools

    monkeypatch.setattr(agent, "_turn_tools", turn_tools)
    for name, value in [
        ("_system_prompt_text", "system"),
        ("_per_turn_cap", None),
        ("_current_level", 0),
        ("_game_tokens_used", 0),
        ("_uses_context_summary", False),
    ]:

        def constant(*args: Any, value: object = value, **kwargs: Any) -> object:
            return value

        monkeypatch.setattr(agent, name, constant)

    def sampling(ctx: Any) -> ullm.backend.GenericSamplingParameters:
        return ullm.backend.GenericSamplingParameters(max_tokens=100)

    def response_manager(**kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(sampling_params=kwargs["sampling_params"])

    monkeypatch.setattr(agent, "_sampling", sampling)
    monkeypatch.setattr(ullm.manager, "ResponseManager", response_manager)
    seen: dict[str, Any] = {}

    class Constructed(Exception):
        pass

    def manager(**kwargs: Any) -> NoReturn:
        seen.update(kwargs)
        raise Constructed

    monkeypatch.setattr(platypus.llm, "DiagnosticTurnManager", manager)
    with pytest.raises(Constructed):
        await agent._run_turn_block(
            cast(LLMMoveContext, SimpleNamespace(backend=object(), diag=None)),
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="play")),
            remaining_calls=1,
        )
    assert seen["classify_tool_calls"] is classifies
    assert (seen["per_turn_cap_lifted"] is None) is classifies


def test_native_factory_classification_tracks_the_final_agent_flag() -> None:
    @dataclass
    class NativeBackend:
        classify_tool_calls: bool = False

    factory = SimpleNamespace(backend_cls=NativeBackend, backend_kwargs={})
    agent = DuckAgent(classify_tool_calls=True)
    agent._sync_tool_classification(factory)
    assert factory.backend_kwargs == {"classify_tool_calls": True}
    agent.classify_tool_calls = False
    agent._sync_tool_classification(factory)
    assert factory.backend_kwargs == {"classify_tool_calls": False}


def test_classification_rejects_an_unsupported_factory_before_startup() -> None:
    agent = DuckAgent(classify_tool_calls=True)
    with pytest.raises(ValueError, match="classifier-capable backend"):
        agent._sync_tool_classification(SimpleNamespace(backend_kwargs={}))


async def test_summary_manager_classifies_before_its_response_observer(monkeypatch: pytest.MonkeyPatch) -> None:
    import test_context_summarization as summaries

    agent = summaries._summarizing_agent()
    agent.classify_tool_calls = True
    summaries._history(agent)
    seen: list[dict[str, Any]] = []
    observed: list[ullm.conv.LLMResponse] = []
    agent.response_observer = observed.append
    original = platypus.llm.DiagnosticTurnManager

    def manager(**kwargs: Any) -> Any:
        seen.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(platypus.llm, "DiagnosticTurnManager", manager)
    malformed = ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="missing_tool", arguments={}))
    backend = summaries._CountingScript(
        responses=[malformed, summaries._response(summaries.SUMMARY), summaries._response("DONE")],
        ordinary_count=agent._context_budget_tokens() + 1,
        compacted_count=500,
        summary_count=500,
    )
    await summaries._run_block(agent, backend)
    compactions = [item for item in seen if item.get("label") == "compaction"]
    assert compactions and all(item.get("classify_tool_calls") is True for item in compactions)
    assert any(response.malformed_call_count > 0 for response in observed)
