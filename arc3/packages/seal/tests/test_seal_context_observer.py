"""U5: public context observations preserve evidence and reject unsafe requests."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.backend
import ullm.continuation
import ullm.conv

from seal.agent import SealAgent
from seal.state import MemoryEntry


@dataclass
class Recorder:
    """Keep lifecycle evidence without depending on the agent's private state."""

    events: list[tuple[Any, ...]] = field(default_factory=lambda: list[tuple[Any, ...]]())

    def prepare_request(self, params: ullm.backend.GenericSamplingParameters) -> None:
        self.events.append(("prepare",))

    def prompt_measured(self, count: int, *, epoch: int, trigger: int, compacting: bool) -> None:
        self.events.append(("count", count, epoch, compacting))

    def history_rewritten(self, *, epoch: int) -> None:
        self.events.append(("rewrite", epoch))

    def request_prepared(
        self, params: ullm.backend.GenericSamplingParameters, *, epoch: int, prompt_tokens: int, target: int
    ) -> None:
        self.events.append(("ready", epoch, prompt_tokens))
        params.request_context["observed_epoch"] = epoch


def _live_agent(monkeypatch: pytest.MonkeyPatch, observer: Recorder) -> SealAgent:
    agent = SealAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
        journal=True,
        token_native_context=True,
        compact_before_request_only=True,
        require_exact_context=True,
        context_observer=observer,
        context_window_tokens=4096,
        compact_to_tokens=1500,
        safety_margin_tokens=0,
    )
    agent.messages = [
        ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="play")),
        ullm.conv.ResponseTurn(elements=[], status="intermediate"),
    ]
    agent._live_conversation = ullm.conv.Conversation(elements=agent.messages, status="intermediate")
    agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=512)

    async def count(*args: Any) -> int:
        size = 300 + sum(len(element.to_string()) for element in agent.messages)
        return ullm.backend.TokenCount(size, list(range(size)))

    monkeypatch.setattr(agent, "_live_backend", SimpleNamespace(token_count=count))
    return agent


async def test_pre_request_compaction_observes_exact_evidence_and_keeps_observer_picklable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = Recorder()
    agent = _live_agent(monkeypatch, observer)
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    response = ullm.conv.LLMResponse(reasoning_content=ullm.conv.StringContent(text="reasoning " * 700))
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    turn.elements.append(response)
    before = list(turn.elements)
    observer.events.clear()
    await agent._persist_history()
    assert turn.elements == before and observer.events == []
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    rewrite = next(i for i, event in enumerate(observer.events) if event[0] == "rewrite")
    evidence = observer.events[rewrite - 1]
    assert evidence[0] == "count" and isinstance(evidence[1], ullm.backend.TokenCount)
    assert evidence[2:] == (0, True)
    assert evidence[1] > agent._context_budget_tokens()
    assert observer.events[-1][0:2] == ("ready", 1)
    assert observer.events[-1][2] <= agent.compact_to_tokens
    assert agent._live_params is not None
    assert agent._live_params.request_context["observed_epoch"] == 1
    assert pickle.loads(pickle.dumps(observer)).events[-1] == observer.events[-1]


@pytest.mark.parametrize("failure", [None, "drift"])
async def test_strict_count_never_falls_back_to_an_estimate(
    monkeypatch: pytest.MonkeyPatch,
    failure: str | None,
) -> None:
    observer = Recorder()
    agent = _live_agent(monkeypatch, observer)

    async def unavailable(*args: Any) -> None:
        if failure:
            raise ullm.continuation.HistoryDriftError("changed prefix")

    monkeypatch.setattr(agent, "_live_backend", SimpleNamespace(token_count=unavailable))
    with pytest.raises((RuntimeError, ullm.continuation.HistoryDriftError)):
        await agent._trim_live_messages()
    assert not any(event[0] == "ready" for event in observer.events)


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("memory_size", [500, 2000, 4000])
async def test_compaction_counts_journal_restored_from_pruned_memory_calls(
    monkeypatch: pytest.MonkeyPatch, memory_size: int, strict: bool
) -> None:
    observer = Recorder()
    agent = _live_agent(monkeypatch, observer)
    agent.evict_turns = 1
    agent.require_exact_context = strict

    async def count(*args: Any) -> int:
        size = 300 + len(agent.memory_prompt_snapshot or "") + sum(len(e.to_string()) for e in agent.messages)
        return ullm.backend.TokenCount(size, list(range(size)))

    monkeypatch.setattr(agent, "_live_backend", SimpleNamespace(token_count=count))
    await agent._trim_live_messages()
    await agent._observe_request_attempt(0)
    entry = MemoryEntry(index=1, response=1, step=1, level=0, last_action=None, text="m" * memory_size)
    agent.memory_entries = [entry]
    agent.memory_entry_count = 1
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    turn.elements.extend(
        [
            ullm.conv.LLMResponse(reasoning_content=ullm.conv.StringContent(text="thought " * 800)),
            ullm.conv.ToolResult(
                call_id="memory1",
                tool_name="memory",
                content=ullm.conv.StringContent(text=agent.prompts.memory_append_acknowledgement(entry)),
            ),
            *[ullm.conv.LLMResponse(final_content=ullm.conv.StringContent(text="tail " * 20)) for _ in range(6)],
        ]
    )
    observer.events.clear()
    if memory_size == 4000 or (strict and memory_size == 2000):
        error = "compaction.*target" if strict else "context budget"
        with pytest.raises(RuntimeError, match=error):
            await agent._trim_live_messages()
        assert not any(event[0] == "ready" for event in observer.events)
    else:
        await agent._trim_live_messages()
        await agent._observe_request_attempt(0)
        assert observer.events[-1][0:2] == ("ready", 1)
        if memory_size == 500:
            assert observer.events[-1][2] <= agent.compact_to_tokens
            assert turn.elements, "compaction should retain the rounds that still fit"
        else:
            assert observer.events[-1][2] <= agent._context_budget_tokens()
            assert not turn.elements, "an irreducible journal must exhaust pruning before returning"
    assert [event for event in observer.events if event[0] == "rewrite"] == [("rewrite", 0)]
    assert entry.text in (agent.memory_prompt_snapshot or "")
    assert agent.memory_entries == [entry]
