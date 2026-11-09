"""U5/A2: observe actual requests and veto rewrites across retries and summary resets."""

from __future__ import annotations

import asyncio
import copy
import random
from dataclasses import dataclass, field
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager
from duck.agent import DuckAgent

from platypus import LLMConfig
from platypus.context import Produced, SubmitAnswer
from platypus.llm import build_llm_context


def _collect_into(submissions: list[Any]) -> SubmitAnswer:
    """A submit channel that keeps every answer, in order."""

    def submit(submission: Any, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    return submit


def _context(backend: Any) -> Any:
    submissions: list[Any] = []
    ctx, _meter = build_llm_context(
        game_id="observed",
        submissions=submissions,
        rng=random.Random(7),
        backend=backend,
        submit=_collect_into(submissions),
    )
    return ctx


@dataclass
class RequestObserver:
    events: list[tuple[Any, ...]] = field(default_factory=lambda: list[tuple[Any, ...]]())
    requests: int = 0
    reject_rewrite: bool = False
    reject_request: bool = False

    def prepare_request(self, params: ullm.backend.GenericSamplingParameters) -> None:
        params.request_context.pop("observed", None)
        self.events.append(("prepare",))

    def prompt_measured(self, count: int, *, epoch: int, trigger: int, compacting: bool) -> None:
        self.events.append(("count", count, epoch, trigger, compacting))

    def history_rewritten(self, *, epoch: int) -> None:
        self.events.append(("rewrite", epoch))
        if self.reject_rewrite:
            raise RuntimeError("observer rejected rewrite")

    def request_prepared(
        self, params: ullm.backend.GenericSamplingParameters, *, epoch: int, prompt_tokens: int, target: int
    ) -> None:
        if self.reject_request:
            raise RuntimeError("observer rejected request")
        self.requests += 1
        params.request_context["observed"] = {
            "request": self.requests,
            "epoch": epoch,
            "count": prompt_tokens,
            "seed": params.seed,
            "max_tokens": params.max_tokens,
        }
        self.events.append(("ready", epoch, target))


@dataclass
class ObservedBackend(ullm.backends.stub.StubBackend):
    requests: list[dict[str, Any]] = field(default_factory=lambda: list[dict[str, Any]]())
    malformed_first: bool = False
    lift_after_first: bool = False

    async def _token_count(self, sampling_params: Any, conversation: ullm.conv.Conversation) -> int:
        await asyncio.sleep(0)
        size = len(conversation.to_string()) // 4
        return ullm.backend.TokenCount(size, list(range(size)))

    async def _text_token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _prepare_input(
        self, sampling_params: Any, turn: Any, conversation: ullm.conv.Conversation, send_back_reasoning: Any
    ) -> Any:
        metadata = copy.deepcopy(sampling_params.request_context["observed"])
        assert metadata["request"] == len(self.requests) + 1
        assert metadata["epoch"] == conversation.history_epoch
        assert metadata["count"] == len(conversation.to_string()) // 4
        assert metadata["seed"] == sampling_params.seed
        assert metadata["max_tokens"] == sampling_params.max_tokens
        self.requests.append({**metadata, "text": conversation.to_string()})
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        if len(self.requests) == 1:
            if self.malformed_first:
                raise ullm.backend.MalformedOutputError("bad first draw")
            if self.lift_after_first:
                element.stop_reason = "max_tokens"
                element.token_usage = ullm.backend.TokenUsage(prompt_tokens=1, completion_tokens=10)


def _ordinary(observer: RequestObserver) -> DuckAgent:
    return DuckAgent(
        reply_cap_tokens=512,
        images="off",
        show_changed_answers=False,
        max_tokens_per_game=None,
        token_native_context=True,
        require_exact_context=True,
        compact_before_request_only=True,
        context_observer=observer,
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
    )


@pytest.mark.parametrize("reject_rewrite", [False, True])
async def test_malformed_retry_observes_each_actual_prompt_and_the_restoration(reject_rewrite: bool) -> None:
    observer = RequestObserver(reject_rewrite=reject_rewrite)
    agent = _ordinary(observer)
    backend = ObservedBackend.from_script("bad", "ok", malformed_first=True)
    assert isinstance(backend, ObservedBackend)
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="play"))
    if reject_rewrite:
        with pytest.raises(RuntimeError, match="observer rejected rewrite"):
            await agent._run_turn_block(_context(backend), [prompt], remaining_calls=1)
        assert prompt.content.to_string() == "play\n"
        assert agent.history_epoch == 0
    else:
        await agent._run_turn_block(_context(backend), [prompt], remaining_calls=1)
        assert prompt.content.to_string() == "play"
        assert agent.history_epoch == 1
    assert len(backend.requests) == 2
    first, retry = backend.requests
    assert retry["seed"] == first["seed"] + 1
    assert "play\n\n" in retry["text"] and retry["text"] != first["text"]
    assert [event[0] for event in observer.events if event[0] != "count"] == [
        "prepare",
        "ready",
        "prepare",
        "ready",
        "rewrite",
    ]
    assert observer.events[-1] == ("rewrite", 0)
    # The per-game subscription must unwind even when an observer rejects a rewrite.
    assert agent._live_conversation is not None
    agent._live_conversation.mark_history_rewritten()
    assert observer.events[-1] == ("rewrite", 0)


@pytest.mark.parametrize("reject_rewrite", [False, True])
async def test_cap_rollback_notifies_before_removal_and_reprepares(reject_rewrite: bool) -> None:
    observer = RequestObserver(reject_rewrite=reject_rewrite)
    agent = _ordinary(observer)
    backend = ObservedBackend.from_script("discarded", "kept", lift_after_first=True)
    assert isinstance(backend, ObservedBackend)
    conversation = ullm.conv.Conversation(elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="play"))])
    params = ullm.backend.GenericSamplingParameters(max_tokens=512)
    agent.messages = conversation.elements
    agent._live_conversation, agent._live_backend, agent._live_params = conversation, backend, params
    manager = ullm.manager.TurnManager(
        conversation=conversation,
        response_manager=ullm.manager.ResponseManager(
            backend=backend,
            conversation=conversation,
            sampling_params=params,
            before_attempt=agent._observe_request_attempt,
        ),
        before_round=agent._trim_live_messages,
        max_generated_tokens_per_turn=10,
        per_turn_cap_lifted=lambda: bool(backend.requests),
    )
    with ullm.conv.observe_history_rewrites(agent._observe_history_rewrite):
        if reject_rewrite:
            with pytest.raises(RuntimeError, match="observer rejected rewrite"):
                await manager.run()
            assert "discarded" in conversation.last_turn.to_string()
            assert len(backend.requests) == 1 and conversation.history_epoch == 0
        else:
            await manager.run()
            assert len(backend.requests) == 2 and conversation.history_epoch == 1
            assert [request["epoch"] for request in backend.requests] == [0, 1]
            assert backend.requests[0]["max_tokens"] == 10
            assert backend.requests[1]["max_tokens"] == 512
            assert "discarded" not in backend.requests[1]["text"]
    assert [event for event in observer.events if event[0] == "rewrite"] == [("rewrite", 0)]


@pytest.mark.parametrize("exact", [False, True])
async def test_irreducible_prompt_reuses_final_measurement(monkeypatch: pytest.MonkeyPatch, exact: bool) -> None:
    observer = RequestObserver()
    agent = _ordinary(observer)
    agent.context_window_tokens = 4096
    agent.compact_to_tokens = 1000
    agent.require_exact_context = exact
    prompt = ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="x" * 20000))
    conversation = ullm.conv.Conversation(elements=[prompt])
    agent._live_conversation = conversation
    agent._live_backend = ObservedBackend()
    agent._live_params = ullm.backend.GenericSamplingParameters(max_tokens=512)
    final_events: list[tuple[Any, ...]] = []
    original_prune = agent._prune_current_turn

    async def prune(*args: Any, **kwargs: Any) -> int:
        result = await original_prune(*args, **kwargs)
        if not result:
            observer.events = final_events
        return result

    monkeypatch.setattr(agent, "_prune_current_turn", prune)
    expected = "compaction cannot meet target" if exact else "context budget"
    with pytest.raises(RuntimeError, match=expected) as failure:
        await agent._trim_messages_for_context(conversation.elements, enforce=True)
    counts = [event[1] for event in final_events if event[0] == "count"]
    assert len(counts) == 1
    assert f"{counts[0]} tokens" in str(failure.value)
    assert conversation.elements == [prompt]
