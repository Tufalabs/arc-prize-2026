"""``malformed_nudges``: a play reply refused on every re-sample is dropped and nudged, not crashed.

ullm re-samples a reply with no text and no tool call up to ``malformed_retries``
times and then raises, which crashes the game. In practice such replies loop
inside their reasoning, and a context that loops four times in a row loops on.
With nudges left, the duck drops the refused draws as the retries did, keeps the
turn's completed rounds, and queues a nudge that says the reply was discarded --
the model never sees it -- beside what stands of the answer. Played through a
scripted stub on ullm's own retry path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
from duck import CheckMode, DuckAgent
from duck.prompts import DuckPrompts, PromptModes
from duck.python_tool import DuckPythonTool
from duck.state import AnswerReading, AnswerSlot

import taaf.game
import taaf.grid
from platypus.context import Produced
from platypus.llm import LLMConfig, build_llm_context

LOOP = "Let me try to think about the generator differently.\n\nOK.\n"


def _view() -> taaf.game.GameView:
    grid = taaf.grid.Grid(np.zeros((1, 1), dtype=np.int8))
    return taaf.game.GameView(game_id="malformed-nudge", given=((grid, grid),), questions=(grid,))


def _looping() -> ullm.conv.LLMResponse:
    """A reply that went in circles inside its reasoning and stopped there."""
    return ullm.conv.LLMResponse(
        reasoning_content=ullm.conv.StringContent(text=LOOP * 20),
        final_content=ullm.conv.StringContent(text=""),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=40, num_requests=1),
    )


def _text(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=3, num_requests=1),
    )


def _python() -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[ullm.conv.ToolCallContent(tool_name="python", arguments={"code": "print('a round')"})]
        ),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=5, num_requests=1),
    )


@dataclass
class _Script(ullm.backends.stub.StubBackend):
    """Scripted replies, empty ones allowed; records each request's user messages
    and whether a refused draw's text reached it."""

    seen: list[list[str]] = field(default_factory=lambda: list[list[str]](), init=False)
    loop_seen: bool = field(default=False, init=False)

    def _check_constraints(self) -> None:
        assert self.responses
        ullm.backend.Backend._check_constraints(self)

    def _prepare_input(self, sampling_params: Any, turn: Any, conversation: Any, send_back_reasoning: str) -> Any:
        self.seen.append([e.content.to_string() for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)])
        self.loop_seen = self.loop_seen or LOOP in conversation.to_string()
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


def _agent(*, nudges: int, answered: bool) -> DuckAgent:
    agent = DuckAgent(
        reply_cap_tokens=256,
        max_tokens_per_game=10_000,
        malformed_retries=3,
        malformed_nudges=nudges,
        # An unanswered entry nudges until something stops it.
        max_llm_calls_per_move=2,
        llm_config=LLMConfig(
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=None)}
        ),
    )
    if answered:
        grid = taaf.grid.Grid(np.ones((1, 1), dtype=np.int8))
        agent.reading = AnswerReading(bound=True, rows=((AnswerSlot(state="ok", grid=grid),),))
    agent._n_questions = 1  # pyright: ignore[reportPrivateUsage]
    return agent


def _context(backend: ullm.backend.Backend) -> Any:
    def ignore(_submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced

    ctx, _meter = build_llm_context(
        game_id="malformed-nudge", submissions=[], rng=random.Random(5), backend=backend, submit=ignore
    )
    return ctx


async def _entry(agent: DuckAgent, ctx: Any, *, opening: bool) -> tuple[int, int]:
    """One analyzer entry; one whose answer stands runs only on an opening
    prompt (the audit's, in play), or on queued feedback."""
    prompts = [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="CONTINUE"))] if opening else None
    return await agent._run_analyzer_entry(ctx, _view(), opening=prompts)  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize("answered", [False, True])
async def test_a_reply_refused_on_every_retry_is_dropped_and_nudged(answered: bool) -> None:
    agent = _agent(nudges=1, answered=answered)
    backend = _Script(responses=[_looping()] * 4 + [_text("Done.")], on_exhausted="repeat_last")
    ctx = _context(backend)

    await _entry(agent, ctx, opening=answered)

    # The four refused draws left nothing in the history, the turn they would
    # have opened included, and the nudge waits as the next message: an
    # answered game goes on to it rather than ending.
    assert not any(isinstance(element, ullm.conv.ResponseTurn) for element in agent.messages)
    expected = DuckPrompts().malformed_nudge(agent._prompt_modes(), [] if answered else agent.reading.problems(1))  # pyright: ignore[reportPrivateUsage]
    assert [[c.to_string() for c in group] for group in agent._pending_tool_feedback] == [[expected]]  # pyright: ignore[reportPrivateUsage]

    await _entry(agent, ctx, opening=False)

    # Four refused draws, then the request the nudge closes.
    assert backend.seen[4][-1] == expected
    assert not backend.loop_seen, "a refused draw reached a request"


async def test_completed_rounds_of_the_turn_stay(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[str] = []

    async def run(self: DuckPythonTool) -> None:
        ran.append(str(self.arguments.get("code")))

    monkeypatch.setattr(DuckPythonTool, "run", run)
    agent = _agent(nudges=1, answered=False)
    backend = _Script(responses=[_python(), *[_looping()] * 4, _text("Done.")], on_exhausted="repeat_last")
    ctx = _context(backend)

    await _entry(agent, ctx, opening=False)

    assert ran == ["print('a round')"]
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn) and turn.status == "processed"
    assert [type(e).__name__ for e in turn.elements] == ["LLMResponse", "ToolResult"]
    assert len(agent._pending_tool_feedback) == 1  # pyright: ignore[reportPrivateUsage]

    await _entry(agent, ctx, opening=False)

    assert backend.seen[5][-1].startswith("Your last reply was discarded")
    assert not backend.loop_seen


async def test_off_the_refusal_crashes_the_game() -> None:
    agent = _agent(nudges=0, answered=False)
    backend = _Script(responses=[_looping()] * 4 + [_text("unused")])

    with pytest.raises(ullm.backend.MalformedOutputError, match="no final content"):
        await _entry(agent, _context(backend), opening=False)


async def test_past_the_count_the_refusal_crashes_the_game() -> None:
    agent = _agent(nudges=1, answered=False)
    backend = _Script(responses=[_looping()] * 8 + [_text("unused")])
    ctx = _context(backend)

    await _entry(agent, ctx, opening=False)
    with pytest.raises(ullm.backend.MalformedOutputError, match="no final content"):
        await _entry(agent, ctx, opening=False)


def test_the_nudge_says_the_reply_was_discarded_and_what_stands() -> None:
    surface = DuckPrompts()
    modes = PromptModes()

    answered = surface.malformed_nudge(modes, [])
    assert answered.startswith(
        "Your last reply was discarded: it went in circles and ended with no answer and no tool call."
    )
    assert "The answer in `test_out` stands." in answered
    open_ = surface.malformed_nudge(modes, ["test_out[0][0] is not set"])
    assert "The answer is not complete:\n  - test_out[0][0] is not set" in open_
    assert open_.endswith("Take a different approach, and make your next step a `python` call.")
    mandatory = surface.malformed_nudge(PromptModes(check_mode=CheckMode.MANDATORY), ["x"])
    assert mandatory.endswith(surface.mandatory_answer_repair())


def test_a_retained_draw_cannot_be_nudged() -> None:
    agent = DuckAgent(preserve_failed_draws=True, malformed_retries=0, malformed_nudges=1)
    with pytest.raises(ValueError, match="malformed_nudges=0"):
        agent.prepare_for_backend()
