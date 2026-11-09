"""A reply the reply cap cut off is not the model stopping: with the answer
standing it is nudged to continue, and only a reply that ended on its own ends
the analyzer entry."""

from __future__ import annotations

import numpy as np
import pytest
import ullm.backend
import ullm.conv
from duck.agent import DuckAgent
from duck.prompts import DuckPrompts
from duck.state import AnswerReading, AnswerSlot

import taaf.game
import taaf.grid

_GRID = taaf.grid.Grid(np.zeros((1, 1), dtype=np.int8))


def _view() -> taaf.game.GameView:
    return taaf.game.GameView(game_id="cut-reply", given=((_GRID, _GRID),), questions=(_GRID,))


def _reply(stop_reason: ullm.conv.StopReason) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="still reasoning about the second attempt"),
        stop_reason=stop_reason,
        token_usage=ullm.backend.TokenUsage(completion_tokens=24_576, num_requests=1),
    )


def _script(monkeypatch: pytest.MonkeyPatch, replies: list[ullm.conv.LLMResponse]) -> list[list[str]]:
    """Stand in for the LLM block. The first block writes a complete answer, as
    a python call in it would; each block ends on the next scripted reply with
    no tool call. Returns the prompts each block was opened with."""
    opened: list[list[str]] = []
    queue = list(replies)

    async def fake_block(self: DuckAgent, ctx: object, prompts: list[ullm.conv.UserPrompt], **_: object) -> int:
        del ctx
        opened.append([prompt.content.to_string() for prompt in prompts])
        self.messages.extend(prompts)
        self.reading = AnswerReading(bound=True, rows=((AnswerSlot(state="ok", grid=_GRID),),))
        self.messages.append(ullm.conv.ResponseTurn(elements=[queue.pop(0)], status="processed"))
        return 1

    def fake_state_prompts(self: DuckAgent, view: taaf.game.GameView) -> list[ullm.conv.UserPrompt]:
        del self, view
        return [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="STATE"))]

    monkeypatch.setattr(DuckAgent, "_run_turn_block", fake_block)
    monkeypatch.setattr(DuckAgent, "_build_state_prompts", fake_state_prompts)
    return opened


def _agent() -> DuckAgent:
    agent = DuckAgent(max_tokens_per_game=None)
    agent._n_questions = 1
    return agent


async def test_a_cut_reply_with_the_answer_standing_is_nudged_to_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    opened = _script(monkeypatch, [_reply("max_tokens"), _reply("max_tokens"), _reply("stop")])
    calls, nudges = await agent._run_analyzer_entry(None, _view())  # type: ignore[arg-type]
    assert (calls, nudges) == (3, 2)
    assert opened[1] == opened[2] == [DuckPrompts().answered_nudge]


async def test_a_reply_that_ended_on_its_own_ends_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent()
    opened = _script(monkeypatch, [_reply("stop"), _reply("stop")])
    calls, nudges = await agent._run_analyzer_entry(None, _view())  # type: ignore[arg-type]
    assert (calls, nudges) == (1, 0)
    assert opened == [["STATE"]]
