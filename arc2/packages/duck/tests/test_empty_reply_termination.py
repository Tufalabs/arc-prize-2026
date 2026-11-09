"""``terminate_on_empty_reply``: a retained reply with no text and no tool call
ends the game as a give-up instead of opening another nudged block."""

from __future__ import annotations

import numpy as np
import pytest
import ullm.backend
import ullm.conv
from duck.agent import DuckAgent

import taaf.game
import taaf.grid
from platypus.contract import GiveUpError


def _view() -> taaf.game.GameView:
    grid = taaf.grid.Grid(np.zeros((1, 1), dtype=np.int8))
    return taaf.game.GameView(game_id="empty-reply", given=((grid, grid),), questions=(grid,))


def _empty_reply() -> ullm.conv.LLMResponse:
    """What a training backend keeps for a one-token ``<|im_end|>`` draw."""
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.EmptyContent(),
        stop_reason="malformed",
        malformed_error="Model produced no final content (no text response, no tool call).",
        malformed_call_count=1,
        token_usage=ullm.backend.TokenUsage(completion_tokens=1, num_requests=1),
    )


def _text_reply(text: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=3, num_requests=1),
    )


def _agent(**kwargs: object) -> DuckAgent:
    agent = DuckAgent(max_tokens_per_game=None, preserve_failed_draws=True, **kwargs)  # type: ignore[arg-type]
    agent._n_questions = 1
    return agent


def _script(monkeypatch: pytest.MonkeyPatch, agent: DuckAgent, replies: list[ullm.conv.LLMResponse]) -> list[int]:
    """Stand in for the LLM block: append the next scripted reply as a processed
    turn, with the manager's feedback prompt trailing a malformed one."""
    blocks: list[int] = []
    queue = list(replies)

    async def fake_block(self: DuckAgent, ctx: object, prompts: list[ullm.conv.UserPrompt], **_: object) -> int:
        del ctx
        self.messages.extend(prompts)
        reply = queue.pop(0)
        elements: list[ullm.conv.Content | ullm.conv.LLMResponse | ullm.conv.UserPrompt] = [reply]
        if reply.malformed_call_count:
            elements.append(
                ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="Malformed reply: no text and no tool call."))
            )
        self.messages.append(ullm.conv.ResponseTurn(elements=elements, status="processed"))  # type: ignore[arg-type]
        blocks.append(len(prompts))
        return 1

    def fake_state_prompts(self: DuckAgent, view: taaf.game.GameView) -> list[ullm.conv.UserPrompt]:
        del self, view
        return [ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="STATE"))]

    monkeypatch.setattr(DuckAgent, "_run_turn_block", fake_block)
    monkeypatch.setattr(DuckAgent, "_build_state_prompts", fake_state_prompts)
    return blocks


def test_the_flag_is_off_by_default() -> None:
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).terminate_on_empty_reply
        is False
    )


def test_an_empty_retained_reply_is_recognised_behind_its_feedback_prompt() -> None:
    agent = _agent()
    agent.messages = [
        ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="STATE")),
        ullm.conv.ResponseTurn(
            elements=[_empty_reply(), ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="feedback"))],  # type: ignore[list-item]
            status="processed",
        ),
    ]
    assert agent._last_turn_was_empty_reply() is True
    agent.messages[-1] = ullm.conv.ResponseTurn(elements=[_text_reply("no tools, but text")], status="processed")
    assert agent._last_turn_was_empty_reply() is False
    capped = _text_reply("")
    capped.stop_reason = "max_tokens"
    agent.messages[-1] = ullm.conv.ResponseTurn(elements=[capped], status="processed")
    assert agent._last_turn_was_empty_reply() is False


async def test_off_the_entry_nudges_past_an_empty_reply(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(max_llm_calls_per_move=2)
    blocks = _script(monkeypatch, agent, [_empty_reply(), _text_reply("still thinking")])
    calls, nudges = await agent._run_analyzer_entry(None, _view())  # type: ignore[arg-type]
    assert (calls, nudges) == (2, 1)
    assert blocks == [1, 1]
    nudge = agent.messages[-2]
    assert isinstance(nudge, ullm.conv.UserPrompt)


async def test_on_the_first_empty_reply_gives_the_game_up(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(terminate_on_empty_reply=True, max_llm_calls_per_move=2)
    blocks = _script(monkeypatch, agent, [_empty_reply(), _text_reply("never requested")])
    with pytest.raises(GiveUpError, match="empty reply"):
        await agent._run_analyzer_entry(None, _view())  # type: ignore[arg-type]
    assert blocks == [1]
    # The retained draw stays in history with its malformed count for the trainer.
    turn = agent.messages[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    assert isinstance(turn.elements[0], ullm.conv.LLMResponse)
    assert turn.elements[0].malformed_call_count == 1


async def test_on_a_text_reply_without_tools_is_still_nudged(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _agent(terminate_on_empty_reply=True, max_llm_calls_per_move=2)
    _script(monkeypatch, agent, [_text_reply("thinking aloud"), _text_reply("more")])
    calls, nudges = await agent._run_analyzer_entry(None, _view())  # type: ignore[arg-type]
    assert (calls, nudges) == (2, 1)
