"""The audit turn: a game that would stop with its answer standing is handed
the puzzle restated with that answer and its last reply, once, and ends at the
next stop, stating as its own confidence (``answer_confidence``) the number its
final reply gave, where the prompt asked for one; a game without the audit
states none. Nothing before the turn differs from a game without it."""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import ullm.backend
import ullm.backends.stub
import ullm.conv
from duck import CheckMode, DuckAgent
from duck.prompts import DuckPrompts
from duck.python_tool import DuckPythonTool

import taaf.game
import taaf.grid
from platypus.context import Produced
from platypus.contract import HANDICAP_HINT
from platypus.llm import LLMConfig, build_llm_context

RULE = "The rule: mirror each row."


def _view() -> taaf.game.GameView:
    """A tiny mirror task with one question."""
    rng = np.random.default_rng(0)
    grids = [taaf.grid.Grid(rng.integers(0, 10, size=(3, 4), dtype=np.int8)) for _ in range(3)]
    given = tuple((g, taaf.grid.Grid(np.fliplr(g.data))) for g in grids[:2])
    return taaf.game.GameView(game_id="audit", given=given, questions=(grids[2],))


def _call(code: str, tokens: int = 10) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code}),
        token_usage=ullm.backend.TokenUsage(completion_tokens=tokens, num_requests=1),
    )


def _say(text: str, tokens: int = 10) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text=text),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(completion_tokens=tokens, num_requests=1),
    )


#: A game that answers and stops, then, audited, adds a second attempt and stops again.
SCRIPT = (
    _call("test_out[0][0] = np.fliplr(test_in[0])"),
    _say(RULE),
    _call("test_out[0][1] = test_in[0]"),
    _say("1. Support ... 4. Confidence: \\boxed{70}"),
)


def _agent(**kwargs: object) -> DuckAgent:
    return DuckAgent(
        reply_cap_tokens=64,
        check_mode=CheckMode.ENABLED,
        repl=DuckPythonTool(check_mode=CheckMode.ENABLED, isolate_network=False),
        exact_token_counts=False,
        # The stub has no reasoning parser to take the register injection out
        # of the reply the audit quotes.
        inject_text_on_first_prompt="",
        inject_text_on_later_prompts="",
        llm_config=LLMConfig(sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=None)}),
        **kwargs,  # type: ignore[arg-type]
    )


class _TemperatureScript(ullm.backends.stub.StubBackend):
    """A scripted backend that keeps each request's temperature."""

    temperatures: list[float | None]

    def __post_init__(self) -> None:
        self.temperatures = []
        super().__post_init__()

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> object:
        self.temperatures.append(getattr(sampling_params, "temperature", None))
        return super()._prepare_input(sampling_params, turn, conversation, send_back_reasoning)


async def _play(
    agent: DuckAgent, *responses: ullm.conv.LLMResponse
) -> tuple[_TemperatureScript, list[taaf.game.Submission]]:
    backend = cast(_TemperatureScript, _TemperatureScript.from_script(*responses))
    submissions: list[taaf.game.Submission] = []

    def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        del produced
        submissions.append(submission)

    ctx, _meter = build_llm_context(
        game_id="audit", submissions=submissions, rng=random.Random(7), backend=backend, submit=submit
    )
    try:
        await agent._do_move(ctx, _view())
    finally:
        if agent.repl is not None:
            await agent.repl.clear_state()
    return backend, submissions


def _user_texts(agent: DuckAgent) -> list[str]:
    return [element.content.to_string() for element in agent.messages if isinstance(element, ullm.conv.UserPrompt)]


def _audit_text(agent: DuckAgent) -> str:
    """The audit prompt, which the changed answers of the audit's own call
    follow in the history."""
    (audit,) = [text for text in _user_texts(agent) if text.startswith(DuckPrompts().audit_opening)]
    return audit


async def test_a_game_that_would_stop_is_audited_once_then_ends() -> None:
    agent = _agent(max_tokens_per_game=None, audit_turn=True, prompts=DuckPrompts(audit_confidence_line=True))
    backend, submissions = await _play(agent, *SCRIPT)
    assert backend._call_index == 4
    audit = _audit_text(agent)
    assert f"> {RULE}" in audit
    assert "test_out[0][1]: empty" in audit
    question = _view().questions[0]
    assert submissions[-1].attempts[0] == (taaf.grid.Grid(np.fliplr(question.data)), question)
    assert agent.answer_confidence() == 70
    assert "4. Confidence." in audit


async def test_the_confidence_is_asked_and_read_only_under_its_line() -> None:
    """Without its line: three questions, and the reply's boxed 70 read as no
    confidence, since none was asked for."""
    agent = _agent(max_tokens_per_game=None, audit_turn=True, prompts=DuckPrompts(audit_confidence_line=False))
    await _play(agent, *SCRIPT)
    audit = _audit_text(agent)
    assert "3. Your answer." in audit
    assert "Confidence." not in audit
    assert not any(line.startswith("4. ") for line in audit.splitlines())
    assert agent.answer_confidence() is None


async def test_a_game_without_the_audit_states_no_confidence_whatever_it_replies() -> None:
    agent = _agent(max_tokens_per_game=None, audit_turn=False, prompts=DuckPrompts(audit_confidence_line=True))
    await _play(agent, SCRIPT[0], SCRIPT[3])
    assert agent.answer_confidence() is None


async def test_the_audit_changes_nothing_before_it_fires() -> None:
    audited = _agent(max_tokens_per_game=None, audit_turn=True)
    await _play(audited, *SCRIPT)
    plain = _agent(max_tokens_per_game=None, audit_turn=False)
    backend, _submissions = await _play(plain, *SCRIPT[:2])
    assert backend._call_index == 2
    assert audited._system_prompt_text() == plain._system_prompt_text()
    before = [element.to_string() for element in plain.messages]
    assert [element.to_string() for element in audited.messages[: len(before)]] == before


async def test_a_spent_budget_ends_the_game_unaudited() -> None:
    agent = _agent(max_tokens_per_game=20, audit_turn=True)
    backend, _submissions = await _play(agent, *SCRIPT)
    assert backend._call_index == 2
    assert agent._audit_quote is None


def _profiles(agent: DuckAgent) -> DuckAgent:
    """The play profile at 0.6 and an audit profile at 0.2."""
    agent.llm_config.sampling_profiles = {
        "play": ullm.backend.GenericSamplingParameters(max_tokens=None, temperature=0.6),
        "audit": ullm.backend.GenericSamplingParameters(max_tokens=None, temperature=0.2),
    }
    return agent


async def test_the_audit_turn_and_what_follows_sample_on_the_audit_profile() -> None:
    agent = _profiles(_agent(max_tokens_per_game=None, audit_turn=True, audit_profile="audit"))
    agent._validate_coherence()
    backend, _submissions = await _play(agent, *SCRIPT)
    assert backend.temperatures == [0.6, 0.6, 0.2, 0.2]


async def test_without_an_audit_profile_the_audit_samples_on_the_play_profile() -> None:
    agent = _profiles(_agent(max_tokens_per_game=None, audit_turn=True))
    backend, _submissions = await _play(agent, *SCRIPT)
    assert backend.temperatures == [0.6] * 4


def test_an_audit_profile_must_exist() -> None:
    agent = _agent(audit_turn=True, audit_profile="audit")
    with pytest.raises(ValueError, match="names no sampling profile"):
        agent._validate_coherence()


def test_the_audit_is_refused_under_a_mandatory_check() -> None:
    agent = DuckAgent(audit_turn=True, check_mode=CheckMode.MANDATORY)
    with pytest.raises(ValueError, match="audit_turn"):
        agent._validate_coherence()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Confidence: \\boxed{85}", 85),
        ("\\boxed{ 70% } then \\boxed{60\\%}", 60),
        ("\\boxed{72.6}", 73),
        ("\\boxed{120}", None),
        ("no number here", None),
    ],
)
def test_the_confidence_is_the_last_boxed_number(text: str, expected: int | None) -> None:
    assert DuckPrompts(audit_confidence_line=True).read_audit_confidence(text) == expected
    assert DuckPrompts(audit_confidence_line=False).read_audit_confidence(text) is None


class _Offers:
    """A move context that records what each slot offer reports."""

    def __init__(self) -> None:
        self.hints: list[dict[str, Any]] = []
        self.backend = SimpleNamespace(total_usage=SimpleNamespace(completion_tokens=123))

    async def yield_slot(self, hints: dict[str, Any]) -> None:
        self.hints.append(dict(hints))


async def test_the_audit_handicap_rides_every_offer_from_the_audits_first_compaction() -> None:
    """Not before the audit, not at a compaction before it, not at the audit's
    first offer without one: from its first compaction on, and on every offer
    after it."""
    agent = _agent(audit_turn=True, audit_handicap_tokens=10_000_000)
    ctx = _Offers()
    for audited, compacted in ((False, True), (True, False), (True, True), (True, False)):
        agent._audit_quote = "the rule" if audited else None
        agent._last_trim_fired = compacted
        await agent._offer_slot(cast(Any, ctx))
    assert [hints.get(HANDICAP_HINT) for hints in ctx.hints] == [None, None, 10_000_000, 10_000_000]
    assert ctx.hints[2]["compacted"] and ctx.hints[2]["tokens"] == 123


async def test_no_audit_handicap_asks_for_nothing() -> None:
    agent = _agent(audit_turn=True)
    ctx = _Offers()
    agent._audit_quote = "the rule"
    agent._last_trim_fired = True
    await agent._offer_slot(cast(Any, ctx))
    assert HANDICAP_HINT not in ctx.hints[0]


def test_a_negative_audit_handicap_is_refused() -> None:
    with pytest.raises(ValueError, match="audit_handicap_tokens"):
        _agent(audit_turn=True, audit_handicap_tokens=-1)._validate_fields()
