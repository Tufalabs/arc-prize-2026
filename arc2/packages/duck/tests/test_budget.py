"""The per-game token budget: what bounds a game, in place of a wall clock.

Seconds are not the game's to spend — what a second buys depends on the
fleet — so a game is bounded by what it generates. Upstream budgets per
LEVEL and leaves the game budget off; a task has no levels, so the game
budget is the whole of it, and upstream's rule that a game budget never
lifts when the queue drains carries over as there being no lift at all.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
import ullm.backend
import ullm.conv
from duck import CheckMode, DuckAgent
from duck.prompts import DuckPrompts

import platypus.diagnostics
import taaf.game
from platypus import GiveUpError
from platypus.context import Produced
from platypus.llm import build_llm_context


def _ignore(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
    """These tests never answer; the budget does not care that they do not."""
    del submission, produced


def context(spent: int = 0) -> Any:
    """A context whose backend has already been billed ``spent`` tokens."""
    backend = ullm.backend.DummyBackend()
    backend.total_usage.completion_tokens = spent
    ctx, _meter = build_llm_context(
        game_id="budget", submissions=[], rng=random.Random(0), backend=backend, submit=_ignore
    )
    return ctx


def test_none_disables_the_budget() -> None:
    unbounded = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=None)
    assert unbounded._per_turn_cap(context(spent=10**9)) is None


def test_the_cap_is_what_is_left() -> None:
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=3_000)
    assert agent._per_turn_cap(context()) == 3_000, "nothing spent yet: the whole allowance"
    assert agent._per_turn_cap(context(spent=1_800)) == 1_200


def test_a_spent_budget_gives_up_rather_than_running_another_turn() -> None:
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=3_000)
    with pytest.raises(GiveUpError, match="per-game token budget exhausted"):
        agent._per_turn_cap(context(spent=3_000))


def test_a_turn_is_never_handed_a_ceiling_of_zero() -> None:
    """A turn capped at zero generates nothing, so the manager appends a
    synthetic placeholder; the placeholder carries no tool call and no usage,
    so the entry nudges into another identical turn and the spend never
    advances to end it. The ceiling must be positive, or the turn must not
    happen — there is no third option that terminates.
    """
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=1_000)
    assert agent._per_turn_cap(context(spent=999)) == 1, "a sliver of allowance is still a runnable turn"
    with pytest.raises(GiveUpError, match="per-game token budget exhausted"):
        agent._per_turn_cap(context(spent=1_000))
    # And overspent, which a truncated turn can leave behind.
    with pytest.raises(GiveUpError):
        agent._per_turn_cap(context(spent=1_500))


def test_the_budget_follows_the_backend_meter() -> None:
    """Completion tokens billed to this game's private client, including a
    round no longer retained in the conversation — so trimming the prompt can
    never make a game look cheaper than it was."""
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=5_000)
    assert agent._game_tokens_used(context(spent=1_250)) == 1_250


def test_a_turn_is_charged_as_the_meter_delta(monkeypatch: pytest.MonkeyPatch) -> None:
    """What a turn cost is the meter's movement across it, reported on the
    turn's span; a meter that moved backwards is a backend bug, not a
    refund."""
    events: list[dict[str, Any]] = []

    class _Recorder:
        def event(self, kind: str, **values: Any) -> None:
            events.append({"kind": kind, **values})

    monkeypatch.setattr(platypus.diagnostics, "current_span", lambda: _Recorder())
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=5_000)
    ctx = context(spent=1_250)
    agent._charge_budget(ctx, game_tokens_before=500)
    assert events == [
        {"kind": "budget", "tokens_added": 750, "tokens_used_this_game": 1_250, "max_tokens_per_game": 5_000}
    ]
    with pytest.raises(RuntimeError, match="moved backwards"):
        agent._charge_budget(ctx, game_tokens_before=5_000)


def test_the_call_cap_bounds_an_entry_not_the_game() -> None:
    """Upstream's ``max_llm_calls_per_move`` ends one analyzer entry, after
    which the loop re-enters with a fresh state prompt; it is not a give-up.
    Zero, the default, leaves the token budget as the only bound."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).max_llm_calls_per_move
        == 0
    )
    capped = DuckAgent(reply_cap_tokens=None, max_llm_calls_per_move=2, max_tokens_per_game=None)
    assert capped._per_turn_cap(context(spent=10**9)) is None, "the call cap is not a token cap"


def test_the_shipped_default_bounds_a_game() -> None:
    """A duck built with no arguments must still stop on its own."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).max_tokens_per_game
        is not None
    )


def test_the_budget_warning_fires_once_at_the_first_crossing() -> None:
    """A late message can trigger a rushed long think, so it is one sentence,
    sent once: the first tool result after the remaining budget drops under
    the threshold, and never again in the game."""
    agent = DuckAgent(reply_cap_tokens=None, max_tokens_per_game=10_000, budget_warning_tokens=3_000)
    assert agent._budget_note(context(spent=5_000)) is None, "above the threshold: nothing"
    note = agent._budget_note(context(spent=7_500))
    assert note is not None and "2,500" in note
    assert agent._budget_note(context(spent=8_000)) is None, "warned already"
    assert DuckAgent(reply_cap_tokens=None, max_tokens_per_game=10_000)._budget_note(context(spent=9_900)) is None, (
        "off by default"
    )
    assert (
        DuckAgent(reply_cap_tokens=None, max_tokens_per_game=None, budget_warning_tokens=3_000)._budget_note(context())
        is None
    )


def test_mandatory_budget_warning_uses_the_checked_submission_channel() -> None:
    agent = DuckAgent(
        reply_cap_tokens=None,
        check_mode=CheckMode.MANDATORY,
        max_tokens_per_game=10_000,
        budget_warning_tokens=3_000,
        audit_turn=False,
        show_changed_answers=False,
    )
    note = agent._budget_note(context(spent=7_500))
    assert note is not None and "2,500" in note
    assert "`check(fn)`" in note and "test_out" not in note


@pytest.mark.parametrize(
    "spent,allowance,expected",
    [
        (12_345, 90_000, "about 77k of 90k"),
        (89_700, 90_000, "about under 1k of 90k"),
        (95_000, 90_000, "about 0k of 90k"),
        (12_345, None, None),
    ],
)
def test_standing_budget_status_uses_the_agents_meter(spent: int, allowance: int | None, expected: str | None) -> None:
    agent = DuckAgent(prompts=DuckPrompts(budget_line=True), max_tokens_per_game=allowance)
    note = agent._budget_note(context(spent))
    if expected is None:
        assert note is None
    else:
        assert note is not None and expected in note
    assert DuckAgent(max_tokens_per_game=allowance)._budget_note(context(spent)) is None


def test_standing_budget_status_survives_the_one_shot_warning() -> None:
    agent = DuckAgent(prompts=DuckPrompts(budget_line=True), max_tokens_per_game=10_000, budget_warning_tokens=3_000)
    assert agent._budget_note(context(5_000)) == "[budget] Tokens left on this puzzle: about 5k of 10k."
    first = agent._budget_note(context(7_500))
    assert first is not None and first.startswith("[budget] Tokens left on this puzzle: about 2k of 10k.\n")
    assert "2,500" in first
    assert agent._budget_note(context(8_000)) == "[budget] Tokens left on this puzzle: about 2k of 10k."


async def test_the_first_reply_cap_binds_the_first_call_only() -> None:
    """The turn manager rewrites ``max_tokens`` before every call from a
    value captured at turn start, so the cap has to be applied inside the
    call itself — and put back, or the lowered ceiling would follow the
    profile into the next block."""
    from duck.agent import _FirstReplyCappedResponseManager  # pyright: ignore[reportPrivateUsage]

    backend = ullm.backend.DummyBackend()
    seen: list[int | None] = []
    original_respond = backend.respond

    def spying_respond(params: Any, conversation: Any) -> Any:
        seen.append(params.max_tokens)
        return original_respond(params, conversation)

    backend.respond = spying_respond  # pyright: ignore[reportAttributeAccessIssue]
    params = ullm.backend.GenericSamplingParameters(max_tokens=16_384)
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys")),
        elements=[ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go"))],
        status="intermediate",
    )
    manager = _FirstReplyCappedResponseManager(
        backend=backend, conversation=conversation, sampling_params=params, first_reply_max_tokens=4_096
    )
    # The turn manager opens a turn before each call; stand in for it here.
    for _ in range(2):
        conversation.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
        await manager.run()
        manager.rearm()
        conversation.elements[-1].status = "processed"
    assert seen == [4_096, 16_384]
    assert params.max_tokens == 16_384, "the profile's ceiling is restored after the capped call"
    uncapped = _FirstReplyCappedResponseManager(
        backend=backend, conversation=conversation, sampling_params=params, first_reply_max_tokens=None
    )
    conversation.elements.append(ullm.conv.ResponseTurn(elements=[], status="intermediate"))
    await uncapped.run()
    assert seen[-1] == 16_384


async def test_the_slot_offer_reports_the_live_spend_and_the_swap_point() -> None:
    """What the host ranks on is what the duck hands it: the backend's own
    completion meter, live to the token, and whether the trim that just ran
    fired — so a parked game is priced at what it has generated, and a free
    swap is named as one."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    ctx = context(spent=1234)
    offers: list[dict[str, Any]] = []

    async def record(hints: Any) -> None:
        offers.append(dict(hints))

    ctx.yield_slot = record
    agent._last_trim_fired = True
    await agent._offer_slot(ctx)
    agent._last_trim_fired = False
    ctx.backend.total_usage.completion_tokens = 1300
    await agent._offer_slot(ctx)
    assert offers == [{"tokens": 1234, "compacted": True}, {"tokens": 1300, "compacted": False}]
