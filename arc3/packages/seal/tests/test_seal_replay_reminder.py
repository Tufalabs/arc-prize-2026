"""A repeated attempt pointed out (``replay_reminder``): the reminder names the
attempt and says the level is winnable, and the prompt says ``history``
already holds what a replay would show. Which RESET closes a repeat is
``platypus.attempts``'s, tested there."""

from __future__ import annotations

import dataclasses

import pytest

from seal.prompts import STANDALONE_MODES, SealPrompts
from seal.state import StepSummary


def _prompt(repeated: list[str] | None, *, on: bool, level_transition: bool = False) -> str:
    summary = StepSummary(
        executed_count=1,
        executed_actions=["RESET"],
        level=1,
        level_transition=level_transition,
        run_complete=False,
        game_over=False,
        repeated=repeated,
    )
    return SealPrompts().user_prompt_text(
        dataclasses.replace(STANDALONE_MODES, replay_reminder=on),
        reset_reason=None,
        summary=summary,
        any_actions_executed=True,
        current_step=1,
        current_level=1,
        observed_max_level=1,
        valid_actions=[],
        valid_actions_observed=True,
        game_over_active=False,
    )


def test_the_reminder_names_the_attempt_and_says_the_level_is_winnable() -> None:
    text = _prompt(["UP", "DOWN", "RESET"], on=True)
    assert (
        "You have now repeated UP, DOWN, RESET from the start of the level exactly as before. The game is "
        "deterministic, so this cost you score while you could have just gotten everything from `history`.\n\n"
        "If you're just repeating actions because you've given up, keep in mind that the level is absolutely "
        "winnable. Review your assumptions and strategy."
    ) in text


def test_a_long_attempt_is_named_by_its_ends() -> None:
    labels = [f"MOUSE(row={i}, col=0)" for i in range(14)] + ["RESET"]
    text = _prompt(labels, on=True)
    assert "repeated MOUSE(row=0, col=0), " in text
    assert "MOUSE(row=4, col=0), ... (5 more) ..., MOUSE(row=10, col=0)" in text
    assert "MOUSE(row=13, col=0), RESET from the start" in text


@pytest.mark.parametrize(("on", "level_transition"), [(False, False), (True, True)])
def test_no_reminder_with_the_switch_off_or_on_a_new_level(on: bool, level_transition: bool) -> None:
    assert "You have now repeated" not in _prompt(["UP", "RESET"], on=on, level_transition=level_transition)


@pytest.mark.parametrize("on", [True, False])
def test_the_history_hint_rides_the_switch(on: bool) -> None:
    text = SealPrompts()._runtime_state_addendum(  # pyright: ignore[reportPrivateUsage]
        dataclasses.replace(STANDALONE_MODES, replay_reminder=on, file_tool_names=("view", "create"))
    )
    assert ("including those before a RESET" in text) == on
