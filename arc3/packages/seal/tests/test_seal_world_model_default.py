"""The world-model default (``world_model_default``): with an optional world
model, the system prompt's world-model section states attempting one as the
default, and so does the user prompt of every level's first turn, the game's
first included; off, or with a mandatory model, neither does."""

from __future__ import annotations

import dataclasses

import pytest

from seal.prompts import STANDALONE_MODES, SealPrompts
from seal.state import StepSummary

LINE = "at least attempt to write one, and only stop if you are convinced it's not useful on this game"


def _modes(*, on: bool, optional: bool = True):
    return dataclasses.replace(
        STANDALONE_MODES, world_model=True, world_model_optional=optional, world_model_default=on
    )


def _user(*, on: bool, summary: StepSummary | None, any_actions_executed: bool) -> str:
    return SealPrompts().user_prompt_text(
        _modes(on=on),
        reset_reason=None,
        summary=summary,
        any_actions_executed=any_actions_executed,
        current_step=1,
        current_level=1,
        observed_max_level=1,
        valid_actions=[],
        valid_actions_observed=True,
        game_over_active=False,
    )


def _summary(*, level_transition: bool) -> StepSummary:
    return StepSummary(
        executed_count=1,
        executed_actions=["UP"],
        level=1,
        level_transition=level_transition,
        run_complete=False,
        game_over=False,
    )


@pytest.mark.parametrize(("on", "optional", "shown"), [(True, True, True), (False, True, False), (True, False, False)])
def test_the_system_prompt_states_the_default(on: bool, optional: bool, shown: bool) -> None:
    """The system prompt's world-model section, which carries the line."""
    text = SealPrompts()._world_model_addendum(_modes(on=on, optional=optional))
    assert text.count(LINE) == (1 if shown else 0)
    if shown:  # right after the optional opening
        assert text.index(LINE) < text.index("`test_world_model()` replays")


def test_the_games_first_turn_states_the_default() -> None:
    assert LINE in _user(on=True, summary=None, any_actions_executed=False)


def test_a_new_levels_first_turn_states_the_default() -> None:
    assert LINE in _user(on=True, summary=_summary(level_transition=True), any_actions_executed=True)


def test_a_turn_within_a_level_does_not() -> None:
    assert LINE not in _user(on=True, summary=_summary(level_transition=False), any_actions_executed=True)


def test_off_no_level_start_states_it() -> None:
    assert LINE not in _user(on=False, summary=None, any_actions_executed=False)
    assert LINE not in _user(on=False, summary=_summary(level_transition=True), any_actions_executed=True)
