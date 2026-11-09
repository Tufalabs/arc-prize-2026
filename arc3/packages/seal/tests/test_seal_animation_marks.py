"""How a batch's moves are marked for their animation frames, past what the scripted reference reaches.

The reference ``seal_scripted_animation`` shows the marks in play; these pin
the line's edge cases: the fully marked move inside the first five, at the
first elided place, and a long batch whose gap hid no animated move. And the
cleaning both the kernel and the host count by.
"""

from __future__ import annotations

import dataclasses
import operator

from seal.frame_delta import kept_animation
from seal.prompts import STANDALONE_MODES, SealPrompts
from seal.state import StepSummary


def _line(animations: list[tuple[int, int | None] | None]) -> str:
    """The executed-actions line for moves named A0, A1, ... with these animations, animations described."""
    summary = StepSummary(
        executed_count=len(animations),
        executed_actions=[f"A{position}" for position in range(len(animations))],
        level=1,
        level_transition=False,
        run_complete=False,
        game_over=False,
        animations=animations,
    )
    modes = dataclasses.replace(STANDALONE_MODES, animation_description="available", animation_marks=True)
    return (
        SealPrompts()
        .user_prompt_text(
            modes,
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
        .splitlines()[1]
    )


def test_a_long_batch_whose_marked_move_is_among_the_first_five() -> None:
    """The fully marked move needs no place of its own; the gap counts the animated moves it hid."""
    animations: list[tuple[int, int | None] | None] = [None] * 12
    animations[2], animations[6] = (3, 4), (2, 8)
    assert _line(animations) == (
        "Executed actions (first 5 and last 5): A0, A1, A2 (3 animation frames, history[4].animation_frames, use "
        "`describe_animation(4)` for a concise code description), A3, A4, ... (2 more, 1 with animation frames) ..., "
        "A7, A8, A9, A10, A11."
    )


def test_a_marked_move_just_past_the_first_five_opens_the_gap() -> None:
    """At the first elided place there is nothing before it to elide."""
    animations: list[tuple[int, int | None] | None] = [None] * 12
    animations[5] = (3, 7)
    assert _line(animations) == (
        "Executed actions (first 5, the move with its animation named, and last 5): A0, A1, A2, A3, A4, "
        "A5 (3 animation frames, history[7].animation_frames, use `describe_animation(7)` for a concise code "
        "description), ..., A7, A8, A9, A10, A11."
    )


def test_a_long_batch_without_animations_reads_as_before() -> None:
    """No animated move: the line the seal always wrote."""
    assert _line([None] * 11) == "Executed actions (first 5 and last 5): A0, A1, A2, A3, A4, ..., A6, A7, A8, A9, A10."


def test_a_move_whose_turn_left_with_its_level_keeps_its_count() -> None:
    """No history index: the short mark, and the full mark goes to the first move that has one."""
    assert _line([(3, None), (2, 1)]) == (
        "Executed actions: A0 (3 anim frames), A1 (2 animation frames, history[1].animation_frames, use "
        "`describe_animation(1)` for a concise code description)."
    )


def test_kept_animation_drops_repeats_and_the_settled_hold() -> None:
    """A leading copy of the board before, a repeated frame, and trailing copies of the settled frame go."""
    assert kept_animation("b", ["b", "x", "x", "y", "x", "s", "s"], "s", operator.eq) == ["x", "y", "x"]
    assert kept_animation(None, ["b", "b"], "s", operator.eq) == ["b"]
    assert kept_animation("b", ["s"], "s", operator.eq) == []
