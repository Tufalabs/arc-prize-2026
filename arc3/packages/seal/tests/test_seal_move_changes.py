"""A batch's moves listed one per line with what each changed (``move_changes``)."""

from __future__ import annotations

import base64
import dataclasses

import numpy as np

from seal.agent import _move_change  # pyright: ignore[reportPrivateUsage]
from seal.prompts import STANDALONE_MODES, SealPrompts
from seal.state import FramePayload, MoveChange, StepSummary


def _report(
    names: list[str], changes: list[MoveChange | None], animations: list[tuple[int, int | None] | None] | None = None
) -> str:
    """The executed-actions report for these moves, animations described."""
    summary = StepSummary(
        executed_count=len(names),
        executed_actions=names,
        level=1,
        level_transition=False,
        run_complete=False,
        game_over=False,
        animations=animations or [],
        changes=changes,
    )
    modes = dataclasses.replace(STANDALONE_MODES, animation_description="available", animation_marks=True)
    text = SealPrompts().user_prompt_text(
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
    lines = text.splitlines()
    end = next(i for i, line in enumerate(lines) if i > 1 and not line.startswith(("- ", "...")))
    return "\n".join(lines[1:end])


def test_each_move_on_its_own_line_with_its_change_and_its_animation_mark() -> None:
    """A changed move, an inert one, and an animated one whose mark follows the change in the same brackets."""
    report = _report(
        ["MOUSE(row=19, col=11) on color 9", "UP", "DOWN"],
        [(52, (9, 10), (1, 11)), (0, (), ()), (8, (5, 11), (5, 11))],
        [None, None, (3, 4)],
    )
    assert report == (
        "Executed actions:\n"
        "- MOUSE(row=19, col=11) on color 9 (52 pix changed, colors 9,10->1,11)\n"
        "- UP (no change)\n"
        "- DOWN (8 pix changed, colors 5,11->5,11; 3 animation frames, history[4].animation_frames, use "
        "`describe_animation(4)` for a concise code description)"
    )


def test_a_level_win_carries_no_change_and_a_long_batch_keeps_its_gap_on_a_line() -> None:
    """The winning move's board is the next level's; the elided moves are one line."""
    names = [f"A{position}" for position in range(12)]
    changes: list[MoveChange | None] = [(1, (3,), (4,))] * 11 + [None]
    assert _report(names, changes) == "\n".join(
        [
            "Executed actions (first 5 and last 5):",
            *(f"- A{position} (1 pix changed, colors 3->4)" for position in range(5)),
            "...",
            *(f"- A{position} (1 pix changed, colors 3->4)" for position in range(7, 11)),
            "- A11",
        ]
    )


def test_the_change_counts_cells_and_their_colours_before_and_after() -> None:
    """Two cells change, from 5 and 9 to 9 and 5: a move."""

    def frame(grid: np.ndarray) -> FramePayload:
        return FramePayload(
            grid_b64=base64.b64encode(grid.astype(np.uint8).tobytes()).decode("ascii"),
            shape=list(grid.shape),
            step=0,
            level=1,
        )

    before = np.full((4, 4), 5)
    before[1, 1] = 9
    after = np.full((4, 4), 5)
    after[1, 2] = 9
    assert _move_change(frame(before), frame(after)) == (2, (5, 9), (5, 9))
    assert _move_change(None, frame(after)) is None
