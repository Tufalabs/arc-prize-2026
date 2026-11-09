"""Animations from a recorded run, as the seal's kernel holds them.

A state's ``raw.frame`` is the move's animation followed by the frame it
settled on. The seal's kernel keeps a turn's animation frames with
consecutive repeats dropped, a leading frame equal to the board before the
move included, and trailing frames equal to the settled frame dropped
(``seal.harness.game.History``); this module does the same, so a block
here is the block the model would read. A level's history is its own: the
turn index counts from the level's first state, and the move that wins a
level is left out, since its turn never reaches a history the model is
prompted with.

This is notebook tooling: it reads the run's pickles through ``taaf``,
which framediff itself never imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from run_transitions import load_states

from framediff.grid import Grid

#: The run the notebook reads: the seal's search-and-delta run, 4 passes of 25 games.
DEFAULT_RUN = Path(
    "/data/shared/research/arc_3_results/shared/seal_search_ab/seal_search_delta_r2/seal_20260918_104712"
)

__all__ = ["DEFAULT_RUN", "Animation", "animations", "first_and_longest", "load_states"]


@dataclass
class Animation:
    """One move's animation as the kernel keeps it, with the frames around it and where it came from."""

    game_id: str
    game_index: int
    level: int
    turn: int
    action: str
    outcome: str
    before: Grid
    frames: list[Grid]
    settled: Grid
    #: How many frames the engine sent before the kernel's trimming.
    raw_length: int

    @property
    def title(self) -> str:
        """One line naming the animation."""
        return (
            f"{self.game_id} (game {self.game_index}), level {self.level}, history[{self.turn}]: "
            f"{self.action}, {self.outcome}, {len(self.frames)} frames ({self.raw_length} sent)"
        )

    @property
    def before_name(self) -> str:
        """The expression the model holds the frame before the move under."""
        return f"history[{self.turn - 1}].frame.grid"

    @property
    def frame_name(self) -> str:
        """The expression of one animation frame, ``{k}`` where its index goes."""
        return f"history[{self.turn}].animation_frames[{{k}}].grid"


def _kept(before: Grid, sent: list[Grid], settled: Grid) -> list[Grid]:
    """The animation frames the kernel keeps: no frame equal to the one kept before it, none trailing equal to
    the settled frame."""
    kept: list[Grid] = []
    previous = before
    for frame in sent:
        if np.array_equal(frame, previous):
            continue
        kept.append(frame)
        previous = frame
    while kept and np.array_equal(kept[-1], settled):
        kept.pop()
    return kept


def _action(state: Any) -> str:
    """The action that led to a state, as the engine names it."""
    action = state.previous_action
    return "none" if action is None else str(action.id.name)


def animations(game: list[Any], game_index: int) -> list[Animation]:
    """Every move of one game run that the kernel would hold an animation for, in play order."""
    found: list[Animation] = []
    level_start = 0
    for index in range(1, len(game)):
        state, previous = game[index], game[index - 1]
        if int(state.raw.levels_completed) != int(previous.raw.levels_completed):
            level_start = index
            continue
        sent = [np.asarray(frame, dtype=np.int8) for frame in state.raw.frame]
        before = np.asarray(previous.raw.frame[-1], dtype=np.int8)
        frames = _kept(before, sent[:-1], sent[-1])
        if not frames:
            continue
        found.append(
            Animation(
                game_id=str(state.raw.game_id),
                game_index=game_index,
                level=int(state.raw.levels_completed) + 1,
                turn=index - level_start,
                action=_action(state),
                outcome=str(state.raw.state).rsplit(".", 1)[-1],
                before=before,
                frames=frames,
                settled=sent[-1],
                raw_length=len(sent) - 1,
            )
        )
    return found


def first_and_longest(states: list[list[Any]]) -> dict[str, tuple[Animation, Animation]]:
    """Per game id, over all its runs in order: its first animation, and its longest, the earliest at a tie."""
    chosen: dict[str, tuple[Animation, Animation]] = {}
    for game_index, game in enumerate(states):
        for animation in animations(game, game_index):
            if animation.game_id not in chosen:
                chosen[animation.game_id] = (animation, animation)
            elif len(animation.frames) > len(chosen[animation.game_id][1].frames):
                chosen[animation.game_id] = (chosen[animation.game_id][0], animation)
    return dict(sorted(chosen.items()))
