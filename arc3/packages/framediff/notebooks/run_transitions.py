"""Real transitions from a recorded run, sampled one move at a time.

A run's ``intermediate_states.pkl`` holds every state of every game: the
frame the model saw after each action, with the level it was on. A
transition is one action within a level: the frame before it, the frame
after it, and the frame the level opened with. The first frame of a level
is never a transition (the decomposer never runs on one), so a level with
no move after its start has no transitions.

Sampling picks a game, then a level of it, then a move of that level,
uniformly at each step, so a long level weighs the same as a short one.
This is notebook tooling: it reads the run's pickles through ``taaf``, which
framediff itself never imports.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from framediff.grid import Grid

#: The run the demo samples from: the seal's 4-pass, 25-game world-model run.
DEFAULT_RUN = Path("/data/shared/research/arc_3_results/shared/seal_wm_full/seal_20260916_220732")


@dataclass(frozen=True)
class Transition:
    """One action of one level: the frames around it and where it came from."""

    game_id: str
    game_index: int
    level: int
    move: int
    moves_in_level: int
    action: str
    outcome: str
    previous: Grid
    current: Grid
    level_start: Grid

    @property
    def title(self) -> str:
        """One line naming the transition."""
        return (
            f"{self.game_id} (game {self.game_index}), level {self.level}, "
            f"move {self.move} of {self.moves_in_level}: {self.action}, {self.outcome}"
        )


@dataclass(frozen=True)
class _Level:
    """The state indices of one level of one game: its start and every move after it."""

    level: int
    start: int
    moves: tuple[int, ...]


def load_states(run: Path = DEFAULT_RUN) -> list[list[Any]]:
    """Every game's states, in play order, from the run's pickle."""
    with (run / "intermediate_states.pkl").open("rb") as handle:
        return pickle.load(handle)


def _levels(states: list[Any]) -> list[_Level]:
    """The levels a game's states cover, keeping only those with a move after their start."""
    levels: list[_Level] = []
    for index, state in enumerate(states):
        level = int(state.raw.levels_completed)
        if not levels or level != levels[-1].level:
            levels.append(_Level(level, index, ()))
        else:
            levels[-1] = _Level(level, levels[-1].start, (*levels[-1].moves, index))
    return [level for level in levels if level.moves]


def _grid(state: Any) -> Grid:
    """The frame the model saw for a state: the last of its animation."""
    return np.asarray(state.raw.frame[-1], dtype=np.int8)


def _action(state: Any) -> str:
    """The action that led to a state, as the engine names it."""
    action = state.previous_action
    return "none" if action is None else str(action.id.name)


def sample_transitions(states: list[list[Any]], seed: int, count: int) -> list[Transition]:
    """``count`` transitions, a game then a level then a move drawn uniformly, from ``seed``."""
    rng = np.random.default_rng(seed)
    per_game = [_levels(game) for game in states]
    playable = [index for index, levels in enumerate(per_game) if levels]
    found: list[Transition] = []
    while len(found) < count:
        game_index = int(rng.choice(playable))
        level = per_game[game_index][int(rng.integers(len(per_game[game_index])))]
        index = int(level.moves[int(rng.integers(len(level.moves)))])
        game = states[game_index]
        state = game[index]
        found.append(
            Transition(
                game_id=str(state.raw.game_id),
                game_index=game_index,
                level=level.level,
                move=level.moves.index(index) + 1,
                moves_in_level=len(level.moves),
                action=_action(state),
                outcome=str(state.raw.state).rsplit(".", 1)[-1],
                previous=_grid(game[index - 1]),
                current=_grid(state),
                level_start=_grid(game[level.start]),
            )
        )
    return found


def transition(states: list[list[Any]], game_index: int, level: int, move: int) -> Transition:
    """One named transition: the ``move``-th move (from 1) of ``level`` in the game at ``game_index``."""
    game = states[game_index]
    found = [entry for entry in _levels(game) if entry.level == level]
    if not found:
        raise ValueError(f"game {game_index} has no moves on level {level}")
    index = found[0].moves[move - 1]
    state = game[index]
    return Transition(
        game_id=str(state.raw.game_id),
        game_index=game_index,
        level=level,
        move=move,
        moves_in_level=len(found[0].moves),
        action=_action(state),
        outcome=str(state.raw.state).rsplit(".", 1)[-1],
        previous=_grid(game[index - 1]),
        current=_grid(state),
        level_start=_grid(game[found[0].start]),
    )
