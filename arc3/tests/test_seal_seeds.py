"""The example world models are real: each one passes ``test_world_model()``
on the game it describes.

A game starts with one of them as its ``world_model.py``, so the model reads
it as a model that works. These are the two games they model, as TAAF games,
and a played sequence on each — winning, through a RESET, never touching a
move the example declines with ``Unknown`` — replayed through the same
validation the gate runs.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arcengine
import numpy as np
from taaf.game import Game, GameState, RunSession

from platypus.contract import Action
from seal import seeds
from seal.action_names import to_model_action
from seal.agent import turn_payload
from seal.harness import game as harness_game
from seal.harness import test as harness_test
from seal.state import FramePayload, TimelineEntry


def _state(self: Game, grid: np.ndarray, state: arcengine.GameState, actions: list[int]) -> GameState:
    """One frame of a one-level game."""
    raw = arcengine.FrameDataRaw(
        game_id=self.game_id or "",
        state=state,
        levels_completed=1 if state == arcengine.GameState.WIN else 0,
        win_levels=1,
        available_actions=actions,
    )
    raw.frame = [grid]
    return GameState(raw=raw)


def _bar(grid: np.ndarray, left: int, width: int, energy: int, capacity: int) -> None:
    """The energy bar: a track, filled in proportion to the energy left."""
    grid[60:62, left : left + width] = 3
    grid[60:62, left : left + round(width * energy / capacity)] = 11


@dataclass
class RecolourGame(Game):
    """The keyboard example's game.

    A square in the middle of the board: LEFT darkens it by one colour index,
    RIGHT lightens it; colour 15 wins, colour 0 loses. Every move spends one of
    24 units of energy, and running out loses. UP moves the square up two rows,
    and a marker in the top-right corner cycles through three colours as moves
    are made — the two things the example does not model.
    """

    label: str = field(default="recolour_game", init=True)
    _colour: int = field(default=8, init=False, repr=False)
    _energy: int = field(default=24, init=False, repr=False)
    _top: int = field(default=28, init=False, repr=False)
    _moves: int = field(default=0, init=False, repr=False)

    _ACTIONS = [1, 3, 4]

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = 1
        self.base_actions_per_level = [7]
        self.hint = "recolour the square"
        self.grid_size = (64, 64)
        return self._reset()

    def _reset(self) -> GameState:
        self._colour, self._energy, self._top, self._moves = 8, 24, 28, 0
        return self._frame(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            return self._reset()
        self._moves += 1
        self._energy -= 1
        if action.id == arcengine.GameAction.ACTION1:
            self._top = max(8, self._top - 2)
        elif action.id == arcengine.GameAction.ACTION3:
            self._colour -= 1
        elif action.id == arcengine.GameAction.ACTION4:
            self._colour += 1
        if self._colour == 15:
            return self._frame(arcengine.GameState.WIN)
        if self._colour == 0 or self._energy <= 0:
            return self._frame(arcengine.GameState.GAME_OVER)
        return self._frame(arcengine.GameState.NOT_FINISHED)

    def _frame(self, state: arcengine.GameState) -> GameState:
        grid = np.zeros((64, 64), dtype=np.int8)
        grid[self._top : self._top + 6, 29:35] = self._colour
        grid[2:6, 58:62] = 6 + self._moves % 3
        _bar(grid, 8, 48, self._energy, 24)
        return _state(self, grid, state, self._ACTIONS)


@dataclass
class TileFlipGame(Game):
    """The MOUSE example's game.

    Twelve 16x16 tiles in the top 48 rows, three rows of four, each a grey or
    yellow body inside a one-cell border of colour 4. A click anywhere on a
    tile flips its body; all yellow wins. Every click spends one of 16 units of
    energy, and running out loses. A click below the tiles spends energy and
    changes nothing — what the example does not model.
    """

    label: str = field(default="tile_flip_game", init=True)
    _bodies: np.ndarray = field(default_factory=lambda: np.zeros((3, 4), dtype=np.int8), init=False, repr=False)
    _energy: int = field(default=16, init=False, repr=False)

    #: Yellow at the start: the rest are grey.
    _START_YELLOW = ((0, 0), (1, 2))

    def _start_game(self, session: RunSession) -> GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = 1
        self.base_actions_per_level = [10]
        self.hint = "flip the tiles"
        self.grid_size = (64, 64)
        return self._reset()

    def _reset(self) -> GameState:
        self._bodies = np.full((3, 4), 5, dtype=np.int8)
        for i, j in self._START_YELLOW:
            self._bodies[i, j] = 11
        self._energy = 16
        return self._frame(arcengine.GameState.NOT_FINISHED)

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.RESET:
            return self._reset()
        self._energy -= 1
        if action.id == arcengine.GameAction.ACTION6:
            row, col = int(action.data["y"]), int(action.data["x"])
            if row < 48:
                i, j = row // 16, col // 16
                self._bodies[i, j] = 11 if self._bodies[i, j] == 5 else 5
        if (self._bodies == 11).all():
            return self._frame(arcengine.GameState.WIN)
        if self._energy <= 0:
            return self._frame(arcengine.GameState.GAME_OVER)
        return self._frame(arcengine.GameState.NOT_FINISHED)

    def _frame(self, state: arcengine.GameState) -> GameState:
        grid = np.zeros((64, 64), dtype=np.int8)
        for i in range(3):
            for j in range(4):
                grid[16 * i : 16 * i + 16, 16 * j : 16 * j + 16] = 4
                grid[16 * i + 1 : 16 * i + 15, 16 * j + 1 : 16 * j + 15] = self._bodies[i, j]
        _bar(grid, 16, 32, self._energy, 16)
        return _state(self, grid, state, [6])


_OUTCOMES = {
    arcengine.GameState.WIN: "won",
    arcengine.GameState.GAME_OVER: "lost",
    arcengine.GameState.NOT_FINISHED: "playing",
}


def _payload(state: GameState, action: Action | None) -> dict[str, Any]:
    """One turn as the host sends it to the kernel, through the host's own
    serialization, so the history an example passes on is the one a game
    gives it."""
    grid = np.asarray(state.raw.frame[-1], dtype=np.int8)
    frame = FramePayload(
        grid_b64=base64.b64encode(grid.astype(np.uint8).tobytes()).decode(),
        shape=list(grid.shape),
        step=0,
        level=1,
    )
    return turn_payload(
        TimelineEntry(action=action, frame=frame, animation=[frame], outcome=_OUTCOMES[state.raw.state], level=1)
    )


def _played(game: Game, actions: list[Action]) -> harness_game.History:
    """The level's history after playing ``actions`` from the start."""
    history = harness_game.History()
    state = game._start_game(None)  # pyright: ignore[reportPrivateUsage, reportArgumentType]
    history._append_turn(_payload(state, None))  # pyright: ignore[reportPrivateUsage]
    for action in actions:
        state = game._execute_action(action.to_action_input())  # pyright: ignore[reportPrivateUsage]
        history._append_turn(_payload(state, action))  # pyright: ignore[reportPrivateUsage]
    return history


def _validate(tmp_path: Path, mouse: bool, game: Game, actions: list[Action]) -> harness_test.TestResult:
    """The example for ``mouse`` validated over ``actions`` played on ``game``."""
    path = tmp_path / "world_model.py"
    path.write_text(seeds.seed_source(mouse=mouse))
    history = _played(game, actions)
    names = [to_model_action(f"ACTION{i}") for i in (1, 3, 4)] if not mouse else ["MOUSE"]
    return harness_test.validate_world_model(str(path), history, ["RESET", *names])


def test_the_keyboard_example_models_the_recolour_game(tmp_path: Path) -> None:
    moves = ["ACTION4", "ACTION3", "RESET"] + ["ACTION4"] * 7
    result = _validate(tmp_path, False, RecolourGame(), [Action(name) for name in moves])
    assert not result, repr(result)
    # The marker is declined on every frame, and nothing else is loose.
    assert repr(result).endswith(
        "wildcards: render declined cells with -1 on 10 turns (up to 16 per turn, first history[0])."
    )


def test_the_keyboard_example_declines_up(tmp_path: Path) -> None:
    result = _validate(tmp_path, False, RecolourGame(), [Action("ACTION4"), Action("ACTION1")])
    assert "do_move raised Unknown('UP has not been played yet') on a move that IS in history" in repr(result)


def _click(i: int, j: int) -> Action:
    """A click on the middle of tile ``(i, j)``."""
    return Action("ACTION6", row=16 * i + 8, col=16 * j + 8)


def test_the_mouse_example_models_the_tile_flip_game(tmp_path: Path) -> None:
    grey = [(i, j) for i in range(3) for j in range(4) if (i, j) not in TileFlipGame._START_YELLOW]
    # One tile flipped and back, then a RESET, then every grey tile once —
    # the last one on its border, which counts as the tile.
    clicks = [_click(0, 1), _click(0, 1), Action("RESET")] + [_click(i, j) for i, j in grey[:-1]]
    last_i, last_j = grey[-1]
    clicks.append(Action("ACTION6", row=16 * last_i, col=16 * last_j))
    result = _validate(tmp_path, True, TileFlipGame(), clicks)
    assert not result, repr(result)
    assert "Passed loosely" not in repr(result)


def test_the_mouse_example_declines_a_click_below_the_tiles(tmp_path: Path) -> None:
    result = _validate(tmp_path, True, TileFlipGame(), [_click(0, 1), Action("ACTION6", row=52, col=20)])
    assert "no click below the tiles has been tried (row 52)" in repr(result)
