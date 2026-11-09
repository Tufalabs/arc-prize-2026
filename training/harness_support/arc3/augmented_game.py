"""An ARC-3 game whose every frame is recolored by one fixed color permutation.

Worker side only: imports the ARC-3 TAAF and arcengine. The permutation is
drawn in ``harness_support/augmentation.py``; this applies it at TAAF's two
subclass hooks, where every frame of the run enters the framework: the
initial state (``_start_game``, including a fast-forwarded start level) and
each step (``_execute_action``, every level, every animation frame). TAAF's
``current_state``, its run record and the states handed to the agent are all
built from what these hooks return, so the agent never sees an unpermuted
frame. Actions pass to the engine unchanged, and level progress (the score)
is the engine's own.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import arcengine
import numpy as np
from taaf.game import GameState
from taaf.game_api import GameAPI

from harness_support.augmentation import ARC3_PALETTE_SIZE, permute_colors

# TAAF's ``Game.execute_action`` puts a frame of this color before a level
# win the engine reported with a single frame (its safety net for
# less-conformant games).
_LEVEL_WIN_FILL = 14


@dataclass
class ColorPermutedGameAPI(GameAPI):
    """``GameAPI`` that shows color ``c`` as ``color_map[c]`` in every frame."""

    color_map: tuple[int, ...] = field(default=(), kw_only=True)

    def __post_init__(self) -> None:
        parent = getattr(super(), "__post_init__", None)
        if parent is not None:
            parent()
        if sorted(self.color_map) != list(range(ARC3_PALETTE_SIZE)):
            raise ValueError("an ARC-3 color map permutes all 16 palette colors")
        self.color_map = tuple(int(color) for color in self.color_map)

    def _start_game(self, session):
        return GameState(raw=self._recolored(super()._start_game(session).raw))

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        # The level before this action: TAAF updates its current state only
        # after the hook returns (a checkpoint replay has none yet).
        before = self._current_state.levels_completed if self._current_state else None
        raw = self._recolored(super()._execute_action(action).raw)
        if (
            before is not None
            and raw.levels_completed > before
            and raw.state != arcengine.GameState.WIN
            and len(raw.frame) <= 1
        ):
            # Supply TAAF's level-win frame in the permuted palette, so the
            # safety net finds two frames and adds no unpermuted one.
            width, height = self.grid_size
            fill = self.color_map[_LEVEL_WIN_FILL]
            raw.frame = [np.full((height, width), fill, dtype=np.int8), *raw.frame]
        return GameState(raw=raw)

    def _recolored(self, raw: arcengine.FrameDataRaw) -> arcengine.FrameDataRaw:
        """A copy of the engine's frame data with recolored frames.

        A copy, because the engine wrapper keeps the object it returned as
        its observation, and new arrays, because the engine writes into the
        buffers it hands back.
        """
        recolored = raw.model_copy()
        recolored.frame = [permute_colors(frame, self.color_map) for frame in raw.frame]
        return recolored
