from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "debug_identify_the_agent-0001"

COLOR_EMPTY = 0
COLOR_BLUE_AGENT = 9
COLOR_GREEN_TARGET = 14


@dataclass(frozen=True)
class LevelSpec:
    grid_size: int
    agent_start: tuple[int, int]
    target_pos: tuple[int, int]


LEVELS = (
    LevelSpec(grid_size=5, agent_start=(3, 1), target_pos=(1, 3)),
    LevelSpec(grid_size=15, agent_start=(13, 1), target_pos=(1, 13)),
)

ACTION_TO_DELTA = {
    int(GameAction.ACTION1.value): (-1, 0),
    int(GameAction.ACTION2.value): (1, 0),
    int(GameAction.ACTION3.value): (0, -1),
    int(GameAction.ACTION4.value): (0, 1),
}


def _token(color: int) -> np.ndarray:
    return np.array([[int(color)]], dtype=np.int8)


def _build_level(spec: LevelSpec, index: int) -> Level:
    return Level(
        name=f"Level {index + 1}",
        grid_size=(spec.grid_size, spec.grid_size),
        sprites=[
            Sprite(
                pixels=np.full((spec.grid_size, spec.grid_size), COLOR_EMPTY, dtype=np.int8),
                name="floor",
                x=0,
                y=0,
                layer=0,
                tags=["floor", "sys_static"],
                collidable=False,
            ),
            Sprite(
                pixels=_token(COLOR_GREEN_TARGET),
                name="target",
                x=spec.target_pos[1],
                y=spec.target_pos[0],
                layer=1,
                tags=["target"],
                collidable=False,
            ),
            Sprite(
                pixels=_token(COLOR_BLUE_AGENT),
                name="agent",
                x=spec.agent_start[1],
                y=spec.agent_start[0],
                layer=2,
                tags=["agent"],
                collidable=True,
            ),
        ],
        data={"level_index": index},
    )


class DebugIdentifyTheAgent(ARCBaseGame):
    def __init__(self, seed: int = 0):
        self._agent: Sprite | None = None
        self._agent_pos = LEVELS[0].agent_start
        levels = [_build_level(spec, index) for index, spec in enumerate(LEVELS)]
        camera = Camera(width=LEVELS[-1].grid_size, height=LEVELS[-1].grid_size, background=COLOR_EMPTY)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=len(levels),
            available_actions=[1, 2, 3, 4],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = LEVELS[self.level_index]
        self._agent_pos = spec.agent_start
        agents = level.get_sprites_by_name("agent")
        if not agents:
            raise RuntimeError("debug_identify_the_agent level is missing the agent sprite.")
        self._agent = agents[0]
        self._sync_agent()

    def _sync_agent(self) -> None:
        if self._agent is None:
            return
        row, col = self._agent_pos
        self._agent.set_position(col, row)

    def _move_agent(self, row_delta: int, col_delta: int) -> None:
        spec = LEVELS[self.level_index]
        row, col = self._agent_pos
        self._agent_pos = (
            max(0, min(spec.grid_size - 1, row + row_delta)),
            max(0, min(spec.grid_size - 1, col + col_delta)),
        )
        self._sync_agent()

    def step(self) -> None:
        action_id = int(getattr(self.action.id, "value", self.action.id))
        delta = ACTION_TO_DELTA.get(action_id)
        if delta is not None:
            self._move_agent(delta[0], delta[1])

        if self._agent_pos == LEVELS[self.level_index].target_pos:
            self.next_level()

        self.complete_action()
