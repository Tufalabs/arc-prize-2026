from __future__ import annotations

import random
from collections import deque

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

GAME_ID = "memory-0001"

GRID_WIDTH = 13
GRID_HEIGHT = 13
LEVEL_COUNT = 9

COLOR_BG = 0
COLOR_FLOOR = 1
COLOR_WALL = 5
COLOR_PLAYER = 9

PAD_PALETTE = tuple(
    color for color in (2, 3, 4, 6, 7, 8, 10, 11, 12, 13, 14, 15) if color not in {COLOR_BG, COLOR_PLAYER}
)
PLAYER_START = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
MAX_SOLVER_SEED_ATTEMPTS = 1024

ACTION_UP_ID = int(GameAction.ACTION1.value)
ACTION_DOWN_ID = int(GameAction.ACTION2.value)
ACTION_LEFT_ID = int(GameAction.ACTION3.value)
ACTION_RIGHT_ID = int(GameAction.ACTION4.value)
ACTION_SPACE_ID = int(GameAction.ACTION5.value)
ACTION_CLICK_ID = int(GameAction.ACTION6.value)

ACTION_TO_DELTA: dict[int, tuple[int, int]] = {
    ACTION_UP_ID: (0, -1),
    ACTION_DOWN_ID: (0, 1),
    ACTION_LEFT_ID: (-1, 0),
    ACTION_RIGHT_ID: (1, 0),
}

INTERIOR_CELLS: tuple[tuple[int, int], ...] = tuple(
    (x, y) for y in range(1, GRID_HEIGHT - 1) for x in range(1, GRID_WIDTH - 1) if (x, y) != PLAYER_START
)


def _solid(width: int, height: int, color: int) -> np.ndarray:
    return np.full((height, width), int(color), dtype=np.int8)


def _build_level(level_idx: int) -> Level:
    floor = Sprite(
        pixels=_solid(GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR),
        name="floor",
        x=0,
        y=0,
        layer=0,
        tags=["floor"],
        collidable=False,
    )

    wall_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
    wall_pixels[0, :] = COLOR_WALL
    wall_pixels[-1, :] = COLOR_WALL
    wall_pixels[:, 0] = COLOR_WALL
    wall_pixels[:, -1] = COLOR_WALL
    walls = Sprite(pixels=wall_pixels, name="walls", x=0, y=0, layer=1, tags=["wall", "blocker"], collidable=True)

    player = Sprite(
        pixels=[[COLOR_PLAYER]],
        name="player",
        x=int(PLAYER_START[0]),
        y=int(PLAYER_START[1]),
        layer=3,
        tags=["player"],
        collidable=True,
    )

    return Level(
        name=f"Memory L{level_idx + 1}",
        grid_size=(GRID_WIDTH, GRID_HEIGHT),
        sprites=[floor, walls, player],
        data={"pads_required": int(level_idx + 1)},
    )


class Memory(ARCBaseGame):
    def __init__(self, seed: int | None = None):
        seed_value = 0 if seed is None else int(seed)
        if len(PAD_PALETTE) < LEVEL_COUNT:
            raise ValueError("Memory requires at least one unique non-player/non-background pad color per level.")

        self._seed_rng = random.Random(None if seed is None else int(seed))
        self._episode_seed: int | None = None
        self._sequence_colors: list[int] = []
        self._sequence_positions: list[tuple[int, int]] = []
        self._solver_programs: dict[int, list[int]] = {}

        self._last_level_index: int | None = None
        self._episode_finished = False

        self._pads_required = 1
        self._expected_order = 1
        self._pad_by_order: dict[int, Sprite] = {}
        self._player: Sprite | None = None
        self._floor: Sprite | None = None
        self._walls: Sprite | None = None
        self._awaiting_final_click = False
        self._awaiting_space = False

        levels = [_build_level(level_idx) for level_idx in range(LEVEL_COUNT)]
        camera = Camera(x=0, y=0, width=GRID_WIDTH, height=GRID_HEIGHT, background=COLOR_FLOOR, letter_box=COLOR_FLOOR)
        super().__init__(
            game_id=GAME_ID,
            levels=levels,
            camera=camera,
            win_score=LEVEL_COUNT,
            available_actions=[
                ACTION_UP_ID,
                ACTION_DOWN_ID,
                ACTION_LEFT_ID,
                ACTION_RIGHT_ID,
                ACTION_SPACE_ID,
                ACTION_CLICK_ID,
            ],
            seed=seed_value,
        )

    def _build_sequence_from_seed(self, seed: int) -> tuple[list[int], list[tuple[int, int]]]:
        rng = random.Random(int(seed))
        colors = [int(color) for color in rng.sample(PAD_PALETTE, LEVEL_COUNT)]
        positions = [tuple(cell) for cell in rng.sample(INTERIOR_CELLS, LEVEL_COUNT)]
        return colors, positions

    @staticmethod
    def _in_playfield(x: int, y: int) -> bool:
        return 0 < x < GRID_WIDTH - 1 and 0 < y < GRID_HEIGHT - 1

    def _shortest_path_actions(
        self, start: tuple[int, int], target: tuple[int, int], blocked_cells: set[tuple[int, int]]
    ) -> list[int] | None:
        if start == target:
            return []

        blocked = set(blocked_cells)
        blocked.discard(start)
        blocked.discard(target)

        queue: deque[tuple[int, int]] = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        previous_action: dict[tuple[int, int], int] = {}

        while queue:
            x, y = queue.popleft()
            for action_id, (dx, dy) in ACTION_TO_DELTA.items():
                nx, ny = int(x + dx), int(y + dy)
                cell = (nx, ny)
                if not self._in_playfield(nx, ny):
                    continue
                if cell in blocked:
                    continue
                if cell in previous:
                    continue

                previous[cell] = (x, y)
                previous_action[cell] = int(action_id)
                if cell == target:
                    queue.clear()
                    break
                queue.append(cell)

        if target not in previous:
            return None

        actions: list[int] = []
        cursor = target
        while previous[cursor] is not None:
            actions.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[assignment]
        actions.reverse()
        return actions

    def _build_solver_programs(self, positions: list[tuple[int, int]]) -> dict[int, list[int]] | None:
        programs: dict[int, list[int]] = {}
        for level_idx in range(LEVEL_COUNT):
            required = int(level_idx + 1)
            cursor = tuple(PLAYER_START)
            program: list[int] = []

            for order_idx in range(required):
                target = tuple(positions[order_idx])
                blockers = {tuple(cell) for cell in positions[order_idx + 1 : required]}
                segment = self._shortest_path_actions(cursor, target, blockers)
                if segment is None:
                    return None
                program.extend(segment)
                cursor = target

            # Every level now uses explicit finalize actions: click current avatar cell, then space to advance.
            program.append(ACTION_CLICK_ID)
            program.append(ACTION_SPACE_ID)
            programs[level_idx] = program

        return programs

    def _find_next_solvable_episode(self) -> tuple[int, list[int], list[tuple[int, int]], dict[int, list[int]]]:
        for _ in range(MAX_SOLVER_SEED_ATTEMPTS):
            candidate_seed = int(self._seed_rng.randrange(0, 2_147_483_647))
            colors, positions = self._build_sequence_from_seed(candidate_seed)
            programs = self._build_solver_programs(positions)
            if programs is not None:
                return candidate_seed, colors, positions, programs

        fallback_seed = 0
        while True:
            colors, positions = self._build_sequence_from_seed(fallback_seed)
            programs = self._build_solver_programs(positions)
            if programs is not None:
                return fallback_seed, colors, positions, programs
            fallback_seed += 1

    def _start_new_episode_layout(self) -> None:
        (self._episode_seed, self._sequence_colors, self._sequence_positions, self._solver_programs) = (
            self._find_next_solvable_episode()
        )
        self._episode_finished = False

    def solver_program_for_level(self, level_idx: int | None = None) -> list[int]:
        idx = int(self.level_index if level_idx is None else level_idx)
        program = self._solver_programs.get(idx)
        if program is None:
            raise RuntimeError(f"No solver program prepared for memory level {idx}.")
        return [int(action_id) for action_id in program]

    def final_pad_position_for_level(self, level_idx: int | None = None) -> tuple[int, int]:
        idx = int(self.level_index if level_idx is None else level_idx)
        if idx < 0 or idx >= len(self._sequence_positions):
            raise RuntimeError(f"No final pad position prepared for memory level {idx}.")
        return tuple(int(v) for v in self._sequence_positions[idx])

    def _should_start_new_episode(self, current_level_idx: int) -> bool:
        if not self._sequence_positions or not self._solver_programs:
            return True
        if current_level_idx != 0:
            return False
        if self._episode_finished:
            return True
        return bool(self._last_level_index is not None and int(self._last_level_index) > 0)

    def _spawn_level_pads(self) -> None:
        for sprite in list(self.current_level.get_sprites_by_tag("pad")):
            self.current_level.remove_sprite(sprite)

        self._pad_by_order.clear()
        for order in range(1, self._pads_required + 1):
            x, y = self._sequence_positions[order - 1]
            color = int(self._sequence_colors[order - 1])
            sprite = Sprite(
                pixels=[[color]],
                name=f"pad_{order}",
                x=int(x),
                y=int(y),
                layer=2,
                tags=["pad", f"seq_{order}"],
                collidable=False,
            )
            self.current_level.add_sprite(sprite)
            self._pad_by_order[order] = sprite

    def on_set_level(self, level: Level) -> None:
        current_level_idx = int(self.level_index)
        if self._should_start_new_episode(current_level_idx):
            self._start_new_episode_layout()

        floors = level.get_sprites_by_name("floor")
        self._floor = floors[0] if floors else None
        if self._floor is None:
            raise RuntimeError("Memory level is missing the floor sprite.")

        walls = level.get_sprites_by_name("walls")
        self._walls = walls[0] if walls else None
        if self._walls is None:
            raise RuntimeError("Memory level is missing the walls sprite.")

        players = level.get_sprites_by_name("player")
        self._player = players[0] if players else None
        if self._player is None:
            raise RuntimeError("Memory level is missing the player sprite.")
        self._reset_level_visuals()
        self._player.set_position(int(PLAYER_START[0]), int(PLAYER_START[1]))

        self._pads_required = int(level.get_data("pads_required") or 1)
        self._pads_required = max(1, min(self._pads_required, LEVEL_COUNT))
        self._expected_order = 1
        self._awaiting_final_click = False
        self._awaiting_space = False
        self._spawn_level_pads()

        self._last_level_index = current_level_idx
        self._episode_finished = False

    def _reset_level_visuals(self) -> None:
        if self._floor is not None:
            self._floor.pixels = _solid(GRID_WIDTH, GRID_HEIGHT, COLOR_FLOOR)

        if self._walls is not None:
            wall_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
            wall_pixels[0, :] = COLOR_WALL
            wall_pixels[-1, :] = COLOR_WALL
            wall_pixels[:, 0] = COLOR_WALL
            wall_pixels[:, -1] = COLOR_WALL
            self._walls.pixels = wall_pixels

        if self._player is not None:
            self._player.pixels = np.array([[COLOR_PLAYER]], dtype=np.int8)

    def _set_whiteout_visuals(self) -> None:
        blank_color = int(getattr(self.camera, "background", COLOR_FLOOR))
        if self._floor is not None:
            self._floor.pixels = _solid(GRID_WIDTH, GRID_HEIGHT, blank_color)
        if self._walls is not None:
            wall_pixels = np.full((GRID_HEIGHT, GRID_WIDTH), -1, dtype=np.int8)
            wall_pixels[0, :] = COLOR_WALL
            wall_pixels[-1, :] = COLOR_WALL
            wall_pixels[:, 0] = COLOR_WALL
            wall_pixels[:, -1] = COLOR_WALL
            self._walls.pixels = wall_pixels
        if self._player is not None:
            self._player.pixels = np.array([[blank_color]], dtype=np.int8)

    def _pad_order_at_player(self) -> int | None:
        if self._player is None:
            return None
        px, py = int(self._player.x), int(self._player.y)
        for order, sprite in self._pad_by_order.items():
            if int(sprite.x) == px and int(sprite.y) == py:
                return int(order)
        return None

    def _handle_pad_contact(self) -> bool:
        order = self._pad_order_at_player()
        if order is None:
            return False

        if order != self._expected_order:
            self.lose()
            return True

        if order >= self._pads_required:
            self._awaiting_final_click = True
            return True

        sprite = self._pad_by_order.pop(order, None)
        if sprite is not None:
            self.current_level.remove_sprite(sprite)

        self._expected_order += 1
        return False

    def _parse_click_cell(self) -> tuple[int, int] | None:
        payload = self.action.data if isinstance(self.action.data, dict) else None
        if payload is None:
            return None
        try:
            raw_x = int(payload.get("x", -1))
            raw_y = int(payload.get("y", -1))
        except Exception:
            return None

        if 0 <= raw_x < GRID_WIDTH and 0 <= raw_y < GRID_HEIGHT:
            return (raw_x, raw_y)

        grid = self.camera.display_to_grid(raw_x, raw_y)
        if grid is None:
            return None
        return (int(grid[0]), int(grid[1]))

    def _consume_final_click(self, action_id: int) -> bool:
        if not self._awaiting_final_click:
            return False
        if action_id != ACTION_CLICK_ID or self._player is None:
            return True

        click = self._parse_click_cell()
        if click is None:
            return True

        player_cell = (int(self._player.x), int(self._player.y))
        if click != player_cell:
            return True

        sprite = self._pad_by_order.pop(self._pads_required, None)
        if sprite is not None:
            self.current_level.remove_sprite(sprite)

        self._expected_order = self._pads_required + 1
        self._awaiting_final_click = False
        self._awaiting_space = True
        self._set_whiteout_visuals()
        return True

    def _consume_space_confirmation(self, action_id: int) -> bool:
        if not self._awaiting_space:
            return False
        if action_id != ACTION_SPACE_ID:
            return True

        self._awaiting_space = False
        if int(self.level_index) >= LEVEL_COUNT - 1:
            self._episode_finished = True
        self.next_level()
        return True

    @staticmethod
    def _action_id(action_obj: object) -> int:
        return int(getattr(action_obj, "value", action_obj))

    def step(self) -> None:
        if self._player is None:
            self.complete_action()
            return

        action_id = self._action_id(self.action.id)

        if self._consume_final_click(action_id):
            self.complete_action()
            return

        if self._consume_space_confirmation(action_id):
            self.complete_action()
            return

        delta = ACTION_TO_DELTA.get(action_id)
        if delta is not None:
            dx, dy = int(delta[0]), int(delta[1])
            self.try_move_sprite(self._player, dx, dy)
            if self._handle_pad_contact():
                self.complete_action()
                return

        self.complete_action()
