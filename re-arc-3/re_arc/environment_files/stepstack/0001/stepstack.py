from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, RenderableUserDisplay, Sprite

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_UNDO = 5
ACTION_CLICK = 6

VIEW_SIZE = 64
BOARD_WIDTH = 15
BOARD_HEIGHT = 11
CELL_SIZE = 4
BOARD_LEFT = 2
BOARD_TOP = 10

COLOR_EXIT_INSIDE = 0
COLOR_TERRAIN_TOP = 1
COLOR_TERRAIN = 3
COLOR_BACKGROUND = 4
COLOR_UNDO = 6
COLOR_INVALID = 8
COLOR_AVATAR = 9
COLOR_FACE = 10
COLOR_BLOCK_TOP = 11
COLOR_BLOCK = 12
COLOR_BLOCK_OUTLINE = 13
COLOR_EXIT = 14


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec(
        "The Falling Step",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            "...............",
            "...............",
            "...............",
            "......B....E...",
            "..........#####",
            "..>.......#####",
            "###############",
        ),
        9,
    ),
    LevelSpec(
        "Outside First",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            "...............",
            ".............E.",
            "............###",
            "..........B.###",
            "........#######",
            "..>..B.....####",
            "###############",
        ),
        11,
    ),
    LevelSpec(
        "Borrowed Stair",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            "...............",
            "............B..",
            "...........##.E",
            "..........#...#",
            ".........#....#",
            ".......B...>B.#",
            "###############",
        ),
        25,
    ),
    LevelSpec(
        "The Load-Bearing Block",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            ".........B.....",
            ".........B.....",
            "........##..E..",
            ".......#...####",
            "......#....####",
            "...B...>...####",
            "###############",
        ),
        26,
    ),
    LevelSpec(
        "Finish the High Work",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            "..........B.B..",
            ".........####.E",
            "........#.....#",
            ".......#......#",
            "......#.....#.#",
            "....B....>B...#",
            "###############",
        ),
        45,
    ),
    LevelSpec(
        "The Delayed Riser",
        (
            "...............",
            "...............",
            "...............",
            "...............",
            "..........B.B.E",
            ".........####.#",
            "........#...B.#",
            ".......#...##.#",
            "......#...#...#",
            "..B.B.>.......#",
            "###############",
        ),
        58,
    ),
    LevelSpec(
        "Five-High Handoff",
        (
            "...............",
            "...............",
            "..........B.B..",
            ".........####.E",
            "........#...B.#",
            ".......#....B.#",
            "......#....##.#",
            ".....#....#...#",
            "....#....#....#",
            ".B.B.>........#",
            "###############",
        ),
        66,
    ),
)


@dataclass(frozen=True)
class StableState:
    player: tuple[int, int]
    facing: int
    blocks: tuple[tuple[int, tuple[int, int]], ...]


@dataclass
class VisualFrame:
    block_pixels: dict[int, tuple[int, int]]
    player_cell: tuple[int, int]
    feedback_cells: tuple[tuple[int, int], ...] = ()
    feedback_color: int | None = None
    braced_face: bool = False
    completion_color: int | None = None


class StepstackDisplay(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Stepstack | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[3:7, 2:62] = COLOR_TERRAIN
        frame[3, 2:62] = COLOR_TERRAIN_TOP
        frame[6, 2:62] = COLOR_TERRAIN_TOP
        frame[3:7, 2] = COLOR_TERRAIN_TOP
        frame[3:7, 61] = COLOR_TERRAIN_TOP
        fill = int(58 * game.remaining_steps / max(1, game.step_budget))
        if fill > 0:
            frame[4:6, 3 : 3 + fill] = COLOR_EXIT

        visual = game._active_visual
        if visual is not None and visual.feedback_color is not None:
            for cell in visual.feedback_cells:
                x, y = _cell_top_left(*cell)
                frame[y : y + CELL_SIZE, x : x + CELL_SIZE] = visual.feedback_color
        if visual is not None and visual.braced_face:
            px, py = _cell_top_left(*visual.player_cell)
            face_x = px + 3 if game.facing > 0 else px
            frame[py + 1 : py + 3, face_x] = COLOR_EXIT_INSIDE
        return frame


def _cell_top_left(x: int, y: int) -> tuple[int, int]:
    return BOARD_LEFT + x * CELL_SIZE, BOARD_TOP + y * CELL_SIZE


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


def _player_pixels(facing: int) -> np.ndarray:
    pixels = np.array(
        [
            [-1, COLOR_AVATAR, COLOR_AVATAR, -1],
            [COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, COLOR_FACE],
            [COLOR_AVATAR, COLOR_AVATAR, COLOR_AVATAR, COLOR_FACE],
            [-1, COLOR_AVATAR, COLOR_AVATAR, -1],
        ],
        dtype=np.int8,
    )
    return pixels if facing > 0 else np.fliplr(pixels)


def _block_pixels() -> np.ndarray:
    return np.array(
        [
            [COLOR_BLOCK_OUTLINE, COLOR_BLOCK_TOP, COLOR_BLOCK_TOP, COLOR_BLOCK_OUTLINE],
            [COLOR_BLOCK, COLOR_BLOCK, COLOR_BLOCK, COLOR_BLOCK],
            [COLOR_BLOCK, COLOR_BLOCK, COLOR_BLOCK, COLOR_BLOCK],
            [COLOR_BLOCK_OUTLINE, COLOR_BLOCK, COLOR_BLOCK, COLOR_BLOCK_OUTLINE],
        ],
        dtype=np.int8,
    )


def _exit_pixels(color: int = COLOR_EXIT) -> np.ndarray:
    return np.array(
        [
            [color, color, color, color],
            [color, COLOR_EXIT_INSIDE, COLOR_EXIT_INSIDE, color],
            [color, COLOR_EXIT_INSIDE, COLOR_EXIT_INSIDE, color],
            [color, COLOR_EXIT_INSIDE, COLOR_EXIT_INSIDE, color],
        ],
        dtype=np.int8,
    )


def _terrain_sprite(terrain: set[tuple[int, int]]) -> Sprite:
    pixels = np.full((VIEW_SIZE, VIEW_SIZE), -1, dtype=np.int8)
    for x, y in terrain:
        px, py = _cell_top_left(x, y)
        pixels[py : py + CELL_SIZE, px : px + CELL_SIZE] = COLOR_TERRAIN
    for x, y in terrain:
        if (x, y - 1) not in terrain:
            px, py = _cell_top_left(x, y)
            pixels[py, px : px + CELL_SIZE] = COLOR_TERRAIN_TOP
    return Sprite(pixels, name="terrain", x=0, y=0, layer=-5, collidable=False, tags=["terrain"])


def _parse_level(spec: LevelSpec) -> Level:
    if len(spec.rows) != BOARD_HEIGHT or any(len(row) != BOARD_WIDTH for row in spec.rows):
        raise ValueError(f"{spec.name} must be a {BOARD_WIDTH}x{BOARD_HEIGHT} ASCII layout")

    terrain: set[tuple[int, int]] = set()
    block_cells: list[tuple[int, int]] = []
    player: tuple[int, int] | None = None
    facing = 1
    exit_cell: tuple[int, int] | None = None
    for y, row in enumerate(spec.rows):
        for x, char in enumerate(row):
            if char == "#":
                terrain.add((x, y))
            elif char == "B":
                block_cells.append((x, y))
            elif char in "<>":
                player = (x, y)
                facing = -1 if char == "<" else 1
            elif char == "E":
                exit_cell = (x, y)

    if player is None or exit_cell is None:
        raise ValueError(f"{spec.name} requires exactly one player and exit")

    exit_x, exit_y = _cell_top_left(*exit_cell)
    player_x, player_y = _cell_top_left(*player)
    sprites: list[Sprite] = [
        _terrain_sprite(terrain),
        Sprite(_exit_pixels(), name="exit", x=exit_x, y=exit_y, layer=-2, collidable=False, tags=["exit"]),
    ]
    for block_id, cell in enumerate(block_cells):
        x, y = _cell_top_left(*cell)
        sprites.append(
            Sprite(_block_pixels(), name=f"block_{block_id}", x=x, y=y, layer=2, collidable=False, tags=["block"])
        )
    sprites.append(
        Sprite(
            _player_pixels(facing), name="player", x=player_x, y=player_y, layer=3, collidable=False, tags=["player"]
        )
    )
    return Level(
        sprites=sprites,
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={
            "terrain": tuple(sorted(terrain)),
            "block_cells": tuple(block_cells),
            "player": player,
            "facing": facing,
            "exit": exit_cell,
            "step_budget": spec.baseline_actions * 6,
        },
    )


class Stepstack(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._display = StepstackDisplay()
        levels = [_parse_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "stepstack-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self._display]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UNDO, ACTION_CLICK],
            seed,
        )
        self._display.game = self

    def on_set_level(self, _level: Level) -> None:
        self.terrain = set(self.current_level.get_data("terrain"))
        initial_blocks = self.current_level.get_data("block_cells")
        self.blocks = {block_id: tuple(cell) for block_id, cell in enumerate(initial_blocks)}
        self.player = tuple(self.current_level.get_data("player"))
        self.facing = int(self.current_level.get_data("facing"))
        self.exit_cell = tuple(self.current_level.get_data("exit"))
        self.step_budget = int(self.current_level.get_data("step_budget"))
        self.remaining_steps = self.step_budget
        self.undo_stack: list[StableState] = []
        self.won = False
        self._animation_frames: list[VisualFrame] = []
        self._active_visual: VisualFrame | None = None
        self._finish_with_win = False
        self.player_sprite = self.current_level.get_sprites_by_tag("player")[0]
        self.exit_sprite = self.current_level.get_sprites_by_tag("exit")[0]
        self.block_sprites = {
            int(sprite.name.split("_")[-1]): sprite for sprite in self.current_level.get_sprites_by_tag("block")
        }
        self._sync_sprites()

    def _stable_state(self) -> StableState:
        return StableState(self.player, self.facing, tuple(sorted(self.blocks.items())))

    def _restore_state(self, state: StableState) -> None:
        self.player = state.player
        self.facing = state.facing
        self.blocks = dict(state.blocks)
        self._sync_sprites()

    def _sync_sprites(self) -> None:
        self.player_sprite.set_position(*_cell_top_left(*self.player))
        self.player_sprite.pixels = _player_pixels(self.facing)
        for block_id, sprite in self.block_sprites.items():
            sprite.set_position(*_cell_top_left(*self.blocks[block_id]))
        self.exit_sprite.pixels = _exit_pixels()

    def _occupied_blocks(self, blocks: dict[int, tuple[int, int]] | None = None) -> set[tuple[int, int]]:
        return set((blocks or self.blocks).values())

    def _inside(self, cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < BOARD_WIDTH and 0 <= cell[1] < BOARD_HEIGHT

    def _empty(self, cell: tuple[int, int], blocks: dict[int, tuple[int, int]] | None = None) -> bool:
        return self._inside(cell) and cell not in self.terrain and cell not in self._occupied_blocks(blocks)

    def _supported(self, cell: tuple[int, int], blocks: dict[int, tuple[int, int]] | None = None) -> bool:
        return cell in self.terrain or cell in self._occupied_blocks(blocks)

    def _gravity(
        self, blocks: dict[int, tuple[int, int]], player: tuple[int, int]
    ) -> tuple[dict[int, tuple[int, int]], list[tuple[dict[int, tuple[int, int]], set[int]]], bool]:
        work = dict(blocks)
        ticks: list[tuple[dict[int, tuple[int, int]], set[int]]] = []
        while True:
            by_column: dict[int, list[tuple[int, int]]] = {}
            for block_id, (x, y) in work.items():
                by_column.setdefault(x, []).append((y, block_id))

            moving: set[int] = set()
            occupied = set(work.values())
            for x, entries in by_column.items():
                entries.sort()
                component: list[tuple[int, int]] = []
                for item in entries:
                    if component and item[0] != component[-1][0] + 1:
                        bottom_y = component[-1][0]
                        if (x, bottom_y + 1) not in self.terrain and (x, bottom_y + 1) not in occupied:
                            moving.update(block_id for _, block_id in component)
                        component = []
                    component.append(item)
                if component:
                    bottom_y = component[-1][0]
                    if (x, bottom_y + 1) not in self.terrain and (x, bottom_y + 1) not in occupied:
                        moving.update(block_id for _, block_id in component)

            if not moving:
                return work, ticks, False
            for block_id in moving:
                x, y = work[block_id]
                destination = (x, y + 1)
                if not self._inside(destination) or destination == player:
                    return blocks, [], True
            before = dict(work)
            for block_id in moving:
                x, y = work[block_id]
                work[block_id] = (x, y + 1)
            ticks.append((before, moving))

    def _pixel_positions(self, blocks: dict[int, tuple[int, int]]) -> dict[int, tuple[int, int]]:
        return {block_id: _cell_top_left(*cell) for block_id, cell in blocks.items()}

    def _gravity_frames(self, ticks: list[tuple[dict[int, tuple[int, int]], set[int]]]) -> list[VisualFrame]:
        frames: list[VisualFrame] = []
        for before, moving in ticks:
            base = self._pixel_positions(before)
            for offset in range(1, CELL_SIZE + 1):
                positions = dict(base)
                for block_id in moving:
                    x, y = positions[block_id]
                    positions[block_id] = (x, y + offset)
                frames.append(VisualFrame(positions, self.player))
        return frames

    def _push_frames(
        self, before: dict[int, tuple[int, int]], block_id: int, direction: int, player_cell: tuple[int, int]
    ) -> list[VisualFrame]:
        base = self._pixel_positions(before)
        frames: list[VisualFrame] = []
        for offset in range(1, CELL_SIZE + 1):
            positions = dict(base)
            x, y = positions[block_id]
            positions[block_id] = (x + direction * offset, y)
            frames.append(VisualFrame(positions, player_cell))
        return frames

    def _completion_frames(self) -> list[VisualFrame]:
        positions = self._pixel_positions(self.blocks)
        return [
            VisualFrame(positions, self.player, completion_color=color)
            for color in (COLOR_EXIT_INSIDE, COLOR_EXIT, COLOR_EXIT_INSIDE, COLOR_EXIT)
        ]

    def _queue(self, frames: list[VisualFrame], *, win_after: bool = False) -> None:
        self._animation_frames = frames
        self._finish_with_win = win_after
        self._play_animation_frame()

    def _play_animation_frame(self) -> None:
        visual = self._animation_frames.pop(0)
        self._active_visual = visual
        for block_id, pixel in visual.block_pixels.items():
            self.block_sprites[block_id].set_position(*pixel)
        self.player_sprite.set_position(*_cell_top_left(*visual.player_cell))
        self.player_sprite.pixels = _player_pixels(self.facing)
        self.exit_sprite.pixels = _exit_pixels(
            COLOR_EXIT if visual.completion_color is None else visual.completion_color
        )
        if self._animation_frames:
            return
        if self._finish_with_win:
            self.won = True
            self.next_level()
        self.complete_action()

    def _spend_step(self) -> bool:
        self.remaining_steps -= 1
        if self.remaining_steps > 0:
            return False
        self._animation_frames.clear()
        self._active_visual = None
        self._sync_sprites()
        self.lose()
        self.complete_action()
        return True

    def _finish_logical_action(self, frames: list[VisualFrame]) -> None:
        self._sync_sprites()
        if self._spend_step():
            return
        if self.player == self.exit_cell:
            frames.extend(self._completion_frames())
            self._queue(frames, win_after=True)
        elif frames:
            final_positions = self._pixel_positions(self.blocks)
            last = frames[-1]
            if (
                last.block_pixels != final_positions
                or last.player_cell != self.player
                or last.feedback_color is not None
                or last.braced_face
            ):
                frames.append(VisualFrame(final_positions, self.player))
            self._queue(frames)
        else:
            self._active_visual = None
            self.complete_action()

    def _invalid_feedback(self, cells: set[tuple[int, int]]) -> list[VisualFrame]:
        positions = self._pixel_positions(self.blocks)
        flash = VisualFrame(positions, self.player, tuple(sorted(cells)), COLOR_INVALID)
        return [flash, flash, VisualFrame(positions, self.player)]

    def _resolve_horizontal(self, direction: int, before_state: StableState) -> None:
        old_player = self.player
        old_blocks = dict(self.blocks)
        self.facing = direction
        adjacent = (old_player[0] + direction, old_player[1])
        beyond = (old_player[0] + 2 * direction, old_player[1])
        block_at = {cell: block_id for block_id, cell in self.blocks.items()}
        frames: list[VisualFrame] = []
        translated = False

        if self._empty(adjacent) and self._supported((adjacent[0], adjacent[1] + 1)):
            tentative, ticks, collision = self._gravity(self.blocks, adjacent)
            if not collision:
                self.player = adjacent
                self.blocks = tentative
                frames.extend(self._gravity_frames(ticks))
                translated = True
        elif adjacent in block_at:
            pushed_id = block_at[adjacent]
            if self._empty(beyond):
                pushed = dict(self.blocks)
                pushed[pushed_id] = beyond
                normal_blocks, normal_ticks, normal_collision = self._gravity(pushed, adjacent)
                braced = False
                if normal_collision:
                    normal_blocks, normal_ticks, braced_collision = self._gravity(pushed, old_player)
                    if braced_collision:
                        normal_collision = True
                    else:
                        normal_collision = False
                        braced = True
                if not normal_collision:
                    self.player = old_player if braced else adjacent
                    self.blocks = normal_blocks
                    frames.extend(self._push_frames(old_blocks, pushed_id, direction, old_player))
                    if braced:
                        positions = self._pixel_positions(pushed)
                        frames.append(VisualFrame(positions, old_player, braced_face=True))
                    frames.extend(self._gravity_frames(normal_ticks))
                    translated = True
            if not translated:
                occupied = self._occupied_blocks()
                supported_column = {adjacent, beyond}
                above = (adjacent[0], adjacent[1] - 1)
                while above in occupied:
                    supported_column.add(above)
                    above = (above[0], above[1] - 1)
                frames = self._invalid_feedback(supported_column)

        state_changed = self._stable_state() != before_state
        if state_changed:
            self.undo_stack.append(before_state)
        if not translated:
            self.blocks = old_blocks
            self.player = old_player
        self._finish_logical_action(frames)

    def _resolve_vertical(self, up: bool, before_state: StableState) -> None:
        d = self.facing
        x, y = self.player
        side = (x + d, y)
        frames: list[VisualFrame] = []
        destination: tuple[int, int] | None = None
        feedback: set[tuple[int, int]] = set()
        if up:
            landing = (x + d, y - 1)
            if self._supported(side) and self._empty(landing):
                destination = landing
            elif self._supported(side):
                feedback = {side, landing}
        else:
            landing = (x + d, y + 1)
            support = (x + d, y + 2)
            if self._empty(side) and self._empty(landing) and self._supported(support):
                destination = landing

        if destination is not None:
            settled, ticks, collision = self._gravity(self.blocks, destination)
            if not collision:
                self.player = destination
                self.blocks = settled
                frames.extend(self._gravity_frames(ticks))
        if self._stable_state() != before_state:
            self.undo_stack.append(before_state)
        elif feedback:
            frames = self._invalid_feedback(feedback)
        self._finish_logical_action(frames)

    def _undo(self) -> None:
        if self.undo_stack:
            self._restore_state(self.undo_stack.pop())
            self._finish_logical_action([])
            return
        positions = self._pixel_positions(self.blocks)
        edge_cells = tuple(
            (x, y)
            for x in range(BOARD_WIDTH)
            for y in range(BOARD_HEIGHT)
            if x in (0, BOARD_WIDTH - 1) or y in (0, BOARD_HEIGHT - 1)
        )
        flash = VisualFrame(positions, self.player, edge_cells, COLOR_UNDO)
        self._finish_logical_action([flash, flash, VisualFrame(positions, self.player)])

    def step(self) -> None:
        if self._animation_frames:
            self._play_animation_frame()
            return
        action = _action_id(self.action.id)
        if action == 0:
            self._active_visual = None
            self._sync_sprites()
            self.complete_action()
            return
        if self.won:
            self.complete_action()
            return
        if action == ACTION_CLICK:
            self._spend_step()
            if self.remaining_steps > 0:
                self.complete_action()
            return
        if action == ACTION_UNDO:
            self._undo()
            return

        before_state = self._stable_state()
        if action == ACTION_LEFT:
            self._resolve_horizontal(-1, before_state)
        elif action == ACTION_RIGHT:
            self._resolve_horizontal(1, before_state)
        elif action == ACTION_UP:
            self._resolve_vertical(True, before_state)
        elif action == ACTION_DOWN:
            self._resolve_vertical(False, before_state)
        else:
            self.complete_action()
