from __future__ import annotations

import gc
from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

VIEW_SIZE = 64

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_CLICK = 6
ACTION_UNDO = 7
MOVE_DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

COLOR_WHITE = 0
COLOR_WALL = 2
COLOR_PLINTH = 3
COLOR_FLOOR = 4
COLOR_OUTSIDE = 5
EMBLEM_COLORS = {"R": 8, "B": 9, "Y": 11, "G": 14, "P": 15}
MOTIFS = {
    "B": ((1, 0), (1, 0)),
    "Y": ((1, 1), (0, 0)),
    "R": ((1, 0), (0, 1)),
    "G": ((0, 1), (1, 0)),
    "P": ((1, 0), (1, 1)),
}
PAD_SYMBOLS = {"b": "B", "y": "Y", "r": "R", "p": "P"}


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    agent: tuple[int, int, str]
    statues: tuple[tuple[int, int, str], ...]
    baseline_actions: int


LEVEL_SPECS = (
    LevelSpec("Make Room", ("#####", "#####", "#b.##", "#####", "#####"), (2, 3, "B"), ((3, 3, "B"),), 1),
    LevelSpec(
        "The Emblem Stays Behind",
        ("#####", "#b.##", "#..##", "#..##", "#####", "#####"),
        (2, 4, "B"),
        ((3, 4, "Y"),),
        9,
    ),
    LevelSpec(
        "Same Color, Same Door",
        ("#######", "#######", "#.B.###", "#b..###", "#######", "#######"),
        (3, 4, "Y"),
        ((4, 4, "B"),),
        6,
    ),
    LevelSpec(
        "Two-Gate Circuit",
        (
            "##########",
            "#...B....#",
            "#...#....#",
            "#b..#.y..#",
            "#...#....#",
            "#...#....#",
            "#...Y....#",
            "##########",
        ),
        (2, 5, "R"),
        ((3, 5, "Y"), (8, 6, "B")),
        28,
    ),
    LevelSpec(
        "No Statue on the Threshold",
        (
            "#############",
            "#....#......#",
            "#....B......#",
            "#...#Bb.....#",
            "#....#......#",
            "#r...###YY###",
            "#....#......#",
            "#....R..y...#",
            "#....R......#",
            "#....#......#",
            "#############",
        ),
        (2, 7, "G"),
        ((3, 7, "B"), (7, 3, "Y"), (11, 9, "R")),
        40,
    ),
    LevelSpec(
        "Unsolve to Proceed",
        (
            "##############",
            "#..#.#.......#",
            "#..#.#..r....#",
            "#..#.B.......#",
            "#..b.B.......#",
            "#..#.##YY##RR#",
            "#..#.#.......#",
            "#..#.#..y....#",
            "#..#.#.......#",
            "#..#.#.......#",
            "#..#.#.......#",
            "##############",
        ),
        (3, 5, "G"),
        ((4, 5, "B"), (11, 5, "Y"), (12, 10, "R")),
        54,
    ),
    LevelSpec(
        "Leave the Blue Pad Empty",
        (
            "################",
            "#...#..#.......#",
            "#..p#..G.......#",
            "#......G.......#",
            "#...#..#.......#",
            "#...#..B.b.....#",
            "##PP#####Y######",
            "#......#.......#",
            "#...r..#.......#",
            "#......R...y...#",
            "#......R.......#",
            "#......#.......#",
            "#......#.......#",
            "################",
        ),
        (4, 4, "G"),
        ((5, 4, "B"), (14, 5, "Y"), (14, 11, "R"), (2, 11, "P")),
        85,
    ),
)


def _geometry(spec: LevelSpec) -> tuple[int, int, int]:
    width = len(spec.rows[0])
    height = len(spec.rows)
    pitch = min(8, VIEW_SIZE // width, VIEW_SIZE // height)
    return pitch, (VIEW_SIZE - width * pitch) // 2, (VIEW_SIZE - height * pitch) // 2


def _cell_rect(spec: LevelSpec, cell: tuple[int, int]) -> tuple[int, int, int]:
    pitch, origin_x, origin_y = _geometry(spec)
    x, y = cell
    return origin_x + (x - 1) * pitch, origin_y + (y - 1) * pitch, pitch


def _draw_center_motif(frame: np.ndarray, x0: int, y0: int, pitch: int, emblem: str, color: int) -> None:
    motif = MOTIFS[emblem]
    mx = x0 + (pitch - 2) // 2
    my = y0 + (pitch - 2) // 2
    for row in range(2):
        for col in range(2):
            if motif[row][col]:
                frame[my + row, mx + col] = color


def _build_board(spec: LevelSpec) -> np.ndarray:
    frame = np.full((VIEW_SIZE, VIEW_SIZE), COLOR_OUTSIDE, dtype=np.int8)
    pitch, origin_x, origin_y = _geometry(spec)
    for row_index, row in enumerate(spec.rows):
        for col_index, symbol in enumerate(row):
            x0 = origin_x + col_index * pitch
            y0 = origin_y + row_index * pitch
            if symbol == "#":
                frame[y0 : y0 + pitch, x0 : x0 + pitch] = COLOR_WALL
                continue
            frame[y0 : y0 + pitch, x0 : x0 + pitch] = COLOR_FLOOR
            if symbol in EMBLEM_COLORS:
                frame[y0 : y0 + pitch, x0 : x0 + pitch] = EMBLEM_COLORS[symbol]
                _draw_center_motif(frame, x0, y0, pitch, symbol, COLOR_WHITE)
            elif symbol in PAD_SYMBOLS:
                emblem = PAD_SYMBOLS[symbol]
                color = EMBLEM_COLORS[emblem]
                if pitch >= 5:
                    frame[y0, x0 : x0 + pitch] = color
                    frame[y0 + pitch - 1, x0 : x0 + pitch] = color
                    frame[y0 : y0 + pitch, x0] = color
                    frame[y0 : y0 + pitch, x0 + pitch - 1] = color
                else:
                    for px, py in (
                        (x0, y0),
                        (x0 + pitch - 1, y0),
                        (x0, y0 + pitch - 1),
                        (x0 + pitch - 1, y0 + pitch - 1),
                    ):
                        frame[py, px] = color
                _draw_center_motif(frame, x0, y0, pitch, emblem, color)
    return frame


def _build_level(spec: LevelSpec) -> Level:
    board = Sprite(
        _build_board(spec), name="board", x=0, y=0, layer=-10, collidable=False, tags=["board", "sys_static"]
    )
    return Level(sprites=[board], grid_size=(64, 64), data={"spec": spec}, name=spec.name)


def _action_id(value: object) -> int:
    return int(getattr(value, "value", value))


class ExchangeRoomsView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ExchangeRooms | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        animation = game.animation
        if animation and animation["kind"] == "swap" and int(animation["phase"]) < 3:
            self._draw_swap(frame, game, animation)
        else:
            for statue in game.statues:
                self._draw_occupant(frame, game.spec, statue["position"], statue["emblem"], is_statue=True)
            agent_offset = (0, 0)
            if animation and animation["kind"] == "bump" and int(animation["phase"]) == 0:
                agent_offset = tuple(animation["delta"])
            self._draw_occupant(
                frame, game.spec, game.agent_position, game.agent_emblem, is_statue=False, pixel_offset=agent_offset
            )

        self._draw_pad_overlays(frame, game)
        self._draw_feedback(frame, game)
        self._draw_budget(frame, game)
        return frame

    def _draw_occupant(
        self,
        frame: np.ndarray,
        spec: LevelSpec,
        cell: tuple[int, int],
        emblem: str,
        *,
        is_statue: bool,
        pixel_offset: tuple[int, int] = (0, 0),
        anchored_panel: tuple[int, int] | None = None,
    ) -> None:
        cell_x, cell_y, pitch = _cell_rect(spec, cell)
        scale = max(1, pitch // 4)
        left = cell_x + (pitch - 4 * scale) // 2 + pixel_offset[0]
        top = cell_y + (pitch - 4 * scale) // 2 + pixel_offset[1]
        color = EMBLEM_COLORS[emblem]
        mask = (".CC.", "CCCC", "CCCC", ".DD." if is_statue else ".CC.")
        for row, line in enumerate(mask):
            for col, mark in enumerate(line):
                if mark == ".":
                    continue
                value = COLOR_PLINTH if mark == "D" else color
                x0 = left + col * scale
                y0 = top + row * scale
                self._fill_clipped(frame, x0, y0, scale, scale, value)

        panel_cell = anchored_panel if anchored_panel is not None else cell
        panel_x, panel_y, panel_pitch = _cell_rect(spec, panel_cell)
        panel_scale = max(1, panel_pitch // 4)
        panel_left = panel_x + (panel_pitch - 4 * panel_scale) // 2 + panel_scale
        panel_top = panel_y + (panel_pitch - 4 * panel_scale) // 2 + panel_scale
        for row in range(2):
            for col in range(2):
                value = COLOR_WHITE if MOTIFS[emblem][row][col] else color
                self._fill_clipped(
                    frame,
                    panel_left + col * panel_scale,
                    panel_top + row * panel_scale,
                    panel_scale,
                    panel_scale,
                    value,
                )

    def _draw_swap(self, frame: np.ndarray, game: ExchangeRooms, animation: dict[str, object]) -> None:
        statue = game.statues[int(animation["statue_index"])]
        old_agent = tuple(animation["agent_position"])
        old_statue = tuple(animation["statue_position"])
        dx, dy = tuple(animation["delta"])
        phase = int(animation["phase"])
        pitch = _geometry(game.spec)[0]
        shift = 0 if phase == 0 else (max(1, pitch // 3) if phase == 1 else max(1, pitch // 2))
        self._draw_occupant(
            frame,
            game.spec,
            old_statue,
            str(animation["statue_emblem"]),
            is_statue=True,
            pixel_offset=(-dx * shift, -dy * shift),
            anchored_panel=old_statue,
        )
        self._draw_occupant(
            frame,
            game.spec,
            old_agent,
            str(animation["agent_emblem"]),
            is_statue=False,
            pixel_offset=(dx * shift, dy * shift),
            anchored_panel=old_agent,
        )
        del statue

    def _draw_pad_overlays(self, frame: np.ndarray, game: ExchangeRooms) -> None:
        statues = {tuple(item["position"]): item for item in game.statues}
        for cell, target in game.pads.items():
            x0, y0, pitch = _cell_rect(game.spec, cell)
            color = EMBLEM_COLORS[target]
            corners = ((x0, y0), (x0 + pitch - 1, y0), (x0, y0 + pitch - 1), (x0 + pitch - 1, y0 + pitch - 1))
            for px, py in corners:
                frame[py, px] = color
            statue = statues.get(cell)
            if statue is not None and statue["emblem"] == target:
                mid = pitch // 2
                for px, py in ((x0 + mid, y0), (x0 + mid, y0 + pitch - 1), (x0, y0 + mid), (x0 + pitch - 1, y0 + mid)):
                    frame[py, px] = COLOR_WHITE
        if game.feedback == "complete":
            for cell in game.pads:
                x0, y0, pitch = _cell_rect(game.spec, cell)
                x1, y1 = x0 + pitch - 1, y0 + pitch - 1
                frame[y0, x0 + 1 : x1] = COLOR_WHITE
                frame[y1, x0 + 1 : x1] = COLOR_WHITE
                frame[y0 + 1 : y1, x0] = COLOR_WHITE
                frame[y0 + 1 : y1, x1] = COLOR_WHITE

    def _draw_feedback(self, frame: np.ndarray, game: ExchangeRooms) -> None:
        animation = game.animation
        if animation and animation["kind"] == "gate":
            phase = int(animation["phase"])
            if phase < 2:
                cell = tuple(animation["target"])
                if phase == 0:
                    x0, y0, pitch = _cell_rect(game.spec, cell)
                    mid = pitch // 2
                    for px, py in (
                        (x0 + mid, y0),
                        (x0 + mid, y0 + pitch - 1),
                        (x0, y0 + mid),
                        (x0 + pitch - 1, y0 + mid),
                    ):
                        frame[py, px] = COLOR_WHITE
                else:
                    self._outline_cell(frame, game.spec, cell, COLOR_WHITE)
        if game.feedback == "bump":
            for cell in game.feedback_cells:
                self._outline_cell(frame, game.spec, tuple(cell), COLOR_WHITE)
        elif game.feedback == "failed":
            frame[0, :] = EMBLEM_COLORS["R"]
            frame[62, :] = EMBLEM_COLORS["R"]
            frame[:63, 0] = EMBLEM_COLORS["R"]
            frame[:63, 63] = EMBLEM_COLORS["R"]

    def _draw_budget(self, frame: np.ndarray, game: ExchangeRooms) -> None:
        frame[63, :] = COLOR_PLINTH
        filled = int(VIEW_SIZE * max(0, game.remaining_steps) / max(1, game.step_budget))
        if filled:
            frame[63, :filled] = COLOR_WHITE

    def _outline_cell(self, frame: np.ndarray, spec: LevelSpec, cell: tuple[int, int], color: int) -> None:
        x0, y0, pitch = _cell_rect(spec, cell)
        x1 = x0 + pitch - 1
        y1 = y0 + pitch - 1
        if 0 <= y0 < VIEW_SIZE:
            frame[y0, max(0, x0) : min(VIEW_SIZE, x1 + 1)] = color
        if 0 <= y1 < VIEW_SIZE:
            frame[y1, max(0, x0) : min(VIEW_SIZE, x1 + 1)] = color
        if 0 <= x0 < VIEW_SIZE:
            frame[max(0, y0) : min(VIEW_SIZE, y1 + 1), x0] = color
        if 0 <= x1 < VIEW_SIZE:
            frame[max(0, y0) : min(VIEW_SIZE, y1 + 1), x1] = color

    @staticmethod
    def _fill_clipped(frame: np.ndarray, x: int, y: int, width: int, height: int, color: int) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(VIEW_SIZE, x + width), min(VIEW_SIZE, y + height)
        if x0 < x1 and y0 < y1:
            frame[y0:y1, x0:x1] = color


class ExchangeRooms(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        # Frame animations allocate several short-lived arrays. Collect between
        # episodes so Python does not pause a timed input midway through a swap.
        gc.collect()
        self._view = ExchangeRoomsView()
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "exchange_rooms-0001",
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_OUTSIDE, COLOR_OUTSIDE, [self._view]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UNDO],
            seed,
        )
        self._view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.terrain: dict[tuple[int, int], tuple[str, str | None]] = {}
        self.pads: dict[tuple[int, int], str] = {}
        for y, row in enumerate(self.spec.rows, start=1):
            for x, symbol in enumerate(row, start=1):
                if symbol == "#":
                    self.terrain[(x, y)] = ("wall", None)
                elif symbol in EMBLEM_COLORS:
                    self.terrain[(x, y)] = ("gate", symbol)
                elif symbol in PAD_SYMBOLS:
                    self.terrain[(x, y)] = ("pad", PAD_SYMBOLS[symbol])
                    self.pads[(x, y)] = PAD_SYMBOLS[symbol]
                else:
                    self.terrain[(x, y)] = ("floor", None)
        self.agent_position = (self.spec.agent[0], self.spec.agent[1])
        self.agent_emblem = self.spec.agent[2]
        self.statues = [{"position": (x, y), "emblem": emblem} for x, y, emblem in self.spec.statues]
        self.step_budget = self.spec.baseline_actions * 6
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[tuple[int, int], str, tuple[tuple[tuple[int, int], str], ...]]] = []
        self.animation: dict[str, object] | None = None
        self.feedback = "none"
        self.feedback_cells: tuple[tuple[int, int], ...] = ()

    def _snapshot(self) -> tuple[tuple[int, int], str, tuple[tuple[tuple[int, int], str], ...]]:
        return (
            self.agent_position,
            self.agent_emblem,
            tuple((tuple(statue["position"]), str(statue["emblem"])) for statue in self.statues),
        )

    def _restore(self, snapshot: tuple[tuple[int, int], str, tuple[tuple[tuple[int, int], str], ...]]) -> None:
        self.agent_position, self.agent_emblem, statue_states = snapshot
        for statue, (position, emblem) in zip(self.statues, statue_states, strict=True):
            statue["position"] = position
            statue["emblem"] = emblem

    def _statue_index_at(self, cell: tuple[int, int]) -> int | None:
        for index, statue in enumerate(self.statues):
            if tuple(statue["position"]) == cell:
                return index
        return None

    def _all_pads_correct(self) -> bool:
        for cell, target in self.pads.items():
            index = self._statue_index_at(cell)
            if index is None or self.statues[index]["emblem"] != target:
                return False
        return True

    def _spend_action(self) -> None:
        self.remaining_steps = max(0, self.remaining_steps - 1)

    def _finish_action(self) -> None:
        if self._all_pads_correct():
            self.feedback = "complete"
            self.next_level()
            self.complete_action()
            return
        if self.remaining_steps <= 0:
            self.feedback = "failed"
            self.lose()
        self.complete_action()

    def _advance_animation(self) -> None:
        animation = self.animation
        if animation is None:
            return
        kind = str(animation["kind"])
        phase = int(animation["phase"]) + 1
        animation["phase"] = phase
        if kind == "swap" and phase == 3:
            index = int(animation["statue_index"])
            statue = self.statues[index]
            statue["position"] = tuple(animation["agent_position"])
            statue["emblem"] = str(animation["agent_emblem"])
            self.agent_position = tuple(animation["statue_position"])
            self.agent_emblem = str(animation["statue_emblem"])
            self.animation = None
            self._finish_action()
        elif kind == "gate" and phase == 2:
            self.agent_position = tuple(animation["target"])
            self.animation = None
            self.feedback = "none"
            self._finish_action()
        elif kind == "bump" and phase == 1:
            self.feedback = "bump"
            self.feedback_cells = tuple(animation["highlight_cells"])
            self.animation = None
            self._finish_action()

    def _start_move(self, delta: tuple[int, int]) -> None:
        dx, dy = delta
        target = (self.agent_position[0] + dx, self.agent_position[1] + dy)
        terrain = self.terrain.get(target, ("wall", None))
        statue_index = self._statue_index_at(target)

        if statue_index is not None:
            if self.terrain[self.agent_position][0] == "gate":
                self.animation = {
                    "kind": "bump",
                    "phase": 0,
                    "delta": delta,
                    "highlight_cells": (self.agent_position, target),
                }
                return
            statue = self.statues[statue_index]
            self.undo_history.append(self._snapshot())
            self.animation = {
                "kind": "swap",
                "phase": 0,
                "delta": delta,
                "statue_index": statue_index,
                "agent_position": self.agent_position,
                "agent_emblem": self.agent_emblem,
                "statue_position": tuple(statue["position"]),
                "statue_emblem": statue["emblem"],
            }
            return

        if terrain[0] == "wall" or (terrain[0] == "gate" and terrain[1] != self.agent_emblem):
            self.animation = {"kind": "bump", "phase": 0, "delta": delta, "highlight_cells": (target,)}
            return

        self.undo_history.append(self._snapshot())
        if terrain[0] == "gate":
            self.animation = {"kind": "gate", "phase": 0, "target": target}
            return
        self.agent_position = target
        self._finish_action()

    def step(self) -> None:
        if self.animation is not None:
            self._advance_animation()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self.complete_action()
            return

        self.feedback = "none"
        self.feedback_cells = ()
        self._spend_action()

        if action == ACTION_UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self._finish_action()
            return
        if action in MOVE_DELTAS:
            self._start_move(MOVE_DELTAS[action])
            return

        # SPACE and valid in-bounds clicks are intentionally inert.
        self._finish_action()
