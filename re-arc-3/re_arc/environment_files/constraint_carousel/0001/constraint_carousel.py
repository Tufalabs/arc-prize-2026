from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

VIEW_SIZE = 64
TOKEN_SIZE = 4
CELL_PITCH = 10

COLOR_COMPLETE = 0
COLOR_OVERLAP = 2
COLOR_SPENT = 3
COLOR_SOCKET = 4
COLOR_BACKGROUND = 5
COLOR_MAGENTA = 6
COLOR_BLUE = 9
COLOR_LIVE = 10
COLOR_YELLOW = 11
COLOR_GREEN = 14
TOKEN_COLORS = (COLOR_MAGENTA, COLOR_BLUE, COLOR_YELLOW, COLOR_GREEN)

ACTION_CLICK = 6
ACTION_UNDO = 7

CHAR_TO_COLOR = {"M": COLOR_MAGENTA, "B": COLOR_BLUE, "Y": COLOR_YELLOW, "G": COLOR_GREEN}

LIVE_ARROW = (".aaaa.", "a....a", ".....a", ".....a", "..aaaa", "...a..")


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, ...]
    goals: tuple[tuple[int, int], ...]
    pivots: tuple[tuple[int, int], ...]
    solution: tuple[tuple[int, int], ...]

    @property
    def budget(self) -> int:
        return 6 * len(self.solution)


LEVEL_SPECS = (
    LevelSpec("One Bright Neighbor", ("BGM", "MYY"), ((0, 0), (0, 1)), ((0, 0),), ((0, 0),)),
    LevelSpec("Same Four, New Places", ("MBY", "MYG"), ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (0, 1))),
    LevelSpec("Handoff", ("MGY", "MYM", "MBB"), ((0, 1), (1, 0)), ((0, 0), (0, 1), (1, 1)), ((0, 0), (1, 1), (0, 1))),
    LevelSpec(
        "Choose a Turn",
        ("MBGB", "MYYB", "-GBG", "-GMY"),
        ((0, 1), (0, 2), (2, 1)),
        ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)),
        ((1, 1), (0, 0), (1, 2), (0, 1), (2, 2)),
    ),
    LevelSpec(
        "Break the Good",
        ("MBG-", "MYMG", "GBMB", "YYGY"),
        ((0, 0), (1, 0), (1, 1), (2, 0), (2, 1)),
        ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((2, 0), (1, 1), (0, 0), (0, 1), (1, 0), (2, 1), (1, 2), (2, 2)),
    ),
    LevelSpec(
        "Two Feeds, One Release",
        ("-BMMY", "BGMGB", "MBYGM", "YGYY-", "-BMB-"),
        ((0, 1), (0, 2), (1, 1), (1, 3), (2, 0), (3, 1), (3, 2)),
        ((0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)),
        ((0, 3), (2, 2), (1, 2), (2, 1), (2, 0), (1, 0), (0, 1), (0, 2), (1, 1)),
    ),
    LevelSpec(
        "Full Carousel",
        ("--MBBG", "-GMGMY", "BBYGBG", "MYGBYB", "-BMYYM", "---MG-"),
        ((0, 2), (1, 1), (1, 4), (2, 2), (2, 3), (3, 1), (3, 2)),
        ((0, 4), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 2), (3, 4), (4, 3)),
        ((4, 3), (2, 0), (1, 2), (2, 1), (1, 3), (3, 2), (0, 4), (3, 4), (2, 3), (2, 2), (2, 4)),
    ),
)


def _origin(spec: LevelSpec) -> tuple[int, int]:
    width = len(spec.rows[0])
    height = len(spec.rows)
    return 3 + (VIEW_SIZE - CELL_PITCH * width) // 2, 3 + (VIEW_SIZE - CELL_PITCH * height) // 2


def _parse_tokens(rows: tuple[str, ...]) -> dict[tuple[int, int], int]:
    return {
        (row, col): CHAR_TO_COLOR[symbol]
        for row, line in enumerate(rows)
        for col, symbol in enumerate(line)
        if symbol != "-"
    }


def _frame_counts(tokens: dict[tuple[int, int], int], frame: tuple[int, int]) -> dict[int, int]:
    row, col = frame
    cells = ((row, col), (row, col + 1), (row + 1, col), (row + 1, col + 1))
    return {color: sum(tokens[cell] == color for cell in cells) for color in TOKEN_COLORS}


class ConstraintCarouselView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: ConstraintCarousel | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[:, :] = COLOR_BACKGROUND
        goal_tokens = game._pip_tokens()
        self._draw_frames(frame, game, goal_tokens)
        self._draw_brackets(frame, game)
        self._draw_tokens(frame, game)
        self._draw_pips(frame, game, goal_tokens)
        self._draw_sockets(frame, game)
        self._draw_budget(frame, game)
        if game.edge_flash:
            frame[0, :] = COLOR_OVERLAP
            frame[-2, :] = COLOR_OVERLAP
            frame[:, 0] = COLOR_OVERLAP
            frame[:, -1] = COLOR_OVERLAP
        return frame

    def _draw_frames(self, frame: np.ndarray, game: ConstraintCarousel, tokens: dict[tuple[int, int], int]) -> None:
        incomplete: list[tuple[int, int]] = []
        complete: list[tuple[int, int]] = []
        for goal in game.goals:
            (complete if all(count == 1 for count in _frame_counts(tokens, goal).values()) else incomplete).append(goal)
        for color, goals in ((COLOR_OVERLAP, incomplete), (COLOR_COMPLETE, complete)):
            for row, col in goals:
                left = game.x0 + CELL_PITCH * col - 3
                top = game.y0 + CELL_PITCH * row - 3
                right = left + 19
                bottom = top + 19
                frame[top : top + 2, left : right + 1] = color
                frame[bottom - 1 : bottom + 1, left : right + 1] = color
                frame[top : bottom + 1, left : left + 2] = color
                frame[top : bottom + 1, right - 1 : right + 1] = color

    def _draw_corner_marks(
        self, frame: np.ndarray, game: ConstraintCarousel, pivot: tuple[int, int], color: int
    ) -> None:
        row, col = pivot
        left = game.x0 + CELL_PITCH * col - 1
        top = game.y0 + CELL_PITCH * row - 1
        right = left + 15
        bottom = top + 15
        frame[top, left : left + 3] = color
        frame[top : top + 3, left] = color
        frame[top, right - 2 : right + 1] = color
        frame[top : top + 3, right] = color
        frame[bottom, left : left + 3] = color
        frame[bottom - 2 : bottom + 1, left] = color
        frame[bottom, right - 2 : right + 1] = color
        frame[bottom - 2 : bottom + 1, right] = color

    def _draw_brackets(self, frame: np.ndarray, game: ConstraintCarousel) -> None:
        for pivot in game.pivots:
            if pivot in game.spent:
                self._draw_corner_marks(frame, game, pivot, COLOR_SPENT)
        for pivot in game.pivots:
            if pivot not in game.spent:
                self._draw_corner_marks(frame, game, pivot, COLOR_LIVE)

    def _draw_tokens(self, frame: np.ndarray, game: ConstraintCarousel) -> None:
        moving = set()
        if game.motion_amount is not None and game.animation_snapshot is not None:
            moving = set(game._pivot_cells(game.animation_pivot))
        for cell, color in game.tokens.items():
            if cell not in moving:
                self._token(frame, game, cell, color)

        if moving and game.animation_snapshot is not None:
            row, col = game.animation_pivot
            directions = {
                (row, col): (game.motion_amount, 0),
                (row, col + 1): (0, game.motion_amount),
                (row + 1, col + 1): (-game.motion_amount, 0),
                (row + 1, col): (0, -game.motion_amount),
            }
            for cell in game._pivot_cells(game.animation_pivot):
                dx, dy = directions[cell]
                self._token(frame, game, cell, game.animation_snapshot[cell], dx, dy)

    def _token(
        self, frame: np.ndarray, game: ConstraintCarousel, cell: tuple[int, int], color: int, dx: int = 0, dy: int = 0
    ) -> None:
        row, col = cell
        x = game.x0 + CELL_PITCH * col + dx
        y = game.y0 + CELL_PITCH * row + dy
        frame[y : y + TOKEN_SIZE, x : x + TOKEN_SIZE] = color

    def _draw_pips(self, frame: np.ndarray, game: ConstraintCarousel, tokens: dict[tuple[int, int], int]) -> None:
        for row, col in game.goals:
            left = game.x0 + CELL_PITCH * col - 3
            top = game.y0 + CELL_PITCH * row - 3
            positions = ((left, top), (left + 17, top), (left + 17, top + 17), (left, top + 17))
            counts = _frame_counts(tokens, (row, col))
            for color, (x, y) in zip(TOKEN_COLORS, positions, strict=True):
                frame[y : y + 3, x : x + 3] = color
                count = counts[color]
                if count == 0:
                    frame[y + 1, x + 1] = COLOR_SOCKET
                elif count >= 2:
                    frame[y : y + 3, x + 1] = COLOR_BACKGROUND

    def _draw_sockets(self, frame: np.ndarray, game: ConstraintCarousel) -> None:
        for row, col in game.pivots:
            x = game.x0 + CELL_PITCH * col + 4
            y = game.y0 + CELL_PITCH * row + 4
            frame[y : y + 6, x : x + 6] = COLOR_SOCKET
            if (row, col) not in game.spent:
                for yy, line in enumerate(LIVE_ARROW):
                    for xx, symbol in enumerate(line):
                        if symbol == "a":
                            frame[y + yy, x + xx] = COLOR_LIVE
            else:
                shift = game.spent_shake_shift if game.shaken_pivot == (row, col) else 0
                for offset in range(4):
                    for xx, yy in ((1 + offset + shift, 1 + offset), (4 - offset + shift, 1 + offset)):
                        if 0 <= xx < 6:
                            frame[y + yy, x + xx] = COLOR_OVERLAP

    def _draw_budget(self, frame: np.ndarray, game: ConstraintCarousel) -> None:
        frame[63, :] = COLOR_SPENT
        filled = int(VIEW_SIZE * game.remaining_steps / max(1, game.step_budget))
        if filled:
            frame[63, :filled] = COLOR_LIVE


class ConstraintCarousel(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self.view = ConstraintCarouselView()
        levels = [Level(grid_size=(VIEW_SIZE, VIEW_SIZE), data={"spec": spec}, name=spec.name) for spec in LEVEL_SPECS]
        camera = Camera(0, 0, VIEW_SIZE, VIEW_SIZE, COLOR_BACKGROUND, COLOR_BACKGROUND, [self.view])
        super().__init__(
            "constraint_carousel-0001",
            levels,
            camera,
            False,
            len(levels),
            [1, 2, 3, 4, 5, ACTION_CLICK, ACTION_UNDO],
            seed=seed,
        )
        self.view.game = self

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.tokens = _parse_tokens(self.spec.rows)
        self.goals = set(self.spec.goals)
        self.pivots = set(self.spec.pivots)
        self.spent: set[tuple[int, int]] = set()
        self.x0, self.y0 = _origin(self.spec)
        self.step_budget = self.spec.budget
        self.remaining_steps = self.step_budget
        self.undo_history: list[tuple[dict[tuple[int, int], int], set[tuple[int, int]], bool]] = []
        self.solved = self._is_solved()
        self.animation_snapshot: dict[tuple[int, int], int] | None = None
        self.animation_pivot = (-1, -1)
        self.motion_amount: int | None = None
        self.motion_index = -1
        self.edge_flash = False
        self.shaken_pivot: tuple[int, int] | None = None
        self.spent_shake_shift = 0
        self.shake_phase = 0

    @staticmethod
    def _pivot_cells(pivot: tuple[int, int]) -> tuple[tuple[int, int], ...]:
        row, col = pivot
        return ((row, col), (row, col + 1), (row + 1, col + 1), (row + 1, col))

    def _pip_tokens(self) -> dict[tuple[int, int], int]:
        if self.motion_amount in (3, 7) and self.animation_snapshot is not None:
            prior = dict(self.tokens)
            prior.update(self.animation_snapshot)
            return prior
        return self.tokens

    def _is_solved(self) -> bool:
        return all(all(count == 1 for count in _frame_counts(self.tokens, goal).values()) for goal in self.goals)

    def _clicked_pivot(self) -> tuple[int, int] | None:
        display_x = int(self.action.data.get("x", 0))
        display_y = int(self.action.data.get("y", 0))
        grid_position = self.camera.display_to_grid(display_x, display_y)
        if grid_position is None:
            return None
        click_x, click_y = grid_position
        for row, col in sorted(self.pivots):
            x = self.x0 + CELL_PITCH * col + 4
            y = self.y0 + CELL_PITCH * row + 4
            if x <= click_x < x + 6 and y <= click_y < y + 6:
                return row, col
        return None

    def _rotate(self, pivot: tuple[int, int]) -> None:
        row, col = pivot
        old = {cell: self.tokens[cell] for cell in self._pivot_cells(pivot)}
        self.animation_snapshot = old
        self.animation_pivot = pivot
        self.tokens[(row, col)] = old[(row + 1, col)]
        self.tokens[(row, col + 1)] = old[(row, col)]
        self.tokens[(row + 1, col + 1)] = old[(row, col + 1)]
        self.tokens[(row + 1, col)] = old[(row + 1, col + 1)]
        self.spent.add(pivot)
        self.remaining_steps -= 1
        self.solved = self._is_solved()
        self.motion_index = 0
        self.motion_amount = 3

    def _finish_rotation(self) -> None:
        self.animation_snapshot = None
        self.motion_amount = None
        if self.solved:
            self.next_level()
        elif self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _play_transient_frame(self) -> bool:
        if self.motion_amount is not None:
            if self.motion_index < 2:
                self.motion_index += 1
                self.motion_amount = (3, 7, 10)[self.motion_index]
            else:
                self._finish_rotation()
            return True
        if self.shaken_pivot is not None:
            if self.shake_phase == 0:
                self.shake_phase = 1
                self.spent_shake_shift = -1
                self.complete_action()
            elif self.shake_phase == 1:
                self.shaken_pivot = None
                self.spent_shake_shift = 0
                self.shake_phase = 2
                return False
            return True
        if self.edge_flash:
            self.edge_flash = False
            self.complete_action()
            return True
        return False

    def step(self) -> None:
        if self._play_transient_frame():
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        if self.action.id == GameAction.ACTION7:
            if self.undo_history:
                tokens, spent, solved = self.undo_history.pop()
                self.tokens = dict(tokens)
                self.spent = set(spent)
                self.solved = solved
            self.complete_action()
            return

        if self.action.id != GameAction.ACTION6:
            self.edge_flash = True
            return

        pivot = self._clicked_pivot()
        if pivot is None:
            self.edge_flash = True
            return
        if pivot in self.spent:
            self.shaken_pivot = pivot
            self.spent_shake_shift = 1
            self.shake_phase = 0
            return

        self.undo_history.append((dict(self.tokens), set(self.spent), self.solved))
        self._rotate(pivot)
