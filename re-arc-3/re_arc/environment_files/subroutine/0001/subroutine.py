from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

GAME_ID = "subroutine-0001"
VIEW_SIZE = 64

ACTION_SPACE = [1, 2, 3, 4, 5, 6]
ACTION_SPACEBAR = 5
ACTION_CLICK = 6

WHITE = 0
LIGHT_GRAY = 1
GRAY = 2
DARK_GRAY = 3
BLACK = 5
RED = 8
BLUE = 9
YELLOW = 11

TARGET_COLORS = {"Y": 11, "O": 12, "B": 9, "C": 10, "G": 14, "M": 6, "Pk": 7, "Pu": 15}

COMPONENT_COLOR = {"R": 8, "B": 9, "G": 14, "P": 15}
COMPONENT_BY_COLOR = {value: key for key, value in COMPONENT_COLOR.items()}
TRAY_POSITIONS = [(2 + 6 * index, 56) for index in range(10)]
BASELINE_ACTIONS = [7, 9, 13, 13, 15, 21, 19, 25]
STEP_BUDGETS = [count * 6 for count in BASELINE_ACTIONS]


@dataclass(frozen=True)
class Token:
    token_id: str
    kind: str
    color: int
    fixed: bool = False


@dataclass(frozen=True)
class ComponentSpec:
    key: str
    x: int
    y: int
    slots: int


@dataclass(frozen=True)
class LevelSpec:
    name: str
    finish_mode: str
    target: tuple[int, ...]
    components: tuple[ComponentSpec, ...]
    slots: dict[str, Token | None]
    tray: dict[str, Token | None]
    energy: int | None = None


def _lit(name: str, color_key: str) -> Token:
    return Token(name, "lit", TARGET_COLORS[color_key])


def _ptr(name: str, color_key: str, *, fixed: bool = False) -> Token:
    color = COMPONENT_COLOR["P"] if color_key == "Pu" else TARGET_COLORS[color_key]
    return Token(name, "ptr", color, fixed=fixed)


LEVEL_SPECS = (
    LevelSpec(
        name="Direct Literals",
        finish_mode="EXACT_RETURN",
        target=(11, 14, 12),
        components=(ComponentSpec("R", 20, 16, 3),),
        slots={"R0": None, "R1": None, "R2": None},
        tray={"T0": _lit("L1_Y", "Y"), "T1": _lit("L1_O", "O"), "T2": _lit("L1_G", "G")},
    ),
    LevelSpec(
        name="Fixed Pointer",
        finish_mode="EXACT_RETURN",
        target=(11, 10, 6, 12),
        components=(ComponentSpec("R", 18, 16, 3), ComponentSpec("B", 24, 28, 2)),
        slots={"R0": None, "R1": _ptr("L2_PtrB_fixed", "B", fixed=True), "R2": None, "B0": None, "B1": None},
        tray={"T0": _lit("L2_O", "O"), "T1": _lit("L2_Y", "Y"), "T2": _lit("L2_M", "M"), "T3": _lit("L2_C", "C")},
    ),
    LevelSpec(
        name="Movable Pointer",
        finish_mode="EXACT_RETURN",
        target=(7, 11, 14, 12, 6),
        components=(ComponentSpec("R", 15, 16, 4), ComponentSpec("B", 21, 28, 2)),
        slots={"R0": None, "R1": None, "R2": None, "R3": None, "B0": None, "B1": None},
        tray={
            "T0": _lit("L3_O", "O"),
            "T1": _ptr("L3_PtrB", "B"),
            "T2": _lit("L3_Pk", "Pk"),
            "T3": _lit("L3_M", "M"),
            "T4": _lit("L3_G", "G"),
            "T5": _lit("L3_Y", "Y"),
        },
    ),
    LevelSpec(
        name="Nested Calls",
        finish_mode="EXACT_RETURN",
        target=(11, 10, 7, 14, 12),
        components=(ComponentSpec("R", 18, 15, 3), ComponentSpec("B", 10, 27, 2), ComponentSpec("G", 34, 39, 2)),
        slots={
            "R0": None,
            "R1": None,
            "R2": None,
            "B0": None,
            "B1": _ptr("L4_PtrG_fixed", "G", fixed=True),
            "G0": None,
            "G1": None,
        },
        tray={
            "T0": _ptr("L4_PtrB", "B"),
            "T1": _lit("L4_G", "G"),
            "T2": _lit("L4_Y", "Y"),
            "T3": _lit("L4_O", "O"),
            "T4": _lit("L4_C", "C"),
            "T5": _lit("L4_Pk", "Pk"),
        },
    ),
    LevelSpec(
        name="Repeated Calls",
        finish_mode="EXACT_RETURN",
        target=(11, 10, 14, 12, 10, 14, 7),
        components=(ComponentSpec("R", 9, 15, 5), ComponentSpec("B", 24, 29, 2)),
        slots={"R0": None, "R1": None, "R2": None, "R3": None, "R4": None, "B0": None, "B1": None},
        tray={
            "T0": _ptr("L5_PtrB_a", "B"),
            "T1": _lit("L5_Y", "Y"),
            "T2": _lit("L5_G", "G"),
            "T3": _lit("L5_O", "O"),
            "T4": _ptr("L5_PtrB_b", "B"),
            "T5": _lit("L5_Pk", "Pk"),
            "T6": _lit("L5_C", "C"),
        },
    ),
    LevelSpec(
        name="Color Identity",
        finish_mode="EXACT_RETURN",
        target=(11, 6, 10, 14, 12, 7, 15, 9, 10, 14),
        components=(
            ComponentSpec("R", 2, 15, 7),
            ComponentSpec("B", 47, 25, 2),
            ComponentSpec("G", 18, 35, 1),
            ComponentSpec("P", 35, 45, 2),
        ),
        slots={
            "R0": None,
            "R1": None,
            "R2": None,
            "R3": None,
            "R4": None,
            "R5": None,
            "R6": None,
            "B0": _lit("L6_C", "C"),
            "B1": _lit("L6_G", "G"),
            "G0": None,
            "P0": None,
            "P1": None,
        },
        tray={
            "T0": _lit("L6_Y", "Y"),
            "T1": _lit("L6_O", "O"),
            "T2": _lit("L6_B", "B"),
            "T3": _lit("L6_M", "M"),
            "T4": _lit("L6_Pk", "Pk"),
            "T5": _lit("L6_Pu", "Pu"),
            "T6": _ptr("L6_PtrG", "G"),
            "T7": _ptr("L6_PtrB_a", "B"),
            "T8": _ptr("L6_PtrPu", "Pu"),
            "T9": _ptr("L6_PtrB_b", "B"),
        },
        energy=18,
    ),
    LevelSpec(
        name="Recursion",
        finish_mode="EARLY_ON_TARGET",
        target=(11, 6, 7, 12, 10, 14, 10, 14, 10, 14, 10, 14),
        components=(
            ComponentSpec("R", 17, 15, 4),
            ComponentSpec("G", 5, 29, 1),
            ComponentSpec("P", 18, 29, 2),
            ComponentSpec("B", 39, 29, 3),
        ),
        slots={
            "R0": _lit("L7_C", "C"),
            "R1": None,
            "R2": None,
            "R3": None,
            "G0": _lit("L7_Y", "Y"),
            "P0": None,
            "P1": None,
            "B0": _lit("L7_M", "M"),
            "B1": None,
            "B2": None,
        },
        tray={
            "T0": _lit("L7_G", "G"),
            "T1": _lit("L7_Pk", "Pk"),
            "T2": _lit("L7_O", "O"),
            "T3": _ptr("L7_PtrG", "G"),
            "T4": _ptr("L7_PtrPu", "Pu"),
            "T5": _ptr("L7_PtrB_a", "B"),
            "T6": _ptr("L7_PtrB_b", "B"),
        },
        energy=18,
    ),
    LevelSpec(
        name="Dependency Network",
        finish_mode="EARLY_ON_TARGET",
        target=(11, 7, 12, 10, 6, 14, 15, 11, 7, 12, 10, 6, 14, 9, 15, 9, 15, 9),
        components=(
            ComponentSpec("R", 4, 15, 4),
            ComponentSpec("B", 39, 25, 3),
            ComponentSpec("P", 24, 35, 2),
            ComponentSpec("G", 5, 45, 5),
        ),
        slots={
            "R0": None,
            "R1": None,
            "R2": None,
            "R3": None,
            "B0": _lit("L8_Y", "Y"),
            "B1": None,
            "B2": None,
            "P0": _lit("L8_C", "C"),
            "P1": None,
            "G0": _lit("L8_B", "B"),
            "G1": _ptr("L8_PtrPu_fixed", "Pu", fixed=True),
            "G2": None,
            "G3": None,
            "G4": None,
        },
        tray={
            "T0": _lit("L8_Pk", "Pk"),
            "T1": _lit("L8_O", "O"),
            "T2": _lit("L8_M", "M"),
            "T3": _lit("L8_G", "G"),
            "T4": _lit("L8_Pu_a", "Pu"),
            "T5": _lit("L8_Pu_b", "Pu"),
            "T6": _ptr("L8_PtrG_a", "G"),
            "T7": _ptr("L8_PtrG_b", "G"),
            "T8": _ptr("L8_PtrB_a", "B"),
            "T9": _ptr("L8_PtrB_b", "B"),
        },
        energy=18,
    ),
)


class SubroutineView(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: Subroutine | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame
        frame[:, :] = WHITE
        self._draw_targets(frame, game)
        for component in game.components:
            self._draw_component(frame, game, component)
        self._draw_tray(frame, game)
        self._draw_energy(frame, game)
        self._draw_step_bar(frame, game)
        return frame

    def _draw_targets(self, frame: np.ndarray, game: Subroutine) -> None:
        for index, color in enumerate(game.target):
            x = 5 + 6 * (index % 9)
            y = 1 + 6 * (index // 9)
            _draw_hollow_square(frame, x, y, color)
            if index < game.trace_len:
                frame[y + 1 : y + 4, x + 1 : x + 4] = color
            if game.fail_index == index:
                frame[y + 1 : y + 4, x + 1 : x + 4] = RED

    def _draw_component(self, frame: np.ndarray, game: Subroutine, component: ComponentSpec) -> None:
        width = 6 * component.slots + 3
        color = COMPONENT_COLOR[component.key]
        x, y = component.x, component.y
        frame[y : y + 9, x : x + width] = color
        frame[y + 1 : y + 8, x + 1 : x + width - 1] = WHITE
        for index in range(component.slots):
            well = f"{component.key}{index}"
            self._draw_well(frame, game, well, x + 2 + 6 * index, y + 2)

    def _draw_tray(self, frame: np.ndarray, game: Subroutine) -> None:
        for index, (x, y) in enumerate(TRAY_POSITIONS):
            self._draw_well(frame, game, f"T{index}", x, y)

    def _draw_well(self, frame: np.ndarray, game: Subroutine, well: str, x: int, y: int) -> None:
        frame[y : y + 5, x : x + 5] = LIGHT_GRAY
        _draw_rect_outline(frame, x, y, 5, 5, DARK_GRAY)
        token = game.tokens_by_well.get(well)
        if token is not None:
            _draw_token(frame, x, y, token)
            if token.fixed:
                _draw_clamps(frame, x, y, BLACK if game.clamp_flash else DARK_GRAY)
        if game.selected_well == well:
            _draw_rect_outline(frame, x - 1, y - 1, 7, 7, BLACK)

    def _draw_energy(self, frame: np.ndarray, game: Subroutine) -> None:
        if game.energy_max is None:
            return
        for index in range(game.energy_max):
            y = 1 + 3 * index
            if y + 2 > 55:
                break
            color = YELLOW if index < game.energy else GRAY
            frame[y : y + 2, 62:64] = color

    def _draw_step_bar(self, frame: np.ndarray, game: Subroutine) -> None:
        width = 58
        x, y = 3, 62
        frame[y:64, x : x + width] = GRAY
        filled = max(0, min(width, int(width * game.steps_left / max(1, game.step_budget))))
        if filled:
            frame[y:64, x : x + filled] = YELLOW


def _draw_rect_outline(frame: np.ndarray, x: int, y: int, width: int, height: int, color: int) -> None:
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(VIEW_SIZE, x + width), min(VIEW_SIZE, y + height)
    if x0 >= x1 or y0 >= y1:
        return
    frame[y0, x0:x1] = color
    frame[y1 - 1, x0:x1] = color
    frame[y0:y1, x0] = color
    frame[y0:y1, x1 - 1] = color


def _draw_hollow_square(frame: np.ndarray, x: int, y: int, color: int) -> None:
    frame[y : y + 5, x : x + 5] = WHITE
    _draw_rect_outline(frame, x, y, 5, 5, color)


def _draw_clamps(frame: np.ndarray, x: int, y: int, color: int) -> None:
    frame[y, x] = color
    frame[y, x + 1] = color
    frame[y + 1, x] = color
    frame[y + 4, x + 4] = color
    frame[y + 4, x + 3] = color
    frame[y + 3, x + 4] = color


def _draw_token(frame: np.ndarray, x: int, y: int, token: Token) -> None:
    if token.kind == "lit":
        frame[y : y + 5, x : x + 5] = token.color
        _draw_rect_outline(frame, x, y, 5, 5, DARK_GRAY)
        return
    frame[y : y + 5, x : x + 5] = WHITE
    _draw_rect_outline(frame, x, y, 5, 5, token.color)
    frame[y + 2, x + 2] = BLACK


def _solid_sprite(color: int) -> Sprite:
    return Sprite(np.full((VIEW_SIZE, VIEW_SIZE), color, dtype=np.int8), name="canvas", layer=-100, collidable=False)


def _build_level(index: int, spec: LevelSpec) -> Level:
    return Level(
        sprites=[_solid_sprite(WHITE)],
        grid_size=(VIEW_SIZE, VIEW_SIZE),
        name=spec.name,
        data={"spec": spec, "index": index, "step_budget": STEP_BUDGETS[index]},
    )


def _action_id(action_id: object) -> int:
    value = getattr(action_id, "value", action_id)
    return int(value)


class Subroutine(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._view = SubroutineView()
        self._view.game = self
        levels = [_build_level(index, spec) for index, spec in enumerate(LEVEL_SPECS)]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, VIEW_SIZE, VIEW_SIZE, background=WHITE, letter_box=WHITE, interfaces=[self._view]),
            False,
            len(levels),
            ACTION_SPACE,
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        self.spec: LevelSpec = level.get_data("spec")
        self.components = list(self.spec.components)
        self.target = tuple(self.spec.target)
        self.tokens_by_well = dict(self.spec.slots)
        self.tokens_by_well.update({f"T{index}": None for index in range(10)})
        self.tokens_by_well.update(self.spec.tray)
        self.selected_well: str | None = None
        self.trace_len = 0
        self.fail_index: int | None = None
        self.clamp_flash = 0
        self.energy_max = self.spec.energy
        self.energy = self.spec.energy if self.spec.energy is not None else 0
        self.step_budget = int(level.get_data("step_budget"))
        self.steps_left = self.step_budget
        self.run_frames: list[tuple[int, int | None]] = []
        self.pending_success = False

    def step(self) -> None:
        if self.run_frames:
            self._play_run_frame()
            return

        action = _action_id(self.action.id)
        if action == GameAction.RESET.value:
            self.complete_action()
            return

        if action in {1, 2, 3, 4}:
            self.complete_action()
            return

        if self.steps_left <= 0:
            self.lose()
            self.complete_action()
            return

        self.steps_left -= 1
        if action == ACTION_CLICK:
            self._handle_click()
        elif action == ACTION_SPACEBAR:
            self._handle_run()

        if self.clamp_flash > 0:
            self.clamp_flash -= 1
        if self.steps_left <= 0 and not self.pending_success:
            self.lose()
            self.complete_action()
            return
        if not self.run_frames:
            self.complete_action()

    def _handle_click(self) -> None:
        x = int(self.action.data.get("x", 0))
        y = int(self.action.data.get("y", 0))
        well = self._well_at(x, y)
        if well is None:
            self.selected_well = None
            return
        token = self.tokens_by_well.get(well)
        if self.selected_well is None:
            if token is None:
                return
            if token.fixed:
                self.clamp_flash = 2
                return
            self.selected_well = well
            return
        if well == self.selected_well:
            self.selected_well = None
            return
        if token is not None and token.fixed:
            self.clamp_flash = 2
            return
        moving = self.tokens_by_well.get(self.selected_well)
        if moving is None or moving.fixed:
            self.selected_well = None
            return
        if self.energy_max is not None and self.energy <= 0:
            return
        self.tokens_by_well[self.selected_well], self.tokens_by_well[well] = token, moving
        self.selected_well = None
        self.trace_len = 0
        self.fail_index = None
        if self.energy_max is not None:
            self.energy -= 1

    def _handle_run(self) -> None:
        if self.energy_max is not None:
            if self.energy <= 0:
                return
            self.energy -= 1
        self.selected_well = None
        success, matched, failed = self._execute_program()
        self.pending_success = success
        self.fail_index = None
        self.trace_len = 0
        self.run_frames = [(index, None) for index in range(1, matched + 1)]
        if failed is not None:
            self.run_frames.append((matched, failed))
        if not self.run_frames:
            self.run_frames.append((0, 0))

    def _play_run_frame(self) -> None:
        self.trace_len, self.fail_index = self.run_frames.pop(0)
        if self.run_frames:
            return
        if self.pending_success:
            self.next_level()
            self.complete_action()
            return
        self.complete_action()

    def _well_at(self, x: int, y: int) -> str | None:
        for index, (well_x, well_y) in enumerate(TRAY_POSITIONS):
            if well_x <= x < well_x + 5 and well_y <= y < well_y + 5:
                return f"T{index}"
        for component in self.components:
            for index in range(component.slots):
                well_x = component.x + 2 + 6 * index
                well_y = component.y + 2
                if well_x <= x < well_x + 5 and well_y <= y < well_y + 5:
                    return f"{component.key}{index}"
        return None

    def _component_slots(self, component_key: str) -> list[Token | None]:
        component = next(item for item in self.components if item.key == component_key)
        return [self.tokens_by_well.get(f"{component.key}{index}") for index in range(component.slots)]

    def _execute_program(self) -> tuple[bool, int, int | None]:
        component_keys = {component.key for component in self.components}
        target_index = 0
        instructions = 0
        stack: list[tuple[str, int]] = [("R", 0)]

        while stack:
            if instructions > 256:
                return False, target_index, min(target_index, len(self.target) - 1)
            if len(stack) > 64:
                return False, target_index, min(target_index, len(self.target) - 1)
            component_key, pc = stack[-1]
            slots = self._component_slots(component_key)
            if pc >= len(slots):
                stack.pop()
                continue
            token = slots[pc]
            stack[-1] = (component_key, pc + 1)
            instructions += 1
            if token is None:
                continue
            if token.kind == "lit":
                if target_index >= len(self.target):
                    return False, len(self.target), len(self.target) - 1
                if token.color != self.target[target_index]:
                    return False, target_index, target_index
                target_index += 1
                if self.spec.finish_mode == "EARLY_ON_TARGET" and target_index == len(self.target):
                    return True, target_index, None
                continue
            callee = COMPONENT_BY_COLOR.get(token.color)
            if callee is None or callee not in component_keys:
                return False, target_index, min(target_index, len(self.target) - 1)
            stack.append((callee, 0))

        if target_index == len(self.target):
            return True, target_index, None
        return False, target_index, target_index


__all__ = ["Subroutine"]
