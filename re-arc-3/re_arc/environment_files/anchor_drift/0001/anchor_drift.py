from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay

GAME_ID = "anchor_drift-0001"

ACTION_UP = 1
ACTION_DOWN = 2
ACTION_LEFT = 3
ACTION_RIGHT = 4
ACTION_SPACE = 5
ACTION_UNDO = 7

DELTAS = {ACTION_UP: (0, -1), ACTION_DOWN: (0, 1), ACTION_LEFT: (-1, 0), ACTION_RIGHT: (1, 0)}

BOARD_SIZE = 10
CELL = 6
MARGIN = 2

WATER = 10
REEF = 4
REEF_DARK = 3
WHITE = 0
ROPE = 13
DARK = 5
SAILOR = 9
PURPLE = 2
ANCHOR = PURPLE
BAR_FULL = 14
BAR_EMPTY = 8


@dataclass
class Crate:
    id: str
    color: int
    pos: tuple[int, int]
    dock: tuple[int, int]
    group: str | None = None
    target_color: int | None = None
    dock_color: int | None = None


AnchorDriftSnapshot = tuple[
    tuple[int, int],
    str,
    tuple[int, int] | None,
    dict[str, tuple[int, tuple[int, int], tuple[int, int], str | None, int | None, int | None]],
    dict[str, tuple[list[str], str]],
]


class AnchorDriftHud(RenderableUserDisplay):
    def __init__(self) -> None:
        self.game: AnchorDrift | None = None

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        game = self.game
        if game is None:
            return frame

        frame[:, :] = DARK
        frame[MARGIN : MARGIN + BOARD_SIZE * CELL, MARGIN : MARGIN + BOARD_SIZE * CELL] = WATER
        self._draw_docks(frame, game)
        self._draw_color_pads(frame, game)
        self._draw_reefs(frame, game)
        self._draw_anchor(frame, game)
        self._draw_ropes(frame, game)
        self._draw_crates(frame, game)
        self._draw_tethers(frame, game)
        self._draw_sailor(frame, game.sailor)
        self._draw_carried_anchor(frame, game)
        self._draw_step_bar(frame, game.remaining_steps, game.step_budget)
        if game.flash_invalid > 0:
            frame[0:2, :] = PURPLE
            frame[62:64, :] = PURPLE
        return frame

    def _cell_rect(self, pos: tuple[int, int]) -> tuple[int, int]:
        return MARGIN + pos[0] * CELL, MARGIN + pos[1] * CELL

    def _center(self, pos: tuple[int, int]) -> tuple[int, int]:
        x, y = self._cell_rect(pos)
        return x + 3, y + 3

    def _crate_center(self, game: AnchorDrift, crate: Crate) -> tuple[int, int]:
        x, y = game.crate_animation_pixels.get(crate.id, self._cell_rect(crate.pos))
        return x + 3, y + 3

    def _draw_docks(self, frame: np.ndarray, game: AnchorDrift) -> None:
        for crate in game.crates.values():
            x, y = self._cell_rect(crate.dock)
            color = crate.dock_color or crate.target_color or crate.color
            frame[y, x + 1 : x + 5] = color
            frame[y + 5, x + 1 : x + 5] = color
            frame[y + 1 : y + 5, x] = color
            frame[y + 1 : y + 5, x + 5] = color
            frame[y + 2 : y + 4, x + 2 : x + 4] = color if crate.pos == crate.dock else WATER

    def _draw_reefs(self, frame: np.ndarray, game: AnchorDrift) -> None:
        for pos in game.reefs:
            x, y = self._cell_rect(pos)
            frame[y : y + 6, x : x + 6] = REEF
            frame[y + 1 : y + 5, x + 1 : x + 5] = REEF_DARK
            frame[y + 2 : y + 4, x + 2 : x + 4] = REEF

    def _draw_color_pads(self, frame: np.ndarray, game: AnchorDrift) -> None:
        for pos, color in game.color_pads.items():
            x, y = self._cell_rect(pos)
            frame[y : y + 6, x : x + 6] = REEF_DARK
            frame[y, x : x + 6] = WHITE
            frame[y + 5, x : x + 6] = WHITE
            frame[y : y + 6, x] = WHITE
            frame[y : y + 6, x + 5] = WHITE
            frame[y + 2 : y + 4, x + 2 : x + 4] = color

    def _draw_anchor(self, frame: np.ndarray, game: AnchorDrift) -> None:
        if game.anchor_state != "dropped" or game.anchor_pos is None:
            return
        x, y = self._cell_rect(game.anchor_pos)
        frame[y + 1 : y + 5, x + 1 : x + 5] = ANCHOR
        frame[y, x + 2 : x + 4] = ANCHOR
        frame[y + 5, x + 2 : x + 4] = ANCHOR
        frame[y + 2 : y + 4, x] = ANCHOR
        frame[y + 2 : y + 4, x + 5] = ANCHOR
        frame[y + 2 : y + 4, x + 2 : x + 4] = WATER

    def _draw_ropes(self, frame: np.ndarray, game: AnchorDrift) -> None:
        for group in game.groups.values():
            crates = [game.crates[cid] for cid in group["members"]]
            for left, right in pairwise(crates):
                x1, y1 = self._crate_center(game, left)
                x2, y2 = self._crate_center(game, right)
                if x1 == x2:
                    lo, hi = sorted((y1, y2))
                    frame[lo : hi + 1, x1] = ROPE
                elif y1 == y2:
                    lo, hi = sorted((x1, x2))
                    frame[y1, lo : hi + 1] = ROPE

    def _draw_tethers(self, frame: np.ndarray, game: AnchorDrift) -> None:
        if game.anchor_state != "dropped" or game.anchor_pos is None:
            return
        for crate in game._anchored_crates():
            x1, y1 = self._crate_center(game, crate)
            x2, y2 = self._center(game.anchor_pos)
            if x1 == x2:
                lo, hi = sorted((y1, y2))
                frame[lo - 2 : hi + 3, x1 - 1] = PURPLE
                frame[lo - 2 : hi + 3, x1 + 1] = PURPLE
            elif y1 == y2:
                lo, hi = sorted((x1, x2))
                frame[y1 - 1, lo - 2 : hi + 3] = PURPLE
                frame[y1 + 1, lo - 2 : hi + 3] = PURPLE

    def _draw_crates(self, frame: np.ndarray, game: AnchorDrift) -> None:
        for crate in game.crates.values():
            x, y = game.crate_animation_pixels.get(crate.id, self._cell_rect(crate.pos))
            self._draw_crate(frame, crate, x, y)
            if crate.pos == crate.dock:
                self._draw_arrived_marker(frame, crate, x, y)

    def _draw_crate(self, frame: np.ndarray, crate: Crate, x: int, y: int) -> None:
        if x < 0 or y < 0 or x + CELL > frame.shape[1] or y + CELL > frame.shape[0]:
            return
        frame[y + 1 : y + 6, x + 1 : x + 6] = DARK
        frame[y + 1 : y + 5, x + 1 : x + 5] = crate.color
        frame[y + 3, x + 1 : x + 5] = DARK
        frame[y + 2, x + 2 : x + 4] = crate.color

    def _draw_arrived_marker(self, frame: np.ndarray, crate: Crate, x: int, y: int) -> None:
        target_color = crate.dock_color or crate.target_color or crate.color
        frame[y, x + 1 : x + 5] = WHITE
        frame[y + 5, x + 1 : x + 5] = WHITE
        frame[y + 1 : y + 5, x] = WHITE
        frame[y + 1 : y + 5, x + 5] = WHITE
        frame[y + 2, x + 2 : x + 4] = target_color
        frame[y + 4, x + 2 : x + 4] = target_color

    def _draw_sailor(self, frame: np.ndarray, pos: tuple[int, int]) -> None:
        x, y = self._cell_rect(pos)
        frame[y + 1, x + 2 : x + 4] = WHITE
        frame[y + 2 : y + 5, x + 2 : x + 4] = SAILOR
        frame[y + 3, x + 1 : x + 5] = SAILOR
        frame[y + 5, x + 1] = SAILOR
        frame[y + 5, x + 4] = SAILOR

    def _draw_carried_anchor(self, frame: np.ndarray, game: AnchorDrift) -> None:
        if game.anchor_state != "held":
            return
        x, y = self._cell_rect(game.sailor)
        frame[y, x + 4] = ANCHOR
        frame[y + 1, x + 3 : x + 6] = ANCHOR
        frame[y + 2, x + 4] = WATER
        frame[y + 2, x + 3] = ANCHOR
        frame[y + 2, x + 5] = ANCHOR

    def _draw_step_bar(self, frame: np.ndarray, remaining: int, budget: int) -> None:
        frame[63, 2:62] = BAR_EMPTY
        fill = int(60 * max(0, remaining) / max(1, budget))
        if fill:
            frame[63, 2 : 2 + fill] = BAR_FULL


class AnchorDrift(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = AnchorDriftHud()
        self._hud.game = self
        levels = [Level(sprites=[], grid_size=(64, 64), data={"spec": spec}, name=spec["name"]) for spec in LEVEL_SPECS]
        super().__init__(
            GAME_ID,
            levels,
            Camera(0, 0, 64, 64, WATER, DARK, [self._hud]),
            False,
            len(levels),
            [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_SPACE, ACTION_UNDO],
            seed=seed,
        )

    def on_set_level(self, level: Level) -> None:
        spec = level.get_data("spec")
        self.sailor = tuple(spec["sailor"])
        self.anchor_state = spec["anchor"][0]
        self.anchor_pos = tuple(spec["anchor"][1]) if len(spec["anchor"]) > 1 else None
        self.fixed_anchor = bool(spec.get("fixed_anchor", False))
        self.requires_anchor = bool(spec.get("requires_anchor", False))
        self.reefs = {tuple(pos) for pos in spec["reefs"]}
        self.color_pads = {tuple(item["pos"]): int(item["color"]) for item in spec.get("color_pads", [])}
        self.crates = {
            item["id"]: Crate(
                item["id"],
                item["color"],
                tuple(item["pos"]),
                tuple(item["dock"]),
                item.get("group"),
                item.get("target_color"),
                item.get("dock_color", item.get("target_color", item["color"])),
            )
            for item in spec["crates"]
        }
        self.groups = {
            gid: {"members": list(data["members"]), "active": data["active"]}
            for gid, data in spec.get("groups", {}).items()
        }
        self.step_budget = int(spec["budget"])
        self.remaining_steps = self.step_budget
        self.flash_invalid = 0
        self.crate_animation_pixels: dict[str, tuple[int, int]] = {}
        self._pivot_animation_frames: list[dict[str, tuple[int, int]]] = []
        self._pivot_final_positions: dict[str, tuple[int, int]] | None = None
        self._pivot_final_sailor: tuple[int, int] | None = None
        self.undo_history: list[AnchorDriftSnapshot] = []

    def step(self) -> None:
        if self._pivot_final_positions is not None:
            self._advance_pivot_animation()
            return

        if self.action.id == GameAction.RESET:
            self.flash_invalid = 0
            self.crate_animation_pixels = {}
            self._pivot_animation_frames = []
            self._pivot_final_positions = None
            self._pivot_final_sailor = None
            self.complete_action()
            return

        action_id = int(self.action.id.value)
        changed = False
        if action_id in DELTAS:
            self._push_undo_snapshot()
            changed = self._move_or_push(DELTAS[action_id])
        elif action_id == ACTION_SPACE:
            self._push_undo_snapshot()
            changed = self._space()
        elif action_id == ACTION_UNDO:
            changed = self._undo()

        self.remaining_steps -= 1
        self.flash_invalid = 0 if changed else 1
        if self._pivot_final_positions is None:
            self._finish_resolved_action()

    def _push_undo_snapshot(self) -> None:
        crates = {
            cid: (crate.color, crate.pos, crate.dock, crate.group, crate.target_color, crate.dock_color)
            for cid, crate in self.crates.items()
        }
        groups = {gid: (list(data["members"]), str(data["active"])) for gid, data in self.groups.items()}
        self.undo_history.append((self.sailor, self.anchor_state, self.anchor_pos, crates, groups))

    def _undo(self) -> bool:
        if not self.undo_history:
            return False
        sailor, anchor_state, anchor_pos, crates, groups = self.undo_history.pop()
        self.sailor = sailor
        self.anchor_state = anchor_state
        self.anchor_pos = anchor_pos
        self.crates = {
            cid: Crate(cid, color, pos, dock, group, target_color, dock_color)
            for cid, (color, pos, dock, group, target_color, dock_color) in crates.items()
        }
        self.groups = {gid: {"members": list(members), "active": active} for gid, (members, active) in groups.items()}
        self.flash_invalid = 0
        self.crate_animation_pixels = {}
        self._pivot_animation_frames = []
        self._pivot_final_positions = None
        self._pivot_final_sailor = None
        return True

    def _space(self) -> bool:
        if self.anchor_state == "held":
            if self.sailor in self.reefs or self.sailor in self._dock_positions() or self._crate_at(self.sailor):
                return False
            self.anchor_state = "dropped"
            self.anchor_pos = self.sailor
            return True
        if self.anchor_state == "dropped" and self.sailor == self.anchor_pos:
            if self.fixed_anchor:
                return False
            self.anchor_state = "held"
            self.anchor_pos = None
            return True
        return False

    def _move_or_push(self, delta: tuple[int, int]) -> bool:
        target = self._add(self.sailor, delta)
        if not self._in_bounds(target) or target in self.reefs:
            return False
        crate = self._crate_at(target)
        if crate is None:
            self.sailor = target
            return True
        if crate.group is not None:
            self.groups[crate.group]["active"] = crate.id

        moving_ids = self.groups[crate.group]["members"] if crate.group is not None else [crate.id]
        old_active_pos = crate.pos
        if self._slide_step_hits_anchor(moving_ids, delta):
            return False
        new_positions = self._pivot_targets(moving_ids, crate, delta)
        pivoting = new_positions is not None
        if not pivoting:
            new_positions = self._slide_targets(moving_ids, delta)
        if new_positions is None or old_active_pos in new_positions.values():
            return False
        if pivoting:
            self._start_pivot_animation(new_positions, old_active_pos)
            return True
        for cid, pos in new_positions.items():
            self.crates[cid].pos = pos
        self._apply_color_pads(new_positions)
        self.sailor = old_active_pos
        return True

    def _start_pivot_animation(
        self, final_positions: dict[str, tuple[int, int]], final_sailor: tuple[int, int]
    ) -> None:
        mid_frame = {}
        final_frame = {}
        for cid, final_pos in final_positions.items():
            old_x, old_y = self._hud._cell_rect(self.crates[cid].pos)
            final_x, final_y = self._hud._cell_rect(final_pos)
            mid_frame[cid] = ((old_x + final_x) // 2, (old_y + final_y) // 2)
            final_frame[cid] = (final_x, final_y)
        self.crate_animation_pixels = mid_frame
        self._pivot_animation_frames = [final_frame]
        self._pivot_final_positions = final_positions
        self._pivot_final_sailor = final_sailor

    def _advance_pivot_animation(self) -> None:
        if self._pivot_animation_frames:
            self.crate_animation_pixels = self._pivot_animation_frames.pop(0)
            return
        assert self._pivot_final_positions is not None
        assert self._pivot_final_sailor is not None
        for cid, pos in self._pivot_final_positions.items():
            self.crates[cid].pos = pos
        self._apply_color_pads(self._pivot_final_positions)
        self.sailor = self._pivot_final_sailor
        self.crate_animation_pixels = {}
        self._pivot_animation_frames = []
        self._pivot_final_positions = None
        self._pivot_final_sailor = None
        self._finish_resolved_action()

    def _finish_resolved_action(self) -> None:
        if self._is_solved():
            self.complete_action()
            self.next_level()
            return
        if self.remaining_steps <= 0:
            self.lose()
        self.complete_action()

    def _slide_targets(self, moving_ids: list[str], delta: tuple[int, int]) -> dict[str, tuple[int, int]] | None:
        positions = {cid: self.crates[cid].pos for cid in moving_ids}
        moved = False
        while True:
            trial = {cid: self._add(pos, delta) for cid, pos in positions.items()}
            if not self._legal_group_positions(trial, moving_ids, check_sailor=False):
                break
            positions = trial
            moved = True
        return positions if moved else None

    def _slide_step_hits_anchor(self, moving_ids: list[str], delta: tuple[int, int]) -> bool:
        if self.anchor_state != "dropped" or self.anchor_pos is None:
            return False
        return any(self._add(self.crates[cid].pos, delta) == self.anchor_pos for cid in moving_ids)

    def _pivot_targets(
        self, moving_ids: list[str], active: Crate, delta: tuple[int, int]
    ) -> dict[str, tuple[int, int]] | None:
        if self.anchor_state != "dropped" or self.anchor_pos is None:
            return None
        if not self._anchored_pivot_crates(moving_ids):
            return None

        candidates: list[tuple[int, dict[str, tuple[int, int]]]] = []
        for clockwise in (True, False):
            positions = self._rotated_group_positions(moving_ids, clockwise)
            if active.pos in positions.values():
                continue
            if not self._legal_group_positions(positions, moving_ids, check_sailor=True):
                continue
            score = self._pivot_alignment_score(active.pos, positions[active.id], delta)
            candidates.append((score, positions))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def _rotated_group_positions(self, moving_ids: list[str], clockwise: bool) -> dict[str, tuple[int, int]]:
        assert self.anchor_pos is not None
        positions = {}
        for cid in moving_ids:
            crate = self.crates[cid]
            rel = (crate.pos[0] - self.anchor_pos[0], crate.pos[1] - self.anchor_pos[1])
            rr = (-rel[1], rel[0]) if clockwise else (rel[1], -rel[0])
            positions[cid] = (self.anchor_pos[0] + rr[0], self.anchor_pos[1] + rr[1])
        return positions

    def _pivot_alignment_score(
        self, old_active_pos: tuple[int, int], new_active_pos: tuple[int, int], delta: tuple[int, int]
    ) -> int:
        movement = (new_active_pos[0] - old_active_pos[0], new_active_pos[1] - old_active_pos[1])
        return movement[0] * delta[0] + movement[1] * delta[1]

    def _anchored_pivot_crates(self, moving_ids: list[str]) -> list[Crate]:
        if self.anchor_state != "dropped" or self.anchor_pos is None:
            return []
        crates = []
        for cid in moving_ids:
            crate = self.crates[cid]
            if self._adjacent(crate.pos, self.anchor_pos):
                crates.append(crate)
        return crates

    def _legal_group_positions(
        self, positions: dict[str, tuple[int, int]], moving_ids: list[str], check_sailor: bool
    ) -> bool:
        if len(set(positions.values())) != len(positions):
            return False
        moving = set(moving_ids)
        for pos in positions.values():
            if not self._in_bounds(pos) or pos in self.reefs or self._is_anchor_blocker(pos, moving_ids):
                return False
            other = self._crate_at(pos)
            if other is not None and other.id not in moving:
                return False
            if check_sailor and pos == self.sailor:
                return False
        return True

    def _is_solved(self) -> bool:
        if self.requires_anchor and self.anchor_state != "dropped":
            return False
        if self.requires_anchor and (
            self.anchor_pos is None
            or not any(self._adjacent(crate.pos, self.anchor_pos) for crate in self.crates.values())
        ):
            return False
        occupied_docks: set[tuple[int, int]] = set()
        for source in self.crates.values():
            target = self._crate_at(source.dock)
            if target is None or target.color != self._dock_target_color(source):
                return False
            occupied_docks.add(source.dock)
        return all(crate.pos in occupied_docks for crate in self.crates.values())

    def _dock_target_color(self, crate: Crate) -> int:
        return crate.target_color or crate.dock_color or crate.color

    def _apply_color_pads(self, positions: dict[str, tuple[int, int]]) -> None:
        for cid, pos in positions.items():
            color = self.color_pads.get(pos)
            if color is not None:
                self.crates[cid].color = color

    def _crate_at(self, pos: tuple[int, int]) -> Crate | None:
        return next((crate for crate in self.crates.values() if crate.pos == pos), None)

    def _dock_positions(self) -> set[tuple[int, int]]:
        return {crate.dock for crate in self.crates.values()}

    def _anchored_crates(self) -> list[Crate]:
        if self.anchor_state != "dropped" or self.anchor_pos is None:
            return []
        crates = []
        for crate in self.crates.values():
            if self._adjacent(crate.pos, self.anchor_pos):
                crates.append(crate)
        return crates

    def _is_anchor_blocker(self, pos: tuple[int, int], moving_ids: list[str]) -> bool:
        del moving_ids
        return self.anchor_state == "dropped" and self.anchor_pos is not None and pos == self.anchor_pos

    def _display_to_cell(self, x: int, y: int) -> tuple[int, int] | None:
        gx = (x - MARGIN) // CELL
        gy = (y - MARGIN) // CELL
        if 0 <= gx < BOARD_SIZE and 0 <= gy < BOARD_SIZE:
            return gx, gy
        return None

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < BOARD_SIZE and 0 <= pos[1] < BOARD_SIZE

    def _adjacent(self, a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def _add(self, a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
        return a[0] + b[0], a[1] + b[1]


LEVEL_SPECS = [
    {
        "name": "The Drift",
        "sailor": (2, 5),
        "anchor": ("absent",),
        "reefs": [(8, 4)],
        "crates": [{"id": "O", "color": 12, "pos": (4, 4), "dock": (7, 4)}],
        "budget": 36,
    },
    {
        "name": "Drop the Hinge",
        "sailor": (3, 4),
        "anchor": ("dropped", (4, 3)),
        "reefs": [],
        "crates": [
            {"id": "O", "color": 12, "pos": (3, 3), "dock": (4, 4)},
            {"id": "Y", "color": 11, "pos": (2, 4), "dock": (0, 4)},
        ],
        "budget": 40,
    },
    {
        "name": "The Raft Hinge",
        "sailor": (1, 1),
        "anchor": ("dropped", (5, 2)),
        "reefs": [(1, 2), (5, 6), (3, 8), (5, 7)],
        "crates": [
            {"id": "O", "color": 12, "pos": (2, 1), "dock": (4, 3), "group": "G1"},
            {"id": "Y", "color": 11, "pos": (2, 2), "dock": (5, 3), "group": "G1"},
            {"id": "G", "color": 14, "pos": (3, 6), "dock": (4, 7)},
        ],
        "groups": {"G1": {"members": ["O", "Y"], "active": "Y"}},
        "budget": 120,
    },
    {
        "name": "Turn Only",
        "sailor": (0, 7),
        "anchor": ("dropped", (7, 7)),
        "requires_anchor": True,
        "reefs": [(3, 2), (4, 5)],
        "crates": [
            {"id": "P", "color": 2, "pos": (2, 4), "dock": (1, 5)},
            {"id": "B", "color": 7, "pos": (6, 5), "dock": (5, 4)},
            {"id": "Y", "color": 11, "pos": (1, 7), "dock": (6, 9)},
        ],
        "budget": 62,
    },
    {
        "name": "Around the Post",
        "sailor": (3, 6),
        "anchor": ("held",),
        "reefs": [(8, 4), (6, 5), (1, 6), (7, 7), (7, 8), (2, 9)],
        "color_pads": [{"pos": (4, 4), "color": 12}],
        "crates": [{"id": "O", "color": 14, "pos": (3, 5), "dock": (6, 6), "target_color": 12}],
        "budget": 192,
    },
    {
        "name": "Which End Pulls",
        "sailor": (1, 2),
        "anchor": ("held",),
        "requires_anchor": True,
        "reefs": [(7, 2), (5, 6), (6, 6), (0, 5), (1, 9), (2, 9), (8, 8)],
        "color_pads": [{"pos": (6, 2), "color": 14}],
        "crates": [
            {"id": "O", "color": 12, "pos": (2, 2), "dock": (2, 7), "group": "G1"},
            {"id": "Y", "color": 11, "pos": (3, 2), "dock": (2, 6), "group": "G1", "target_color": 14},
        ],
        "groups": {"G1": {"members": ["O", "Y"], "active": "Y"}},
        "budget": 216,
    },
    {
        "name": "Temporary Wall",
        "sailor": (3, 5),
        "anchor": ("held",),
        "reefs": [
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (8, 1),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
            (4, 4),
            (9, 5),
            (5, 6),
            (9, 6),
            (7, 7),
            (7, 8),
        ],
        "color_pads": [{"pos": (8, 2), "color": 14}],
        "crates": [
            {"id": "Y", "color": 11, "pos": (4, 5), "dock": (5, 5), "target_color": 14},
            {"id": "B", "color": 7, "pos": (1, 7), "dock": (1, 1), "group": "G1"},
            {"id": "P", "color": 2, "pos": (1, 8), "dock": (1, 2), "group": "G1"},
        ],
        "groups": {"G1": {"members": ["B", "P"], "active": "P"}},
        "budget": 288,
    },
    {
        "name": "Harbor Weave",
        "sailor": (0, 1),
        "anchor": ("held",),
        "requires_anchor": True,
        "reefs": [
            (4, 0),
            (5, 0),
            (6, 0),
            (0, 3),
            (1, 3),
            (2, 3),
            (4, 3),
            (6, 3),
            (7, 3),
            (8, 3),
            (9, 3),
            (4, 4),
            (4, 5),
            (5, 5),
            (1, 7),
            (2, 7),
            (4, 7),
            (4, 8),
        ],
        "color_pads": [{"pos": (7, 1), "color": 14}, {"pos": (6, 4), "color": 14}],
        "crates": [
            {"id": "Y", "color": 11, "pos": (1, 1), "dock": (7, 2), "target_color": 14},
            {"id": "B", "color": 7, "pos": (1, 6), "dock": (6, 7), "group": "G1", "dock_color": 2},
            {"id": "P", "color": 2, "pos": (2, 6), "dock": (6, 8), "group": "G1", "target_color": 14},
        ],
        "groups": {"G1": {"members": ["B", "P"], "active": "P"}},
        "budget": 240,
    },
]
