from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from arcengine import ARCBaseGame, Camera, Level, Sprite

UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO = 1, 2, 3, 4, 5, 6, 7
VIEW = 64
TIERS = ("H", "M", "L")
DECK_Y = {"H": 17, "M": 35, "L": 53}
COLORS = {
    "back": 1,
    "platform": 2,
    "stone": 3,
    "structure": 4,
    "background": 5,
    "b_frame": 6,
    "b_rope": 7,
    "warning": 8,
    "a_frame": 9,
    "a_rope": 10,
    "player": 11,
    "ceiling": 13,
    "goal": 14,
}


def sx(x: int) -> int:
    return 4 + (5 * x) // 2


def _run(tier: str, start: int, end: int) -> tuple[tuple[int, str], ...]:
    return tuple((x, tier) for x in range(start, end))


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    left_bays: tuple[int, int]
    right_bays: tuple[int, int]
    initial: tuple[str, str]


@dataclass(frozen=True)
class LevelSpec:
    name: str
    rows: tuple[str, str, str]
    pairs: tuple[PairSpec, ...]
    horizontal: tuple[tuple[int, str], ...]
    vertical: tuple[tuple[int, str, str], ...]
    player: tuple[int, str]
    stones: tuple[tuple[str, tuple[int, str]], ...]
    goal: tuple[int, str]
    baseline: int
    full_blockers: tuple[tuple[str, str, str], ...] = ()
    head_blockers: tuple[tuple[str, str, str, int], ...] = ()


# ASCII legend: # absent; . normal deck; : recessed deck; a/A are pair A's
# left/right cages; b/B are pair B's. P, O, and G mark static player, stone,
# and goal nodes. Attached entities remain in `stones` because their identity
# and the cage beneath them are ambiguous in a single character.
LEVEL_SPECS = (
    LevelSpec(
        "Passenger Weight",
        ("#######################", "#P...aa####AA##########", "#####aa..G#############"),
        (PairSpec("A", (5, 6), (11, 12), ("M", "M")),),
        _run("M", 1, 6) + _run("L", 5, 9),
        (),
        (1, "M"),
        (),
        (9, "L"),
        8,
    ),
    LevelSpec(
        "Leave Something Behind",
        ("#########..############", "#####aa.O.PAA..########", "#######################"),
        (PairSpec("A", (5, 6), (11, 12), ("M", "M")),),
        _run("M", 5, 10) + _run("M", 9, 14) + _run("H", 9, 12),
        ((9, "M", "H"),),
        (10, "M"),
        (("S", (8, "M")),),
        (14, "M"),
        11,
    ),
    LevelSpec(
        "Vacate the Bridge",
        ("#..OP::....############", "#....##....##.G########", "###########AA.#########"),
        (PairSpec("A", (5, 6), (11, 12), ("H", "L")),),
        _run("H", 1, 10) + _run("M", 1, 4) + _run("M", 7, 14) + _run("L", 11, 13),
        ((1, "H", "M"), (4, "H", "M"), (10, "H", "M")),
        (4, "H"),
        (("S1", (12, "L")), ("S2", (3, "H"))),
        (14, "M"),
        19,
    ),
    LevelSpec(
        "Rejected Height",
        ("####aa...::....########", "#...aa.P.AA.O.G########", "#...aa#####....########"),
        (PairSpec("A", (4, 5), (9, 10), ("M", "M")),),
        _run("M", 1, 14) + _run("L", 1, 5) + _run("L", 9, 14) + _run("H", 4, 14),
        ((14, "H", "M"), (14, "M", "L"), (3, "M", "L")),
        (7, "M"),
        (("S", (12, "M")),),
        (1, "L"),
        29,
        (("A", "left", "L"),),
    ),
    LevelSpec(
        "Relay Court",
        ("P.aa..::....::..BB..G##", "##aa..#####.####BB#####", "#####.AA....bb#########"),
        (PairSpec("A", (2, 3), (6, 7), ("H", "L")), PairSpec("B", (12, 13), (16, 17), ("L", "H"))),
        _run("H", 0, 20) + _run("M", 2, 5) + _run("L", 5, 13),
        ((5, "H", "M"), (5, "M", "L"), (11, "H", "M"), (11, "M", "L")),
        (0, "H"),
        (("C", (7, "L")), ("D", (12, "L"))),
        (20, "H"),
        43,
    ),
    LevelSpec(
        "Reverse Relay",
        ("G.Oaa..::....::..BB.P##", "#########.#####..BB####", "...####AA....bb.#######"),
        (PairSpec("A", (3, 4), (7, 8), ("H", "L")), PairSpec("B", (13, 14), (17, 18), ("L", "H"))),
        _run("H", 0, 20) + _run("M", 15, 18) + _run("L", 7, 15) + _run("L", 0, 2),
        ((15, "H", "M"), (15, "M", "L"), (9, "H", "M"), (9, "M", "L"), (2, "H", "L"), (0, "H", "L")),
        (20, "H"),
        (("C", (13, "L")), ("D", (8, "L")), ("E", (2, "H"))),
        (0, "H"),
        45,
    ),
    LevelSpec(
        "The Low Ceiling",
        ("G.aa..::.##.::.O.::...#", "########P...::...####.#", "######AA..O.#####BB...#"),
        (PairSpec("A", (2, 3), (6, 7), ("H", "L")), PairSpec("B", (12, 13), (17, 18), ("H", "L"))),
        _run("H", 0, 8) + _run("H", 11, 21) + _run("M", 8, 16) + _run("L", 6, 13) + _run("L", 17, 21),
        ((8, "H", "M"), (8, "M", "L"), (11, "H", "M"), (16, "H", "M"), (21, "H", "M"), (21, "M", "L")),
        (8, "L"),
        (("a", (7, "L")), ("b", (10, "L")), ("d", (18, "L")), ("E", (15, "H"))),
        (0, "H"),
        53,
        (),
        (("B", "right", "M", 18),),
    ),
)


def _parse_rows(rows: tuple[str, str, str]) -> tuple[set[tuple[int, str]], set[tuple[int, str]]]:
    normal: set[tuple[int, str]] = set()
    back: set[tuple[int, str]] = set()
    for tier, row in zip(TIERS, rows, strict=True):
        if len(row) != 23:
            raise ValueError(f"ASCII row must contain 23 cells, got {len(row)}: {row!r}")
        for x, symbol in enumerate(row):
            if symbol == ":":
                back.add((x, tier))
            elif symbol in ".POG":
                normal.add((x, tier))
    return normal, back


def _build_level(spec: LevelSpec) -> Level:
    _parse_rows(spec.rows)
    board = Sprite(
        np.full((VIEW, VIEW), COLORS["background"], dtype=np.int8),
        name="court",
        x=0,
        y=0,
        layer=0,
        collidable=False,
        tags=["court"],
    )
    return Level([board], (VIEW, VIEW), {"spec": spec}, spec.name)


def _tier_path(start: str, end: str) -> tuple[str, ...]:
    a, b = TIERS.index(start), TIERS.index(end)
    lo, hi = sorted((a, b))
    return TIERS[lo : hi + 1]


class CounterweightCourt(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        super().__init__(
            "counterweight_court-0001",
            levels,
            Camera(0, 0, VIEW, VIEW, COLORS["background"], COLORS["background"]),
            False,
            len(levels),
            [UP, DOWN, LEFT, RIGHT, SPACE, CLICK, UNDO],
            seed,
        )

    def on_set_level(self, _level: Level) -> None:
        self.spec: LevelSpec = self.current_level.get_data("spec")
        self.board = self.current_level.get_sprites_by_tag("court")[0]
        self.normal_nodes, self.back_nodes = _parse_rows(self.spec.rows)
        self.horizontal_edges = {frozenset(((x, tier), (x + 1, tier))) for x, tier in self.spec.horizontal}
        self.vertical_edges = {frozenset(((x, tier_a), (x, tier_b))) for x, tier_a, tier_b in self.spec.vertical}
        self.player = self.spec.player
        self.stones = dict(self.spec.stones)
        self.pair_tiers = {pair.pair_id: [pair.initial[0], pair.initial[1]] for pair in self.spec.pairs}
        self.steps_used = 0
        self.step_budget = self.spec.baseline * 6
        self.feedback = "none"
        self.selected_object = None
        self.cursor_cell = None
        self._animation: list[dict[str, object]] = []
        self._finish_after_animation = False
        self._reset_after_animation = False
        self.undo_history: list[tuple[tuple[int, str], dict[str, tuple[int, str]], dict[str, list[str]]]] = []
        self._sync_frame()

    def _pair(self, pair_id: str) -> PairSpec:
        return next(pair for pair in self.spec.pairs if pair.pair_id == pair_id)

    def _cage_at(self, node: tuple[int, str], tiers: dict[str, list[str]] | None = None) -> tuple[str, str] | None:
        pair_tiers = self.pair_tiers if tiers is None else tiers
        x, tier = node
        for pair in self.spec.pairs:
            left_tier, right_tier = pair_tiers[pair.pair_id]
            if tier == left_tier and x in pair.left_bays:
                return pair.pair_id, "left"
            if tier == right_tier and x in pair.right_bays:
                return pair.pair_id, "right"
        return None

    def _active(self, node: tuple[int, str], tiers: dict[str, list[str]] | None = None) -> bool:
        return node in self.normal_nodes or node in self.back_nodes or self._cage_at(node, tiers) is not None

    def _edge(self, start: tuple[int, str], end: tuple[int, str], vertical: bool) -> bool:
        edges = self.vertical_edges if vertical else self.horizontal_edges
        return frozenset((start, end)) in edges and self._active(start) and self._active(end)

    def _snapshot(self) -> tuple[tuple[int, str], dict[str, tuple[int, str]], dict[str, list[str]]]:
        return self.player, dict(self.stones), {key: list(value) for key, value in self.pair_tiers.items()}

    def _restore(self, state: tuple[tuple[int, str], dict[str, tuple[int, str]], dict[str, list[str]]]) -> None:
        player, stones, pair_tiers = state
        self.player = player
        self.stones = dict(stones)
        self.pair_tiers = {key: list(value) for key, value in pair_tiers.items()}

    def _attached(
        self, player: tuple[int, str], stones: dict[str, tuple[int, str]], tiers: dict[str, list[str]]
    ) -> dict[tuple[str, str], list[str]]:
        result: dict[tuple[str, str], list[str]] = {}
        for entity, node in [("player", player), *stones.items()]:
            cage = self._cage_at(node, tiers)
            if cage is not None:
                result.setdefault(cage, []).append(entity)
        return result

    def _targets(self, attached: dict[tuple[str, str], list[str]]) -> dict[str, list[str]]:
        targets: dict[str, list[str]] = {}
        for pair in self.spec.pairs:
            left = len(attached.get((pair.pair_id, "left"), ()))
            right = len(attached.get((pair.pair_id, "right"), ()))
            if left > right:
                targets[pair.pair_id] = ["L", "H"]
            elif left < right:
                targets[pair.pair_id] = ["H", "L"]
            else:
                targets[pair.pair_id] = ["M", "M"]
        return targets

    def _settled_entities(
        self,
        player: tuple[int, str],
        stones: dict[str, tuple[int, str]],
        attached: dict[tuple[str, str], list[str]],
        targets: dict[str, list[str]],
    ) -> tuple[tuple[int, str], dict[str, tuple[int, str]]]:
        final_player = player
        final_stones = dict(stones)
        for (pair_id, side), names in attached.items():
            target_tier = targets[pair_id][0 if side == "left" else 1]
            for name in names:
                if name == "player":
                    final_player = (player[0], target_tier)
                else:
                    final_stones[name] = (stones[name][0], target_tier)
        return final_player, final_stones

    def _validate_settlement(
        self,
        provisional_player: tuple[int, str],
        provisional_stones: dict[str, tuple[int, str]],
        old_tiers: dict[str, list[str]],
        attached: dict[tuple[str, str], list[str]],
        targets: dict[str, list[str]],
        final_player: tuple[int, str],
        final_stones: dict[str, tuple[int, str]],
    ) -> dict[str, object] | None:
        provisional_entities = {"player": provisional_player, **provisional_stones}
        for pair in self.spec.pairs:
            for side_index, side in enumerate(("left", "right")):
                old_tier = old_tiers[pair.pair_id][side_index]
                target = targets[pair.pair_id][side_index]
                if old_tier == target:
                    continue
                bays = pair.left_bays if side == "left" else pair.right_bays
                for swept in _tier_path(old_tier, target):
                    if (pair.pair_id, side, swept) in self.spec.full_blockers:
                        return {"pair": pair.pair_id, "side": side, "tier": swept, "kind": "full"}
                    for entity in attached.get((pair.pair_id, side), ()):
                        bay_x = provisional_entities[entity][0]
                        if (pair.pair_id, side, swept, bay_x) in self.spec.head_blockers:
                            return {"pair": pair.pair_id, "side": side, "tier": swept, "kind": "head", "entity": entity}
                    attached_names = set(attached.get((pair.pair_id, side), ()))
                    for entity, node in provisional_entities.items():
                        if entity not in attached_names and node[0] in bays and node[1] == swept:
                            return {
                                "pair": pair.pair_id,
                                "side": side,
                                "tier": swept,
                                "kind": "occupant",
                                "entity": entity,
                            }

        final_entities = [final_player, *final_stones.values()]
        if len(set(final_entities)) != len(final_entities):
            return {
                "kind": "occupancy",
                "node": next(node for node in final_entities if final_entities.count(node) > 1),
            }
        for node in final_entities:
            if not self._active(node, targets):
                return {"kind": "occupancy", "node": node}
        occupied_cages: set[tuple[int, str]] = set()
        for pair in self.spec.pairs:
            for side_index, bays in enumerate((pair.left_bays, pair.right_bays)):
                tier = targets[pair.pair_id][side_index]
                nodes = {(x, tier) for x in bays}
                if occupied_cages & nodes:
                    return {"kind": "cages", "node": next(iter(occupied_cages & nodes))}
                occupied_cages |= nodes
        return None

    def _attempt_direction(self, action: int) -> tuple[bool, dict[str, object] | None, tuple | None]:
        old_player, old_stones, old_tiers = self._snapshot()
        x, tier = old_player
        if action in (LEFT, RIGHT):
            dx = -1 if action == LEFT else 1
            target = (x + dx, tier)
            if not self._edge(old_player, target, False):
                return False, {"kind": "edge", "from": old_player, "to": target}, None
            stone_name = next((name for name, node in old_stones.items() if node == target), None)
            provisional_stones = dict(old_stones)
            if stone_name is not None:
                beyond = (target[0] + dx, tier)
                if (
                    not self._edge(target, beyond, False)
                    or beyond in old_stones.values()
                    or beyond == old_player
                    or not self._active(beyond)
                ):
                    return False, {"kind": "edge", "from": target, "to": beyond}, None
                provisional_stones[stone_name] = beyond
            provisional_player = target
        else:
            direction = -1 if action == UP else 1
            candidates = []
            for edge in self.vertical_edges:
                if old_player not in edge:
                    continue
                destination = next(iter(edge - {old_player}))
                if (TIERS.index(destination[1]) - TIERS.index(tier)) * direction > 0:
                    candidates.append(destination)
            if len(candidates) != 1 or not self._edge(old_player, candidates[0], True):
                guessed = (x, TIERS[max(0, min(2, TIERS.index(tier) + direction))])
                return False, {"kind": "edge", "from": old_player, "to": guessed}, None
            provisional_player = candidates[0]
            if provisional_player in old_stones.values():
                return False, {"kind": "edge", "from": old_player, "to": provisional_player}, None
            provisional_stones = dict(old_stones)

        attached = self._attached(provisional_player, provisional_stones, old_tiers)
        targets = self._targets(attached)
        final_player, final_stones = self._settled_entities(provisional_player, provisional_stones, attached, targets)
        rejection = self._validate_settlement(
            provisional_player, provisional_stones, old_tiers, attached, targets, final_player, final_stones
        )
        if rejection is not None:
            return False, rejection, None
        self.player = final_player
        self.stones = final_stones
        self.pair_tiers = targets
        context = (old_player, old_stones, old_tiers, provisional_player, provisional_stones, attached, targets)
        return True, None, context

    def _movement_frames(self, context: tuple) -> list[dict[str, object]]:
        old_player, _old_stones, old_tiers, provisional_player, _provisional_stones, attached, targets = context
        max_frames = 1
        for pair in self.spec.pairs:
            for index in (0, 1):
                distance = abs(TIERS.index(old_tiers[pair.pair_id][index]) - TIERS.index(targets[pair.pair_id][index]))
                max_frames = max(max_frames, 8 if distance == 2 else 4 if distance == 1 else 1)
        stair = (
            old_player[0] == provisional_player[0]
            and abs(TIERS.index(old_player[1]) - TIERS.index(provisional_player[1])) == 2
        )
        if stair:
            max_frames = max(max_frames, 4)
        frames: list[dict[str, object]] = []
        for frame_index in range(1, max_frames + 1):
            ratio = frame_index / max_frames
            cage_y: dict[tuple[str, str], int] = {}
            entity_y: dict[str, int] = {}
            for pair in self.spec.pairs:
                for side_index, side in enumerate(("left", "right")):
                    start_y = DECK_Y[old_tiers[pair.pair_id][side_index]]
                    end_y = DECK_Y[targets[pair.pair_id][side_index]]
                    cage_y[(pair.pair_id, side)] = round(start_y + (end_y - start_y) * ratio)
                    for entity in attached.get((pair.pair_id, side), ()):
                        entity_y[entity] = cage_y[(pair.pair_id, side)]
            if stair:
                entity_y["player"] = round(
                    DECK_Y[old_player[1]] + (DECK_Y[provisional_player[1]] - DECK_Y[old_player[1]]) * ratio
                )
            frames.append({"cage_y": cage_y, "entity_y": entity_y})
        return frames

    def _blocked_frames(self, feedback: dict[str, object], count: int) -> list[dict[str, object]]:
        return [{"feedback": feedback, "feedback_on": index % 2 == 0} for index in range(count)]

    def _queue(self, frames: list[dict[str, object]], *, finish: bool = False, reset: bool = False) -> None:
        self._animation = frames or [{}]
        self._finish_after_animation = finish
        self._reset_after_animation = reset

    def _play_animation_frame(self) -> None:
        override = self._animation.pop(0)
        self._sync_frame(override)
        if self._animation:
            return
        if self._reset_after_animation:
            self.on_set_level(self.current_level)
            self.complete_action()
            return
        if self._finish_after_animation:
            self.next_level()
            self.complete_action()
            return
        self.feedback = "none"
        self.complete_action()

    def step(self) -> None:
        if self._animation:
            self._play_animation_frame()
            return
        action = int(getattr(self.action.id, "value", self.action.id))
        if action == 0:
            self._sync_frame()
            self.complete_action()
            return

        self.steps_used += 1
        success = False
        if action == UNDO:
            if self.undo_history:
                self._restore(self.undo_history.pop())
            self.feedback = "none"
            frames = [{}]
        elif action in (UP, DOWN, LEFT, RIGHT):
            before = self._snapshot()
            success, rejection, context = self._attempt_direction(action)
            if success:
                self.undo_history.append(before)
                self.feedback = "none"
                frames = self._movement_frames(context)
            else:
                assert rejection is not None
                self.feedback = "rejected_settlement" if rejection.get("kind") not in {"edge"} else "blocked_edge"
                frames = self._blocked_frames(rejection, 5 if self.feedback == "rejected_settlement" else 2)
        else:
            self.feedback = "blocked_edge"
            frames = self._blocked_frames({"kind": "noop", "node": self.player}, 2)

        if success and self.player == self.spec.goal:
            self.feedback = "completion"
            frames.extend({"pulse": index % 2 == 0} for index in range(6))
            self._queue(frames, finish=True)
        elif self.steps_used >= self.step_budget:
            self.feedback = "budget_failure"
            frames = [{"failure_flash": index % 2 == 0} for index in range(8)]
            self._queue(frames, reset=True)
        else:
            self._queue(frames)
        self._play_animation_frame()

    def _draw_line(self, frame: np.ndarray, a: tuple[int, str], b: tuple[int, str], color: int, thick: int = 1) -> None:
        x1, y1 = sx(a[0]), DECK_Y[a[1]]
        x2, y2 = sx(b[0]), DECK_Y[b[1]]
        if y1 == y2:
            lo, hi = sorted((x1, x2))
            frame[y1 - thick + 1 : y1 + 1, lo : hi + 1] = color
            return
        steps = max(abs(x2 - x1), abs(y2 - y1))
        for index in range(steps + 1):
            ratio = index / max(1, steps)
            x = round(x1 + (x2 - x1) * ratio)
            y = round(y1 + (y2 - y1) * ratio)
            frame[max(0, y - thick + 1) : y + 1, max(0, x - thick + 1) : x + 1] = color

    def _draw_static(self, frame: np.ndarray) -> None:
        for edge in self.horizontal_edges:
            a, b = tuple(edge)
            if a in self.back_nodes and b in self.back_nodes:
                self._draw_line(frame, a, b, COLORS["back"], 1)
            elif a in self.normal_nodes and b in self.normal_nodes:
                self._draw_line(frame, a, b, COLORS["platform"], 2)
        for node in self.normal_nodes:
            x, tier = node
            frame[DECK_Y[tier] - 1 : DECK_Y[tier] + 1, sx(x) - 1 : sx(x) + 2] = COLORS["platform"]
        for node in self.back_nodes:
            x, tier = node
            frame[DECK_Y[tier], sx(x) - 1 : sx(x) + 2] = COLORS["back"]
        for edge in self.vertical_edges:
            a, b = tuple(edge)
            x = sx(a[0])
            y1, y2 = sorted((DECK_Y[a[1]], DECK_Y[b[1]]))
            if abs(TIERS.index(a[1]) - TIERS.index(b[1])) == 2:
                self._draw_line(frame, (a[0], a[1]), (b[0], b[1]), COLORS["platform"], 2)
            else:
                frame[y1 : y2 + 1, x - 2] = COLORS["structure"]
                frame[y1 : y2 + 1, x + 2] = COLORS["structure"]
                for y in range(y1 + 3, y2, 4):
                    frame[y, x - 2 : x + 3] = COLORS["back"]
        for pair in self.spec.pairs:
            for bays in (pair.left_bays, pair.right_bays):
                left, right = sx(bays[0]) - 1, sx(bays[1]) + 1
                for tier in TIERS:
                    y = DECK_Y[tier]
                    frame[y - 1 : y + 1, left - 2 : left] = COLORS["platform"]
                    frame[y - 1 : y + 1, right + 1 : right + 3] = COLORS["stone"]

    def _draw_goal(self, frame: np.ndarray, pulse: bool) -> None:
        del pulse
        x, tier = self.spec.goal
        px, py = sx(x), DECK_Y[tier]
        color = COLORS["goal"]
        frame[py - 6 : py + 1, px - 2] = color
        frame[py - 6 : py + 1, px + 2] = color
        frame[py - 6, px - 2 : px + 3] = color

    def _draw_cage(self, frame: np.ndarray, pair: PairSpec, side: str, deck: int) -> None:
        bays = pair.left_bays if side == "left" else pair.right_bays
        color = COLORS["a_frame"] if pair.pair_id == "A" else COLORS["b_frame"]
        x1, x2 = sx(bays[0]) - 1, sx(bays[1]) + 1
        frame[deck - 7, x1 : x2 + 1] = color
        frame[deck - 7 : deck + 1, x1] = color
        frame[deck - 7 : deck + 1, x2] = color
        frame[deck - 1 : deck + 1, x1 : x2 + 1] = color

    def _draw_pair(self, frame: np.ndarray, pair: PairSpec, cage_y: dict[tuple[str, str], int]) -> None:
        left_y = cage_y.get((pair.pair_id, "left"), DECK_Y[self.pair_tiers[pair.pair_id][0]])
        right_y = cage_y.get((pair.pair_id, "right"), DECK_Y[self.pair_tiers[pair.pair_id][1]])
        left_x = (sx(pair.left_bays[0]) + sx(pair.left_bays[1])) // 2
        right_x = (sx(pair.right_bays[0]) + sx(pair.right_bays[1])) // 2
        center = (left_x + right_x) // 2
        rope = COLORS["a_rope"] if pair.pair_id == "A" else COLORS["b_rope"]
        frame[7, left_x : right_x + 1] = rope
        frame[7 : left_y - 6, left_x] = rope
        frame[7 : right_y - 6, right_x] = rope
        frame[5:10, center - 2 : center + 3] = COLORS["structure"]
        frame[6:9, center - 1 : center + 2] = rope
        self._draw_cage(frame, pair, "left", left_y)
        self._draw_cage(frame, pair, "right", right_y)

    def _draw_blockers(self, frame: np.ndarray, feedback: dict[str, object] | None, flash_on: bool) -> None:
        for pair_id, side, tier in self.spec.full_blockers:
            pair = self._pair(pair_id)
            bays = pair.left_bays if side == "left" else pair.right_bays
            x1, x2 = sx(bays[0]) - 2, sx(bays[1]) + 3
            y = DECK_Y[tier]
            color = (
                COLORS["warning"] if flash_on and feedback and feedback.get("kind") == "full" else COLORS["structure"]
            )
            frame[y - 4 : y + 2, x1:x2] = color
            frame[y - 5, x1 + 1 : x2 - 1] = COLORS["stone"]
        for pair_id, side, tier, bay_x in self.spec.head_blockers:
            del pair_id, side
            x, y = sx(bay_x), DECK_Y[tier]
            color = COLORS["warning"] if flash_on and feedback and feedback.get("kind") == "head" else COLORS["stone"]
            frame[y - 7 : y - 5, x - 2 : x + 3] = color
            frame[y - 5, x - 2 : x + 3] = COLORS["ceiling"]

    def _draw_entities(self, frame: np.ndarray, entity_y: dict[str, int], pulse: bool) -> None:
        for name, (x, tier) in self.stones.items():
            y = entity_y.get(name, DECK_Y[tier])
            px = sx(x)
            frame[y - 4 : y - 1, px - 1 : px + 1] = COLORS["stone"]
            frame[y - 4, px - 1 : px + 1] = COLORS["back"]
        x, tier = self.player
        y = entity_y.get("player", DECK_Y[tier])
        px = sx(x)
        color = COLORS["goal"] if pulse else COLORS["player"]
        frame[y - 4 : y - 1, px - 1 : px + 1] = color

    def _draw_feedback(self, frame: np.ndarray, feedback: dict[str, object], flash_on: bool) -> None:
        if not flash_on:
            return
        kind = feedback.get("kind")
        if kind in {"edge", "noop"}:
            start = feedback.get("from", feedback.get("node", self.player))
            end = feedback.get("to", start)
            self._draw_line(frame, start, end, COLORS["warning"], 2)
            px, py = sx(start[0]), DECK_Y[start[1]] - 7
            frame[py : py + 3, px - 1 : px + 2] = COLORS["warning"]
            return
        pair_id = feedback.get("pair")
        side = feedback.get("side")
        tier = feedback.get("tier")
        if pair_id and side and tier:
            pair = self._pair(str(pair_id))
            bays = pair.left_bays if side == "left" else pair.right_bays
            x1, x2 = sx(bays[0]) - 2, sx(bays[1]) + 2
            y = DECK_Y[str(tier)]
            frame[y - 8, x1 : x2 + 1] = COLORS["warning"]
            frame[y - 8 : y + 1, x1] = COLORS["warning"]
            frame[y - 8 : y + 1, x2] = COLORS["warning"]
            frame[y, x1 : x2 + 1] = COLORS["warning"]
            entity = feedback.get("entity")
            if entity:
                node = self.player if entity == "player" else self.stones[str(entity)]
                px = sx(node[0])
                frame[y - 5 : y, px - 2] = COLORS["warning"]
                frame[y - 5 : y, px + 1] = COLORS["warning"]
                frame[y - 5, px - 2 : px + 2] = COLORS["warning"]
                frame[y - 1, px - 2 : px + 2] = COLORS["warning"]

    def _sync_frame(self, override: dict[str, object] | None = None) -> None:
        override = override or {}
        frame = np.full((VIEW, VIEW), COLORS["background"], dtype=np.int8)
        if override.get("failure_flash"):
            frame[3:64, :] = COLORS["warning"]
        self._draw_static(frame)
        pulse = bool(override.get("pulse"))
        self._draw_goal(frame, pulse)
        cage_y = override.get("cage_y") or {}
        for pair in self.spec.pairs:
            self._draw_pair(frame, pair, cage_y)
        feedback = override.get("feedback")
        flash_on = bool(override.get("feedback_on"))
        self._draw_blockers(frame, feedback, flash_on)
        self._draw_entities(frame, override.get("entity_y") or {}, pulse)
        if feedback:
            self._draw_feedback(frame, feedback, flash_on)

        remaining = max(0, self.step_budget - self.steps_used)
        filled = max(0, min(56, ceil(56 * remaining / self.step_budget)))
        frame[1:3, 4:60] = COLORS["structure"]
        if filled:
            bar_color = COLORS["warning"] if remaining <= self.step_budget / 10 else COLORS["back"]
            frame[1:3, 4 : 4 + filled] = bar_color
        self.board.pixels = frame
