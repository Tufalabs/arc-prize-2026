from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent, camera_grid_to_display

DIR_BY_ACTION = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

IDX_TO_DIR = {0: (1, 0), 1: (0, 1), 2: (-1, 0), 3: (0, -1)}

REFLECT = {(0, 0): 3, (0, 1): 1, (1, 0): 2, (1, 1): 0, (2, 0): 1, (2, 1): 3, (3, 0): 0, (3, 1): 2}


def _cell_bit(width: int, x: int, y: int) -> int:
    return 1 << (y * width + x)


def _iter_boulder_cells(top_left: tuple[int, int]):
    bx, by = top_left
    yield (bx, by)
    yield (bx + 1, by)
    yield (bx, by + 1)
    yield (bx + 1, by + 1)


class BoulderMirrorBeamComboDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level = game.current_level

        width = int(level.get_data("width"))
        height = int(level.get_data("height"))

        walls = {tuple(int(v) for v in item) for item in (level.get_data("walls") or [])}
        emitter = tuple(int(v) for v in level.get_data("emitter"))
        receiver_cells = {tuple(int(v) for v in item) for item in (level.get_data("receiver_cells") or [])}

        mirror_entries = list(level.get_data("mirrors") or [])
        mirror_cells: list[tuple[int, int]] = []
        mirror_start_mask = 0
        for idx, entry in enumerate(mirror_entries):
            mx = int(entry["x"])
            my = int(entry["y"])
            mirror_cells.append((mx, my))
            if int(entry["orientation"]) & 1:
                mirror_start_mask |= 1 << idx

        boulder_start = tuple(sorted(tuple(int(v) for v in item) for item in (level.get_data("boulders") or [])))
        player_start = tuple(int(v) for v in level.get_data("player_start"))
        time_limit = max(1, int(level.get_data("time_limit") or 1))

        solid_static = set(walls)
        solid_static.add(emitter)
        solid_static.update(receiver_cells)
        solid_static.update(mirror_cells)

        mirror_index_by_cell = {cell: idx for idx, cell in enumerate(mirror_cells)}

        def in_bounds(x: int, y: int) -> bool:
            return 0 <= x < width and 1 <= y < height

        def boulder_cells(boulders: tuple[tuple[int, int], ...]) -> set[tuple[int, int]]:
            cells: set[tuple[int, int]] = set()
            for top_left in boulders:
                cells.update(_iter_boulder_cells(top_left))
            return cells

        def trace_beam(mirror_mask: int, boulders: tuple[tuple[int, int], ...]) -> tuple[int, bool]:
            b_cells = boulder_cells(boulders)

            x, y = emitter
            direction = 0
            seen: set[tuple[int, int, int]] = set()

            beam_bits = 0
            receiver_lit = False

            while True:
                state = (x, y, direction)
                if state in seen:
                    break
                seen.add(state)

                dx, dy = IDX_TO_DIR[direction]
                nx = x + dx
                ny = y + dy
                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    break

                next_cell = (nx, ny)
                if next_cell in walls:
                    break
                if next_cell in b_cells:
                    break
                if next_cell in receiver_cells:
                    receiver_lit = True
                    break

                beam_bits |= _cell_bit(width, nx, ny)

                mirror_idx = mirror_index_by_cell.get(next_cell)
                if mirror_idx is not None:
                    orient = 1 if (mirror_mask & (1 << mirror_idx)) else 0
                    direction = REFLECT[(direction, orient)]

                x, y = nx, ny

            return beam_bits, receiver_lit

        def find_boulder_idx_at(x: int, y: int, boulders: tuple[tuple[int, int], ...]) -> int:
            for idx, top_left in enumerate(boulders):
                if (x, y) in set(_iter_boulder_cells(top_left)):
                    return idx
            return -1

        def can_place_boulder(
            new_top_left: tuple[int, int], *, moved_idx: int, boulders: tuple[tuple[int, int], ...]
        ) -> bool:
            candidate = set(_iter_boulder_cells(new_top_left))
            for cx, cy in candidate:
                if not in_bounds(cx, cy):
                    return False
                if (cx, cy) in solid_static:
                    return False

            for idx, other in enumerate(boulders):
                if idx == moved_idx:
                    continue
                if candidate.intersection(set(_iter_boulder_cells(other))):
                    return False

            return True

        beam_cache: dict[tuple[int, tuple[tuple[int, int], ...]], tuple[int, bool]] = {}

        def cached_trace(mirror_mask: int, boulders: tuple[tuple[int, int], ...]) -> tuple[int, bool]:
            key = (int(mirror_mask), boulders)
            cached = beam_cache.get(key)
            if cached is not None:
                return cached
            solved = trace_beam(mirror_mask, boulders)
            beam_cache[key] = solved
            return solved

        initial = (int(player_start[0]), int(player_start[1]), boulder_start, int(mirror_start_mask), -1, 0)

        queue = deque([(initial, 0)])
        previous: dict[tuple, tuple | None] = {initial: None}
        prev_action: dict[tuple, tuple[int, int | None]] = {}

        goal_state = None

        while queue:
            state, depth = queue.popleft()
            if depth >= time_limit:
                continue
            px, py, boulders, mirror_mask, pending_idx, has_prev_beam = state

            action_choices: list[tuple[int, int | None]] = [(1, None), (2, None), (3, None), (4, None), (5, None)]
            for dx, dy in IDX_TO_DIR.values():
                mx = px + dx
                my = py + dy
                midx = mirror_index_by_cell.get((mx, my))
                if midx is not None:
                    action_choices.append((6, midx))

            for action_id, click_idx in action_choices:
                npx = px
                npy = py
                nboulders = boulders

                if action_id in DIR_BY_ACTION:
                    dx, dy = DIR_BY_ACTION[action_id]
                    tx = px + dx
                    ty = py + dy
                    if in_bounds(tx, ty):
                        hit_idx = find_boulder_idx_at(tx, ty, boulders)
                        if hit_idx >= 0:
                            btop = boulders[hit_idx]
                            new_top = (btop[0] + dx, btop[1] + dy)
                            if can_place_boulder(new_top, moved_idx=hit_idx, boulders=boulders):
                                tmp = list(boulders)
                                tmp[hit_idx] = new_top
                                nboulders = tuple(tmp)
                                npx, npy = tx, ty
                        elif (tx, ty) not in solid_static:
                            npx, npy = tx, ty

                if has_prev_beam:
                    previous_beam_bits, _ = cached_trace(mirror_mask, boulders)
                else:
                    previous_beam_bits = 0

                nmirror_mask = mirror_mask
                if pending_idx >= 0:
                    nmirror_mask ^= 1 << pending_idx

                beam_bits, receiver_lit = cached_trace(nmirror_mask, nboulders)
                lethal_bits = beam_bits & previous_beam_bits

                if lethal_bits & _cell_bit(width, npx, npy):
                    continue

                if receiver_lit:
                    goal_state = (
                        npx,
                        npy,
                        nboulders,
                        nmirror_mask,
                        click_idx if action_id == 6 and click_idx is not None else -1,
                        1,
                    )
                    previous[goal_state] = state
                    prev_action[goal_state] = (action_id, click_idx)
                    queue.clear()
                    break

                npending = click_idx if action_id == 6 and click_idx is not None else -1
                next_state = (npx, npy, nboulders, nmirror_mask, npending, 1)

                if next_state not in previous:
                    previous[next_state] = state
                    prev_action[next_state] = (action_id, click_idx)
                    queue.append((next_state, depth + 1))

        if goal_state is None:
            raise RuntimeError("boulder_mirror_beam_combo DSL could not solve current level")

        action_plan: list[tuple[int, int | None]] = []
        cursor = goal_state
        while previous[cursor] is not None:
            action_plan.append(prev_action[cursor])
            cursor = previous[cursor]  # type: ignore[index]
        action_plan.reverse()

        program: list[tuple[int, dict[str, int]]] = []
        for action_id, click_idx in action_plan:
            if action_id != 6 or click_idx is None:
                program.append((int(action_id), {}))
                continue
            mx, my = mirror_cells[click_idx]
            dx, dy = camera_grid_to_display(game.camera, int(mx), int(my))
            program.append((6, {"x": int(dx), "y": int(dy)}))

        if not program:
            program = [(5, {})]

        return program


AGENT_CLASS = BoulderMirrorBeamComboDslAgent
