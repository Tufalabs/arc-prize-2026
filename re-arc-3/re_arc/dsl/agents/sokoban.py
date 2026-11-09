from __future__ import annotations

from collections import deque

from ..core import CachedProgramDslAgent

MOVE_ACTIONS: tuple[tuple[int, tuple[int, int]], ...] = ((1, (0, -1)), (2, (0, 1)), (3, (-1, 0)), (4, (1, 0)))
WAIT_ACTION_ID = 1


def _is_corner_deadlock(
    crate: tuple[int, int], walls: set[tuple[int, int]], targets: set[tuple[int, int]], width: int, height: int
) -> bool:
    if crate in targets:
        return False
    x, y = crate

    def is_wall(cell: tuple[int, int]) -> bool:
        cx, cy = cell
        if cx < 0 or cy < 0 or cx >= width or cy >= height:
            return True
        return cell in walls

    up = is_wall((x, y - 1))
    down = is_wall((x, y + 1))
    left = is_wall((x - 1, y))
    right = is_wall((x + 1, y))
    return (up and left) or (up and right) or (down and left) or (down and right)


class SokobanDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    @property
    def _agent_tag(self) -> str:
        return "sokoban"

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        layout = dict(getattr(env._game, "solver_layout", {}) or {})
        width = int(layout.get("width") or 0)
        height = int(layout.get("height") or 0)
        walls = {tuple(map(int, c)) for c in layout.get("walls", ())}
        targets = {tuple(map(int, c)) for c in layout.get("targets", ())}
        player = tuple(map(int, layout.get("player", ())))
        crates = tuple(sorted(tuple(map(int, c)) for c in layout.get("crates", ())))

        if len(player) != 2 or not crates or not targets:
            raise RuntimeError("sokoban solver received invalid level layout.")

        start_state = (player, crates)

        queue: deque[tuple[tuple[int, int], tuple[tuple[int, int], ...]]] = deque([start_state])
        previous: dict[
            tuple[tuple[int, int], tuple[tuple[int, int], ...]],
            tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None,
        ] = {start_state: None}
        previous_action: dict[tuple[tuple[int, int], tuple[tuple[int, int], ...]], int] = {}

        goal_state: tuple[tuple[int, int], tuple[tuple[int, int], ...]] | None = None

        while queue:
            state = queue.popleft()
            (px, py), crates_tuple = state
            crate_set = set(crates_tuple)

            if crate_set == targets:
                goal_state = state
                break

            for action_id, (dx, dy) in MOVE_ACTIONS:
                nx, ny = px + dx, py + dy
                next_player = (nx, ny)

                if nx < 0 or ny < 0 or nx >= width or ny >= height:
                    continue
                if next_player in walls:
                    continue

                next_crates = set(crate_set)
                if next_player in crate_set:
                    bx, by = nx + dx, ny + dy
                    beyond = (bx, by)
                    if bx < 0 or by < 0 or bx >= width or by >= height:
                        continue
                    if beyond in walls or beyond in crate_set:
                        continue
                    next_crates.remove(next_player)
                    next_crates.add(beyond)

                    if _is_corner_deadlock(beyond, walls, targets, width, height):
                        continue

                next_state = (next_player, tuple(sorted(next_crates)))
                if next_state in previous:
                    continue

                previous[next_state] = state
                previous_action[next_state] = int(action_id)
                queue.append(next_state)

        if goal_state is None:
            raise RuntimeError("sokoban solver failed to find a level program.")

        plan: list[int] = []
        cursor = goal_state
        while previous[cursor] is not None:
            plan.append(previous_action[cursor])
            cursor = previous[cursor]  # type: ignore[assignment]
        plan.reverse()

        # Sokoban applies the previous action input each frame, so append one
        # extra no-op-ish input to flush the final planned move.
        plan.append(WAIT_ACTION_ID)

        return [(int(action_id), {}) for action_id in plan]


AGENT_CLASS = SokobanDslAgent
