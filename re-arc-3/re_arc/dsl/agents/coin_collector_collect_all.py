from __future__ import annotations

from collections import deque

from arcengine import GameAction

from ..core import CachedProgramDslAgent
from ..solvers.grid import shortest_path_actions

WAIT = int(GameAction.ACTION5.value)


def _bfs_distances(
    start: tuple[int, int], *, width: int, height: int, blocked: set[tuple[int, int]]
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
    queue: deque[tuple[int, int]] = deque([start])
    dist: dict[tuple[int, int], int] = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)):
            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue
            if (nx, ny) in blocked or (nx, ny) in dist:
                continue
            dist[(nx, ny)] = dist[(x, y)] + 1
            parent[(nx, ny)] = (x, y)
            queue.append((nx, ny))

    return dist, parent


class CoinCollectorCollectAllDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = getattr(env, "_game", None)
        if game is None:
            raise RuntimeError("coin_collector_collect_all DSL expects TransitionRewardEnv with _game")

        level = getattr(game, "current_level", None)
        if level is None:
            raise RuntimeError("coin_collector_collect_all DSL missing current level")

        grid_size = tuple(level.grid_size)
        width = int(grid_size[0])
        height = int(grid_size[1])
        player_start_raw = list(level.get_data("player_start") or [0, 1])
        walls_raw = list(level.get_data("walls") or [])
        coins_raw = list(level.get_data("coins") or [])

        start = (int(player_start_raw[0]), int(player_start_raw[1]))
        walls = {
            (int(cell[0]), int(cell[1])) for cell in walls_raw if isinstance(cell, (list, tuple)) and len(cell) == 2
        }
        coins_remaining = {
            (int(cell[0]), int(cell[1])) for cell in coins_raw if isinstance(cell, (list, tuple)) and len(cell) == 2
        }

        blocked: set[tuple[int, int]] = set()
        for y in range(height):
            for x in range(width):
                if x + 1 >= width or y < 1 or y + 1 >= height:
                    blocked.add((x, y))
                    continue
                overlap_wall = False
                for dy in (0, 1):
                    for dx in (0, 1):
                        if (x + dx, y + dy) in walls:
                            overlap_wall = True
                            break
                    if overlap_wall:
                        break
                if overlap_wall:
                    blocked.add((x, y))

        if start in blocked:
            raise RuntimeError("coin_collector_collect_all DSL start is blocked")

        program: list[tuple[int, dict[str, int]]] = []
        position = start

        while coins_remaining:
            dist, _ = _bfs_distances(position, width=width, height=height, blocked=blocked)

            best_coin: tuple[int, int] | None = None
            best_goal: tuple[int, int] | None = None
            best_dist: int | None = None

            for coin in sorted(coins_remaining):
                cx, cy = coin
                candidate_goals: list[tuple[int, int]] = []
                for dy in (0, 1):
                    for dx in (0, 1):
                        gx, gy = cx - dx, cy - dy
                        if (gx, gy) in blocked:
                            continue
                        candidate_goals.append((gx, gy))
                for goal in candidate_goals:
                    if goal not in dist:
                        continue
                    d = dist[goal]
                    if best_dist is None or d < best_dist:
                        best_dist = d
                        best_coin = coin
                        best_goal = goal

            if best_coin is None or best_goal is None:
                raise RuntimeError("coin_collector_collect_all DSL could not reach all coins")

            actions = shortest_path_actions(position, best_goal, width=width, height=height, blocked=blocked)
            if actions is None:
                raise RuntimeError("coin_collector_collect_all DSL shortest_path_actions failed")

            for action_id in actions:
                program.append((int(action_id), {}))

            position = best_goal

            px, py = position
            covered = {(px + dx, py + dy) for dy in (0, 1) for dx in (0, 1)}
            coins_remaining -= covered

        # Buffer extra waits for per-level win flash transitions.
        for _ in range(12):
            program.append((WAIT, {}))

        return program


AGENT_CLASS = CoinCollectorCollectAllDslAgent
