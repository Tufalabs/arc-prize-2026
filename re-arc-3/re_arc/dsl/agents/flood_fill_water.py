from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent

_ENV_MOD = import_module("re_arc.environment_files.flood_fill_water.0001.floodfillwater")
LEVEL_MODELS = _ENV_MOD.LEVEL_MODELS
_advance_state = _ENV_MOD._advance_state
_to_xy = _ENV_MOD._to_xy
initial_state = _ENV_MOD.initial_state


@dataclass(frozen=True)
class _Node:
    state: object
    g: int
    h: int
    parent: int
    action: int


class FloodFillWaterDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_MODELS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        model = env._game._model
        if model is None:
            raise RuntimeError("flood_fill_water DSL requires loaded level model")
        start = initial_state(model)

        def key_of(state) -> tuple:
            return (
                int(state.player),
                tuple(int(x) for x in state.crates),
                int(state.source_on),
                int(state.gate_open),
                int(state.sensor_wet),
                int(state.pending_gate_open),
                int(state.foam),
                int(state.settled),
            )

        def heuristic(state) -> int:
            px, py = _to_xy(int(state.player), int(model.width))
            best = 10**9
            for eidx in model.exit_cells:
                ex, ey = _to_xy(int(eidx), int(model.width))
                dist = abs(px - ex) + abs(py - ey)
                if dist < best:
                    best = dist
            return best if best < 10**9 else 0

        nodes: list[_Node] = [_Node(state=start, g=0, h=heuristic(start), parent=-1, action=0)]
        frontier: list[tuple[int, int, int]] = []
        heappush(frontier, (nodes[0].h, 0, 0))

        best_time_left = {key_of(start): int(start.time_left)}
        best_f_seen = nodes[0].h
        serial = 1
        expansions = 0
        goal_idx: int | None = None

        while frontier:
            _, _, node_idx = heappop(frontier)
            node = nodes[node_idx]
            state = node.state

            state_key = key_of(state)
            if int(state.time_left) < int(best_time_left.get(state_key, -1)):
                continue

            expansions += 1
            if expansions > 400000:
                break

            if node.g + node.h > best_f_seen + 48:
                continue

            for action_id in (1, 2, 3, 4, 5):
                next_state, info = _advance_state(model, state, action_id)
                if info.status == "GAME_OVER":
                    continue
                ng = node.g + 1
                nh = heuristic(next_state)

                next_key = key_of(next_state)
                remaining = int(next_state.time_left)
                if remaining <= 0 and info.status != "WIN":
                    continue
                if remaining <= int(best_time_left.get(next_key, -1)):
                    continue
                best_time_left[next_key] = remaining

                next_node_idx = len(nodes)
                nodes.append(_Node(state=next_state, g=ng, h=nh, parent=node_idx, action=action_id))

                if info.status == "WIN":
                    goal_idx = next_node_idx
                    frontier.clear()
                    break

                score = ng + nh
                if score < best_f_seen:
                    best_f_seen = score
                heappush(frontier, (score, serial, next_node_idx))
                serial += 1

        if goal_idx is None:
            raise RuntimeError("flood_fill_water DSL planner failed to find a winning plan")

        actions: list[int] = []
        cursor = goal_idx
        while cursor > 0:
            node = nodes[cursor]
            actions.append(int(node.action))
            cursor = int(node.parent)
        actions.reverse()
        return [(aid, {}) for aid in actions]


AGENT_CLASS = FloodFillWaterDslAgent
