from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from ..core import CachedProgramDslAgent, camera_grid_to_display, observation_level_index

SocketState = int

LEVEL_SPECS = (
    {
        "start": (2, 0),
        "targets": ((4, 2),),
        "nodes": {(2, 0), (2, 2), (4, 2)},
        "sockets": ((2, 1), (3, 2)),
        "budget": 6,
    },
    {
        "start": (2, 0),
        "targets": ((2, 6),),
        "nodes": {(2, 0), (4, 0), (4, 2), (4, 4), (2, 4), (2, 2), (0, 2), (0, 4), (2, 6)},
        "sockets": ((3, 0), (4, 1), (4, 3), (3, 4), (2, 5), (2, 1), (1, 2), (0, 3)),
        "budget": 10,
    },
    {
        "start": (2, 0),
        "targets": ((2, 6), (6, 6)),
        "nodes": {(2, 0), (2, 2), (4, 2), (4, 4), (2, 4), (6, 4), (0, 2), (2, 6), (6, 6)},
        "sockets": ((2, 1), (3, 2), (4, 3), (3, 4), (2, 5), (5, 4), (6, 5), (1, 2)),
        "budget": 13,
    },
)


def _socket_neighbors(socket: tuple[int, int], state: SocketState) -> tuple[tuple[int, int], ...]:
    if state == 1:
        return ((socket[0], socket[1] - 1), (socket[0], socket[1] + 1))
    if state == 2:
        return ((socket[0] - 1, socket[1]), (socket[0] + 1, socket[1]))
    return ()


def _all_targets_hit(spec: dict[str, object], states: tuple[SocketState, ...]) -> bool:
    nodes = set(spec["nodes"])
    sockets = tuple(spec["sockets"])
    start = spec["start"]
    targets = tuple(spec["targets"])

    active_nodes = {start}
    active_sockets: set[tuple[int, int]] = set()
    frontier_nodes = {start}
    frontier_sockets: set[tuple[int, int]] = set()
    phase = "nodes"

    while True:
        if phase == "nodes":
            new_sockets: set[tuple[int, int]] = set()
            for socket, state in zip(sockets, states, strict=True):
                if socket in active_sockets or state == 0:
                    continue
                if any(node in frontier_nodes for node in _socket_neighbors(socket, state)):
                    new_sockets.add(socket)
            if not new_sockets:
                break
            active_sockets.update(new_sockets)
            frontier_sockets = new_sockets
            phase = "sockets"
        else:
            new_nodes: set[tuple[int, int]] = set()
            for socket in frontier_sockets:
                state = states[sockets.index(socket)]
                for node in _socket_neighbors(socket, state):
                    if node in nodes and node not in active_nodes:
                        new_nodes.add(node)
            if not new_nodes:
                break
            active_nodes.update(new_nodes)
            frontier_nodes = new_nodes
            phase = "nodes"

    return all(target in active_nodes for target in targets)


class DominoSocketDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "domino_socket-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _level_index(self, observation):
        return observation_level_index(observation, self.total_levels)

    def _search_states(self, spec: dict[str, object]) -> list[int]:
        sockets = tuple(spec["sockets"])
        budget = int(spec["budget"])
        start = tuple(0 for _ in sockets)
        queue = deque([(start, budget)])
        previous: dict[tuple[tuple[SocketState, ...], int], tuple[tuple[SocketState, ...], int] | None] = {
            (start, budget): None
        }
        action_taken: dict[tuple[tuple[SocketState, ...], int], int] = {}

        def _expand(
            state: tuple[SocketState, ...], remaining: int
        ) -> Iterable[tuple[int, tuple[SocketState, ...], int]]:
            if remaining <= 0:
                return ()
            out = []
            for idx in range(len(state)):
                nxt = list(state)
                nxt[idx] = (nxt[idx] + 1) % 3
                out.append((idx, tuple(nxt), remaining - 1))
            return out

        goal_key: tuple[tuple[SocketState, ...], int] | None = None
        while queue:
            state, remaining = queue.popleft()
            if remaining >= 1 and _all_targets_hit(spec, state):
                goal_key = (state, remaining)
                break
            for action_idx, next_state, next_remaining in _expand(state, remaining):
                key = (next_state, next_remaining)
                if key in previous:
                    continue
                previous[key] = (state, remaining)
                action_taken[key] = action_idx
                queue.append(key)

        if goal_key is None:
            raise RuntimeError("Domino Socket DSL could not find a winning edit plan.")

        actions: list[int] = []
        cursor = goal_key
        while previous[cursor] is not None:
            actions.append(action_taken[cursor])
            cursor = previous[cursor]
        actions.reverse()
        return actions

    def _build_level_program(self, env):
        level_idx = int(env._game.level_index)
        spec = LEVEL_SPECS[level_idx]
        socket_clicks = self._search_states(spec)
        sockets = tuple(spec["sockets"])
        program: list[tuple[int, dict[str, int]]] = []
        for idx in socket_clicks:
            row, col = sockets[idx]
            x = 4 + 7 * col + 2
            y = 8 + 7 * row + 2
            dx, dy = camera_grid_to_display(env._game.camera, x, y)
            program.append((6, {"x": int(dx), "y": int(dy)}))
        program.append((5, {}))
        return program


AGENT_CLASS = DominoSocketDslAgent
