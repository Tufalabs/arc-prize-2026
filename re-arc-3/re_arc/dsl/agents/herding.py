from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from re_arc.dsl.core import CachedProgramDslAgent, camera_grid_to_display
from re_arc.dsl.solvers.search import beam_search

_herding_mod = import_module("re_arc.environment_files.herding.0001.herding")
HerdingState = _herding_mod.HerdingState
ModelAction = _herding_mod.ModelAction
_all_sheep_in_pen = _herding_mod._all_sheep_in_pen
step_model = _herding_mod.step_model


@dataclass(frozen=True)
class _ActionToken:
    kind: str
    dx: int = 0
    dy: int = 0
    x: int = 0
    y: int = 0


def _heuristic(static, state: HerdingState) -> float:
    if _all_sheep_in_pen(static, state):
        return 0.0

    targets = set(static.pen_floor) | set(static.doors)
    outside: list[tuple[int, int]] = []
    total_dist = 0
    for sheep in state.sheep:
        if (sheep.x, sheep.y) in static.pen_floor:
            continue
        on_open_door = any(
            (sheep.x, sheep.y) == door and int(state.door_phase[idx]) == 2 for idx, door in enumerate(static.doors)
        )
        if on_open_door:
            continue
        outside.append((sheep.x, sheep.y))
        total_dist += min(abs(sheep.x - tx) + abs(sheep.y - ty) for tx, ty in targets)

    if not outside:
        return 0.0

    px, py = state.player
    player_term = min(abs(px - sx) + abs(py - sy) for sx, sy in outside)
    return float((len(outside) * 14) + total_dist + player_term)


class HerdingDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        static = game._static
        start_state: HerdingState = game._herding_state

        click_targets = sorted(static.switches)

        action_tokens: list[_ActionToken] = [
            _ActionToken("move", dx=0, dy=-1),
            _ActionToken("move", dx=0, dy=1),
            _ActionToken("move", dx=-1, dy=0),
            _ActionToken("move", dx=1, dy=0),
            _ActionToken("whistle"),
            _ActionToken("noop"),
        ]
        action_tokens.extend(_ActionToken("click", x=x, y=y) for x, y in click_targets)

        def is_goal(state: HerdingState) -> bool:
            return _all_sheep_in_pen(static, state)

        def expand(state: HerdingState):
            if state.time_left <= 0:
                return []
            out = []
            for token in action_tokens:
                if token.kind == "move":
                    model_action = ModelAction(move=(token.dx, token.dy))
                elif token.kind == "whistle":
                    model_action = ModelAction(whistle=True)
                elif token.kind == "click":
                    model_action = ModelAction(click=(token.x, token.y))
                else:
                    model_action = ModelAction()

                stepped = step_model(static, state, model_action, game._level_seed())
                if stepped.lost:
                    continue
                out.append((token, stepped.state, 1.0))
            return out

        plan_tokens = None
        depth_cap = max(1, int(start_state.time_left))
        for width in (120, 220, 360, 520, 800, 1200):
            plan_tokens = beam_search(
                start_state, is_goal, expand, lambda state: _heuristic(static, state), width=width, max_depth=depth_cap
            )
            if plan_tokens is not None:
                break

        if plan_tokens is None:
            raise RuntimeError(f"Herding DSL failed to find a plan for level={getattr(game, '_level_index', '?') + 1}.")

        program: list[tuple[int, dict[str, int]]] = []
        for token in plan_tokens:
            if token.kind == "move":
                if (token.dx, token.dy) == (0, -1):
                    program.append((1, {}))
                elif (token.dx, token.dy) == (0, 1):
                    program.append((2, {}))
                elif (token.dx, token.dy) == (-1, 0):
                    program.append((3, {}))
                elif (token.dx, token.dy) == (1, 0):
                    program.append((4, {}))
                else:
                    raise RuntimeError("Unexpected herding move token.")
            elif token.kind == "whistle":
                program.append((5, {}))
            elif token.kind == "click":
                click_x, click_y = camera_grid_to_display(game.camera, token.x, token.y)
                program.append((6, {"x": int(click_x), "y": int(click_y)}))
            else:
                program.append((6, {"x": -1, "y": -1}))

        if not program:
            raise RuntimeError("Herding DSL produced an empty program.")
        return program


AGENT_CLASS = HerdingDslAgent
