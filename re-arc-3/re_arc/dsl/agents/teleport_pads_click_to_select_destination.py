from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent
from ..solvers.grid import grid_to_display_click

_ENV_MOD = import_module(
    "re_arc.environment_files.teleport_pads_click_to_select_destination.0001.teleportpadsclicktoselectdestination"
)

_deserialize_model = _ENV_MOD._deserialize_model
find_solution_actions = _ENV_MOD.find_solution_actions


class TeleportPadsClickToSelectDestinationDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.current_level
        model = _deserialize_model(level)
        plan = find_solution_actions(model)
        if plan is None:
            raise RuntimeError("teleport_pads_click_to_select_destination DSL could not solve the current level.")

        game = env._game
        program: list[tuple[int, dict[str, int]]] = []
        for kind, payload in plan:
            if kind == "move":
                program.append((int(payload), {}))
                continue
            if kind == "space":
                program.append((5, {}))
                continue
            if kind == "click":
                gx, gy = (int(payload[0]), int(payload[1]))
                program.append((6, grid_to_display_click(game.camera, (gx, gy))))
                continue
            raise RuntimeError(f"Unsupported plan action kind: {kind}")

        return program


AGENT_CLASS = TeleportPadsClickToSelectDestinationDslAgent
