from __future__ import annotations

from arcengine import GameAction

from ..core import CachedProgramDslAgent, camera_grid_to_display

WAIT = int(GameAction.ACTION4.value)
SPACE = int(GameAction.ACTION5.value)
CLICK = int(GameAction.ACTION6.value)


class LaseretchDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level = game.current_level
        raw_solution = list(level.get_data("solution") or [])
        program: list[tuple[int, dict[str, int]]] = []

        for step in raw_solution:
            step_type = str(step.get("type", "")).strip().lower()
            if step_type == "wait":
                count = max(0, int(step.get("n", 1)))
                for _ in range(count):
                    program.append((WAIT, {}))
                continue

            if step_type == "space":
                program.append((SPACE, {}))
                continue

            if step_type in {"click_port", "click_center"}:
                pos = step.get("pos")
                if not isinstance(pos, (list, tuple)) or len(pos) != 2:
                    raise RuntimeError("laseretch DSL invalid click step payload")
                gx, gy = int(pos[0]), int(pos[1])
                dx, dy = camera_grid_to_display(game.camera, gx, gy)
                program.append((CLICK, {"x": int(dx), "y": int(dy)}))
                continue

            raise RuntimeError(f"laseretch DSL unknown step type: {step_type!r}")

        if not program:
            program = [(WAIT, {})]
        return program


AGENT_CLASS = LaseretchDslAgent
