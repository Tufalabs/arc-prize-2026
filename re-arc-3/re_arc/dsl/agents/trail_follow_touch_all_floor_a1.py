from __future__ import annotations

from arcengine import GameAction

from re_arc.dsl.core import CachedProgramDslAgent


def _decode(script: str) -> list[tuple[int, dict[str, int]]]:
    mapping = {
        "U": int(GameAction.ACTION1.value),
        "D": int(GameAction.ACTION2.value),
        "L": int(GameAction.ACTION3.value),
        "R": int(GameAction.ACTION4.value),
        "W": int(GameAction.ACTION5.value),
    }
    return [(mapping[ch], {}) for ch in script if ch in mapping]


LEVEL_SCRIPTS = [
    "RRRDLLLDRRRDLLL",
    "RRRRDLDRDLLLLUURDR",
    "LDRRURRRRURDLLLLLULLUUUURDLDDDDLLLLULD",
    "DRUULLDDLLLLLLDLURRRRRRRRDDDDDLLDRRRRRULLUU",
    "DRURUULDLULDLUDDRDLDRRRRURURRDLLLDDDURRRRURDRRRRRRRDDRUUULLLLLLL",
    "URULLLDDLDDRURDRDLLLUUUUUDRRDDRRURRUULDDDRDLLUUULLLRDRRRRRDRRRULLRURRDDDRRRURDRURDDDWUURDLULLDDLLUURLLDLDLLRRRRRRRRRDR",
]


class TrailFollowTouchAllFloorA1DslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=6)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        idx = int(getattr(game, "_current_level_index", 0))
        if idx < 0 or idx >= len(LEVEL_SCRIPTS):
            raise RuntimeError(f"Unexpected level index {idx} for {self.game_id}.")
        return _decode(LEVEL_SCRIPTS[idx])


AGENT_CLASS = TrailFollowTouchAllFloorA1DslAgent
