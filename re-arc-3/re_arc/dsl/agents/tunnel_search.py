from __future__ import annotations

from ..core import CachedProgramDslAgent

UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4


class TunnelSearchDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        level = env._game.level_index

        if level == 0:
            # D1 (62,42): up tunnel, D2 (34,62): left stub
            return [(UP, {})] * 9 + [(DOWN, {})] * 9 + [(LEFT, {})] * 13 + [(DOWN, {})] * 1

        if level == 1:
            # D1 (42,78): down tunnel, D2 (72,62): corridor, D3 (98,10): up tunnel
            return [(RIGHT, {})] * 15 + [(DOWN, {})] * 8 + [(UP, {})] * 8 + [(RIGHT, {})] * 29 + [(UP, {})] * 26

        if level == 2:
            # D1 (34,18): dead-end up-left, D2 (98,158): down-mid,
            # D4 (178,94): far right, D3 (162,34): up-right
            return (
                [(RIGHT, {})] * 12
                + [(UP, {})] * 38
                + [(DOWN, {})] * 38
                + [(RIGHT, {})] * 32
                + [(DOWN, {})] * 32
                + [(UP, {})] * 32
                + [(RIGHT, {})] * 40
                + [(LEFT, {})] * 8
                + [(UP, {})] * 30
            )

        raise RuntimeError(f"Unsupported tunnel_search level: {level}")


AGENT_CLASS = TunnelSearchDslAgent
