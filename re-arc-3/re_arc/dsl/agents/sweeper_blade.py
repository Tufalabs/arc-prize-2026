from __future__ import annotations

from importlib import import_module

from ..core import CachedProgramDslAgent

_MODULE = import_module("re_arc.environment_files.sweeper_blade.0001.sweeperblade")
LEVEL_SPECS = _MODULE.LEVEL_SPECS

PROGRAMS: dict[int, list[tuple[int, dict[str, int]]]] = {
    0: [(4, {}), (4, {}), (4, {}), (4, {}), (4, {})],
    1: [(4, {}), (4, {}), (4, {}), (4, {}), (4, {}), (4, {}), (4, {})],
    2: [(1, {}), (1, {}), (1, {}), (2, {}), (2, {}), (2, {}), (2, {}), (2, {}), (2, {}), (2, {})],
}


class SweeperBladeDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str = "sweeper_blade-0001"):
        super().__init__(game_id=game_id, total_levels=len(LEVEL_SPECS))

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        del env
        if self._current_level_idx is None:
            raise RuntimeError("sweeper_blade DSL missing current level index.")
        return list(PROGRAMS[self._current_level_idx])


AGENT_CLASS = SweeperBladeDslAgent
