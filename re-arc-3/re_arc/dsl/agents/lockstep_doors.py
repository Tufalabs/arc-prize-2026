from __future__ import annotations

from ..core import CachedProgramDslAgent

PROGRAMS = (
    ("X",),
    ("A", "X"),
    tuple("ABACAX"),
    tuple("ABAC AADCDAACDEC ECAABD DACACEX".replace(" ", "")),
    tuple("AAABEACCEAAE BCAACABAADDAACAECCDFDAE F EDDDFX".replace(" ", "")),
    tuple("CABB BDBBAAECCAEABD DDEFDDFDFDGE BDBBBDBBBAEAEGGX".replace(" ", "")),
    tuple("FACCFCBBEEEDFDDFAAAFAADDEFEGCFDFGEDAAFAAFAAAEFFAAAFEHGHX"),
)


class LockstepDoorsDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=7)

    def _build_level_program(self, env):
        level_index = int(env._game.level_index)
        spec = env._game.spec
        centers = {rotor.name: rotor.center for rotor in spec.rotors}
        centers["X"] = (60, 32)
        return [(6, {"x": centers[name][0], "y": centers[name][1]}) for name in PROGRAMS[level_index]]


AGENT_CLASS = LockstepDoorsDslAgent
