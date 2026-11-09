"""Replay agent driven by a recorded human action list.

The per-game action lists in re_arc/dsl/human_replays/ hold a winning
playthrough per game, split into per-level segments. Official-game replays are
fetched by scripts/fetch_official_human_replays.py (provenance + verification);
in-house games can record one in the player (`make play` -> Record -> "Use
recording for DSL agent"), which writes the same format here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core import DslAgent

REPLAY_DIR = Path(__file__).resolve().parent.parent / "human_replays"


class HumanReplayAgent(DslAgent):
    def __init__(self, game_id: str):
        base = game_id.split("-", 1)[0].lower()
        path = REPLAY_DIR / f"{base}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        recorded_game_id = str(payload["game_id"]).strip().lower()
        if recorded_game_id != game_id.strip().lower():
            raise ValueError(f"{path} holds a replay for {recorded_game_id!r}, not {game_id!r}.")
        levels = payload["levels"]
        super().__init__(game_id=game_id, total_levels=len(levels))
        self._actions = [
            (int(action_id), {str(key): int(value) for key, value in (data or {}).items()})
            for segment in levels
            for action_id, data in segment
        ]
        self._cursor = 0

    def reset_episode(self) -> None:
        super().reset_episode()
        self._cursor = 0

    def next_action(self, _env: Any, _observation: Any) -> tuple[int, dict[str, int]]:
        if self._cursor >= len(self._actions):
            raise RuntimeError(f"{self.game_id}: human replay exhausted after {self._cursor} actions without WIN.")
        action_id, action_data = self._actions[self._cursor]
        self._cursor += 1
        return action_id, action_data
