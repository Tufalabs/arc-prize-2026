"""The arcade's game started from a level checkpoint.

A module of its own because ``taaf.game_api`` imports the arcade, some four
hundred modules, and ``platypus.checkpoints`` is imported by every game's
child process, which never builds a game.
"""

from __future__ import annotations

from dataclasses import dataclass

from taaf.game_api import GameAPI

from platypus.checkpoints import CheckpointStart


@dataclass
class CheckpointGameAPI(CheckpointStart, GameAPI):
    """A local arcade game started from a level checkpoint: the source's
    actions are replayed into the engine before play, and ``stop_at_level``
    ends it where the caller wants (``CheckpointStart``)."""


__all__ = ["CheckpointGameAPI"]
