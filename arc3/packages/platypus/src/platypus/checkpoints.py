"""Level checkpoints (I2 ``"level"``, I4): snapshot a game's agent as each new
level starts, and start a game from such a snapshot.

A checkpoint is one directory per (game, pass, level):

    <job_dir>/checkpoints/<game_id>/p<pass>/L<level>/
        agent.pkl       the per-game agent, pickled whole (A1/A4)
        rng.json        the game's RNG state (``ctx.rng``), for a branch
                        that continues the source's draws exactly
        manifest.json   readable without unpickling: where it was taken,
                        the level's first frame and observation index, and
                        every action played since the level the game
                        started on

A checkpoint exists only to start branched play: a new game that resumes the
agent exactly where the source game stood at that level's start. It is not a
crash-recovery mechanism, so nothing of the source run — traces, diagnostics,
TAAF history — is carried; the branch is a new gameplay from its first action.

A branch game is a ``CheckpointStart`` (``platypus.checkpoint_games`` has the
arcade's). It replays the manifest's actions into a fresh engine before TAAF's
bookkeeping starts, which rebuilds both the engine's state and the moves the
agent's history held; the level's first frame and observation index are what
check that the replay arrived where the source stood.
"""

from __future__ import annotations

import json
import os
import pickle
import random
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import taaf.game

from platypus.agent import Agent
from platypus.context import MoveRecord, new_move_record
from platypus.contract import Action, Move, Observation, partition

CHECKPOINTS_DIR = "checkpoints"
AGENT_FILE = "agent.pkl"
RNG_FILE = "rng.json"
MANIFEST_FILE = "manifest.json"


def checkpoint_dir(root: Path, game_id: str, pass_index: int, level: int) -> Path:
    """Where the checkpoint of one (game, pass) at the start of ``level`` lives."""
    return root / game_id / f"p{pass_index}" / f"L{level}"


@dataclass(frozen=True)
class CheckpointWriter:
    """Writes one (game, pass)'s checkpoints. Plain values only, so it crosses
    into a game child with the rest of the payload."""

    root: str
    game_id: str
    pass_index: int
    # The level the game's moves start from: 0, or a TAAF level window's
    # first level (``start_at_level``), which a replay starts the engine on.
    start_level: int = 0
    # The checkpoint this game itself started from, if any: provenance only,
    # since the moves a branch writes already include the ones it inherited.
    branched_from: str | None = None

    def write(self, agent: Agent, observation: Observation, moves: Sequence[Move], rng: random.Random) -> Path:
        """Snapshot ``agent`` as it stands before its first move on the level
        ``observation`` starts.

        The pickle comes first: an agent holding something that cannot be
        pickled at a level boundary breaks A4, and that must fail the game
        loudly rather than leave a partial checkpoint behind. The directory
        appears whole or not at all — written under a temporary name, then
        renamed."""
        blob = pickle.dumps(agent)
        dest = checkpoint_dir(Path(self.root), self.game_id, self.pass_index, observation.level)
        manifest = {
            "game_id": self.game_id,
            "pass_index": self.pass_index,
            "level": observation.level,
            "observation_index": observation.index,
            "frame": observation.frame.data.tolist(),
            "start_level": self.start_level,
            "actions": [[move.action.name, move.action.row, move.action.col] for move in moves],
            "branched_from": self.branched_from,
            "agent_class": f"{type(agent).__module__}.{type(agent).__qualname__}",
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        staging = dest.with_name(f".{dest.name}.tmp")
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir()
        (staging / AGENT_FILE).write_bytes(blob)
        (staging / RNG_FILE).write_text(json.dumps(rng.getstate()))
        (staging / MANIFEST_FILE).write_text(json.dumps(manifest))
        os.replace(staging, dest)
        return dest


@dataclass(frozen=True)
class Checkpoint:
    """A stored checkpoint, read for starting a game from it."""

    path: Path
    manifest: dict[str, Any]

    @classmethod
    def read(cls, path: str | Path) -> Checkpoint:
        """The checkpoint at ``path``; its manifest is read, the agent is not."""
        path = Path(path)
        return cls(path=path, manifest=json.loads((path / MANIFEST_FILE).read_text()))

    @property
    def level(self) -> int:
        """The level whose start this checkpoint is."""
        return int(self.manifest["level"])

    @property
    def observation_index(self) -> int:
        """The index of the level's first observation in the source game."""
        return int(self.manifest["observation_index"])

    @property
    def start_level(self) -> int:
        """The level the source game started on, where the replay of
        ``actions`` begins."""
        return int(self.manifest["start_level"])

    @property
    def actions(self) -> tuple[Action, ...]:
        """Every action the source game played before this checkpoint, in order."""
        return tuple(Action(name, row=row, col=col) for name, row, col in self.manifest["actions"])

    def rng_state(self) -> tuple[Any, ...]:
        """The source game's RNG state at this checkpoint, in the form
        ``random.Random.setstate`` takes."""
        version, internal, gauss = json.loads((self.path / RNG_FILE).read_text())
        return (version, tuple(internal), gauss)

    def load_agent(self) -> Agent:
        """The agent as it stood; a fresh object on every call, so every pass
        branching from here gets its own."""
        agent = pickle.loads((self.path / AGENT_FILE).read_bytes())
        if not isinstance(agent, Agent):
            raise TypeError(f"{self.path / AGENT_FILE} holds a {type(agent).__name__}, not an Agent")
        return agent

    def check_replay(self, game_id: str, state: taaf.game.GameState, observation_index: int) -> None:
        """Refuse a replay that did not arrive where the source stood: on
        another level, another board, or at another observation index. The
        agent would resume against a different game than the one its history
        describes, which no later reading could detect. Replaying the same
        actions into the same engine arrives by construction, so a refusal
        means the engine is not deterministic, or the checkpoint is another
        game's."""
        if state.levels_completed != self.level:
            raise ValueError(
                f"{game_id}: replaying checkpoint {self.path} reached level index {state.levels_completed}, "
                f"but it was taken at the start of level index {self.level}"
            )
        expected = np.array(self.manifest["frame"], dtype=np.int8)
        if not np.array_equal(state.frame.data, expected):
            raise ValueError(
                f"{game_id}: replaying checkpoint {self.path} reached level index {self.level} "
                "on a different board than the source did"
            )
        if observation_index != self.observation_index:
            raise ValueError(
                f"{game_id}: replaying checkpoint {self.path} numbered the level's first observation "
                f"{observation_index}, but the source numbered it {self.observation_index}"
            )


@dataclass
class CheckpointStart(taaf.game.Game):
    """A local game started from a level checkpoint (I4), mixed in over the
    game class (``platypus.checkpoint_games.CheckpointGameAPI`` for the
    arcade).

    ``_start_game`` replays the source game's actions into a fresh engine
    before TAAF's run bookkeeping starts. That is this game's fast-forward to
    its ``start_at_level``, which the checkpoint sets (TAAF's level-window
    contract): the engine holds the state the source reached, including
    whatever carries across levels, and none of the prefix counts as this
    game's play. The replay also rebuilds the source's moves, which the
    restored agent's ``ctx.history`` continues from.
    """

    checkpoint: str = field(kw_only=True)
    # The source game's moves up to the checkpoint, as the replay rebuilt them.
    history: MoveRecord = field(default_factory=new_move_record, init=False, repr=False)

    def _start_game(self, session: taaf.game.RunSession) -> taaf.game.GameState:
        """Start the engine where the source game did, replay its actions to
        the checkpoint, and refuse a replay that arrives anywhere else."""
        checkpoint = Checkpoint.read(self.checkpoint)
        if self.start_at_level not in (None, checkpoint.level):
            raise ValueError(
                f"checkpoint {checkpoint.path} starts level index {checkpoint.level}, "
                f"not start_at_level={self.start_at_level}"
            )
        self.start_at_level = checkpoint.start_level or None
        state = super()._start_game(session)
        try:
            self.start_at_level = checkpoint.level
            index = 0
            for action in checkpoint.actions:
                state = self._replay(action, state)
                observations = partition(state, action, index + 1)
                self.history.append(Move(action=action, observations=observations))
                index = observations[-1].index
            checkpoint.check_replay(self.game_id, state, index)
        except BaseException:
            # The engine is open, and no GameRun will ever exist to finish it.
            self._finish_game()
            raise
        return taaf.game.GameState(raw=state.raw)

    def _replay(self, action: Action, before: taaf.game.GameState) -> taaf.game.GameState:
        """Execute one action of the prefix on the engine, stamped as
        ``Game.execute_action`` hands a step to its caller — whether it won a
        level, and a level win's lone frame preceded by a green one — since
        those are what ``partition`` splits observations on."""
        action_input = action.to_action_input()
        inner = self._execute_action(action_input)
        just_won_level = inner.levels_completed > before.levels_completed
        if just_won_level and not inner.won and len(inner.raw.frame) <= 1:
            width, height = self.grid_size
            inner.raw.frame = [np.full((height, width), 14, dtype=np.int8), *inner.raw.frame]
        return taaf.game.GameState(raw=inner.raw, previous_action=action_input, just_won_level=just_won_level)


@dataclass(frozen=True)
class GameCheckpoints:
    """One game's checkpoint plumbing, handed to the move loop: the writer
    (``None`` when the run takes no checkpoints), the moves a game started
    from a checkpoint inherits (its history continues them, and its first
    observation is their last), and the RNG state the game's ``ctx.rng``
    resumes from (``None``: its own seed)."""

    writer: CheckpointWriter | None = None
    history: Sequence[Move] = ()
    rng_state: tuple[Any, ...] | None = None


__all__ = [
    "AGENT_FILE",
    "CHECKPOINTS_DIR",
    "MANIFEST_FILE",
    "RNG_FILE",
    "Checkpoint",
    "CheckpointStart",
    "CheckpointWriter",
    "GameCheckpoints",
    "checkpoint_dir",
]
