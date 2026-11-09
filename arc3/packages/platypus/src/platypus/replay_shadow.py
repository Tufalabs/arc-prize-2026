"""Replays a game would only repeat, answered from the record instead.

A model that RESETs a level and plays again what it already played from the
level's start learns nothing new: the game is deterministic, and the frames
are in its history. The game still charges every one of those actions to the
level. So a RESET is answered from the record -- the level's opening board --
and each action after it that follows a path already played on the level
from its start is answered with the state the game returned then; the game
is sent nothing. The first action off the recorded paths catches the game
up first: a RESET, unless the game is still at the level's start, and the
path the model has replayed since, each checked against what was served, and
then the new action, played for real. A replay the model abandons with
another RESET is never sent at all, and neither is a RESET at the level's
start or straight after another.

The paths are a tree: every attempt made on the level from its start, so a
replay of any of them is answered, not only of the last. RESET is taken to
restore the board the level opened on, the same determinism every replay
rests on and every catch-up checks. The tree lives for one level; a level
change starts an empty one.

The model sees no difference, and the move record keeps every move it made;
the game's own record keeps what the engine executed. A catch-up that does
not reproduce what was served means the game is not deterministic after a
RESET, and the model has been shown frames the game would not have produced:
the run's ``replay_shadow_mismatch`` says whether that ends the game or
switches the shadow off for it and plays on from the game's real state.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

import arcengine
import numpy as np
import taaf.game

from platypus.contract import Action

#: What the hosting's execute callback returns: the game's new state and the run's state.
Result = tuple[taaf.game.GameState, str]
Key = tuple[str, int | None, int | None]
RESET = "RESET"


class ReplayShadowMismatch(RuntimeError):
    """A catch-up the game did not reproduce: it is not deterministic after a RESET."""


@dataclass
class _Node:
    """One action on a recorded path: the action, what the game returned for it, and the actions played after it."""

    action: Action | None
    result: Result | None
    children: dict[Key, _Node] = field(default_factory=lambda: dict[Key, "_Node"]())


def _key(action: Action) -> Key:
    """An action as the paths compare it: its name and its coordinates."""
    return (action.name, action.row, action.col)


def _reset_result(opening: taaf.game.GameState) -> Result:
    """What a RESET returns on the level ``opening`` opened: its settled board
    alone, as an ordinary move's state rather than a level win's."""
    raw = copy.deepcopy(opening.raw)
    raw.frame = raw.frame[-1:]
    raw.action_input = arcengine.ActionInput(id=arcengine.GameAction.RESET)
    return dataclasses.replace(opening, raw=raw, previous_action=raw.action_input, just_won_level=False), "playing"


def _differences(served: Result, real: Result) -> list[str]:
    """How a result the game returned differs from the one served in its place; empty when identical."""
    (served_state, served_run), (real_state, real_run) = served, real
    out: list[str] = []
    if served_run != real_run:
        out.append(f"run state {served_run!r} != {real_run!r}")
    if served_state.raw.state != real_state.raw.state:
        out.append(f"game state {served_state.raw.state} != {real_state.raw.state}")
    if served_state.levels_completed != real_state.levels_completed:
        out.append(f"levels completed {served_state.levels_completed} != {real_state.levels_completed}")
    if served_state.available_actions != real_state.available_actions:
        out.append(f"available actions {served_state.available_actions} != {real_state.available_actions}")
    served_frames, real_frames = served_state.raw.frame, real_state.raw.frame
    if len(served_frames) != len(real_frames):
        out.append(f"{len(served_frames)} frames != {len(real_frames)}")
    else:
        unequal = [i for i, (a, b) in enumerate(zip(served_frames, real_frames)) if not np.array_equal(a, b)]
        if unequal:
            out.append(f"frames {unequal} differ")
    return out


class ReplayShadow:
    """The hosting's execute callback, with replays answered from the record.

    After each call, ``served`` says whether the game was spared the action
    and ``caught_up`` lists the actions sent to the game ahead of it to catch
    up, each with the differences its result showed (empty when it matched).
    """

    def __init__(
        self,
        raw_execute: Callable[[Action], Awaitable[Result]],
        mismatch: Literal["crash", "continue"],
        opening: taaf.game.GameState,
    ):
        """Wrap ``raw_execute`` for a game standing at ``opening``, a level's
        start; ``mismatch`` is what a catch-up the game does not reproduce does."""
        self._raw_execute = raw_execute
        self._mismatch = mismatch
        self.enabled = True
        self.served = False
        self.caught_up: list[tuple[Action, list[str]]] = []
        self.served_count = 0
        self.caught_up_count = 0
        self._open_level(opening)

    def _open_level(self, opening: taaf.game.GameState) -> None:
        """Start the tree of a level that opened on ``opening``, the game at its root."""
        self._level = opening.levels_completed
        # What a RESET returns, and the tree of paths from the level's start.
        self._root = _Node(action=None, result=_reset_result(opening))
        # Where the game is on the tree.
        self._live = self._root
        # The path served since a RESET the game was not sent; None while the
        # game is where the model is.
        self._shadow: list[_Node] | None = None

    async def __call__(self, action: Action) -> Result:
        """The state ``action`` leads to: from the record after a RESET while it
        replays a recorded path, else from the game."""
        self.served = False
        self.caught_up = []
        if not self.enabled:
            return await self._raw_execute(action)
        key = _key(action)
        if key[0] == RESET:
            self._shadow = []
            return self._serve(self._root)
        if self._shadow is not None:
            node = self._shadow[-1] if self._shadow else self._root
            child = node.children.get(key)
            if child is not None:
                self._shadow.append(child)
                return self._serve(child)
            await self._catch_up()
            if not self.enabled:
                return await self._raw_execute(action)
        return await self._play(action, key)

    def _serve(self, node: _Node) -> Result:
        """``node``'s recorded result, as a fresh copy: the caller may keep what it is given."""
        assert node.result is not None
        self.served = True
        self.served_count += 1
        return copy.deepcopy(node.result)

    async def _play(self, action: Action, key: Key) -> Result:
        """Send ``action`` to the game and record its result on the tree."""
        result = await self._raw_execute(action)
        state = result[0]
        if state.levels_completed != self._level:
            self._open_level(state)
            return result
        # A snapshot: the engine writes into the frame buffers it hands back.
        child = _Node(action=action, result=copy.deepcopy(result))
        self._live.children[key] = child
        self._live = child
        return result

    async def _catch_up(self) -> None:
        """Bring the game to where the model is: a RESET unless the game is at
        the level's start, then the path served since, each checked."""
        assert self._shadow is not None
        path, self._shadow = self._shadow, None
        steps: list[tuple[Action, _Node]] = [] if self._live is self._root else [(Action(RESET), self._root)]
        steps += [(node.action, node) for node in path if node.action is not None]
        for action, node in steps:
            assert node.result is not None
            real = await self._raw_execute(action)
            differences = _differences(node.result, real)
            self.caught_up.append((action, differences))
            self.caught_up_count += 1
            if differences:
                detail = f"replaying {action.name} did not reproduce what was served: " + "; ".join(differences)
                if self._mismatch == "crash":
                    raise ReplayShadowMismatch(detail)
                # Played on from the game's real state; nothing recorded is trusted again.
                self.enabled = False
                return
            self._live = node
