"""Synthetic tasks for tests and demos — small, deterministic, and never
loaded from disk."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from taaf.game import ArcGame, GameSpec, Pair, RunSession, Submission
from taaf.grid import Grid


def flip_task(game_id: str, *, n_given: int = 3, n_to_answer: int = 1, seed: int = 0, size: int = 4) -> GameSpec:
    """A task whose answer is its question flipped left-right.

    The transformation the example solver applies, so a benchmark of these is
    a solver that scores 100 and a harness that can prove it.
    """
    rng = np.random.default_rng(seed)
    pairs: list[Pair] = []
    for i in range(n_given + n_to_answer):
        q = Grid(rng.integers(0, 10, size=(size, size), dtype=np.int8))
        pairs.append(Pair(question=q, answer=Grid(np.fliplr(q.data)), given=i < n_given))
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def constant_task(game_id: str, *, n_to_answer: int = 1) -> GameSpec:
    """A task no left-right flip can solve: the answer is always a single
    zero cell, whatever the question."""
    zero = Grid(np.zeros((1, 1), dtype=np.int8))
    pairs = [Pair(question=Grid(np.array([[i + 1, i + 2]], dtype=np.int8)), answer=zero, given=True) for i in range(2)]
    pairs += [
        Pair(question=Grid(np.array([[9, 9]], dtype=np.int8)), answer=zero, given=False) for _ in range(n_to_answer)
    ]
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def flip_game(game_id: str = "example_flip", **kwargs: object) -> ArcGame:
    """A playable game over :func:`flip_task`: solved exactly by the example
    solver, which is what makes it the deployment tests' fixture."""
    return ArcGame(spec_data=flip_task(game_id, **kwargs))  # type: ignore[arg-type]


def constant_game(game_id: str = "example_constant", **kwargs: object) -> ArcGame:
    """A playable game over :func:`constant_task`: answerable, never solved by
    a flip."""
    return ArcGame(spec_data=constant_task(game_id, **kwargs))  # type: ignore[arg-type]


@dataclass
class CountingGame(ArcGame):
    """An ``ArcGame`` that records how often each hook ran.

    Tests assert on the lifecycle without a shadow class or a mock: the
    counters are the only thing it adds.
    """

    starts: int = field(default=0, init=False)
    executes: int = field(default=0, init=False)
    finishes: int = field(default=0, init=False)

    def _start_game(self, session: RunSession) -> GameSpec:
        """Count the call, then defer to the base."""
        self.starts += 1
        return super()._start_game(session)

    def _execute_action(self, submission: Submission) -> None:
        """Count the call, then defer to the base."""
        self.executes += 1
        super()._execute_action(submission)

    def _finish_game(self) -> None:
        """Count the call."""
        self.finishes += 1
