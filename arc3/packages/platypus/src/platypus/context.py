"""Per-move context surface (requirements.md §4).

Not a TAAF wrapper: TAAF hands solvers whole ``Game`` objects; the
contract deliberately narrows what agents see to services with no
game/engine handle, so every engine interaction goes through the solver
(I1) and an agent's determinism inputs stay enumerable (A2 — the RNG and
the observations are the only variation sources).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Sequence, overload

import taaf.support

from platypus.contract import Action, GameInfo, Move, Observation
from platypus.diagnostics import SpanLogger
from platypus.slots import SlotView
from platypus.status import NULL_STATUS, StatusHandle


async def no_yield(hints: Mapping[str, Any]) -> None:
    """``ctx.yield_slot`` where no host offers one: return, having done
    nothing.

    The default rather than ``None`` so agent code never guards the call.
    Yielding is an offer, and a host that cannot take it up simply
    declines — which is exactly what a run with no priority scheduler,
    and every in-process test, should do.
    """
    del hints


# The infra-owned move list (I6). Every move holds its full GameState and
# frames, and a long game reaches thousands, while agents read the newest
# one (``ctx.history[-1]``): so each is held compressed, the newest kept
# live, and an older one is rebuilt only when read.
MoveRecord = taaf.support.CompressedList[Move]


def new_move_record() -> MoveRecord:
    """An empty move list for one game."""
    return taaf.support.CompressedList[Move](keep_last=True)


class HistoryView(Sequence[Move]):
    """Read-only, O(1) view over the infra-owned move list (I6, I14).
    Reflects appends live; offers no mutation surface."""

    def __init__(self, moves: Sequence[Move]):
        self._moves = moves

    @overload
    def __getitem__(self, index: int) -> Move: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Move]: ...

    def __getitem__(self, index: int | slice) -> Move | Sequence[Move]:
        return self._moves[index]

    def __len__(self) -> int:
        return len(self._moves)


@dataclass
class MoveContext:
    """Base context, delivered to every ``do_move`` call — all agents.
    Never store it on the agent (A1).

    - ``history``: canonical append-only move record, owned by infra (I6).
    - ``game``: static game facts.
    - ``rng``: the only randomness source (A2); state owned by infra.
    - ``shared``: agent-published cross-game resources (``_host_setup`` —
      not yet wired, always empty).
    - ``diag``: this game's diagnostics span stream (I7). Agents open
      free-form spans below infra's move span; the null stream (default)
      makes every call a no-op, so agent code never checks.
    - ``slots``: how many games want a slot and do not hold one — queued
      to start and parked alike (``platypus.slots``). Read it to know
      whether giving up on this game frees a slot anything can use, or
      merely idles the machine. Read on access, so it is current at the
      moment asked; the default view knows nothing and reports
      ``drained`` false.
    - ``status``: what this game is doing, for the periodic report
      (``platypus.status``). Infra owns the coarse state; an agent uses
      this to refine ``running`` into its own vocabulary, and hands it to
      tools that know about a wait infra cannot see. The null handle
      (default) makes every call a no-op.
    - ``yield_slot``: offer this game's concurrency slot back to the run
      and take it again (``platypus.priority``). Awaits until the host
      re-grants it, which is immediate unless something outranks this
      game. Call it where the game's KV cache is already lost — a
      compaction — so a swap there costs nothing; the ``hints`` mapping
      is what the host ranks on (the duck reports its level and what it
      has spent there). Never a limit and never refused: the game
      continues either way, and a run whose host has no scheduler
      returns from it having done nothing.
    - ``execute``: in-move action execution (§2). Only callable while a
      decision ``do_move`` is running; executes the action against the
      live game immediately and returns the resulting observations (the
      same partitioning a returned action gets — a level transition
      yields two). Each call records a full move; after one or more
      calls ``do_move`` may return ``None``. Set by the move loop.
    """

    history: Sequence[Move]
    game: GameInfo
    rng: random.Random
    shared: Mapping[str, Any]
    diag: SpanLogger = field(default_factory=SpanLogger)
    slots: SlotView = field(default_factory=SlotView)
    status: StatusHandle = NULL_STATUS
    yield_slot: Callable[[Mapping[str, Any]], Awaitable[None]] = no_yield
    execute: Callable[[Action], Awaitable[tuple[Observation, ...]]] | None = None
