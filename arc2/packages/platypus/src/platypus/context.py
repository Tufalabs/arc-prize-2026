"""Per-call context surface.

Not a TAAF wrapper: TAAF hands solvers whole ``Game`` objects, and the
contract deliberately narrows what agents see to services with no game
handle, so every engine interaction goes through the solver and an agent's
determinism inputs stay enumerable — the RNG and the task are the only
variation sources.
"""

from __future__ import annotations

import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, overload

import taaf.game
from platypus.contract import ContractViolationError
from platypus.diagnostics import SpanLogger
from platypus.slots import SlotView
from platypus.status import NULL_STATUS, StatusHandle

#: Which grids of a submission the agent produced itself, indexed like
#: ``Submission.attempts``: ``produced[q][a]`` for attempt ``a`` of question
#: ``q``. ``False`` marks a filler — a placeholder for a question the agent has
#: not answered, or its one grid copied into a second slot.
Produced = tuple[tuple[bool, ...], ...]


class SubmitAnswer(Protocol):
    """The shape of ``MoveContext.submit``."""

    def __call__(self, submission: taaf.game.Submission, /, *, produced: Produced | None = None) -> None:
        """Answer with ``submission``; ``produced`` says which of its grids the
        agent made itself, and ``None`` means every one of them."""
        ...


def all_produced(submission: taaf.game.Submission) -> Produced:
    """The provenance of a submission whose every grid the agent made."""
    return tuple(tuple(True for _ in group) for group in submission.attempts)


def check_produced(submission: taaf.game.Submission, produced: Produced | None) -> Produced:
    """``produced`` resolved against ``submission``, refused unless it has
    exactly one flag per grid."""
    if produced is None:
        return all_produced(submission)
    shape = tuple(len(group) for group in submission.attempts)
    given = tuple(len(flags) for flags in produced)
    if given != shape:
        raise ContractViolationError(f"produced has shape {given}; the submission's attempts have {shape}")
    return tuple(tuple(bool(flag) for flag in flags) for flags in produced)


async def no_yield(hints: Mapping[str, Any]) -> None:
    """``ctx.yield_slot`` where no host offers one: return, having done
    nothing.

    The default rather than ``None`` so agent code never guards the call.
    Yielding is an offer, and a host that cannot take it up simply
    declines — which is exactly what a run with no priority scheduler,
    and every in-process test, should do.
    """
    del hints


class SubmissionView(Sequence[taaf.game.Submission]):
    """Read-only, O(1) view over the infra-owned list of this game's
    submissions, oldest first. Reflects appends live; offers no mutation
    surface."""

    def __init__(self, submissions: list[taaf.game.Submission]):
        self._submissions = submissions

    @overload
    def __getitem__(self, index: int) -> taaf.game.Submission: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[taaf.game.Submission]: ...

    def __getitem__(self, index: int | slice) -> taaf.game.Submission | Sequence[taaf.game.Submission]:
        return self._submissions[index]

    def __len__(self) -> int:
        return len(self._submissions)


@dataclass
class MoveContext:
    """Delivered to every ``do_move`` call. Never store it on the agent.

    - ``submit``: the agent's answer channel, and the only one. Callable any
      number of times while ``do_move`` runs; each call replaces the standing
      answer, is recorded with what it cost, and reaches ``submission.json``
      before anything can kill the run. It returns nothing — the result of an
      answer is its score, and an agent that could read that would have an
      oracle to iterate against. Set by the solver; a call outside a live
      ``do_move`` is a contract violation. ``produced=`` says which grids
      the agent made itself rather than filled in (``Produced``); an agent
      that composes answers from others' (``platypus.multi_take``) reads it,
      and nothing else does.
    - ``history``: every submission this game has made, oldest first,
      infra-owned and append-only.
    - ``game_id``: the task being played.
    - ``pass_index``: which pass copy of the game this is (0-based; 0
      outside a multi-pass benchmark). The copies are still played
      independently — no copy's play may depend on what another saw or did —
      but an agent MAY read this to tell them apart, which is what lets a
      single run cover several configurations by giving each pass its own.
      It is TAAF's stamped ``GameRun.pass_index``, never the entry's
      position in the run's list: shuffling permutes that list, so a
      position encodes play order rather than pass.
    - ``rng``: the only randomness source; state owned by infra.
    - ``diag``: this game's diagnostics span stream. Agents open free-form
      spans below infra's span; the null stream (default) makes every call a
      no-op, so agent code never checks.
    - ``slots``: how many games want a slot and do not hold one — queued
      to start and parked alike (``platypus.slots``). Read it to know
      whether giving up on this game frees a slot anything can use, or
      merely idles the machine. Read on access, so it is current at the
      moment asked; the default view knows nothing and reports
      ``drained`` false.
    - ``status``: what this game is doing, for the periodic report
      (``platypus.status``). Infra owns the coarse state; an agent uses this
      to refine ``running`` into its own vocabulary, and hands it to tools
      that know about a wait infra cannot see. The null handle (default)
      makes every call a no-op.
    - ``yield_slot``: offer this game's concurrency slot back to the run
      and take it again (``platypus.priority``). Awaits until the host
      re-grants it, which is immediate unless something outranks this
      game. Call it where the game's KV cache is already lost — a
      compaction, or the boundary between two takes — so a swap there
      costs nothing; the ``hints`` mapping is what the host ranks on (the
      duck reports what it has spent and whether it just compacted). Never
      a limit and never refused: the game continues either way, and a run
      whose host has no scheduler returns from it having done nothing.
    """

    game_id: str
    history: Sequence[taaf.game.Submission]
    rng: random.Random
    submit: SubmitAnswer
    pass_index: int = 0
    diag: SpanLogger = field(default_factory=SpanLogger)
    slots: SlotView = field(default_factory=SlotView)
    status: StatusHandle = NULL_STATUS
    yield_slot: Callable[[Mapping[str, Any]], Awaitable[None]] = no_yield


__all__ = ["MoveContext", "Produced", "SubmissionView", "SubmitAnswer", "all_produced", "check_produced", "no_yield"]
