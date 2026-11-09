"""``AgentSolver``: generic, LLM-free hosting machinery (I1, I5, I6, I11,
I12-part). Holds no configuration — reads ``agent.run_config`` off the
template.

There is deliberately no LLM code in this module — the LLM layer
(``platypus.llm`` / ``platypus.llm_solver``) plugs into two narrow seams
that must live HERE because both process modes share this hosting path:

- ``_game_llm_backend()``: returns each game's own backend client
  (``None`` in the base class). ``LLMAgentSolver`` overrides it; the
  in-process path mounts the client on the context, the subprocess path
  ships it in the child payload.
- the token meter (I1): an opaque object threaded through the move loop
  so tokens spent since the previous executed action attach to the next
  one, and give-up/cancel tails reach ``finish_game``. The base solver
  only calls ``meter.take()`` — the metering itself is
  ``platypus.llm.TokenMeter``.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import datetime
import os
import random
import secrets
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping, cast

import taaf.game
from taaf.solver import Solver

import platypus.diagnostics
import platypus.host_monitor
import platypus.priority
import platypus.reference
import platypus.status
from platypus.agent import Agent
from platypus.attempts import Attempts
from platypus.checkpoints import CHECKPOINTS_DIR, Checkpoint, CheckpointStart, CheckpointWriter, GameCheckpoints
from platypus.context import HistoryView, MoveContext, MoveRecord, new_move_record
from platypus.contract import (
    Action,
    ContractViolationError,
    GameInfo,
    GameResult,
    GiveUp,
    Move,
    Observation,
    RunConfig,
    action_names,
    partition,
)
from platypus.diagnostics import Span, SpanLogger
from platypus.heartbeat import HeartbeatGame
from platypus.priority import PrioritySlots, StartPacer
from platypus.replay_shadow import ReplayShadow
from platypus.slots import SlotView, write_slots
from platypus.status import StatusHandle, StatusRegistry, status_monitor
from platypus.subproc import SubprocessDiedError, host_game_in_subprocess

# Status-registry key for the sacrificed liveness game. Deliberately not
# a number: hosted games are keyed by their index in the run.
_HEARTBEAT_KEY = "heartbeat"

# Opening of the give-up reason the wall budget produces. Readers match on
# it to tell "this game had a move cancelled under it" from an ordinary
# agent give-up, which is a real distinction for token accounting.
RUNTIME_LIMIT_REASON = "per-game runtime limit"


#: How a game ended by a limit on one level gives up (``_StuckOnLevel``).
STUCK_REASON = "stuck on a level"


class _StuckOnLevel(BaseException):
    """A game past a limit on one level (``RunConfig.max_attempt_repeats``,
    ``max_actions_per_level``), raised out of ``ctx.execute`` to end the move
    there. A BaseException, as a cancellation is: an agent's own error
    handling must not take it for a failed action and play on."""


def _total_level_weight(game: taaf.game.Game) -> float:
    """What a whole game is worth in ARC's own level weights.

    The score weights level ``i`` (1-indexed) by ``i``, so an n-level game
    totals ``n(n+1)/2``. TAAF leaves the count at 0 until the game starts;
    that weighs 1, which prices its tokens like any other game's — and a
    game that has not started has spent nothing to price.
    """
    levels = game.number_of_levels
    return levels * (levels + 1) / 2 or 1.0


def charged_tokens(hints: Mapping[str, Any], *, weight: float, weighting: float) -> float:
    """A game's reported spend, priced by what the level it is on is worth.

    ARC gives level ``i`` (1-indexed) the share ``i / weight`` of a game's
    score, so a token spent there buys score at that rate and is charged
    the inverse::

        level_tokens * (weight / i) ** weighting

    ``weighting`` 0 charges every token alike and the expression collapses
    to the raw spend; 1 is the full price (on a 6-level game, total weight
    21, a level-1 token is charged 21 and a level-2 token 10.5); above 1
    sharpens it. Defined once and used by both the ranking and the lines
    the scheduler prints, so what a run says it charged is what it ranked
    on.
    """
    # The reported level counts levels WON, so the level being played is
    # the next one down and its 1-indexed depth is that plus one. Off by
    # one here would divide a first-level game's tokens by zero.
    depth = int(hints.get("level", 0)) + 1
    return int(hints.get("level_tokens", 0)) * (weight / depth) ** weighting


def _planned_game_id(game: taaf.game.Game) -> str:
    """The id ``game`` plays under, before it has started.

    A TAAF game sets ``game_id`` in ``start_game``; until then it is empty.
    An engine game starts under its ``external_game_id`` when it has one and
    otherwise under its env's id, which its ``env_name`` names when it
    carries the version (every game a sweep lists does); an example game
    under its ``label``.
    """
    if game.game_id:
        return game.game_id
    for name in ("external_game_id", "env_name", "label"):
        value = getattr(game, name, None)
        if value:
            return str(value)
    return ""


def _rank_key(config: RunConfig, games: list[taaf.game.Game]) -> platypus.priority.RankKey:
    """How ``slot_policy`` ranks one game, over what it last reported.

    Evaluated at every comparison rather than folded into a periodic
    order: comparing costs nothing, only swapping does, so ranking on
    anything but the current numbers buys nothing and loses exactly the
    events that matter most — a game that has just won a level is cheap
    from its next comparison, not from the next tick.

    ``least_level_tokens`` reads the spend on the level a game stands on.
    One that has never reported is at zero and therefore first, so before
    anything has won a level the ranking is the run's own play order and
    the scheduler is a round-robin.

    What those tokens are *charged* is ``scheduler_level_weighting`` —
    see :func:`charged_tokens`. At its default of 0 the ranking is the raw
    spend.

    ``least_game_tokens`` reads the game's whole spend (``game_tokens``),
    unweighted: the least-served game first, so the machine is shared
    evenly rather than following progress. A game that has never reported
    is at zero and first, as above.
    """
    if config.slot_policy == "queue_order":
        return lambda key, hints: (int(key),)
    if config.slot_policy == "least_game_tokens":
        return lambda key, hints: (int(hints.get("game_tokens", 0)), int(key))
    by_key = {str(index): game for index, game in enumerate(games)}
    weighting = config.scheduler_level_weighting

    def charged(key: str, hints: Mapping[str, Any]) -> tuple[Any, ...]:
        """The spend on this game's current level, as the ranking sees it.

        The weight is read per comparison rather than tabulated once:
        TAAF fills a game's level count in when the game starts, and a
        table built before the fan-out would price every game at 1 for
        the whole run without ever saying so.
        """
        game = by_key.get(key)
        weight = _total_level_weight(game) if game is not None else 1.0
        return (charged_tokens(hints, weight=weight, weighting=weighting), int(key))

    return charged


def _slot_labels(games: list[taaf.game.Game]) -> dict[str, str]:
    """What the scheduler calls each game on stdout, keyed by slot key.

    The game_id, except where a run plays a game more than once — several
    passes share an id, and two lines naming the same game while meaning
    different plays is worse than a longer name. Only the ambiguous ones
    pay for it, so a single-pass run reads as plain game ids.
    """
    counts: dict[str, int] = {}
    for game in games:
        counts[game.game_id or ""] = counts.get(game.game_id or "", 0) + 1
    labels: dict[str, str] = {}
    for index, game in enumerate(games):
        game_id = game.game_id or str(index)
        run = game.game_run
        if counts.get(game.game_id or "", 0) < 2:
            labels[str(index)] = game_id
        elif run is not None:
            labels[str(index)] = f"{game_id} pass {run.pass_index}"
        else:
            labels[str(index)] = f"{game_id} #{index}"
    return labels


class _MoveExecutor:
    """The ``ctx.execute`` implementation: executes one action against the
    live game (via the process-mode-specific ``execute`` callback), records
    the resulting :class:`Move`, and stamps the move span's ``infra/action``
    event — the SAME recording path whether the agent executed in-move or
    returned the action, so history and diagnostics cannot diverge between
    the two styles. Armed only while a decision ``do_move`` runs."""

    def __init__(
        self,
        raw_execute: Callable[[Action], Awaitable[tuple[taaf.game.GameState, str]]],
        moves: MoveRecord,
        next_index: int,
        shadow: ReplayShadow | None = None,
        max_attempt_repeats: int | None = None,
        max_actions_per_level: int | None = None,
    ):
        self._raw_execute = raw_execute
        self.shadow = shadow
        self.attempts = Attempts()
        self._max_attempt_repeats = max_attempt_repeats
        self._max_actions_per_level = max_actions_per_level
        self._moves = moves
        self._next_index = next_index
        self.run_state = "playing"
        self.last_observations: tuple[Observation, ...] = ()
        self._move_span: Span | None = None
        self._available: tuple[str, ...] = ()
        self.count_this_move = 0

    def arm(self, move_span: Span, available: tuple[str, ...]) -> None:
        """Enable in-move execution for one decision call, against the
        actions the observation it starts from offers."""
        self._move_span = move_span
        self._available = available
        self.count_this_move = 0

    def disarm(self) -> None:
        """Refuse further calls: the decision call this executor served is over."""
        self._move_span = None

    async def __call__(self, action: Action) -> tuple[Observation, ...]:
        span = self._move_span
        if span is None:
            raise ContractViolationError(
                "ctx.execute is only available while a decision do_move is running "
                "(delivery-only calls and on_finish get a refusing executor)"
            )
        if self.run_state != "playing":
            raise ContractViolationError(f"ctx.execute refused: the run is {self.run_state!r}, not playing")
        if not isinstance(cast(object, action), Action):
            raise ContractViolationError(f"ctx.execute takes an Action, got {type(action).__name__}")
        # The same check the returned-action path makes, so a mistake reads
        # the same in either style: the engine would refuse this too, but
        # from inside taaf, naming action ids rather than the valid set.
        if action.name not in self._available:
            raise ContractViolationError(
                f"action {action.name!r} is not available; the observation offers {self._available}"
            )
        game_state, run_state = await (self.shadow or self._raw_execute)(action)
        self.run_state = run_state
        observations = partition(game_state, action, self._next_index)
        self._next_index += len(observations)
        self._moves.append(Move(action=action, observations=observations))
        result = observations[0]
        if self.shadow is not None:
            # Sent to the game ahead of this action, to bring it to where the
            # model already was: not moves of the model's, so events of their own.
            for caught_up, differences in self.shadow.caught_up:
                span.event(
                    "infra/catch_up",
                    name=caught_up.name,
                    row=caught_up.row,
                    col=caught_up.col,
                    matched=not differences,
                    differences="; ".join(differences),
                )
        # ``shadow`` only on a move the game was spared: a run without one
        # records exactly what it always did.
        served = {"shadow": True} if self.shadow is not None and self.shadow.served else {}
        span.event(
            "infra/action",
            name=action.name,
            row=action.row,
            col=action.col,
            outcome=result.outcome,
            level=result.level,
            levels_completed=game_state.levels_completed,
            **served,
        )
        # Per-action result frame (1-based, ordered like the infra/action
        # events) so viewers can show every intermediate frame of a
        # multi-action move; plain frame_after (last wins) is the last
        # action's immediate result for single-frame consumers. At a level
        # split that is the win frame, not the settled state the next move
        # starts from — which appears as that move's frame_at.
        span.artifact(f"frame_after_{self.count_this_move + 1}", result.frame.data, codec="png-grid")
        span.artifact("frame_after", result.frame.data, codec="png-grid")
        self.count_this_move += 1
        self._available = observations[-1].available_actions
        self.last_observations = observations
        self._check_stuck(action, result.level, won_level=result.outcome == "level_won")
        return observations

    def _check_stuck(self, action: Action, level: int, *, won_level: bool) -> None:
        """Count ``action``, played on ``level``, and end the move when the
        level is past a limit; the action itself is already recorded. An
        action that won its level ends nothing: the game plays on into the
        next level, whose count starts over."""
        repeats = self.attempts.record(action, level)
        reason = None
        if self._max_attempt_repeats is not None and repeats >= self._max_attempt_repeats:
            reason = f"the same attempt played {repeats} times on level {level + 1}"
        elif self._max_actions_per_level is not None and self.attempts.actions >= self._max_actions_per_level:
            reason = f"{self.attempts.actions} actions on level {level + 1}"
        if reason is not None and self.run_state == "playing" and not won_level:
            raise _StuckOnLevel(reason)


async def run_move_loop(
    agent: Agent,
    ctx: MoveContext,
    moves: MoveRecord,
    execute: Callable[[Action], Awaitable[tuple[taaf.game.GameState, str]]],
    initial_state: taaf.game.GameState,
    initial_run_state: str,
    number_of_levels: int | None,
    base_actions_per_level: list[int] | None,
    checkpoints: GameCheckpoints = GameCheckpoints(),
) -> GiveUp | None:
    """The §2 loop, shared verbatim by both process modes (I9: agent code
    identical): deliver an observation, then either the agent executes
    actions in-move through ``ctx.execute`` (returning ``None`` when done)
    or returns one action for the loop to execute — both via the same
    ``execute`` callback (which returns the new state and run state) and
    the same move recording. Loops until the game ends or the agent gives
    up. Returns the ``GiveUp`` if the agent gave up, else ``None``
    (terminal game state).

    Diagnostics (I7/I12): the whole loop is the ``infra/game`` span, each
    do_move is an ``infra/move`` span stamped with what happened (executed
    action(s) / gave_up / delivery), and each executed action emits an
    ``infra/action`` event on its move span. Agent spans and llm/turn
    spans nest under the move span via the ambient current span. An agent
    crash unwinds through the open move span, closing it — and the game
    span — with outcome ``error``.

    Checkpoints (I2 ``"level"``): with a writer, the agent is snapshotted
    after every move that leaves the game playing on a new level — before
    the next ``do_move``, so the snapshot is exactly what that call would
    start from. A game started from a checkpoint inherits the source's moves
    (``checkpoints.history``): its ``ctx.history`` continues them, its first
    observation is the one the source settled on, and under
    ``branch_rng="resume"`` its ``ctx.rng`` continues from the source's state
    there."""
    with ctx.diag.span("infra/game", ctx.game.game_id) as game_span:
        game_span.artifact(
            "game_metadata",
            {
                "number_of_levels": number_of_levels,
                "base_actions_per_level": base_actions_per_level,
            },
            codec="json",
        )
        give_up = await _move_loop_inner(
            agent, ctx, moves, execute, initial_state, initial_run_state, game_span, checkpoints
        )
        if give_up is not None:
            game_span.set_outcome("gave_up")
            game_span.kv(give_up_reason=give_up.reason)
        return give_up


def pace_while_busy(ctx: MoveContext, config: RunConfig, paced_gap: float) -> None:
    """Set this game's request gap for its next move from the run's slot tally.

    The backend's own gap while at least ``min_request_gap_running_fraction``
    of the slots are held, none below it. Per move: a move that starts while
    the run is busy keeps its pacing to its end. An unknown tally keeps the
    gap, so a run with no slot publisher paces as it always did.
    """
    backend = getattr(ctx, "backend", None)
    if backend is None:
        return
    running, limit = ctx.slots.running, ctx.slots.limit
    quiet = running is not None and bool(limit) and running < limit * config.min_request_gap_running_fraction
    backend.min_request_gap_s = 0.0 if quiet else paced_gap


async def _move_loop_inner(
    agent: Agent,
    ctx: MoveContext,
    moves: MoveRecord,
    execute: Callable[[Action], Awaitable[tuple[taaf.game.GameState, str]]],
    initial_state: taaf.game.GameState,
    initial_run_state: str,
    game_span: Span,
    checkpoints: GameCheckpoints,
) -> GiveUp | None:
    """The move loop proper; `run_move_loop` wraps it with the wall budget
    and the game-span bookkeeping."""
    if checkpoints.rng_state is not None:
        ctx.rng.setstate(checkpoints.rng_state)
    frame_before: Any = None  # the PREVIOUS move's input frame (None on move 0)
    for move in checkpoints.history:
        moves.append(move)
    if moves:
        # Started from a checkpoint: resume on the observation the source
        # settled on, with the frame that preceded it.
        observation = moves[-1].observation
        previous = [o for move in moves[-2:] for o in move.observations]
        frame_before = previous[-2].frame.data if len(previous) > 1 else None
    else:
        observation = Observation.from_state(
            initial_state, outcome="playing", level=initial_state.levels_completed, index=0, caused_by=None
        )
    shadow = (
        ReplayShadow(execute, agent.run_config.replay_shadow_mismatch, initial_state)
        if agent.run_config.replay_shadow
        else None
    )
    executor = _MoveExecutor(
        execute,
        moves,
        next_index=observation.index + 1,
        shadow=shadow,
        max_attempt_repeats=agent.run_config.max_attempt_repeats,
        max_actions_per_level=agent.run_config.max_actions_per_level,
    )
    executor.run_state = initial_run_state
    ctx.execute = executor
    try:
        return await _play_moves(agent, ctx, moves, executor, observation, frame_before, game_span, checkpoints)
    finally:
        if shadow is not None and (shadow.served_count or shadow.caught_up_count):
            game_span.kv(
                replay_served=shadow.served_count,
                replay_caught_up=shadow.caught_up_count,
                replay_shadow_on=shadow.enabled,
            )


async def _play_moves(
    agent: Agent,
    ctx: MoveContext,
    moves: MoveRecord,
    executor: _MoveExecutor,
    observation: Observation,
    frame_before: Any,
    game_span: Span,
    checkpoints: GameCheckpoints,
) -> GiveUp | None:
    """The moves themselves, from ``observation`` until the game ends or the agent gives up."""
    limit_s = agent.run_config.max_runtime_s_per_game
    # The budget is one thing, on or off: the deadline and the number it was
    # computed from are never meaningful apart.
    budget: _Budget | None = None if limit_s is None else _Budget(limit_s, time.monotonic() + limit_s)
    # The gap this game's backend was built with, before any move turns it off.
    paced_gap = getattr(getattr(ctx, "backend", None), "min_request_gap_s", None)
    while True:
        await asyncio.sleep(0)  # R12.02 yield floor; also where cancellation lands
        if executor.run_state != "playing":
            game_span.set_outcome(executor.run_state)
            return None
        if budget is not None and time.monotonic() >= budget.deadline:
            # Infra-imposed wall budget (RunConfig.max_runtime_s_per_game):
            # stop between moves and score as-is, like the original duck
            # harness's max_runtime_minutes.
            return budget.give_up()
        observations_this_move: list[Observation] = []
        with ctx.diag.span("infra/move", f"move {len(moves)}") as move_span:
            # The move's frame context (rendered as a row by the move
            # chunk): previous input / this input / the action's result.
            # Content-addressed storage, so repeated frames cost nothing.
            if frame_before is not None:
                move_span.artifact("frame_before", frame_before, codec="png-grid")
            move_span.artifact("frame_at", observation.frame.data, codec="png-grid")
            # Enforce the wall budget BY CANCELLING the in-flight move at the
            # deadline, not just between moves: a duck move runs unbounded LLM
            # rounds (max_llm_calls_per_move=0), so a between-moves-only check
            # lets one long move overrun the cap by minutes. wait_for cancels
            # the do_move coroutine so CancelledError propagates into its
            # innermost await (the LLM generation) and aborts it. In-move
            # ``ctx.execute`` calls are inside the same coroutine, so the cap
            # covers them too; actions executed before the cut are recorded.
            executor.arm(move_span, observation.available_actions)
            first_in_move = len(moves)
            if paced_gap is not None:
                pace_while_busy(ctx, agent.run_config, paced_gap)
            try:
                if budget is None:
                    decision = await agent.do_move(ctx, observation)
                else:
                    remaining = budget.remaining()
                    if remaining <= 0:
                        move_span.set_outcome("gave_up")
                        return budget.give_up()
                    try:
                        decision = await asyncio.wait_for(agent.do_move(ctx, observation), remaining)
                    except (asyncio.TimeoutError, TimeoutError):
                        move_span.set_outcome("gave_up")
                        return budget.give_up()
                if isinstance(decision, GiveUp):
                    move_span.set_outcome("gave_up")
                    return decision
                if decision is None:
                    if executor.count_this_move == 0:
                        raise ContractViolationError(
                            "do_move returned None without executing any action via ctx.execute"
                        )
                    move_span.set_outcome(f"executed {executor.count_this_move} in-move")
                else:
                    await executor(decision)
                    move_span.set_outcome(f"executed {decision.name}")
            except _StuckOnLevel as stuck:
                move_span.set_outcome("gave_up")
                game_span.event("infra/stuck", reason=str(stuck))
                return GiveUp(f"{STUCK_REASON}: {stuck}")
            finally:
                executor.disarm()
            for move in moves[first_in_move:]:
                observations_this_move.extend(move.observations)
        if executor.run_state != "playing":
            # Terminal mid-move: the loop-top check reports it next pass.
            game_span.set_outcome(executor.run_state)
            return None
        if decision is not None:
            # Loop-executed action: its non-settled observations still get
            # their delivery-only do_move calls (§2). In-move executed
            # actions returned ALL their observations to the agent from
            # ctx.execute — no second delivery.
            extras = executor.last_observations[:-1]
            previous = (
                observations_this_move[-len(executor.last_observations) - 1].frame.data
                if len(observations_this_move) > len(executor.last_observations)
                else observation.frame.data
            )
            give_up = await _deliver_extras(agent, ctx, extras, previous_frame=previous, budget=budget)
            if give_up is not None:
                return give_up
        # The next move's "before" is the last frame preceding its input:
        # this move's input normally, the preceding frame across a split
        # or an in-move sequence.
        settled = observations_this_move[-1]
        frame_before = (
            observations_this_move[-2].frame.data if len(observations_this_move) > 1 else observation.frame.data
        )
        if checkpoints.writer is not None and executor.run_state == "playing" and settled.level != observation.level:
            # An event on the game span, not a span of its own: taking a
            # checkpoint must leave the run's record as it leaves its play.
            if settled.caused_by is None:
                path = checkpoints.writer.write(agent, settled, moves, ctx.rng)
                game_span.event(
                    "infra/checkpoint", level=settled.level, observation_index=settled.index, path=str(path)
                )
            else:
                # The move played on past the level's first observation, so
                # no moment between moves is that level's start.
                game_span.event("infra/checkpoint", level=settled.level, skipped="the move acted past the level start")
        observation = settled


@dataclass(frozen=True)
class _Budget:
    """The per-game wall budget: the configured limit and the moment it runs
    out. One value, so no path can test the deadline while reporting some
    other number — every give-up on it reads identically in diagnostics."""

    limit_s: float
    deadline: float

    def remaining(self) -> float:
        """Seconds left; zero or less means spent."""
        return self.deadline - time.monotonic()

    def give_up(self) -> GiveUp:
        """The give-up this budget running out produces."""
        return GiveUp(f"{RUNTIME_LIMIT_REASON} ({self.limit_s:.0f}s) reached")


async def _deliver_extras(
    agent: Agent,
    ctx: MoveContext,
    extras: tuple[Observation, ...],
    *,
    previous_frame: Any = None,
    budget: "_Budget | None" = None,
) -> GiveUp | None:
    """Delivery-only do_move calls (§2): every observation reaches the
    agent through the same channel, but a level-win observation is not a
    decision point — the returned action is discarded. A GiveUp is
    honored. ``previous_frame`` (the executing move's input) seeds the
    delivery spans' "before" frame; there is no "after" — a delivery
    executes nothing.

    The wall budget covers these calls too: an agent may do real work on a
    level win (the whale relearns its model there), so a delivery left
    unbounded would be a hole in a cap the rest of the loop enforces."""
    for extra in extras:
        with ctx.diag.span("infra/move", f"deliver {extra.outcome}") as move_span:
            # Deliberately NOT marked trivial here. Whether anything happened
            # is the agent's to say: a delivery is where the whale relearns
            # its model, and a search costs no tokens for a token check to
            # notice. An agent that does nothing on delivery marks its own.
            if previous_frame is not None:
                move_span.artifact("frame_before", previous_frame, codec="png-grid")
            move_span.artifact("frame_at", extra.frame.data, codec="png-grid")
            if budget is None:
                decision = await agent.do_move(ctx, extra)
            else:
                remaining = budget.remaining()
                if remaining <= 0:
                    move_span.set_outcome("gave_up")
                    return budget.give_up()
                try:
                    decision = await asyncio.wait_for(agent.do_move(ctx, extra), remaining)
                except (asyncio.TimeoutError, TimeoutError):
                    move_span.set_outcome("gave_up")
                    return budget.give_up()
            # Stamped inside the span: a give-up honored here ends the game,
            # and I12 wants that outcome on the move span it happened in —
            # not just on the game span the reason eventually reaches.
            if isinstance(decision, GiveUp):
                move_span.set_outcome("gave_up")
                move_span.kv(give_up_reason=decision.reason)
                return decision
            move_span.set_outcome("delivery")
        previous_frame = extra.frame.data
    return None


@dataclass
class AgentSolver(Solver):
    """Drives the move loop (§2) for a plain :class:`Agent` under
    ``taaf.benchmark.Benchmark``."""

    agent: Agent = field(kw_only=True)

    # Run-level diagnostics stream (parent-written): the infra/run span
    # plus one game_hosted event per finished game — references into the
    # per-game streams, not duplicated summaries.
    _run_diag: SpanLogger = field(default_factory=SpanLogger, init=False, repr=False)
    _run_span: Span | None = field(default=None, init=False, repr=False)
    # Where this run writes its streams and blocks — the job dir, or the
    # scratch dir ``RunConfig.diagnostics_scratch`` names — and, for the
    # latter, what archives them into the job dir and which games are done.
    _diag_root: Path | None = field(default=None, init=False, repr=False)
    _archiver: platypus.diagnostics.DiagnosticsArchiver | None = field(default=None, init=False, repr=False)
    _finished_runs: list[str] = field(default_factory=lambda: list[str](), init=False, repr=False)
    # run_id -> the seed that game's RNG was actually built from, so the
    # reference can record it (see _game_seed).
    _game_seeds: dict[str, int] = field(default_factory=lambda: dict[str, int](), init=False, repr=False)

    # The sacrificed liveness game, when RunConfig.heartbeat_game is set.
    _heartbeat: HeartbeatGame | None = field(default=None, init=False, repr=False)

    # What every hosted game is doing, for the periodic status line. Owned
    # by this (parent) process; games write into it directly in-process,
    # and over the control pipe from a subprocess.
    _status: StatusRegistry = field(default_factory=StatusRegistry, init=False, repr=False)

    # The queue tally published for agents to read (``platypus.slots``).
    # Counted here rather than derived from ``_status``: the heartbeat game
    # never enters ``_fan_out``'s list, so counting the list keeps it out
    # without special-casing, and the answer stays O(1).
    _slots_waiting: int = field(default=0, init=False, repr=False)
    _slots_running: int = field(default=0, init=False, repr=False)

    # The run's concurrency, granted by rank (``platypus.priority``).
    # Built per fan-out; None outside one.
    _priority: PrioritySlots | None = field(default=None, init=False, repr=False)
    _start_pacer: StartPacer | None = field(default=None, init=False, repr=False)
    # Slot key (the game's index in the run) -> the game_id the scheduler
    # names it by on stdout. Indices are what stay unique across passes;
    # ids are what a reader recognises.
    _slot_labels: dict[str, str] = field(default_factory=lambda: dict[str, str](), init=False, repr=False)
    # The admitted set as last reported. Crossings are measured against
    # what was said, not against a recomputed "before", so the printed
    # lines compose into a complete account with no gaps between them.
    _announced: set[str] = field(default_factory=lambda: set[str](), init=False, repr=False)

    # Kaggle deploy hooks: taaf.deploy_kaggle duck-reads these attribute
    # names off the solver; the values come from the agent's RunConfig — the
    # engine it links first, then the agent's own.
    @property
    def kaggle_kernel_sources(self) -> list[str]:
        """Kaggle kernel sources, read by TAAF's deploy layer off the solver."""
        engine = self.agent.run_config.kaggle_engine
        return list(engine.kernel_sources() if engine else ()) + list(self.agent.run_config.kaggle_kernel_sources)

    @property
    def kaggle_docker_image(self) -> str | None:
        """The Kaggle image the kernel is pinned to (TAAF R2.65), read by
        TAAF's deploy layer off the solver: the engine's, which its
        wheelhouse was built for; None leaves it to Kaggle."""
        engine = self.agent.run_config.kaggle_engine
        return engine.docker_image if engine else None

    @property
    def kaggle_dataset_sources(self) -> list[str]:
        """Kaggle dataset mounts, read by TAAF's deploy layer off the solver."""
        return list(self.agent.run_config.kaggle_dataset_sources)

    @property
    def kaggle_model_sources(self) -> list[str]:
        """Kaggle model mounts, read by TAAF's deploy layer off the solver.

        Separate from the datasets because Kaggle keeps them apart: a model
        has a framework, a variation and a version in its path, and weights
        published as one cannot be attached as a dataset.
        """
        return list(self.agent.run_config.kaggle_model_sources)

    @property
    def kaggle_setup_commands(self) -> list[str]:
        """Kaggle pre-run shell commands, read by TAAF's deploy layer: the
        engine's install (and overlay) first, then the agent's own."""
        engine = self.agent.run_config.kaggle_engine
        return list(engine.setup_commands() if engine else ()) + list(self.agent.run_config.kaggle_setup_commands)

    @property
    def kaggle_exclude_source_repos(self) -> list[str]:
        """Repos left out of the source bundle, read by TAAF's deploy layer."""
        engine = self.agent.run_config.kaggle_engine
        named = list(self.agent.run_config.kaggle_exclude_source_repos) + list(
            engine.exclude_source_repos() if engine else ()
        )
        return list(dict.fromkeys(named))

    @property
    def kaggle_bundle_files(self) -> list[str]:
        """Files copied into the bundle's payload/, read by TAAF's deploy layer.

        LLM-layer hook: ``LLMAgentSolver`` adds the engine's own payload, the
        overlay it builds on the launcher; the base solver ships only what
        the run config names.
        """
        return list(self.agent.run_config.kaggle_bundle_files)

    def _setup(self) -> None:
        """Validate the run's configuration against the agent, at run start.
        Everything here is a fail-fast: a combination the agent cannot honour
        is an error now rather than a divergence discovered mid-run."""
        config = self.agent.run_config
        # Asked here, at run start, rather than read off the class: an agent's
        # snapshot cadence can depend on configuration that is only settled
        # once the agent is fully built (see Agent.supported_checkpointing).
        supported = self.agent.supported_checkpointing()
        if config.checkpointing not in supported:
            raise ValueError(
                f"checkpointing={config.checkpointing!r} not supported by "
                f"{type(self.agent).__name__} as configured (supports {sorted(supported)})"
            )
        if config.checkpointing == "move":
            raise NotImplementedError('checkpointing="move" is not implemented yet')
        if config.checkpointing == "level" and self.job_dir is None:
            raise ValueError('checkpointing="level" writes into the job dir, and this run has none')
        if config.slot_policy not in ("queue_order", "least_level_tokens", "least_game_tokens"):
            # The Literal annotation is not a runtime check and RunConfig is
            # mutable, so a typo would otherwise fall through to the ranking
            # branch — silently scheduling a run that asked not to be.
            raise ValueError(
                f"slot_policy={config.slot_policy!r} is not a policy; expected "
                '"queue_order", "least_level_tokens" or "least_game_tokens"'
            )
        if config.scheduler_level_weighting < 0:
            # A negative exponent inverts the pricing — it would rank a game
            # deep in a long game as the most expensive thing in the run.
            # Never what anyone means, and silent if it ran.
            raise ValueError(
                f"scheduler_level_weighting={config.scheduler_level_weighting} must be at least 0 "
                "(0 charges every token alike, 1 prices them by level value)"
            )
        if not self.label:
            self.label = config.label or type(self.agent).__name__

    def _status_reporters(self) -> list[Callable[[], str]]:
        """One-line pool reporters for the status tick.

        Asked of the AGENT, because which pools exist is the agent's fact:
        the whale contends for search slots, an agent with a python tool for
        kernels, an agent with neither for nothing. Optional hook — an agent
        that does not define it reports no pools.

        These run in the HOST process, which never acquires the pools itself
        (the games do, in their own subprocesses), so a reporter is expected
        to force-initialize its semaphore and read the SHARED count — see
        ullm.tools.python.pool_status.
        """
        hook: Callable[[], list[Callable[[], str]]] | None = getattr(self.agent, "status_reporters", None)
        return list(hook()) if callable(hook) else []

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        """Host every (game, pass) copy concurrently under the configured
        cap; on cancellation, finish still-playing games and re-raise
        (the taaf solver contract).

        Games are hosted as explicit tasks so that any early exit can wind
        them down before returning. ``asyncio.gather`` resolves as soon as
        the FIRST child is cancelled or raises: it neither waits for its
        siblings' cleanup nor (on a raise) cancels them. Returning there
        would leave sibling games mid-flight — subprocesses unreaped
        (I9's no-orphans guarantee) or, worse, still playing after the run
        has reported failure. So we cancel every sibling and await them
        all before propagating.
        """
        config = self.agent.run_config
        games = self._shuffled(games, config)
        self.start_heartbeat(games)
        games = self.agent_games(games)
        archiving: asyncio.Task[None] | None = None
        monitors: contextlib.AbstractAsyncContextManager[object] = contextlib.nullcontext()
        if self._diagnostics_enabled:
            assert self.job_dir is not None
            self._diag_root = self._diagnostics_root(config)
            self._run_diag = SpanLogger(
                platypus.diagnostics.run_stream_path(self._diag_root),
                platypus.diagnostics.BlockStore(platypus.diagnostics.blocks_dir(self._diag_root)),
            )
            if self._archiver is not None:
                archiving = asyncio.create_task(self._archive_periodically(config.diagnostics_archive_interval_s))
            # Wherever diagnostics are on: what they cost is two idle threads
            # and a line every ten seconds, and a blocked host loop is
            # otherwise invisible. TAAF's periodic save renders the run's page
            # on the loop, on purpose (the runs hold still only between its
            # steps); the engine generates through it on the requests it
            # holds, so that hold is expected rather than reported.
            # Bound to a name of its own: `import taaf.benchmark` would make
            # `taaf` local to this whole function.
            import taaf.benchmark as _taaf_benchmark  # noqa: PLC0415 - host-only; game children import this module

            monitors = platypus.host_monitor.host_monitors(
                self._diag_root, expected=(_taaf_benchmark.Benchmark._periodic_save_loop.__code__,)
            )
        try:
            async with monitors:
                with self._run_diag.span("infra/run", self.label) as run_span:
                    self._run_span = run_span if self._run_diag.active else None
                    with status_monitor(
                        self._status,
                        config.status_interval_s,
                        report_memory=config.report_memory,
                        reporters=self._status_reporters(),
                    ):
                        await self._fan_out(games, config)
        finally:
            # Before the diagnostics teardown: stopping finishes the
            # sacrificed game, and a game left ``playing`` here would be
            # stamped ``crashed`` by the benchmark.
            await self.stop_heartbeat()
            self._run_span = None
            self._run_diag.close()
            self._run_diag = SpanLogger()
            if archiving is not None:
                archiving.cancel()
                await asyncio.gather(archiving, return_exceptions=True)
            await self._finish_archive()

    def _game_checkpoints(self, game: taaf.game.Game, pass_index: int) -> tuple[Checkpoint | None, GameCheckpoints]:
        """The checkpoint this game starts from, if it is a ``CheckpointStart``
        (whose own start replayed and checked it), and the plumbing its move
        loop gets: where it writes its own, the moves it inherits, and the
        random state it resumes under ``branch_rng="resume"``."""
        config = self.agent.run_config
        branch = game if isinstance(game, CheckpointStart) else None
        source = Checkpoint.read(branch.checkpoint) if branch is not None else None
        writer = None
        if config.checkpointing == "level":
            assert self.job_dir is not None
            writer = CheckpointWriter(
                root=str(Path(self.job_dir) / CHECKPOINTS_DIR),
                game_id=game.game_id or "",
                pass_index=pass_index,
                # A branch's moves run from where its source's did.
                start_level=source.start_level if source is not None else game.start_at_level or 0,
                branched_from=str(source.path) if source is not None else None,
            )
        rng_state = source.rng_state() if source is not None and config.branch_rng == "resume" else None
        history = branch.history if branch is not None else ()
        return source, GameCheckpoints(writer=writer, history=history, rng_state=rng_state)

    def _diagnostics_root(self, config: RunConfig) -> Path:
        """Where this run writes its streams and blocks: the scratch dir
        ``diagnostics_scratch`` names, with an archiver to move them into the
        job dir, or the job dir itself when it names none."""
        assert self.job_dir is not None
        job_dir = Path(self.job_dir)
        scratch = os.path.expandvars(config.diagnostics_scratch)
        if not scratch or "$" in scratch:  # empty, or a variable nothing set
            return job_dir
        root = Path(scratch) / "diagnostics"
        self._finished_runs = []
        self._archiver = platypus.diagnostics.DiagnosticsArchiver(root, job_dir)
        platypus.diagnostics.register_scratch(job_dir, root)
        return root

    async def _archive_periodically(self, interval_s: float) -> None:
        """Archive the finished games' diagnostics every ``interval_s``, on a
        thread: the event loop is what serves the games."""
        assert self._archiver is not None
        archiver = self._archiver
        while True:
            await asyncio.sleep(interval_s)
            try:
                t0 = time.monotonic()
                part = await asyncio.to_thread(archiver.archive, list(self._finished_runs))
                if part is not None:
                    print(f"diagnostics: archived into {part.name} in {time.monotonic() - t0:.1f}s", flush=True)
            except Exception:
                traceback.print_exc()

    async def _finish_archive(self) -> None:
        """The run's last part — everything still on scratch — and then the
        scratch dir goes. A failure is printed rather than raised: the games
        are played, and the scratch copy stays for as long as the node keeps it."""
        archiver, self._archiver = self._archiver, None
        if archiver is None:
            return
        try:
            t0 = time.monotonic()
            part = await asyncio.to_thread(archiver.finish)
            if part is not None:
                print(f"diagnostics: archived the rest into {part.name} in {time.monotonic() - t0:.1f}s", flush=True)
        except Exception:
            traceback.print_exc()
        finally:
            assert self.job_dir is not None
            platypus.diagnostics.register_scratch(self.job_dir, None)

    def start_heartbeat(self, games: list[taaf.game.Game]) -> None:
        """Claim the first game as the liveness heartbeat and start it
        (idempotent, no-op when ``heartbeat_game`` is off).

        Called by ``_run_games``, and again — earlier — by any subclass
        with slow setup of its own to do first: a model load is silence
        the watchdog counts, so covering it is most of the point. The
        idempotence is what lets both call sites exist without either
        knowing about the other.
        """
        config = self.agent.run_config
        if not config.heartbeat_game or self._heartbeat is not None or not games:
            return
        if len(games) == 1:
            raise ValueError(
                "run_config.heartbeat_game=True sacrifices the first game, but this run has "
                "only one — the agent would be left with nothing to play"
            )
        self._heartbeat = HeartbeatGame(games[0], config.heartbeat_interval_s)
        self._heartbeat.start()
        # Reported like any other game, so the status line's game count
        # still adds up: a run of 25 that hosts 24 would otherwise read as
        # having lost one. The key is non-numeric, unlike the hosted games'
        # indices, so the two can never collide.
        self._status.register(_HEARTBEAT_KEY, games[0].game_id or "?")
        self._status.set_coarse(_HEARTBEAT_KEY, platypus.status.RUNNING)
        self._status.set_detail(_HEARTBEAT_KEY, "heartbeat")

    async def stop_heartbeat(self) -> None:
        """Finish the sacrificed game, so it lands ``gave_up`` rather than
        being stamped ``crashed`` by the benchmark teardown."""
        heartbeat, self._heartbeat = self._heartbeat, None
        if heartbeat is not None:
            await heartbeat.stop()
            self._status.set_coarse(_HEARTBEAT_KEY, platypus.status.FINISHED)

    def _shuffled(self, games: list[taaf.game.Game], config: RunConfig) -> list[taaf.game.Game]:
        """The run's entries in ``run_config``'s order — shuffled, or as
        given, then by ``play_priority`` where it names any.

        Applied to the whole (game, pass) list before anything claims an
        entry, so the heartbeat's sacrifice is drawn from the shuffle rather
        than pinned to the benchmark's first game. Copies rather than
        shuffling in place: the caller's list is TAAF's.

        The seed is resolved to a concrete integer here and printed, so a run
        that took the default (today's date) still records the order it
        played.
        """
        ordered = games
        if config.shuffle_games:
            seed = config.shuffle_seed
            if seed is None:
                today = datetime.date.today()
                seed = today.year * 10000 + today.month * 100 + today.day
            ordered = list(games)
            random.Random(seed).shuffle(ordered)
            print(f"[games] shuffled {len(ordered)} entries with seed {seed}", flush=True)
        if config.play_priority:
            priority = config.play_priority
            # Stable: games of one priority keep the order above.
            ordered = sorted(ordered, key=lambda game: -priority.get(_planned_game_id(game), 0.0))
            named = sum(1 for game in ordered if _planned_game_id(game) in priority)
            print(f"[games] {named} of {len(ordered)} entries ordered by play priority, highest first", flush=True)
        return ordered

    def agent_games(self, games: list[taaf.game.Game]) -> list[taaf.game.Game]:
        """The games the agent actually gets: everything but the heartbeat.

        Identity, not game_id — with several passes the other copies of the
        same game are still the agent's to play."""
        if self._heartbeat is None:
            return games
        return [game for game in games if game is not self._heartbeat.game]

    @property
    def _diagnostics_enabled(self) -> bool:
        """I7 gating: streams need a job_dir; ``minimal_diagnostics``
        (Kaggle submission) turns the whole facility off at emit."""
        return self.job_dir is not None and not self.minimal_diagnostics

    async def _fan_out(self, games: list[taaf.game.Game], config: RunConfig) -> None:
        """Play every (game, pass) under the concurrency cap. On any failure
        every sibling is cancelled AND awaited to the end of its cleanup —
        gather alone would return while subprocesses were still unreaped."""
        # Slot keys are game indices; a grant is reported by the runs it joins.
        run_ids = {str(i): game.game_run.run_id for i, game in enumerate(games) if game.game_run is not None}
        priority = PrioritySlots(
            config.max_concurrent,
            rank_key=_rank_key(config, games),
            on_grant=lambda key, freed: self._on_slot_granted(
                run_ids.get(key), run_ids.get(freed) if freed is not None else None
            ),
        )
        self._priority = priority
        self._start_pacer = StartPacer(config.start_gap_s)
        self._slot_labels = _slot_labels(games)
        self._announced = set()
        # Counted here rather than off the slot object: a game is waiting
        # from the moment the run owes it a slot, which is before its task
        # has run at all. Starting from the object's zeros would publish a
        # drained queue for as long as that took.
        self._slots_waiting, self._slots_running = len(games), 0
        self._publish_slots(config)
        tasks = [
            asyncio.create_task(self._host_game(game, index, config, priority), name=f"host-game-{game.game_id}")
            for index, game in enumerate(games)
        ]
        reporter = asyncio.create_task(self._report_loop(config), name="slot-report")
        holder = (
            asyncio.create_task(
                platypus.host_monitor.hold_loop(config.loop_hold_s, config.loop_hold_every_s), name="loop-hold"
            )
            if config.loop_hold_s > 0
            else None
        )
        try:
            try:
                await asyncio.gather(*tasks)
            finally:
                for helper in (reporter, holder):
                    if helper is not None:
                        helper.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await helper
        except BaseException as early_exit:
            for task in tasks:
                task.cancel()
            # return_exceptions=True: this wait must not be cut short by a
            # sibling's own failure — every task has to reach the end of its
            # cleanup. Their outcomes are already recorded on the game runs.
            await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(early_exit, asyncio.CancelledError):
                # Only cancellation finishes still-playing games; on a real
                # error they stay "playing" for Benchmark teardown to mark
                # crashed.
                for game in games:
                    if game.game_run is not None and game.game_run.final_score is None:
                        with contextlib.suppress(Exception):
                            game.finish_game()
            raise
        finally:
            # After the sibling wind-down, not before it: a game still
            # unwinding may offer its slot back on the way out, and it
            # should meet a live scheduler that declines rather than a
            # half-torn-down one.
            self._priority = None

    def _on_slot_granted(self, run_id: str | None, freed_by: str | None) -> None:
        """LLM-layer hook: a slot went to ``run_id``, taken from ``freed_by``
        (``None`` when nobody held it). ``LLMAgentSolver`` records it on the
        traces, where it says which game's calls follow which; the base
        solver has nothing to record."""
        del run_id, freed_by

    def _game_llm_backend(self, game: taaf.game.Game) -> Any | None:
        """LLM-layer hook: a fresh per-game backend client, or ``None``
        for LLM-free hosting. Overridden by ``LLMAgentSolver`` (which uses
        the game's identity to point per-game capture/replay streams);
        the base solver never touches ullm."""
        del game
        return None

    def _child_agent(self, agent: Agent) -> Agent:
        """LLM-layer hook: the agent as the CHILD should receive it (I9).

        The base agent crosses whole. ``LLMAgentSolver`` overrides it to
        drop the engine spec, which is a host-side object the child has no
        call for and cannot unpickle cheaply — the ullm-free layers here
        must not know that, which is why this is a hook rather than a
        field the payload builder edits."""
        return agent

    def _game_seed(self, game: taaf.game.Game, game_index: int) -> int:
        """The one place a per-game RNG seed is decided (I11/I13).

        Three sources, in precedence order: an explicit
        ``run_config.rng_seed_per_game`` entry for this game_id; the
        run-level ``rng_seed`` offset by the game's index; OS entropy.

        Always a concrete integer, never ``None``: a seed nobody can observe
        makes an interesting run unreplayable, and drawing one here costs
        nothing. Because the value is explicit, a run can also adopt seeds it
        did not generate — which is what lets a reference replay without
        depending on the game's position in the run's list, all that
        ``rng_seed + game_index`` ever encoded.

        ``game_index`` is the position in the list the agent is handed, so a
        heartbeat game (taken off the front) shifts every derived seed by
        one; ``rng_seed_per_game`` is what pins seeds across that flag.
        """
        config = self.agent.run_config
        override = config.rng_seed_per_game.get(game.game_id or "")
        if override is not None:
            return override
        if config.rng_seed is not None:
            return config.rng_seed + game_index
        return secrets.randbits(63)

    async def _host_game(
        self, game: taaf.game.Game, game_index: int, config: RunConfig, priority: PrioritySlots
    ) -> None:
        """One game: fresh agent copy off the template, per-game RNG and
        context, then the move loop — with agent exceptions handled per
        ``on_agent_error`` and cancellation finishing the game (I5)."""
        # Registered BEFORE the slot is taken, so a game queued behind the
        # concurrency cap is visible as queued rather than as not existing.
        # game_index is unique across the run's game list (pass copies
        # included), which game_id is not.
        status_key = str(game_index)
        self._status.register(status_key, game.game_id or "?")
        # Infra acquires, not the agent: an agent cannot forget to, and the
        # slot is held across everything hosting does — the deepcopy, the
        # backend, the child process — rather than only where agent code runs.
        await priority.acquire(status_key)
        try:
            # Holding the slot, so the pace is paid by the starting game alone.
            if self._start_pacer is not None:
                await self._start_pacer.wait()
            # Every freed slot is accounted for on stdout: under a ranking
            # that puts an unplayed game ahead of any game that has spent
            # anything, most pauses hand the slot to a game starting for
            # the first time, and without this the pauses read as slots
            # vanishing.
            print(f"{self._slot_labels.get(status_key, status_key)} has been started.", flush=True)
            self._status.set_coarse(status_key, platypus.status.RUNNING)
            self._slots_waiting -= 1
            self._slots_running += 1
            self._publish_slots(config)
            run = game.game_run
            pass_index = run.pass_index if run is not None else 0
            source, checkpoints = self._game_checkpoints(game, pass_index)
            if source is None:
                agent = copy.deepcopy(self.agent)  # per-(game, pass) copy off the template (I11)
            else:
                # The agent as the source game left it, under THIS run's
                # run-level configuration — the source's is a record of how
                # that run was hosted, not a say in how this one is.
                agent = source.load_agent()
                agent.run_config = self.agent.run_config
            seed = self._game_seed(game, game_index)
            rng = random.Random(seed)
            if run is not None:
                self._game_seeds[run.run_id] = seed
            moves = new_move_record()
            initial_state = game.current_state
            game_info = GameInfo(
                game_id=game.game_id or "",
                hint=game.hint,
                action_space=action_names(initial_state.available_actions),
                number_of_levels=game.number_of_levels,
                # TAAF's stamp, not `game_index`: the latter is a position in
                # the list AFTER shuffling and the heartbeat's steal, so it
                # says play order rather than which pass this copy is.
                pass_index=pass_index,
            )
            llm_backend = self._game_llm_backend(game)
            try:
                await self._host_game_inner(
                    game, agent, game_info, config, seed, llm_backend, rng, moves, status_key, checkpoints
                )
            finally:
                self._status.set_coarse(status_key, platypus.status.FINISHED)
                self._slots_running -= 1
                self._publish_slots(config)
                # Post-game hook (any terminal path, cancellation included):
                # the LLM solver packages per-game artifacts here.
                self._on_game_hosted(game)
        finally:
            # Retire rather than release: this game will not ask again, so
            # it leaves the population the ranking policy ranks. The slot
            # is granted under the order in force, and the rerank that
            # follows only reports the crossing and refreshes the order —
            # so a run's stdout accounts for every slot that changes
            # hands, not just the ones a tick happened to catch.
            #
            label = self._slot_labels.get(status_key, status_key)
            priority.retire(status_key)
            self._report_schedule(config, trigger=f"{label} finished", retiring=status_key)

    def _yield_handle(
        self, game: taaf.game.Game, status_key: str, config: RunConfig
    ) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
        """One game's ``ctx.yield_slot``: take up the offer, or decline it.

        The agent offers at every point it could be swapped out and says
        whether a swap is free there (``compacted``); this decides. Free
        offers are always taken — under a fixed order they cost nothing,
        because the same game is handed its slot straight back. An offer
        that is not free is taken only once the game has held the slot for
        ``slot_hold_s``, which is the fallback for a game that never
        reaches a free point at all — and which ``None`` retires, leaving
        the free points as the only place a slot changes hands.
        """
        label = self._slot_labels.get(status_key, game.game_id or status_key)
        held_since = time.monotonic()
        weight = _total_level_weight(game)

        async def yield_slot(hints: Mapping[str, Any]) -> None:
            nonlocal held_since
            priority = self._priority
            if priority is None:
                return
            priority.report(status_key, hints)
            hold_s = config.slot_hold_s
            held_long_enough = hold_s is not None and time.monotonic() - held_since >= hold_s
            if not (hints.get("compacted") or held_long_enough):
                return
            # Both numbers, always: at weighting 0 they agree and the line
            # says the pricing is off, and at any other setting the raw
            # spend alone cannot explain why this game lost its slot.
            charged = charged_tokens(hints, weight=weight, weighting=config.scheduler_level_weighting)
            spend = (
                f"at {int(hints.get('level_tokens', 0))} tokens spent in level {int(hints.get('level', 0))}"
                f", charged {charged:.0f}"
            )
            if config.slot_policy == "least_game_tokens":
                # What this policy ranks on; the level's numbers stay for
                # reading the game's progress.
                spend += f", {int(hints.get('game_tokens', 0))} in the game"
            parked = False
            # What the agent had published when it parked. Moving between
            # coarse states clears the refinement, and the agent's handle
            # cannot know that happened — it suppresses a label equal to the
            # one it last sent — so a game that resumes inside the same scope
            # would report the bare coarse state until its next label change.
            detail_at_park: str | None = None

            def park() -> None:
                # ``parked`` means the transition below HAS been applied, so
                # it is set with that state and ahead of the only line here
                # that can fail: printing into a closed stdout raises, and a
                # flag set before the state would have the undo in `finally`
                # reverse a transition that never happened — leaving the
                # tally permanently over-counting and eventually negative.
                nonlocal detail_at_park, parked
                # Its own state, not "running" (which would inflate the
                # count past the run's own cap and keep showing whatever
                # it was doing when it stopped doing anything) and not
                # "waiting for slot" (which is the games nothing has
                # started — a parked game is holding a process, a
                # conversation and a kernel that those are not).
                detail_at_park = self._status.detail(status_key)
                self._status.set_coarse(status_key, platypus.status.PAUSED)
                self._slots_running -= 1
                self._slots_waiting += 1
                self._publish_slots(config)
                parked = True
                # Printed as it happens, not once the wait is over: a line
                # that only appears on resume cannot tell you who is parked.
                print(f"{label} has been paused, {spend}.", flush=True)

            try:
                if await priority.yield_slot(status_key, hints, on_park=park):
                    print(f"{label} has been resumed, {spend}.", flush=True)
            finally:
                # Undone in a finally, and not only where the wait ended
                # well: a cancellation landing on a parked game still
                # unwinds through the hosting path that counts it running,
                # and a tally that goes negative there stays wrong for the
                # rest of the run.
                if parked:
                    self._status.set_coarse(status_key, platypus.status.RUNNING)
                    if detail_at_park is not None:
                        self._status.set_detail(status_key, detail_at_park)
                    self._slots_running += 1
                    self._slots_waiting -= 1
                    self._publish_slots(config)
                # Reset on the grant, not on the offer: the fallback exists
                # to bound how long one game holds a slot, and a game that
                # was just handed one has held it for no time at all.
                held_since = time.monotonic()

        return yield_slot

    async def _report_loop(self, config: RunConfig) -> None:
        """Say where the slots stand every ``slot_report_interval_s``.

        A heartbeat, not a decision: the ranking is live, so nothing waits
        on this loop. It exists because drift no single event caused —
        a game slowly spending its way out of the admitted set — reaches
        the log no other way.
        """
        interval = config.slot_report_interval_s
        while True:
            await asyncio.sleep(interval)
            # Named for what it is. A trigger that reads like a schedule —
            # "30 min tick" — invites the conclusion that the run reshuffles
            # every half hour, which nothing here does.
            self._report_schedule(config, trigger="periodic report")

    def _report_schedule(self, config: RunConfig, *, trigger: str, retiring: str | None = None) -> None:
        """Say who the ranking now puts in the slots, and what crossed.

        Reported as crossings rather than as an order: a run of 110 games
        reshuffles constantly a long way below the cut, and none of it
        means anything. What is worth a line is a game entering or leaving
        the set the ranking says should be running — measured against what
        was last *said*, so the log reads as a sequence of changes with no
        gaps — and ``no change`` when none did, so silence never has to be
        read as either. ``retiring`` is a game whose departure the run has
        already announced by finishing it.
        """
        priority = self._priority
        if priority is None:
            return
        admitted = priority.admitted()
        entered = [key for key in admitted if key not in self._announced]
        left = sorted(self._announced - set(admitted) - ({retiring} if retiring else set()), key=int)
        self._announced = set(admitted)
        changes: list[str] = []
        if entered:
            changes.append("in " + ", ".join(self._slot_labels.get(key, key) for key in entered))
        if left:
            changes.append("out " + ", ".join(self._slot_labels.get(key, key) for key in left))
        print(f"[slots] ({trigger}): {'; '.join(changes) if changes else 'no change'}", flush=True)
        if self._run_span is not None:
            self._run_span.event(
                "infra/schedule",
                trigger=trigger,
                policy=config.slot_policy,
                weighting=config.scheduler_level_weighting,
                active=priority.active,
                running=priority.running,
                waiting=priority.waiting,
                entered=" ".join(self._slot_labels.get(key, key) for key in entered),
                left=" ".join(self._slot_labels.get(key, key) for key in left),
                # Out of line and content-addressed: the order and the spend
                # behind it are what a replay checks the live scheduler
                # against, and a push that changed nothing stores nothing.
                state=self._run_span.store_block(
                    {
                        "order": [self._slot_labels.get(key, key) for key in priority.order()],
                        "hints": {self._slot_labels.get(key, key): priority.hints(key) for key in priority.live_keys()},
                    }
                ),
            )

    def _publish_slots(self, config: RunConfig) -> None:
        """Republish the queue tally, at every point a slot changes hands.

        Driven by the transitions rather than by the status monitor, which
        a ``status_interval_s=None`` run turns off entirely.
        """
        if self.job_dir is not None:
            write_slots(
                self.job_dir,
                waiting=self._slots_waiting,
                running=self._slots_running,
                limit=config.max_concurrent,
            )

    def _on_game_hosted(self, game: taaf.game.Game) -> None:
        """Called once per game after hosting ends (whatever the outcome).
        Base: the run stream's game_hosted event — the reference from the
        run-level stream into the game's own stream. Overrides call
        super()."""
        run = game.game_run
        if run is not None and self._archiver is not None:
            self._finished_runs.append(run.run_id)  # its stream is final: archivable
        if run is not None and self._diagnostics_enabled:
            # TAAF's central diagnostics.html renders this job_dir-relative
            # path as a plain <a href>: it resolves against the viewer
            # server (dynamic) or a static export — one link, both worlds.
            run.solver_analysis_html = f"viewer/g_{run.run_id}.html"
        if self._run_span is not None and run is not None:
            self._run_span.event(
                "game_hosted",
                run_id=run.run_id,
                game_id=game.game_id,
                state=run.state,
                levels_completed=run.levels_completed,
                final_score=run.final_score,
            )

    def _game_diag_paths(self, game: taaf.game.Game) -> tuple[str, str] | None:
        """(stream path, block-store dir) for one game's diagnostics, or
        ``None`` when diagnostics is off. The stream lives in the game-run
        dir (keyed by run_id, unique across passes) under the run's
        diagnostics root — the job dir, or its scratch dir; blocks are
        shared run-wide."""
        run = game.game_run
        if not self._diagnostics_enabled or run is None:
            return None
        assert self._diag_root is not None
        game_dir = platypus.reference.game_run_dir(self._diag_root, run.run_id)
        return (
            str(platypus.diagnostics.game_stream_path(game_dir)),
            str(platypus.diagnostics.blocks_dir(self._diag_root)),
        )

    def _status_publisher(self, status_key: str, config: RunConfig) -> Callable[[str], None] | None:
        """The callback a game's agent-side labels are published through,
        or ``None`` when reporting is off — which is what makes disabling
        it actually free, subprocess pipe traffic included."""
        if config.status_interval_s is None:
            return None
        return lambda label: self._status.set_detail(status_key, label)

    def _status_handle(self, status_key: str, config: RunConfig) -> StatusHandle:
        """In-process games write straight into the parent's registry."""
        publish = self._status_publisher(status_key, config)
        return platypus.status.NULL_STATUS if publish is None else StatusHandle(publish)

    @staticmethod
    def _diag_logger(paths: tuple[str, str] | None) -> SpanLogger:
        """A span logger over the given stream and block store, or the null
        logger when diagnostics are off."""
        if paths is None:
            return SpanLogger()
        stream_path, store_dir = paths
        return SpanLogger(stream_path, platypus.diagnostics.BlockStore(store_dir))

    async def _host_game_inner(
        self,
        game: taaf.game.Game,
        agent: Agent,
        game_info: GameInfo,
        config: RunConfig,
        seed: int,
        llm_backend: Any | None,
        rng: random.Random,
        moves: MoveRecord,
        status_key: str,
        checkpoints: GameCheckpoints = GameCheckpoints(),
    ) -> None:
        """Play one (game, pass) in this process: build the context, run the
        move loop, and report the result. The subprocess mode runs the same
        loop in a child instead (`platypus.subproc`)."""
        yield_slot = self._yield_handle(game, status_key, config)
        if config.process_mode == "subprocess":
            # Outside the handlers below: subprocess failures carry
            # their own policy handling, and the parent-side agent
            # copy must never run on_finish (the child's copy did).
            await self._host_subprocess(
                game, agent, game_info, config, seed, llm_backend, status_key, yield_slot, checkpoints
            )
            return
        initial_state = game.current_state
        diag = self._diag_logger(self._game_diag_paths(game))
        status = self._status_handle(status_key, config)
        slots = SlotView(job_dir=self.job_dir)
        if llm_backend is None:
            ctx: MoveContext = MoveContext(
                history=HistoryView(moves),
                game=game_info,
                rng=rng,
                shared={},
                diag=diag,
                status=status,
                slots=slots,
                yield_slot=yield_slot,
            )
            meter = None
        else:
            import platypus.llm

            ctx, meter = platypus.llm.build_llm_context(
                moves, game_info, rng, llm_backend, diag, status, slots, yield_slot
            )
        try:
            await self._move_loop(game, agent, ctx, moves, initial_state, meter, checkpoints)
        except asyncio.CancelledError:
            trailing = meter.take() if meter is not None else (0, 0)
            with contextlib.suppress(Exception):
                # cancelling task -> state "cancelled"; the interrupted
                # move's tokens go here (I1/I5, R11.03)
                game.finish_game(generated_tokens=trailing[0], uncached_input_tokens=trailing[1])
            await self._finish_agent(game, agent, ctx, self._result(game))
            raise
        except Exception:
            if game.game_run is not None:
                game.game_run.solver_note = f"agent error:\n{traceback.format_exc()}"
            # on_finish still runs, best-effort (I5).
            await self._finish_agent(game, agent, ctx, self._result(game, crashed=True))
            if config.on_agent_error == "raise":
                raise
            # "crash_game": leave the run "playing"; Benchmark's teardown
            # marks it crashed. Other games continue.
        finally:
            if llm_backend is not None:
                import platypus.llm

                platypus.llm.record_replay_consumption(llm_backend, diag)
            diag.close()
            if llm_backend is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(llm_backend.close())

    async def _move_loop(
        self,
        game: taaf.game.Game,
        agent: Agent,
        ctx: MoveContext,
        moves: MoveRecord,
        initial_state: taaf.game.GameState,
        meter: Any = None,
        checkpoints: GameCheckpoints = GameCheckpoints(),
    ) -> None:
        """In-process hosting: run the shared loop directly against the
        game, then finish and report. ``meter`` (a ``platypus.llm.TokenMeter``
        when the agent is LLM-backed) attributes all tokens spent since the
        previous executed action to the next one (I1)."""

        async def execute(action: Action) -> tuple[taaf.game.GameState, str]:
            generated, uncached = meter.take() if meter is not None else (0, 0)
            game_state = game.execute_action(
                action.to_action_input(), generated_tokens=generated, uncached_input_tokens=uncached
            )
            run = game.game_run
            return game_state, (run.state if run is not None else "playing")

        give_up = await run_move_loop(
            agent,
            ctx,
            moves,
            execute,
            initial_state,
            "playing",
            game.number_of_levels,
            game.base_actions_per_level,
            checkpoints,
        )
        if give_up is not None and game.game_run is not None:
            game.game_run.solver_note = f"gave up: {give_up.reason}"
        trailing = meter.take() if meter is not None else (0, 0)
        # Idempotent; a give-up/cancelled move's tokens go here (I1/I5, R11.03).
        game.finish_game(generated_tokens=trailing[0], uncached_input_tokens=trailing[1])
        await self._finish_agent(game, agent, ctx, self._result(game))

    async def _host_subprocess(
        self,
        game: taaf.game.Game,
        agent: Agent,
        game_info: GameInfo,
        config: RunConfig,
        seed: int,
        llm_backend: Any | None = None,
        status_key: str = "",
        yield_slot: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
        checkpoints: GameCheckpoints = GameCheckpoints(),
    ) -> None:
        """Subprocess hosting (I9): the child runs the agent and the same
        move loop; this side executes engine actions and applies the error
        policy. A child that dies without completing the protocol is a
        loud error — never a fallback to in-process."""
        try:
            kind, agent_error = await host_game_in_subprocess(
                game,
                self._child_agent(agent),
                game_info,
                seed,
                config,
                llm_backend,
                diag_paths=self._game_diag_paths(game),
                job_dir=str(self.job_dir) if self.job_dir is not None else None,
                on_status=self._status_publisher(status_key, config),
                on_yield=yield_slot,
                checkpoints=checkpoints,
            )
        except SubprocessDiedError:
            note = f"game subprocess died:\n{traceback.format_exc()}"
            if game.game_run is not None:
                game.game_run.solver_note = note
            if config.on_agent_error == "raise":
                raise
            # "crash_game": leave the run "playing"; Benchmark's teardown
            # marks it crashed. on_finish cannot run — the agent is gone.
            return
        if kind == "agent_error" and config.on_agent_error == "raise":
            raise RuntimeError(f"agent error in game subprocess:\n{agent_error}")

    def _result(self, game: taaf.game.Game, *, crashed: bool = False) -> GameResult:
        """The agent-facing summary handed to ``on_finish``."""
        run = game.game_run
        if run is None:
            return GameResult(state="crashed", levels_completed=0, score=None)
        run_state = run.state
        if crashed or run_state in ("not_started", "playing"):
            return GameResult(state="crashed", levels_completed=game.current_state.levels_completed, score=None)
        return GameResult(
            state=run_state,
            levels_completed=game.current_state.levels_completed,
            score=run.final_score,
        )

    async def _finish_agent(self, game: taaf.game.Game, agent: Agent, ctx: MoveContext, result: GameResult) -> None:
        """Run ``on_finish`` shielded (I5): it must complete even when the
        surrounding task is being cancelled; a failure inside it must not
        mask the game's terminal state."""
        try:
            await asyncio.shield(agent.on_finish(ctx, result))
        except asyncio.CancelledError:
            pass
        except Exception:
            if game.game_run is not None:
                game.game_run.solver_note = (
                    game.game_run.solver_note or ""
                ) + f"\non_finish failed:\n{traceback.format_exc()}"
