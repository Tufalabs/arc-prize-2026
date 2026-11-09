"""``AgentSolver``: generic, LLM-free hosting machinery. Holds no
configuration — reads ``agent.run_config`` off the template.

There is deliberately no LLM code in this module — the LLM layer
(``platypus.llm`` / ``platypus.llm_solver``) plugs into two narrow seams
that must live HERE because both process modes share this hosting path:

- ``_game_llm_backend()``: returns each game's own backend client
  (``None`` in the base class). ``LLMAgentSolver`` overrides it; the
  in-process path mounts the client on the context, the subprocess path
  ships it in the child payload.
- the token meter: an opaque object threaded through hosting so tokens
  spent since the previous submission attach to the next one, and the
  tail after the last one reaches ``finish_game``. The base solver only
  calls ``meter.take()`` — the metering itself is ``platypus.llm.TokenMeter``.
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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import platypus.diagnostics
import platypus.host_monitor
import platypus.priority
import platypus.reference
import platypus.status
import taaf.game
from platypus.agent import Agent
from platypus.context import MoveContext, Produced, SubmissionView, check_produced
from platypus.contract import (
    HANDICAP_HINT,
    TAKE_BOUNDARY_HINT,
    TAKE_INDEX_HINT,
    ContractViolationError,
    GameOutcome,
    GiveUp,
    RunConfig,
)
from platypus.diagnostics import Span, SpanLogger
from platypus.priority import PrioritySlots, StartPacer
from platypus.slots import SlotView, write_slots
from platypus.status import StatusHandle, StatusRegistry, status_monitor
from platypus.subproc import SubprocessDiedError, host_game_in_subprocess
from taaf.solver import Solver

# Opening of the give-up reason the wall budget produces. Readers match on
# it to tell "this game was cut under it" from an ordinary agent give-up,
# which is a real distinction for token accounting.
RUNTIME_LIMIT_REASON = "per-game runtime limit"


class Submitter:
    """The ``ctx.submit`` implementation: validates one answer against the
    task, passes it to the live game through the process-mode-specific
    callback, and records it in diagnostics — infra's record, so it exists
    for every agent without one logging anything:

    - an ``infra/submit`` event on the move span: the tally (how many, in
      what order);
    - an ``infra/submit`` span at the ambient position in the agent's own
      tree, carrying the submitted grids: what was answered, and where in
      the agent's work it happened;
    - ``final_<q>_attempt_<a>`` artifacts on the move span, last wins: the
      standing answer, for the task row — the counterpart of the frame a
      move's action used to come back with.

    ``produced`` is checked against the submission and goes no further: TAAF
    scores grids, and the agent that composes answers from it has read it
    before it gets here.

    Armed only while ``do_move`` runs. Nothing is returned to the agent: the
    result of an answer is its score, and reading that would be an oracle.
    """

    def __init__(
        self,
        raw_submit: Callable[[taaf.game.Submission], None],
        submissions: list[taaf.game.Submission],
        n_to_answer: int,
        diag: SpanLogger,
    ):
        self._raw_submit = raw_submit
        self._submissions = submissions
        self._n_to_answer = n_to_answer
        self._diag = diag
        self._span: Span | None = None

    def arm(self, span: Span) -> None:
        """Enable submission for the one ``do_move`` call this serves."""
        self._span = span

    def disarm(self) -> None:
        """Refuse further calls: the call this submitter served is over."""
        self._span = None

    def __call__(self, submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
        span = self._span
        if span is None:
            raise ContractViolationError(
                "ctx.submit is only available while do_move is running "
                "(on_finish and anything after the return get a refusing submitter)"
            )
        # Agent code is not type-checked against this signature at runtime,
        # so the guard earns its keep even though the annotation promises it.
        if not isinstance(cast(object, submission), taaf.game.Submission):
            raise ContractViolationError(f"ctx.submit takes a Submission, got {type(submission).__name__}")
        # The same check TAAF makes, one frame earlier: a shape bug should
        # fail where the agent made it, naming what the task asked for. In
        # subprocess mode this is the only place it CAN fail loudly for the
        # agent — nothing comes back from the parent to tell it otherwise.
        if len(submission.attempts) != self._n_to_answer:
            raise ContractViolationError(
                f"submission covers {len(submission.attempts)} pair(s); this task has {self._n_to_answer} to answer"
            )
        check_produced(submission, produced)
        self._raw_submit(submission)
        self._submissions.append(submission)
        index = len(self._submissions) - 1
        span.event("infra/submit", index=index)
        # Under the ambient span, not the move span: an answer made from
        # inside an agent's own span (a tool call, a search) shows up there,
        # in sequence with the work that produced it.
        with self._diag.span("infra/submit", f"answer {index + 1}") as submit_span:
            submit_span.kv(index=index)
            for q, attempts in enumerate(submission.attempts):
                for a, grid in enumerate(attempts):
                    submit_span.artifact(f"ask_{q}_attempt_{a + 1}", grid.data, codec="png-grid")
                    span.artifact(f"final_{q}_attempt_{a + 1}", grid.data, codec="png-grid")


@dataclass
class _Budget:
    """The per-game wall budget: the configured limit and the moment it runs
    out. One value, so no path can test the deadline while reporting some
    other number — every cut on it reads identically in diagnostics.

    The clock stops while the game is parked (``pause`` / ``resume``): a
    game that has offered its slot back holds no GPU and spends nothing,
    and the net exists to catch a game spinning inside its call, not one
    waiting for the run to hand it a slot again.
    """

    limit_s: float
    deadline: float
    paused_at: float | None = None

    def remaining(self) -> float:
        """Seconds left; zero or less means spent. Frozen while paused."""
        now = time.monotonic() if self.paused_at is None else self.paused_at
        return self.deadline - now

    def pause(self) -> None:
        """Stop the clock, at this moment."""
        self.paused_at = time.monotonic()

    def resume(self) -> None:
        """Start the clock again, pushing the deadline out by the pause."""
        if self.paused_at is not None:
            self.deadline += time.monotonic() - self.paused_at
            self.paused_at = None

    def give_up(self) -> GiveUp:
        """The give-up this budget running out produces."""
        return GiveUp(f"{RUNTIME_LIMIT_REASON} ({self.limit_s:.0f}s) reached")


async def run_game(
    agent: Agent,
    ctx: MoveContext,
    view: taaf.game.GameView,
    submitter: Submitter,
    limit_s: float | None,
) -> GiveUp | None:
    """Play one game: a single ``do_move`` under the wall budget, shared
    verbatim by both process modes so agent code is identical.

    Returns the ``GiveUp`` if the agent gave up or the budget ran out, else
    ``None``.

    Diagnostics: the whole game is an ``infra/game`` span, the call is an
    ``infra/move`` span stamped with what happened, and each answer emits an
    ``infra/submit`` event on it. Agent spans and llm/turn spans nest under
    the move span via the ambient current span. An agent crash unwinds
    through the open spans, closing them with outcome ``error``.
    """
    with ctx.diag.span("infra/game", ctx.game_id) as game_span:
        game_span.artifact(
            "task_metadata",
            {"n_given": len(view.given), "n_to_answer": len(view.questions)},
            codec="json",
        )
        give_up = await _play(agent, ctx, view, submitter, limit_s, game_span)
        if give_up is not None:
            game_span.set_outcome("gave_up")
            game_span.kv(give_up_reason=give_up.reason)
        else:
            game_span.set_outcome(f"submitted {len(ctx.history)}")
        return give_up


def pace_while_busy(ctx: MoveContext, config: RunConfig, paced_gap: float) -> None:
    """Set this game's request gap from the run's slot tally.

    The backend's own gap while at least ``min_request_gap_running_fraction``
    of the slots are held, none below it. Decided as the game's call starts,
    and again before each LLM call by an agent with a per-round hook (the
    duck's), since an ARC2 game is one call that can outlive the busy part
    of a run. An unknown tally keeps the gap, so a run with no slot publisher
    paces as it always did.
    """
    backend = getattr(ctx, "backend", None)
    if backend is None:
        return
    running, limit = ctx.slots.running, ctx.slots.limit
    quiet = running is not None and bool(limit) and running < limit * config.min_request_gap_running_fraction
    backend.min_request_gap_s = 0.0 if quiet else paced_gap


async def _play(
    agent: Agent,
    ctx: MoveContext,
    view: taaf.game.GameView,
    submitter: Submitter,
    limit_s: float | None,
    game_span: Span,
) -> GiveUp | None:
    """The call proper; ``run_game`` wraps it with the game-span bookkeeping."""
    del game_span  # closed by the caller; here only to mark the nesting
    # The budget is one thing, on or off: the deadline and the number it was
    # computed from are never meaningful apart.
    budget = None if limit_s is None else _Budget(limit_s, time.monotonic() + limit_s)
    with ctx.diag.span("infra/move", "play") as move_span:
        for i, (question, answer) in enumerate(view.given):
            move_span.artifact(f"given_{i}_question", question.data, codec="png-grid")
            move_span.artifact(f"given_{i}_answer", answer.data, codec="png-grid")
        for i, question in enumerate(view.questions):
            move_span.artifact(f"ask_{i}", question.data, codec="png-grid")
        # Answers submitted before a budget cut are already recorded and
        # scored; ``_under_budget`` is where the cut happens.
        submitter.arm(move_span)
        # The gap this game's backend was built with.
        paced_gap = getattr(getattr(ctx, "backend", None), "min_request_gap_s", None)
        if paced_gap is not None:
            pace_while_busy(ctx, agent.run_config, paced_gap)
        try:
            if budget is None:
                decision = await agent.do_move(ctx, view)
            else:
                try:
                    decision = await _under_budget(agent, ctx, view, budget)
                except TimeoutError:
                    move_span.set_outcome("gave_up")
                    return budget.give_up()
            if isinstance(decision, GiveUp):
                move_span.set_outcome("gave_up")
                move_span.kv(give_up_reason=decision.reason)
                return decision
            move_span.set_outcome(f"submitted {len(ctx.history)}")
            return None
        finally:
            submitter.disarm()


async def _under_budget(agent: Agent, ctx: MoveContext, view: taaf.game.GameView, budget: _Budget) -> GiveUp | None:
    """``do_move`` cut at the budget's deadline, with the clock stopped
    while the game is parked.

    The budget is enforced BY CANCELLING the call: a game is one unbounded
    call, so there is no between-calls check that could catch an overrun.
    ``asyncio.timeout`` cancels the coroutine at its innermost await and
    surfaces the cut as ``TimeoutError``. A park (``ctx.yield_slot``) is
    time the game holds no slot and spends nothing, so the budget pauses
    across it and, on the re-grant, the deadline moves out by exactly the
    parked interval. Wrapped here rather than on the host so both process
    modes measure the park in the process that keeps the deadline. The
    timer is disarmed for the park, not merely re-aimed after it: in
    process the loop stays live while the game waits, and a deadline that
    fell inside the park would otherwise cut a game for time that does
    not count (in a child the offer blocks the whole loop, and disarming
    is a no-op).
    """
    loop = asyncio.get_running_loop()
    offer = ctx.yield_slot
    async with asyncio.timeout(budget.remaining()) as clock:

        async def parked(hints: Mapping[str, Any]) -> None:
            if clock.expired():
                # The cut is already in flight; it lands at this await.
                await offer(hints)
                return
            clock.reschedule(None)
            budget.pause()
            try:
                await offer(hints)
            finally:
                budget.resume()
                clock.reschedule(loop.time() + budget.remaining())

        ctx.yield_slot = parked
        try:
            return await agent.do_move(ctx, view)
        finally:
            ctx.yield_slot = offer


def _rank_key(config: RunConfig, games: list[taaf.game.Game]) -> platypus.priority.RankKey:
    """How ``slot_policy`` ranks one game, over what it last reported.

    Evaluated at every comparison rather than folded into a periodic
    order: comparing costs nothing, only swapping does, so ranking on
    anything but the current numbers buys nothing and loses exactly the
    events that matter most — a game that has just become cheap is cheap
    from its next comparison, not from the next tick.

    ``least_tokens`` reads what a game has spent so far. One that has
    never reported is at zero and therefore first, so before anything has
    spent, the ranking is the run's own play order and the scheduler is a
    round-robin. Every task here is one level worth the same score, so a
    token is a token: there is nothing to price it by — except a take after
    a game's first, which ranks ``take_handicap_tokens`` further back for
    each take before it, so first answers come before second ones, and a
    game that asks to rank further back still (``HANDICAP_HINT``: the duck's
    audit handicap, once its audit has compacted). The handicaps are added
    here rather than to the ``tokens`` a game reports, which stay its true
    spend: the schedule report records the hints the ranking read, and an
    audit reads them.
    """
    del games
    if config.slot_policy == "queue_order":
        return lambda key, hints: (int(key),)
    handicap = config.take_handicap_tokens

    def spent(key: str, hints: Mapping[str, Any]) -> tuple[Any, ...]:
        """The spend on this game, as the ranking sees it."""
        return (
            int(hints.get("tokens", 0))
            + handicap * int(hints.get(TAKE_INDEX_HINT, 0))
            + int(hints.get(HANDICAP_HINT, 0)),
            int(key),
        )

    return spent


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


@dataclass
class AgentSolver(Solver):
    """Hosts an :class:`Agent` for every game of a ``taaf.benchmark.Benchmark``."""

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

    # What every hosted game is doing, for the periodic status line. Owned
    # by this (parent) process; games write into it directly in-process,
    # and over the control pipe from a subprocess.
    _status: StatusRegistry = field(default_factory=StatusRegistry, init=False, repr=False)

    # The queue tally published for agents to read (``platypus.slots``).
    _slots_waiting: int = field(default=0, init=False, repr=False)
    _slots_running: int = field(default=0, init=False, repr=False)
    # The run's slots, for as long as a fan-out is live (``_fan_out``); a
    # game offering its slot back outside one meets ``None`` and is declined.
    _priority: PrioritySlots | None = field(default=None, init=False, repr=False)
    _start_pacer: StartPacer | None = field(default=None, init=False, repr=False)
    # What the scheduler calls each game on stdout, by slot key.
    _slot_labels: dict[str, str] = field(default_factory=lambda: dict[str, str](), init=False, repr=False)
    # The admitted set as it was last reported, so the next report can say
    # what crossed rather than restate the order.
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
        """The Kaggle image the kernel is pinned to, read by TAAF's deploy
        layer off the solver: the engine's, which its wheelhouse was built
        for; None leaves it to Kaggle."""
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
        """Validate the run's configuration and name the run from the agent,
        at run start. A fail-fast: a combination the run cannot honour is an
        error now rather than a divergence discovered mid-run."""
        config = self.agent.run_config
        if config.slot_policy not in ("queue_order", "least_tokens"):
            # The Literal annotation is not a runtime check and RunConfig is
            # mutable, so a typo would otherwise fall through to the ranking
            # branch — silently scheduling a run that asked not to be.
            raise ValueError(
                f'slot_policy={config.slot_policy!r} is not a policy; expected "queue_order" or "least_tokens"'
            )
        if not self.label:
            self.label = config.label or type(self.agent).__name__

    def _status_reporters(self) -> list[Callable[[], str]]:
        """One-line pool reporters for the status tick.

        Asked of the AGENT, because which pools exist is the agent's fact: an
        agent with a python tool contends for kernels, an agent with neither
        for nothing. Optional hook — an agent that does not define it reports
        no pools.

        These run in the HOST process, which never acquires the pools itself
        (the games do, in their own subprocesses), so a reporter is expected
        to force-initialize its semaphore and read the SHARED count.
        """
        hook: Callable[[], list[Callable[[], str]]] | None = getattr(self.agent, "status_reporters", None)
        return list(hook()) if callable(hook) else []

    async def _run_games(self, games: list[taaf.game.Game]) -> None:
        """Host every (game, pass) copy concurrently under the configured
        cap; on cancellation, finish still-playing games and re-raise
        (the taaf solver contract)."""
        config = self.agent.run_config
        games = self._shuffled(games, config)
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
            import taaf.benchmark as _taaf_benchmark

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
            self._run_span = None
            self._run_diag.close()
            self._run_diag = SpanLogger()
            if archiving is not None:
                archiving.cancel()
                await asyncio.gather(archiving, return_exceptions=True)
            await self._finish_archive()

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

    def _shuffled(self, games: list[taaf.game.Game], config: RunConfig) -> list[taaf.game.Game]:
        """The run's entries in ``run_config``'s order — shuffled, or as
        given, then by ``play_priority`` where it names any.

        Copies rather than shuffling in place: the caller's list is TAAF's.
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
            ordered = sorted(ordered, key=lambda game: -priority.get(game.game_id or "", 0.0))
            named = sum(1 for game in ordered if (game.game_id or "") in priority)
            print(f"[games] {named} of {len(ordered)} entries ordered by play priority, highest first", flush=True)
        return ordered

    @property
    def _diagnostics_enabled(self) -> bool:
        """Streams need a job_dir; ``minimal_diagnostics`` (Kaggle
        submission) turns the whole facility off at emit."""
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
                    if game.game_run is not None and not game.game_run.finished:
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
        """LLM-layer hook: the agent as the CHILD should receive it.

        The base agent crosses whole. ``LLMAgentSolver`` overrides it to
        drop the engine spec, which is a host-side object the child has no
        call for and cannot unpickle cheaply — the ullm-free layers here
        must not know that, which is why this is a hook rather than a
        field the payload builder edits."""
        return agent

    def _game_seed(self, game: taaf.game.Game, game_index: int) -> int:
        """The one place a per-game RNG seed is decided.

        Three sources, in precedence order: an explicit
        ``run_config.rng_seed_per_game`` entry for this game_id; the
        run-level ``rng_seed`` offset by the game's index; OS entropy.

        Always a concrete integer, never ``None``: a seed nobody can observe
        makes an interesting run unreplayable, and drawing one here costs
        nothing. Because the value is explicit, a run can also adopt seeds it
        did not generate — which is what lets a reference replay without
        depending on the game's position in the run's list.
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
        context, then the play call — with agent exceptions handled per
        ``on_agent_error`` and cancellation finishing the game."""
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
            agent = copy.deepcopy(self.agent)  # per-(game, pass) copy off the template
            seed = self._game_seed(game, game_index)
            rng = random.Random(seed)
            run = game.game_run
            if run is not None:
                self._game_seeds[run.run_id] = seed
            llm_backend = self._game_llm_backend(game)
            try:
                await self._host_game_inner(game, agent, config, seed, llm_backend, rng, status_key)
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
            label = self._slot_labels.get(status_key, status_key)
            priority.retire(status_key)
            self._report_schedule(config, trigger=f"{label} finished", retiring=status_key)

    def _yield_handle(
        self, game: taaf.game.Game, status_key: str, config: RunConfig
    ) -> Callable[[Mapping[str, Any]], Awaitable[None]]:
        """One game's ``ctx.yield_slot``: take up the offer, or decline it.

        The agent offers at every point it could be swapped out and says
        whether a swap is free there (``compacted``, or a boundary between
        takes); this decides. Free offers are always taken — under a fixed
        order they cost nothing, because the same game is handed its slot
        straight back. An offer that is not free is taken only once the
        game has held the slot for ``slot_hold_s``, which is the fallback
        for a game that never reaches a free point at all — and which
        ``None`` retires, leaving the free points as the only place a slot
        changes hands.
        """
        label = self._slot_labels.get(status_key, game.game_id or status_key)
        held_since = time.monotonic()

        async def yield_slot(hints: Mapping[str, Any]) -> None:
            nonlocal held_since
            priority = self._priority
            if priority is None:
                return
            priority.report(status_key, hints)
            hold_s = config.slot_hold_s
            held_long_enough = hold_s is not None and time.monotonic() - held_since >= hold_s
            if not (hints.get("compacted") or hints.get(TAKE_BOUNDARY_HINT) or held_long_enough):
                return
            spend = f"at {int(hints.get('tokens', 0))} tokens spent"
            # Said the same way on pause and on resume, and never as "before"
            # a take: a game still inside take 1 is before take 2 as well.
            if TAKE_INDEX_HINT in hints:
                take_index = int(hints[TAKE_INDEX_HINT])
                if hints.get(TAKE_BOUNDARY_HINT):
                    spend += f", take {take_index} finished, take {take_index + 1} not yet started"
                else:
                    spend += f", in take {take_index + 1}"
            if hints.get(HANDICAP_HINT):
                spend += f", ranked {int(hints[HANDICAP_HINT]):,} tokens back"
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

        Reported as crossings rather than as an order: a run of 240 games
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
                n_submitted=len(run.history),
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
        config: RunConfig,
        seed: int,
        llm_backend: Any | None,
        rng: random.Random,
        status_key: str,
    ) -> None:
        """Play one (game, pass) in this process: build the context, run the
        call, and report the result. The subprocess mode runs the same call
        in a child instead (``platypus.subproc``)."""
        yield_slot = self._yield_handle(game, status_key, config)
        if config.process_mode == "subprocess":
            # Outside the handlers below: subprocess failures carry
            # their own policy handling, and the parent-side agent
            # copy must never run on_finish (the child's copy did).
            await self._host_subprocess(game, agent, config, seed, llm_backend, status_key, yield_slot)
            return
        diag = self._diag_logger(self._game_diag_paths(game))
        status = self._status_handle(status_key, config)
        slots = SlotView(job_dir=self.job_dir)
        view = game.query()
        submissions: list[taaf.game.Submission] = []
        meter: Any = None

        def raw_submit(submission: taaf.game.Submission) -> None:
            """Charge the tokens spent since the last answer to this one."""
            generated, uncached = meter.take() if meter is not None else (0, 0)
            game.execute_action(submission, generated_tokens=generated, uncached_input_tokens=uncached)

        submitter = Submitter(raw_submit, submissions, len(view.questions), diag)
        # TAAF's stamp, not `game_index`: the latter is a position in the
        # list AFTER shuffling, so it says play order rather than which pass
        # this copy is.
        pass_index = game.game_run.pass_index if game.game_run is not None else 0
        if llm_backend is None:
            ctx = MoveContext(
                game_id=game.game_id,
                history=SubmissionView(submissions),
                pass_index=pass_index,
                rng=rng,
                submit=submitter,
                diag=diag,
                status=status,
                slots=slots,
                yield_slot=yield_slot,
            )
        else:
            import platypus.llm

            ctx, meter = platypus.llm.build_llm_context(
                game.game_id,
                submissions,
                rng,
                llm_backend,
                submitter,
                diag,
                status,
                slots,
                yield_slot,
                pass_index=pass_index,
            )
        try:
            await self._play_and_finish(game, agent, ctx, view, submitter, config, meter)
        except asyncio.CancelledError:
            trailing = meter.take() if meter is not None else (0, 0)
            with contextlib.suppress(Exception):
                # cancelling task -> state "cancelled"; the tokens spent
                # since the last answer go here
                game.finish_game(generated_tokens=trailing[0], uncached_input_tokens=trailing[1])
            await self._finish_agent(game, agent, ctx, self._outcome("cancelled", len(submissions)))
            raise
        except Exception:
            if game.game_run is not None:
                game.game_run.solver_note = f"agent error:\n{traceback.format_exc()}"
            # on_finish still runs, best-effort.
            await self._finish_agent(game, agent, ctx, self._outcome("crashed", len(submissions)))
            if config.on_agent_error == "raise":
                raise
            # "crash_game": leave the run unfinished; Benchmark's teardown
            # marks it crashed — and scores whatever was submitted. Other
            # games continue.
        finally:
            if llm_backend is not None:
                import platypus.llm

                platypus.llm.record_replay_consumption(llm_backend, diag)
            diag.close()
            if llm_backend is not None:
                with contextlib.suppress(Exception):
                    await asyncio.shield(llm_backend.close())

    async def _play_and_finish(
        self,
        game: taaf.game.Game,
        agent: Agent,
        ctx: MoveContext,
        view: taaf.game.GameView,
        submitter: Submitter,
        config: RunConfig,
        meter: Any = None,
    ) -> None:
        """In-process hosting: run the shared call directly against the
        game, then finish and report."""
        give_up = await run_game(agent, ctx, view, submitter, config.max_runtime_s_per_game)
        if give_up is not None and game.game_run is not None:
            game.game_run.solver_note = f"gave up: {give_up.reason}"
        trailing = meter.take() if meter is not None else (0, 0)
        # Idempotent; the tokens spent after the last answer go here.
        game.finish_game(generated_tokens=trailing[0], uncached_input_tokens=trailing[1])
        state = "gave_up" if give_up is not None else "finished"
        await self._finish_agent(game, agent, ctx, self._outcome(state, len(ctx.history)))

    async def _host_subprocess(
        self,
        game: taaf.game.Game,
        agent: Agent,
        config: RunConfig,
        seed: int,
        llm_backend: Any | None = None,
        status_key: str = "",
        yield_slot: Callable[[Mapping[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        """Subprocess hosting: the child runs the agent and the same call;
        this side executes submissions against the game, holds the child's
        slot offers, and applies the error policy. A child that dies without
        completing the protocol is a loud error — never a fallback to
        in-process."""
        try:
            kind, agent_error = await host_game_in_subprocess(
                game,
                self._child_agent(agent),
                seed,
                config,
                llm_backend,
                diag_paths=self._game_diag_paths(game),
                job_dir=str(self.job_dir) if self.job_dir is not None else None,
                on_status=self._status_publisher(status_key, config),
                on_yield=yield_slot,
            )
        except SubprocessDiedError:
            note = f"game subprocess died:\n{traceback.format_exc()}"
            if game.game_run is not None:
                game.game_run.solver_note = note
            if config.on_agent_error == "raise":
                raise
            # "crash_game": leave the run unfinished; Benchmark's teardown
            # marks it crashed. on_finish cannot run — the agent is gone.
            return
        if kind == "agent_error" and config.on_agent_error == "raise":
            raise RuntimeError(f"agent error in game subprocess:\n{agent_error}")

    @staticmethod
    def _outcome(state: Literal["finished", "gave_up", "cancelled", "crashed"], n_submitted: int) -> GameOutcome:
        """The agent-facing summary handed to ``on_finish``.

        Built from how the agent's own call ended, which the caller knows —
        never from ``GameRun.state``, which distinguishes ``won`` from
        ``submitted`` and would hand the agent its own correctness.
        """
        return GameOutcome(state=state, n_submitted=n_submitted)

    async def _finish_agent(self, game: taaf.game.Game, agent: Agent, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Run ``on_finish`` shielded: it must complete even when the
        surrounding task is being cancelled; a failure inside it must not
        mask the game's terminal state."""
        try:
            await asyncio.shield(agent.on_finish(ctx, outcome))
        except asyncio.CancelledError:
            pass
        except Exception:
            if game.game_run is not None:
                game.game_run.solver_note = (
                    game.game_run.solver_note or ""
                ) + f"\non_finish failed:\n{traceback.format_exc()}"


__all__ = ["RUNTIME_LIMIT_REASON", "AgentSolver", "Submitter", "run_game"]
