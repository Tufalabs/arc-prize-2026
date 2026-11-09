"""A game played in two takes, and one answer composed from both — or in up
to ten, queued by how far they agree.

``MultiTakeAgent`` is the agent, and its docstring states the behaviour: the
order a game runs in, the composition rule, how a take ends, and what is
configured where. ``TieredTakeAgent`` is its ten-take variant, and its
docstring states what differs. The rest of the module is their parts:
``compose`` and ``vote``, which make the answer, ``category``, which queues a
game for its next take, and the pieces of the ``[take]`` log.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Self, cast

import ullm.backend

import platypus.diagnostics
import taaf.game
from platypus.context import Produced, SubmissionView, check_produced
from platypus.contract import (
    HANDICAP_HINT,
    TAKE_BOUNDARY_HINT,
    TAKE_INDEX_HINT,
    ContractViolationError,
    GameOutcome,
    GiveUp,
    RunConfig,
)
from platypus.llm import LLMAgent, LLMConfig, LLMMoveContext
from taaf.grid import Grid

#: How many takes a ``MultiTakeAgent`` game plays. The composition rule is
#: stated for two.
N_TAKES = 2

#: The prefix of every stdout line the takes print.
LOG_PREFIX = "[take]"

TakeState = Literal["finished", "gave_up", "cancelled", "crashed"]


@dataclass(frozen=True)
class TakeAnswer:
    """One take's standing answer, and which of its grids it produced."""

    submission: taaf.game.Submission
    produced: Produced

    def grids(self, question: int, take_index: int) -> list[tuple[Grid, str]]:
        """The grids this take produced for one question, in its own order,
        each with the name the composition gives its source."""
        attempts = self.submission.attempts[question]
        return [
            (grid, f"take {take_index + 1} grid {slot + 1}")
            for slot, (grid, made) in enumerate(zip(attempts, self.produced[question], strict=True))
            if made
        ]


@dataclass(frozen=True)
class Composition:
    """The answer the takes compose to."""

    submission: taaf.game.Submission
    #: Which grids of ``submission`` a take produced; a copy or a placeholder
    #: did not come from one.
    produced: Produced
    #: Per question, per attempt, where the grid came from: ``"take 2 grid
    #: 1"``, ``"copy of attempt 1"`` or ``"placeholder"``.
    sources: tuple[tuple[str, ...], ...]


def compose(questions: Sequence[Grid], answers: Sequence[TakeAnswer | None]) -> Composition | None:
    """What the takes' standing answers compose to (``MultiTakeAgent`` states
    the rule), or ``None`` while no take has produced a grid at all."""
    if len(answers) != N_TAKES:
        raise ValueError(f"the composition is stated for {N_TAKES} takes; got {len(answers)}")
    groups: list[tuple[Grid, ...]] = []
    produced: list[tuple[bool, ...]] = []
    sources: list[tuple[str, ...]] = []
    anything = False
    for question_index, question in enumerate(questions):
        first, second = (
            [] if answer is None else answer.grids(question_index, take_index)
            for take_index, answer in enumerate(answers)
        )
        # The head is attempt 1; the rest are attempt 2's candidates in order.
        order = [first[0], *second[:1], *first[1:2], *second[1:2]] if first else second[:2]
        chosen: list[tuple[Grid, str, bool]] = []
        if order:
            anything = True
            head, head_source = order[0]
            chosen.append((head, head_source, True))
            tail = next(((grid, source) for grid, source in order[1:] if grid != head), None)
            if tail is not None:
                chosen.append((tail[0], tail[1], True))
        else:
            chosen.append((question, "placeholder", False))
        while len(chosen) < taaf.game.N_ATTEMPTS:
            chosen.append((chosen[0][0], "copy of attempt 1", False))
        groups.append(tuple(grid for grid, _, _ in chosen))
        sources.append(tuple(source for _, source, _ in chosen))
        produced.append(tuple(made for _, _, made in chosen))
    if not anything:
        return None
    return Composition(
        submission=taaf.game.Submission(attempts=tuple(groups)),
        produced=tuple(produced),
        sources=tuple(sources),
    )


def grid_label(grid: Grid) -> str:
    """A grid as the take log names it: its size and a short content hash, so
    that identical grids read as identical across takes and lines."""
    digest = hashlib.blake2b(repr(grid.shape).encode() + grid.data.tobytes(), digest_size=3).hexdigest()
    return f"{grid.shape[0]}x{grid.shape[1]} #{digest}"


class _Composer:
    """The takes' standing answers, and the composition last submitted."""

    def __init__(self, ctx: LLMMoveContext, questions: tuple[Grid, ...]) -> None:
        self.answers: list[TakeAnswer | None] = [None] * N_TAKES
        #: The confidence each ended take stated, by its index.
        self.confidences: dict[int, int] = {}
        self.composition: Composition | None = None
        self.questions = questions
        self._ctx = ctx

    def record(self, take_index: int, answer: TakeAnswer) -> None:
        """Take a take's new standing answer, and submit the composition if
        its grids or their provenance changed."""
        self.answers[take_index] = answer
        self._resubmit()

    def end(self, take_index: int, confidence: int | None) -> None:
        """A take's play is over, at the confidence it stated. Nothing else
        changes here: a take's answer counts from its first submission on."""
        if confidence is not None:
            self.confidences[take_index] = confidence

    def _compose(self) -> Composition | None:
        """What the standing answers compose to."""
        return compose(self.questions, self.answers)

    def _resubmit(self) -> None:
        """Submit the composition if its grids or their provenance changed."""
        composition = self._compose()
        if composition is None:
            return
        previous = self.composition
        self.composition = composition
        if (
            previous is not None
            and previous.submission == composition.submission
            and previous.produced == composition.produced
        ):
            return
        self._ctx.submit(composition.submission, produced=composition.produced)


@dataclass(kw_only=True)
class MultiTakeAgent(LLMAgent):
    """An agent that plays its game twice — two agents in turn, its **takes** —
    and submits one answer composed from both.

    Every test output is scored on two attempts, and either may be right, so
    the second attempt is worth most when it holds a genuinely different grid;
    an independent second take is where one comes from. A take is not a TAAF
    pass: passes are replicates reported separately, while takes combine into
    the one answer this game submits. To the solver this is an ordinary
    ``LLMAgent``, and both takes run inside the one game process.

    **A game, in order.**

    1. Take 1 plays on a context of its own. The game's backend, RNG,
       diagnostics, status and slot view pass through; the answer channel and
       the submission history are the take's own, and its slot offers reach
       the game marked with the take.
    2. Whenever take 1 submits, that submission becomes its standing answer,
       the takes' answers are composed, and the composition is submitted to
       the game if its grids or their provenance changed. While only take 1
       has answered, the composition is take 1's own answer.
    3. When take 1 ends, a ``[take]`` block goes to stdout — how it ended,
       what it cost, its own confidence in its answer
       (``LLMAgent.answer_confidence``), what it produced per question, and
       where each grid of the standing answer came from — then its
       ``on_finish`` runs and the agent is dropped, freeing its conversation
       and kernel.
    4. ``_start_take`` prepares take 2 from take 1. It does nothing here; a
       subclass whose takes count anything per game — the duck's token budget
       and history epoch — carries it across (``duck.multi_take``).
    5. The game offers its slot back, marked ``take_boundary``: nothing is
       resident there, so the swap is free, and from here ``least_tokens``
       ranks the game ``RunConfig.take_handicap_tokens`` behind its spend.
    6. Take 2 plays the same way, from a start that knows nothing of take 1.
       Its submissions recompose the answer; its end prints its block and runs
       its ``on_finish``.

    **The composition, per test output.** A take's grids are the ones it
    produced (``MoveContext.submit``'s ``produced``), in its own slot order; a
    filler — a placeholder, or a copy of its one grid — never counts.

    - Attempt 1 is take 1's first grid.
    - Attempt 2 is the first of take 2's first grid, take 1's second grid and
      take 2's second grid that is not identical to attempt 1.
    - Where take 1 has no grid for the question, take 2's grids take its
      place, in take 2's order.
    - Where no take has one, the question's input grid stands in, as the duck
      fills one. A missing attempt 2 copies attempt 1.

    **How a take ends.** Finished, given up, crashed under
    ``on_agent_error="crash_game"``, or cut off by ``max_runtime_s_per_take``:
    each hands on to the next take, and the game gives up only if no take
    finished. A crash under ``"raise"`` propagates and a cancellation of the
    game goes straight through; either way ``on_finish`` prints the
    interrupted take's block and finishes it. A take that submits after its
    play has ended is refused. A take's own ``on_finish`` failing is contained
    as the host contains an agent's: printed and recorded, and the next take
    still plays.

    **Limits are per take.** ``max_runtime_s_per_take`` is enforced here, on a
    clock that stops while the game is parked. Token budgets are the takes'
    own; a take that reads the game's one backend meter needs ``_start_take``
    to account for the spend before it.

    **Configuration.** Exactly two takes, each an ``LLMAgent``. Construction
    deep-copies them, so the caller's agents are never modified and
    ``(agent, agent)`` is two independent takes; change a take afterwards
    through ``takes[i]``. The first take's configuration becomes this
    agent's: ``run_config`` and ``llm_config`` *are* its objects, so the
    deployment it was built for is what the solver reads, and passing either
    here is refused. Construction moves the first take's
    ``max_runtime_s_per_game`` into ``max_runtime_s_per_take`` when that is
    unset, since a game-level net sized for one take would cut the pair short.
    Every take plays on that ``run_config``, since hosting settings are the
    game's rather than a take's. The second take plays on the engine the
    first one sized: it points at the same backend factory and prepares with
    ``settle_engine=False``, so it writes nothing onto the engine and is
    refused if it does not fit. ``without_engine`` drops the factory from
    every take on the way into the game child.
    """

    #: How many takes a game plays: ``takes`` must hold exactly this many.
    n_takes: ClassVar[int] = N_TAKES

    takes: tuple[LLMAgent, ...]
    # Wall-clock net per take, enforced here because infra's net
    # (``RunConfig.max_runtime_s_per_game``) bounds the whole game. None is no
    # net.
    max_runtime_s_per_take: float | None = None
    # The take playing right now: what ``on_finish`` has to report and finish
    # if the game ends in the middle of one.
    _playing: _Playing | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Copy the takes, take over the first one's configuration, and settle
        eagerly, so a broken combination is refused now rather than at run
        start."""
        if len(self.takes) != self.n_takes:
            raise ValueError(f"{type(self).__name__} plays {self.n_takes} takes; got {len(self.takes)}")
        for take in self.takes:
            if not isinstance(cast(object, take), LLMAgent):
                raise TypeError(f"a take must be an LLMAgent; got {type(take).__name__}")
        if self.run_config != RunConfig() or self.llm_config != LLMConfig():
            raise ValueError(
                f"{type(self).__name__} takes run_config and llm_config over from its first take; "
                "configure them there, not here"
            )
        self.takes = tuple(copy.deepcopy(take) for take in self.takes)
        self.run_config = self.takes[0].run_config
        self.llm_config = self.takes[0].llm_config
        game_net = self.run_config.max_runtime_s_per_game
        self.run_config.max_runtime_s_per_game = None
        if self.max_runtime_s_per_take is None:
            self.max_runtime_s_per_take = game_net
        self.prepare_for_backend()

    # ------------------------------------------------------------------
    # host side: the engine, the child crossing, the status line
    # ------------------------------------------------------------------

    def prepare_for_backend(self, *, settle_engine: bool = True) -> None:
        """Settle the engine through the first take, then check the others
        fit it.

        Every take plays on this agent's ``run_config``: hosting settings are
        the game's, not a take's. The first take settles its derived engine
        arguments onto the one factory; the others point at that factory and
        prepare with ``settle_engine=False``, so their own checks see the
        engine it sized without moving it. Nothing moves between calls, so
        this does the same however often it runs.
        """
        super().prepare_for_backend(settle_engine=settle_engine)
        first, *others = self.takes
        first.llm_config = self.llm_config
        for take in self.takes:
            take.run_config = self.run_config
        first.prepare_for_backend(settle_engine=settle_engine)
        for take in others:
            take.llm_config.backend_factory = self.llm_config.backend_factory
            take.prepare_for_backend(settle_engine=False)

    def _validate_host_backend(self, backend: ullm.backend.Backend) -> None:
        """Every take validates the backend it will play on."""
        for take in self.takes:
            take._validate_host_backend(backend)

    def needs_images(self) -> bool:
        """Whether any take sends images: the backend serves every take."""
        return any(take.needs_images() for take in self.takes)

    def without_engine(self) -> Self:
        """This agent and its takes, each without the engine spec."""
        child = super().without_engine()
        child.takes = tuple(take.without_engine() for take in self.takes)
        return child

    def status_reporters(self) -> list[Callable[[], str]]:
        """The first take's status reporters, if it publishes any: the solver
        reads them off the hosted agent, which is this one."""
        hook: Callable[[], list[Callable[[], str]]] | None = getattr(self.takes[0], "status_reporters", None)
        return list(hook()) if callable(hook) else []

    # ------------------------------------------------------------------
    # the game
    # ------------------------------------------------------------------

    async def _do_move(self, ctx: LLMMoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Play the takes in turn and submit the composition whenever a take's
        answer changes it.

        A take that ends on its own — finished, given up, out of its time,
        or crashed under ``on_agent_error="crash_game"`` — hands on to the
        next; the game gives up only when every take did. A cancellation of
        the game itself goes straight through, and ``on_finish`` reports and
        finishes the take it interrupted.
        """
        pending = list(self.takes)
        # The takes leave the agent as they play, so a finished one is freed
        # at its boundary rather than kept alive to the end of the game.
        self.takes = ()
        composer = self._new_composer(ctx, view.questions)
        endings: list[str] = []
        previous: LLMAgent | None = None
        for take_index in range(self.n_takes):
            take = pending.pop(0)
            if previous is not None:
                spent = self._game_tokens(ctx)
                self._start_take(take, previous=previous, spent_tokens=spent)
                previous = None
                hints = self._slot_hints(take_index, composer, {"tokens": spent})
                await ctx.yield_slot({**hints, TAKE_BOUNDARY_HINT: True})
            state, note = await self._play_take(ctx, view, take_index, take, composer)
            if state != "finished":
                endings.append(f"take {take_index + 1} {state}: {_last_line(note)}")
            previous = take
        if len(endings) == self.n_takes:
            return GiveUp("; ".join(endings))
        return None

    async def _play_take(
        self,
        ctx: LLMMoveContext,
        view: taaf.game.GameView,
        take_index: int,
        take: LLMAgent,
        composer: _Composer,
    ) -> tuple[TakeState, str]:
        """One take, start to finish: its own context, its play under the
        per-take net, the take log, and its ``on_finish``."""
        loop = asyncio.get_running_loop()
        clock: asyncio.Timeout | None = None
        playing = _Playing(
            take=take,
            take_index=take_index,
            composer=composer,
            view=view,
            tokens_before=self._game_tokens(ctx),
        )

        def submit(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
            """The take's answer channel: recorded here, composed, and passed up
            only as the composition."""
            if not playing.armed:
                raise ContractViolationError(f"take {take_index + 1} submitted after its play ended")
            if len(submission.attempts) != len(view.questions):
                raise ContractViolationError(
                    f"submission covers {len(submission.attempts)} pair(s); "
                    f"this task has {len(view.questions)} to answer"
                )
            flags = check_produced(submission, produced)
            playing.submissions.append(submission)
            composer.record(take_index, TakeAnswer(submission=submission, produced=flags))

        async def yield_slot(hints: Mapping[str, Any]) -> None:
            """The take's slot offer, marked with the take. The per-take net
            stops while the game is parked and resumes with what was left."""
            deadline = clock.when() if clock is not None else None
            remaining = None if deadline is None else deadline - loop.time()
            if clock is not None:
                clock.reschedule(None)
            try:
                await ctx.yield_slot(self._slot_hints(take_index, composer, hints))
            finally:
                if clock is not None and remaining is not None:
                    clock.reschedule(loop.time() + remaining)

        playing.ctx = LLMMoveContext(
            game_id=ctx.game_id,
            history=SubmissionView(playing.submissions),
            pass_index=ctx.pass_index,
            rng=ctx.rng,
            submit=submit,
            diag=ctx.diag,
            slots=ctx.slots,
            status=ctx.status,
            yield_slot=yield_slot,
            backend=ctx.backend,
        )
        self._playing = playing
        state: TakeState
        with ctx.diag.span("take", f"take {take_index + 1}") as span:
            try:
                async with asyncio.timeout(self.max_runtime_s_per_take) as timeout:
                    clock = timeout
                    decision = await take.do_move(playing.ctx, view)
                state, note = ("gave_up", decision.reason) if isinstance(decision, GiveUp) else ("finished", "")
            except TimeoutError:
                if clock is None or not clock.expired():
                    if self.run_config.on_agent_error == "raise":
                        raise
                    state, note = "crashed", traceback.format_exc()
                else:
                    state, note = "cancelled", f"per-take runtime limit ({self.max_runtime_s_per_take:g} s)"
            except Exception:
                if self.run_config.on_agent_error == "raise":
                    raise
                state, note = "crashed", traceback.format_exc()
            finally:
                playing.armed = False
            playing.confidence = take.answer_confidence()
            composer.end(take_index, playing.confidence)
            span.set_outcome(state)
            self._report_take(ctx, playing, state, note)
        await self._finish_take(playing, state)
        return state, note

    def _new_composer(self, ctx: LLMMoveContext, questions: tuple[Grid, ...]) -> _Composer:
        """The composer a game's answers go through."""
        return _Composer(ctx, questions)

    def _slot_hints(self, take_index: int, composer: _Composer, hints: Mapping[str, Any]) -> dict[str, Any]:
        """A slot offer made while take ``take_index`` plays or is about to,
        as the game reports it: marked with the take."""
        del composer
        return {**hints, TAKE_INDEX_HINT: take_index}

    def _start_take(self, take: LLMAgent, *, previous: LLMAgent, spent_tokens: int) -> None:
        """Prepare a take about to play from the one before it, which has
        finished; ``spent_tokens`` is what the game has generated so far.

        Nothing here. A take that counts anything per game rather than per
        take — the duck's token budget and history epoch — needs a subclass
        to carry it across (``duck.multi_take``).
        """
        del take, previous, spent_tokens

    async def _on_finish(  # pyright: ignore[reportIncompatibleMethodOverride] — same covariant ctx narrowing as LLMAgent._do_move
        self, ctx: LLMMoveContext, outcome: GameOutcome
    ) -> None:
        """Report and finish the take that was playing when the game ended, if
        one was; a take that ended on its own was finished at its boundary."""
        playing = self._playing
        if playing is None:
            return
        playing.armed = False
        playing.confidence = playing.take.answer_confidence()
        self._report_take(ctx, playing, outcome.state, "the game ended during this take")
        await self._finish_take(playing, outcome.state)

    def _report_take(self, ctx: LLMMoveContext, playing: _Playing, state: TakeState, note: str) -> None:
        """Put a take's ending on its span and, loudly, on stdout — once."""
        if playing.reported:
            return
        playing.reported = True
        take_tokens = self._game_tokens(ctx) - playing.tokens_before
        platypus.diagnostics.current_span().kv(
            state=state,
            note=note,
            answers=len(playing.submissions),
            confidence=playing.confidence,
            take_tokens=take_tokens,
            game_tokens=self._game_tokens(ctx),
        )
        print(self._take_log(ctx, playing, state, note, take_tokens), flush=True)

    async def _finish_take(self, playing: _Playing, state: TakeState) -> None:
        """The take's own ``on_finish``: its resources go here, at its
        boundary.

        A failure inside it is contained as the host contains an agent's
        (``AgentSolver._finish_agent``): cleanup is best-effort, and a take
        whose cleanup fails has already played, so it must not cost the next
        take its play. It is printed and recorded on the game's span instead.
        What a failed cleanup can leave behind is that take's own — the
        duck's kernel and scratch directory — never the game's backend or
        context, which the next take plays on.
        """
        assert playing.ctx is not None
        try:
            await playing.take.on_finish(playing.ctx, GameOutcome(state=state, n_submitted=len(playing.submissions)))
        except Exception as exc:
            print(
                f"{LOG_PREFIX} take {playing.take_index + 1} on_finish failed; the game plays on:\n"
                f"{traceback.format_exc()}",
                flush=True,
            )
            platypus.diagnostics.current_span().event(
                "take/on_finish_failed", take=playing.take_index + 1, error=f"{type(exc).__name__}: {exc}"
            )
        self._playing = None

    @staticmethod
    def _game_tokens(ctx: LLMMoveContext) -> int:
        """Completion tokens the game's backend has billed so far, every take
        included."""
        return ctx.backend.total_usage.completion_tokens

    def _take_log(self, ctx: LLMMoveContext, playing: _Playing, state: TakeState, note: str, take_tokens: int) -> str:
        """The stdout block a finished take prints: how it ended and what it
        cost, its confidence, what it produced per question, and where every
        grid of the standing composed answer came from."""
        game = f"{ctx.game_id} pass {ctx.pass_index}"
        number = playing.take_index + 1
        ending = f"{state}: {_last_line(note)}" if note else state
        confidence = "no confidence" if playing.confidence is None else f"confidence {playing.confidence}"
        lines = [
            f"{LOG_PREFIX} ==== {game}: take {number} of {self.n_takes} finished ({ending}) ====",
            f"{LOG_PREFIX} {game} take {number}: {take_tokens:,} tokens generated in the take, "
            f"{MultiTakeAgent._game_tokens(ctx):,} in the game; {len(playing.submissions)} answer(s), {confidence}",
        ]
        answer = playing.composer.answers[playing.take_index] if playing.submissions else None
        for question in range(len(playing.view.questions)):
            made = [] if answer is None else answer.grids(question, playing.take_index)
            produced = ", ".join(f"{source} {grid_label(grid)}" for grid, source in made) or "nothing"
            lines.append(f"{LOG_PREFIX} {game} take {number} question {question + 1}: produced {produced}")
        composition = playing.composer.composition
        if composition is None:
            lines.append(f"{LOG_PREFIX} {game} answer: none yet")
        else:
            for question, (grids, sources) in enumerate(
                zip(composition.submission.attempts, composition.sources, strict=True)
            ):
                attempts = ", ".join(
                    f"attempt {slot + 1} = {source} ({grid_label(grid)})"
                    for slot, (grid, source) in enumerate(zip(grids, sources, strict=True))
                )
                lines.append(f"{LOG_PREFIX} {game} answer question {question + 1}: {attempts}")
        return "\n".join(lines)


@dataclass
class _Playing:
    """A take in play, and what reporting and finishing it needs."""

    take: LLMAgent
    take_index: int
    composer: _Composer
    view: taaf.game.GameView
    tokens_before: int
    ctx: LLMMoveContext | None = None
    submissions: list[taaf.game.Submission] = field(default_factory=lambda: list[taaf.game.Submission]())
    armed: bool = True
    reported: bool = False
    #: The take's own confidence, read as its play ends.
    confidence: int | None = None


def _last_line(note: str) -> str:
    """A note's last non-empty line: the exception itself, for a traceback."""
    lines = [line for line in note.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


# ---------------------------------------------------------------------------
# up to ten takes, queued by how far they agree
# ---------------------------------------------------------------------------

#: How many takes a ``TieredTakeAgent`` game plays at most.
N_TIERED_TAKES = 10

#: What each category ranks behind the one before it (``TieredTakeAgent``), in
#: tokens: far past what any game spends with every handicap it carries, so a
#: game in a lower category always ranks first and spend orders a category.
CATEGORY_HANDICAP_TOKENS = 1_000_000_000


def category(ended: Sequence[TakeAnswer | None], n_questions: int) -> int:
    """Where a game queues for its next take, from the answers its ended takes
    stood on (``None``: a take that ended without one); lower plays first.

    0 and 1 are its first and second takes. After those, every test output
    needs a grid 1 two takes produced first — 2 until it has one — and then
    one three takes did, or the first two takes agreeing on it there — 3 until
    it has. A game past both queues a category a take, 4 for its third take.
    """
    played = len(ended)
    if played < 2:
        return played
    firsts = [[_first_grid(answer, question) for answer in ended] for question in range(n_questions)]
    most = [max(Counter(grid for grid in grids if grid is not None).values(), default=0) for grids in firsts]
    if min(most, default=0) < 2:
        return 2
    agreed = [grids[0] is not None and grids[0] == grids[1] for grids in firsts]
    if any(votes < 3 and not pair for votes, pair in zip(most, agreed, strict=True)):
        return 3
    return 2 + played


def category_name(rank: int) -> str:
    """What a category queues, for the take log."""
    if rank < 2:
        return f"take {rank + 1}"
    if rank == 2:
        return "no grid 1 two takes produced yet"
    if rank == 3:
        return "no grid 1 three takes produced, nor the first two agreeing"
    return f"take {rank - 1}, a category a take"


def _first_grid(answer: TakeAnswer | None, question: int) -> Grid | None:
    """The first grid a take produced for a question, if it produced one."""
    grids = [] if answer is None else answer.grids(question, 0)
    return grids[0][0] if grids else None


@dataclass
class _Tally:
    """One candidate grid's votes: takes that produced it first, and second,
    and the highest confidence among each."""

    first: int
    second: int
    source: str
    first_confidence: int = -1
    second_confidence: int = -1

    @property
    def confidence(self) -> int:
        """The confidence it ranks at: of the takes that produced it first,
        or second where none did; -1 where none of them stated one."""
        return self.first_confidence if self.first else self.second_confidence


def _tiebreak(grid: Grid) -> bytes:
    """A stable content order for candidates tied after take 1's preference."""
    return hashlib.blake2b(repr(grid.shape).encode() + grid.data.tobytes(), digest_size=8).digest()


def vote(
    questions: Sequence[Grid],
    answers: Sequence[tuple[int, TakeAnswer]],
    confidences: Mapping[int, int] | None = None,
) -> Composition | None:
    """What several takes' answers vote to, given as ``(take index, answer)``,
    or ``None`` while none has produced a grid.

    Per test output, every grid a take produced is a candidate, ranked by how
    many takes produced it first, then by how many produced it second. Ties
    go to the grid the most confident take produced, in the place that
    counted (``confidences``, each take's own by its index; a take that stated
    none ranks below any that did), then to take 1's first produced grid, then
    its second, then ``_tiebreak`` for grids take 1 did not produce. Attempt 1
    is the top candidate and attempt 2 the next; with one candidate attempt 2
    copies attempt 1, and with none the question's input grid stands in. A
    grid's source names the first take that produced it, and its votes.
    """
    groups: list[tuple[Grid, ...]] = []
    produced: list[tuple[bool, ...]] = []
    sources: list[tuple[str, ...]] = []
    anything = False
    for question_index, question in enumerate(questions):
        tally: dict[Grid, _Tally] = {}
        first_take_order: dict[Grid, int] = {}
        for take_index, answer in answers:
            confidence = -1 if confidences is None else confidences.get(take_index, -1)
            # A take votes once for a grid, however many of its slots hold it.
            named: dict[Grid, str] = {}
            for grid, source in answer.grids(question_index, take_index):
                named.setdefault(grid, source)
            for position, (grid, source) in enumerate(list(named.items())[:2]):
                if take_index == 0:
                    first_take_order[grid] = position
                entry = tally.setdefault(grid, _Tally(first=0, second=0, source=source))
                if position == 0:
                    entry.first += 1
                    entry.first_confidence = max(entry.first_confidence, confidence)
                else:
                    entry.second += 1
                    entry.second_confidence = max(entry.second_confidence, confidence)
        ranked = sorted(
            tally.items(),
            key=lambda item: (
                -item[1].first,
                -item[1].second,
                -item[1].confidence,
                first_take_order.get(item[0], 2),
                _tiebreak(item[0]),
            ),
        )
        chosen = [
            (grid, f"{entry.source}, produced first by {entry.first} and second by {entry.second}", True)
            for grid, entry in ranked[: taaf.game.N_ATTEMPTS]
        ]
        if chosen:
            anything = True
        else:
            chosen.append((question, "placeholder", False))
        while len(chosen) < taaf.game.N_ATTEMPTS:
            chosen.append((chosen[0][0], "copy of attempt 1", False))
        groups.append(tuple(grid for grid, _, _ in chosen))
        sources.append(tuple(source for _, source, _ in chosen))
        produced.append(tuple(made for _, _, made in chosen))
    if not anything:
        return None
    return Composition(
        submission=taaf.game.Submission(attempts=tuple(groups)),
        produced=tuple(produced),
        sources=tuple(sources),
    )


@dataclass(frozen=True)
class _Numbered(TakeAnswer):
    """A take's answer that names its own take wherever ``compose`` puts it."""

    take_index: int = 0

    def grids(self, question: int, take_index: int) -> list[tuple[Grid, str]]:
        """The grids this take produced for one question, named for its own take."""
        del take_index
        return super().grids(question, self.take_index)


class _VotingComposer(_Composer):
    """Up to ``n_takes`` takes' standing answers: the first two count from
    their first answer, a later take's once its play is over. Three counted
    answers or more are voted on (``vote``); fewer compose as two takes do
    (``compose``), each named for its own take."""

    def __init__(self, ctx: LLMMoveContext, questions: tuple[Grid, ...], n_takes: int) -> None:
        super().__init__(ctx, questions)
        self.answers = [None] * n_takes
        self.ended: set[int] = set()

    def end(self, take_index: int, confidence: int | None) -> None:
        """A take's play is over: a later take's answer counts from here."""
        super().end(take_index, confidence)
        self.ended.add(take_index)
        self._resubmit()

    def _compose(self) -> Composition | None:
        """What the counted answers compose or vote to."""
        counted = [
            (index, answer)
            for index, answer in enumerate(self.answers)
            if answer is not None and (index < N_TAKES or index in self.ended)
        ]
        if len(counted) > N_TAKES:
            return vote(self.questions, counted, self.confidences)
        numbered: list[TakeAnswer | None] = [
            _Numbered(submission=answer.submission, produced=answer.produced, take_index=index)
            for index, answer in counted
        ]
        return compose(self.questions, numbered + [None] * (N_TAKES - len(numbered)))


@dataclass(kw_only=True)
class TieredTakeAgent(MultiTakeAgent):
    """A ``MultiTakeAgent`` that plays up to ten takes and queues each by how
    far the takes before it agree. ``takes`` holds ten. Not configurable:
    everything below is fixed, and everything not here is
    ``MultiTakeAgent``'s.

    **The queue.** Before each take the game is put in a category
    (``category``), and the ``least_tokens`` ranking serves a lower category
    before anything in a higher one: the take's slot offers carry the
    category as ``CATEGORY_HANDICAP_TOKENS`` a step on top of whatever
    handicap the take asks for itself. Within a category the ranking is as
    for any game. The categories, numbered as ``category`` and the take log
    number them:

    0. the first take;
    1. the second take;
    2. a further take while some test output has no grid 1 that two takes
       produced first;
    3. a further take while some test output has none that three did, the
       first two takes agreeing on it counting as three;
    4. on from there, a category a take: 4 for the third take, 5 for the
       fourth, up to 11 for the tenth.

    A game may play all ten takes; the ranking decides which games get
    further takes before the clock runs out.

    **The answer.** Takes 1 and 2 count from their first submission, as in
    ``MultiTakeAgent``; a later take counts once its play is over, as does
    everything the categories read. Up to two counted answers compose as two
    takes do (``compose``); three or more are voted on (``vote``), its ties
    broken on the confidence each take stated as it ended, as arm sweep 2's
    arms replayed by the take simulator in ``arc2-experiments``
    (``arm_budget/take_sim``) read it.

    From the pass-budget simulation in ``arc2-experiments``
    (``pass_budget``, the agreed-pair ranking under variant A): a run of ten
    passes a task, rescheduled on 50 slots against the token budget charged.
    """

    n_takes: ClassVar[int] = N_TIERED_TAKES

    def _new_composer(self, ctx: LLMMoveContext, questions: tuple[Grid, ...]) -> _Composer:
        """The voting composer, sized for every take."""
        return _VotingComposer(ctx, questions, self.n_takes)

    def _slot_hints(self, take_index: int, composer: _Composer, hints: Mapping[str, Any]) -> dict[str, Any]:
        """The take's offer, ranked behind every lower category than its game's."""
        rank = category(composer.answers[:take_index], len(composer.questions))
        handicap = int(hints.get(HANDICAP_HINT, 0)) + rank * CATEGORY_HANDICAP_TOKENS
        return {**super()._slot_hints(take_index, composer, hints), HANDICAP_HINT: handicap}

    def _take_log(self, ctx: LLMMoveContext, playing: _Playing, state: TakeState, note: str, take_tokens: int) -> str:
        """``MultiTakeAgent``'s take log, and where the game queues next."""
        log = super()._take_log(ctx, playing, state, note, take_tokens)
        following = playing.take_index + 1
        if following == self.n_takes:
            return log
        rank = category(playing.composer.answers[:following], len(playing.view.questions))
        return (
            f"{log}\n{LOG_PREFIX} {ctx.game_id} pass {ctx.pass_index} take {following + 1} queues in "
            f"category {rank}: {category_name(rank)}"
        )


__all__ = [
    "CATEGORY_HANDICAP_TOKENS",
    "LOG_PREFIX",
    "N_TAKES",
    "N_TIERED_TAKES",
    "Composition",
    "MultiTakeAgent",
    "TakeAnswer",
    "TieredTakeAgent",
    "category",
    "category_name",
    "compose",
    "grid_label",
    "vote",
]
