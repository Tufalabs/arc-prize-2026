"""Core game abstractions: the task types, ``GameRun`` bookkeeping, and the
abstract ``Game`` a solver plays.

A game is one ARC task. The solver reads the task through ``query()`` and
calls ``execute_action`` with two attempts for every pair it has to answer.
It may call it **more than once**: each call replaces the standing answer and
is recorded with what it cost, so a run cut off part-way still submits the
best answer the solver had reached.

The score is recorded once, in ``finish_game``, on the standing answer, and
nothing the solver calls returns one: ``execute_action`` would otherwise hand
it per-pair correctness for an answer it can still replace, which is a scoring
oracle. Whoever watches the run reads a game still in play through
``GameRun.current_score``, by the same rule the finish applies.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

from taaf.grid import Grid

if TYPE_CHECKING:
    import taaf.trace

T = TypeVar("T")

N_ATTEMPTS = 2


# --- the task ----------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One question/answer pair of a task.

    ``given`` marks a demonstration the solver may see in full; a pair that
    is not given is one the solver must answer. ``answer`` is ``None`` only
    where the ground truth genuinely does not exist — a competition rerun,
    or a deliberately withheld run — never as a placeholder.
    """

    question: Grid
    answer: Grid | None
    given: bool

    def __post_init__(self) -> None:
        if self.given and self.answer is None:
            raise ValueError("a given pair must carry its answer")


@dataclass(frozen=True)
class GameSpec:
    """A whole task: its id and its pairs, answers included where known.

    Solvers never receive one of these — they get a :class:`GameView`.
    """

    game_id: str
    pairs: tuple[Pair, ...]

    def __post_init__(self) -> None:
        if not self.game_id:
            raise ValueError("game_id must be non-empty")
        if not any(p.given for p in self.pairs):
            raise ValueError(f"{self.game_id}: a task needs at least one given pair")
        if not self.to_answer:
            raise ValueError(f"{self.game_id}: a task needs at least one pair to answer")

    @property
    def given(self) -> tuple[Pair, ...]:
        """The demonstration pairs, in order."""
        return tuple(p for p in self.pairs if p.given)

    @property
    def to_answer(self) -> tuple[Pair, ...]:
        """The pairs the solver must answer, in order."""
        return tuple(p for p in self.pairs if not p.given)

    @property
    def answers_known(self) -> bool:
        """Whether every to-answer pair carries ground truth, i.e. whether
        this task can be scored at all."""
        return all(p.answer is not None for p in self.to_answer)

    def view(self) -> GameView:
        """The solver-visible projection: everything except the answers to
        the pairs it has to produce."""
        return GameView(
            game_id=self.game_id,
            given=tuple((p.question, p.answer) for p in self.given if p.answer is not None),
            questions=tuple(p.question for p in self.to_answer),
        )

    def withhold_answers(self) -> GameSpec:
        """A copy whose to-answer pairs carry no ground truth — the state a
        competition rerun is in (``simulate_submission``)."""
        return GameSpec(
            game_id=self.game_id,
            pairs=tuple(p if p.given else Pair(question=p.question, answer=None, given=False) for p in self.pairs),
        )


@dataclass(frozen=True)
class GameView:
    """What a solver may see: the demonstration pairs in full, and the
    questions it must answer. There is deliberately no path from here to a
    withheld answer."""

    game_id: str
    given: tuple[tuple[Grid, Grid], ...]
    questions: tuple[Grid, ...]


@dataclass(frozen=True)
class Submission:
    """The solver's answer: ``N_ATTEMPTS`` grids for each pair it had to
    answer, in the same order as :attr:`GameView.questions`."""

    attempts: tuple[tuple[Grid, ...], ...]

    def __post_init__(self) -> None:
        for i, group in enumerate(self.attempts):
            if len(group) != N_ATTEMPTS:
                raise ValueError(f"pair {i}: expected {N_ATTEMPTS} attempts, got {len(group)}")

    def to_json_list(self) -> list[dict[str, list[list[int]]]]:
        """The competition's per-task shape: one dict per pair, keys
        ``attempt_1``..``attempt_N``."""
        return [{f"attempt_{i + 1}": g.to_lists() for i, g in enumerate(group)} for group in self.attempts]


@dataclass(frozen=True)
class PairResult:
    """How one to-answer pair came out. ``correct`` is ``None`` where the
    ground truth was withheld — never ``False``, which would read as a wrong
    answer."""

    correct: bool | None
    matched_attempt: int | None


@dataclass(frozen=True)
class GameResult:
    """How an answer scored, one entry per to-answer pair.

    ``GameRun.result`` holds the one for the standing answer, recorded at
    ``finish_game``. Diagnostics scores earlier answers the same way, from the
    recorded history — see ``taaf.diagnostics``.
    """

    pairs: tuple[PairResult, ...]

    @property
    def scored(self) -> bool:
        """Whether ground truth was available, i.e. whether the numbers below
        mean anything."""
        return all(p.correct is not None for p in self.pairs)

    @property
    def n_correct(self) -> int:
        """How many pairs came back right. 0 on an unscored result — read
        :attr:`scored` first."""
        return sum(1 for p in self.pairs if p.correct)

    @property
    def score(self) -> float | None:
        """Percentage of pairs answered correctly, or ``None`` unscored."""
        if not self.scored:
            return None
        return 100.0 * self.n_correct / len(self.pairs)

    def to_json_dict(self) -> dict[str, Any]:
        """JSON-friendly form for ``benchmark.json``."""
        return {"pairs": [{"correct": p.correct, "matched_attempt": p.matched_attempt} for p in self.pairs]}

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> GameResult:
        """Inverse of :meth:`to_json_dict`."""
        return cls(
            pairs=tuple(PairResult(correct=p["correct"], matched_attempt=p["matched_attempt"]) for p in d["pairs"])
        )


# --- RunSession --------------------------------------------------------------


@dataclass
class RunSession:
    """Per-``Benchmark.run()`` resource cache, and the carrier for run-wide
    settings a ``Game`` needs at start.

    ``withhold_answers`` drops the ground truth from every pair a solver must
    answer, putting the run in the state a competition rerun is actually in.
    Applied here rather than at the dataset, so it reaches every ``Game``
    subclass by construction and one switch cannot be half-applied.

    Owned by ``Benchmark.run``'s frame, never pickled. Standalone callers
    (tests, notebooks) get a fresh session per ``Game.start_game()``.
    """

    record_specs: bool = True
    withhold_answers: bool = False
    trace_root: Path | None = None
    #: Called after every ``execute_action``, so the run's submission file
    #: follows the standing answer instead of trailing it by a sweep. A
    #: ``Game`` has no handle on the ``Benchmark``, and the write has to
    #: happen where the answer changes.
    on_action: Callable[[], None] | None = None

    _resources: dict[Hashable, Any] = field(default_factory=lambda: dict[Hashable, Any](), repr=False)
    _closeables: list[Any] = field(default_factory=lambda: list[Any](), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_or_make(self, key: Hashable, factory: Callable[[], T]) -> T:
        """Return the cached resource for ``key``, building it once."""
        with self._lock:
            if key not in self._resources:
                self._resources[key] = factory()
            return self._resources[key]  # type: ignore[no-any-return]

    def register_closeable(self, resource: Any) -> None:
        """Register something whose ``close()`` runs at :meth:`close`."""
        self._closeables.append(resource)

    def close(self) -> None:
        """Close every registered resource, best-effort."""
        for resource in self._closeables:
            try:
                resource.close()
            except Exception:
                pass
        self._closeables.clear()
        self._resources.clear()


# --- GameRun -----------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One submitted answer with what it cost.

    ``generated_tokens`` / ``uncached_input_tokens`` are what was spent since
    the *previous* answer, so a cumulative sum over the history gives the
    spend at which each answer stood. ``wallclock_seconds`` is already
    cumulative: monotonic seconds since ``Game.start_game()``.
    """

    submission: Submission
    generated_tokens: int
    uncached_input_tokens: int
    wallclock_seconds: float


GameRunState = Literal["not_started", "playing", "won", "submitted", "gave_up", "cancelled", "crashed"]


@dataclass
class GameRun:
    """Per-game run state.

    State machine::

        not_started -> playing     on Game.start_game()
        playing -> cancelled       finish_game with an asyncio cancel outstanding
        playing -> won             finish_game, answered, >= 1 pair correct
        playing -> submitted       finish_game, answered, none correct or unscored
        playing -> gave_up         finish_game, never answered
        playing -> crashed         Benchmark teardown, which stamps the state
                                   and then calls finish_game

    ``execute_action`` does not move the state: it may run many times, and
    which answer counts is not settled until the game finishes.

    The state says **how the game ended**; :attr:`answered` says whether there
    is an answer, and the two are now independent — a game cut off after
    submitting reads ``cancelled`` *and* carries a score. ``cancelled``
    therefore wins over ``won``/``submitted``: being stopped from outside is
    the fact a reader cannot recover any other way, while whether it answered
    is right there in the history.

    ``submitted`` means the solver answered and did not win — either it got
    nothing right, or nothing could be scored. It is separate from
    ``gave_up`` because that one means the solver never answered at all:
    merging them would make give-up statistics meaningless and would label
    every game of an unscored run a give-up.

    Fields:

    - ``game_id`` / ``number_of_levels`` / ``base_actions_per_level`` /
      ``hint``: basic info populated by ``Game.start_game()``.
      ``number_of_levels`` is always 1 — the level is beaten iff at least one
      pair came back right. ``base_actions_per_level`` is always ``None``:
      ARC2 has no action baseline, and diagnostics reads that as "this lane
      steps once, at the end".
    - ``run_id``: globally unique, filename-safe; the join key for traces.
    - ``pass_index``: which pass copy this run is, 0-based.
    - ``state`` / ``levels_completed`` / ``final_score``: progress summary.
      ``final_score`` is a percentage and every finished run has one,
      however it ended: 0.0 means nothing was correct, whether the solver
      answered and got everything wrong or never answered at all, and a run
      stopped by a crash or a cancellation keeps the score its standing
      answer had already reached. ``None`` means the score could not be
      computed — the answer key was withheld — and nothing else. Unset
      until the game finishes; :meth:`current_score` reads a run in play.
    - ``history``: every answer the solver submitted, in order. The last is
      the standing one — what the submission file carries and what
      ``finish_game`` scores. The earlier ones are what let diagnostics plot
      a lane's score against what it had spent when it stood.
    - ``result``: per-pair outcome of the standing answer (every pair wrong
      where there was none), recorded by ``finish_game``;
      :meth:`current_result` reads a run in play.
    - ``spec``: the task played. Kept out of JSON (it is bulky and already
      reproducible from the dataset) and persisted to a sidecar pickle, which
      is what lets diagnostics render the answer sheet.
    - ``final_generated_tokens`` / ``final_uncached_input_tokens``: tokens
      spent on a turn that produced no answer — a give-up or a cancellation.
      Counted in totals, but not as an action.
    """

    game_id: str
    number_of_levels: int = 1
    base_actions_per_level: list[int] | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pass_index: int = 0
    hint: str | None = None
    state: GameRunState = "not_started"
    history: list[ActionRecord] = field(default_factory=lambda: list[ActionRecord]())
    result: GameResult | None = None
    spec: GameSpec | None = field(default=None, repr=False)
    record_specs: bool = field(default=True, repr=False)
    actions_per_level: list[int] = field(default_factory=lambda: [0])
    levels_completed: int = 0
    started_at: datetime | None = None
    started_at_monotonic: float | None = field(default=None, repr=False)
    trace_root: Path | None = field(default=None, repr=False)
    final_score: float | None = None
    solver_note: str | None = None
    solver_analysis_html: str | None = None
    final_generated_tokens: int = 0
    final_uncached_input_tokens: int = 0
    final_wallclock_seconds: float = 0.0
    # Guards finish_game's idempotence. Not `final_score is not None`: an
    # unscored run legitimately finishes with no score, and not `state`
    # either, since the Benchmark teardown pre-stamps `crashed`.
    finished: bool = False

    @property
    def answered(self) -> bool:
        """Whether the solver submitted an answer."""
        return bool(self.history)

    @property
    def standing_submission(self) -> Submission | None:
        """The answer that counts: the last one submitted, or ``None``."""
        return self.history[-1].submission if self.history else None

    def result_against(self, spec: GameSpec) -> GameResult:
        """The standing answer scored against ``spec``: the one rule both the
        finish and a run in play are scored by.

        With no answer every pair is wrong — nothing answered is nothing right,
        a zero rather than an unknown, withheld answer key or not.
        """
        standing = self.standing_submission
        if standing is None:
            return GameResult(pairs=tuple(PairResult(correct=False, matched_attempt=None) for _ in spec.to_answer))
        return score_submission(spec, standing)

    def current_result(self) -> GameResult | None:
        """The run's result as it stands, whatever its state: the recorded
        ``result`` once the game has finished, the standing answer scored
        against the recorded spec before that.

        ``None`` where a run in play has nothing to score against: no spec
        was recorded (``minimal_diagnostics``), or the run was loaded without
        its sidecar.
        """
        if self.finished:
            return self.result
        if self.spec is None:
            return None
        return self.result_against(self.spec)

    def current_score(self) -> float | None:
        """The percentage of pairs right as the run stands, whatever its
        state; 0.0 before any answer. ``None`` only where it cannot be
        computed: the answer key was withheld, or :meth:`current_result` has
        nothing to score against.

        ``final_score`` first: for a finished run it is the score of record,
        and a finished give-up read from ``benchmark.json`` may carry no
        ``result`` to fall back on.
        """
        if self.final_score is not None:
            return self.final_score
        result = self.current_result()
        return None if result is None else result.score

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize for ``benchmark.json``. ``spec`` is dropped — it goes to
        the sidecar pickle."""
        return {
            "game_id": self.game_id,
            "number_of_levels": self.number_of_levels,
            "base_actions_per_level": self.base_actions_per_level,
            "run_id": self.run_id,
            "pass_index": self.pass_index,
            "hint": self.hint,
            "state": self.state,
            "history": [
                {
                    "submission": rec.submission.to_json_list(),
                    "generated_tokens": rec.generated_tokens,
                    "uncached_input_tokens": rec.uncached_input_tokens,
                    "wallclock_seconds": rec.wallclock_seconds,
                }
                for rec in self.history
            ],
            "result": self.result.to_json_dict() if self.result is not None else None,
            "record_specs": self.record_specs,
            "actions_per_level": list(self.actions_per_level),
            "levels_completed": self.levels_completed,
            "final_score": self.final_score,
            "solver_note": self.solver_note,
            "solver_analysis_html": self.solver_analysis_html,
            "final_generated_tokens": self.final_generated_tokens,
            "final_uncached_input_tokens": self.final_uncached_input_tokens,
            "final_wallclock_seconds": self.final_wallclock_seconds,
            "finished": self.finished,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> GameRun:
        """Inverse of :meth:`to_json_dict`; ``spec`` stays ``None``."""
        started_at_raw = d.get("started_at")
        run = cls(
            game_id=d["game_id"],
            number_of_levels=d.get("number_of_levels", 1),
            base_actions_per_level=d.get("base_actions_per_level"),
            run_id=d.get("run_id") or uuid.uuid4().hex,
            pass_index=d.get("pass_index", 0),
            hint=d.get("hint"),
            state=d["state"],
            result=GameResult.from_json_dict(d["result"]) if d.get("result") else None,
            record_specs=bool(d.get("record_specs", True)),
            actions_per_level=list(d.get("actions_per_level") or [0]),
            levels_completed=d.get("levels_completed", 0),
            final_score=d.get("final_score"),
            solver_note=d.get("solver_note"),
            solver_analysis_html=d.get("solver_analysis_html"),
            final_generated_tokens=d.get("final_generated_tokens", 0),
            final_uncached_input_tokens=d.get("final_uncached_input_tokens", 0),
            final_wallclock_seconds=d.get("final_wallclock_seconds", 0.0),
            finished=bool(d.get("finished", False)),
            started_at=datetime.fromisoformat(started_at_raw) if started_at_raw else None,
        )
        for entry in d["history"]:
            run.history.append(
                ActionRecord(
                    submission=Submission(
                        attempts=tuple(
                            tuple(Grid.from_lists(group[f"attempt_{i + 1}"]) for i in range(N_ATTEMPTS))
                            for group in entry["submission"]
                        )
                    ),
                    generated_tokens=entry["generated_tokens"],
                    uncached_input_tokens=entry["uncached_input_tokens"],
                    wallclock_seconds=entry["wallclock_seconds"],
                )
            )
        return run

    def register_trace(self, trace: taaf.trace.Trace) -> Path:
        """Validate and persist one trace for this run. The trace's
        ``run_id`` must equal this run's."""
        from taaf import trace as trace_mod

        assert self.trace_root is not None, (
            "register_trace requires a trace_root; it is set by Game.start_game from RunSession.trace_root"
        )
        trace_mod.validate_trace(trace, self)
        return trace_mod.write_trace(trace, self, self.trace_root)


# --- Game --------------------------------------------------------------------


@dataclass
class Game:
    """Abstract game: one ARC task.

    Subclasses implement ``_start_game``; ``_execute_action`` and
    ``_finish_game`` are optional hooks. The public methods here own the
    ``GameRun`` bookkeeping, submission validation, token logging, scoring
    and the state machine.

    Pickle / deepcopy contract: before ``start_game`` the default walk over
    ``__dict__`` runs as usual, which is what lets ``Benchmark`` copy a game
    per pass. Afterwards ``__getstate__`` raises, and ``__deepcopy__`` raises
    unless ``allow_deepcopy=True``.
    """

    allow_deepcopy: bool = field(default=False, kw_only=True)
    game_id: str = field(default="", init=False)
    number_of_levels: int = field(default=1, init=False)
    base_actions_per_level: list[int] | None = field(default=None, init=False)
    hint: str | None = field(default=None, init=False)
    game_run: GameRun | None = field(default=None, init=False, repr=False)
    _spec: GameSpec | None = field(default=None, init=False, repr=False)
    _on_action: Callable[[], None] | None = field(default=None, init=False, repr=False)

    @property
    def spec(self) -> GameSpec:
        """The task being played. Available after ``start_game``."""
        assert self._spec is not None, "spec requires start_game()"
        return self._spec

    def query(self) -> GameView:
        """What the solver may see. Safe to call at any point while playing;
        it never carries a withheld answer."""
        assert self.game_run is not None, "Call start_game() first"
        return self.spec.view()

    def start_game(self, session: RunSession | None = None) -> GameView:
        """Start the game and return the solver-visible view."""
        assert self.game_run is None, "start_game() already called"
        if session is None:
            session = RunSession()
        self._on_action = session.on_action
        spec = self._start_game(session)
        if session.withhold_answers:
            spec = spec.withhold_answers()
        try:
            if not spec.game_id:
                raise ValueError("_start_game() must return a spec with a game_id")
            self._spec = spec
            self.game_id = spec.game_id
        except BaseException:
            # The subclass may have opened resources but game_run does not
            # exist yet, so finish_game() can never run. Give it the hook.
            self._finish_game()
            raise
        self.game_run = GameRun(
            game_id=self.game_id,
            number_of_levels=self.number_of_levels,
            base_actions_per_level=self.base_actions_per_level,
            hint=self.hint,
            state="playing",
            spec=spec if session.record_specs else None,
            record_specs=session.record_specs,
            started_at=datetime.now(),
            started_at_monotonic=time.monotonic(),
            trace_root=session.trace_root,
        )
        return spec.view()

    def execute_action(
        self,
        submission: Submission,
        generated_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> None:
        """Submit an answer, replacing any earlier one.

        May be called any number of times while the game is playing. Each call
        validates the submission against the task's shape and records what it
        cost; the last one to land is what ``finish_game`` scores and what the
        submission file carries.

        Raises ``ValueError`` on a malformed submission — loudly, rather than
        letting a shape bug read as a wrong answer. Returns nothing: handing
        back per-pair correctness for an answer the solver can still replace
        would be a scoring oracle, which is why the recorded score waits for
        ``finish_game``.
        """
        assert self.game_run is not None, "Call start_game() first"
        assert self.game_run.state == "playing", f"Cannot execute_action in state {self.game_run.state!r}"
        to_answer = self.spec.to_answer
        if len(submission.attempts) != len(to_answer):
            raise ValueError(
                f"submission covers {len(submission.attempts)} pairs; the task has {len(to_answer)} to answer"
            )

        assert self.game_run.started_at_monotonic is not None
        elapsed = time.monotonic() - self.game_run.started_at_monotonic
        self._execute_action(submission)

        self.game_run.history.append(
            ActionRecord(
                submission=submission,
                generated_tokens=generated_tokens,
                uncached_input_tokens=uncached_input_tokens,
                wallclock_seconds=elapsed,
            )
        )
        self.game_run.actions_per_level[0] += 1
        if self._on_action is not None:
            self._on_action()

    def finish_game(self, generated_tokens: int = 0, uncached_input_tokens: int = 0) -> None:
        """Score the standing answer and mark the game finished. Idempotent.

        This is where the score is recorded, on whatever answer stood when the
        game ended — a normal finish, a give-up, or a cancellation. A game that
        never answered scores zero.

        The state records how the game ended: ``cancelled`` when an asyncio
        cancel is outstanding (it wins even over a correct answer — being
        stopped from outside is not recoverable from anything else), else
        ``won`` / ``submitted`` if it answered, else ``gave_up``. A state
        already moved off ``playing`` is left alone, which is how the
        ``Benchmark`` teardown stamps ``crashed`` and still gets a score.

        The token arguments carry what was spent since the last answer, on a
        turn that produced none; they count toward totals but not as an action.
        """
        assert self.game_run is not None, "finish_game() before start_game()"
        run = self.game_run
        if run.finished:
            return
        run.finished = True
        run.final_generated_tokens = generated_tokens
        run.final_uncached_input_tokens = uncached_input_tokens
        if run.started_at_monotonic is not None:
            run.final_wallclock_seconds = time.monotonic() - run.started_at_monotonic
        # Against `self.spec`: the key the game was played against, answers
        # withheld where the run withholds them. Not what a subclass was
        # constructed with, and not the run's copy, which minimal_diagnostics
        # does not record.
        run.result = run.result_against(self.spec)
        if run.result.scored and run.result.n_correct > 0:
            run.levels_completed = 1
        self._finish_game()
        if run.state == "playing":
            cancelling = False
            try:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    cancelling = True
            except RuntimeError:
                cancelling = False  # no running loop: a sync caller gave up
            if cancelling:
                run.state = "cancelled"
            elif not run.answered:
                run.state = "gave_up"
            else:
                run.state = "won" if run.levels_completed else "submitted"
        # Every finished run carries a score, whatever ended it: what it got
        # right is knowable from its standing answer alone. A run cut off by a
        # crash or a cancellation keeps what it had already reached, which can
        # be above zero. The sole exception is a withheld answer key, where the
        # score cannot be computed at all and `result.score` is itself None.
        run.final_score = run.result.score
        self._print_finish_line(run)

    @staticmethod
    def _print_finish_line(run: GameRun) -> None:
        """One auditable line per finished game, the same fields the
        diagnostics per-pass row carries."""
        tokens = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
        answers = f" answers={len(run.history)}" if len(run.history) > 1 else ""
        if run.result is None or not run.answered:
            detail = "answered=no"
        elif run.result.scored:
            detail = f"correct={run.result.n_correct}/{len(run.result.pairs)}"
        else:
            detail = f"correct=?/{len(run.result.pairs)} (unscored)"
        score = "?" if run.final_score is None else f"{run.final_score:.2f}"
        note_suffix = f' note="{run.solver_note}"' if run.solver_note else ""
        print(
            f"[finished] {run.game_id} state={run.state} score={score} {detail} tokens={tokens}{answers}{note_suffix}"
        )

    # --- pickle / deepcopy contract -----------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        if self.game_run is not None:
            raise RuntimeError(f"Cannot pickle {type(self).__name__} after start_game().")
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self, memo: dict[int, Any]) -> Game:
        if self.game_run is not None and not self.allow_deepcopy:
            raise RuntimeError(f"Cannot deepcopy {type(self).__name__} after start_game() unless allow_deepcopy=True.")
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new

    # --- subclass hooks -----------------------------------------------------

    def _start_game(self, session: RunSession) -> GameSpec:
        """Produce the task to play. ``session`` carries run-wide settings and
        any shared resources a subclass needs."""
        raise NotImplementedError("Subclasses must implement _start_game")

    def _execute_action(self, submission: Submission) -> None:
        """Optional per-answer hook. Default: no-op."""

    def _finish_game(self) -> None:
        """Optional teardown. Default: no-op."""


def score_submission(spec: GameSpec, submission: Submission) -> GameResult:
    """Score a submission against a task, taking each pair independently.

    A pair is correct if *any* attempt matches — the attempt index carries no
    meaning across pairs, which is the competition's own rule.
    """
    results: list[PairResult] = []
    for pair, attempts in zip(spec.to_answer, submission.attempts, strict=True):
        if pair.answer is None:
            results.append(PairResult(correct=None, matched_attempt=None))
            continue
        matched = next((i for i, g in enumerate(attempts) if g == pair.answer), None)
        results.append(PairResult(correct=matched is not None, matched_attempt=matched))
    return GameResult(pairs=tuple(results))


@dataclass
class ArcGame(Game):
    """A game over a known :class:`GameSpec` — the only concrete ``Game`` in
    normal use.

    The spec is held from construction (it is small, immutable and picklable),
    so a benchmark's game list fully describes what will be played.
    """

    spec_data: GameSpec

    def _start_game(self, session: RunSession) -> GameSpec:
        """The spec is already in hand; nothing to open."""
        del session
        return self.spec_data


def games_from_specs(specs: Sequence[GameSpec]) -> list[Game]:
    """Wrap task specs as playable games — the one step between a dataset and
    a benchmark."""
    return [ArcGame(spec_data=spec) for spec in specs]
