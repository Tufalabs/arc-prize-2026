"""Core game abstractions (R11.01–R11.05). The R11.06 example subclass
lives in ``taaf.game_examples``.

This module also owns the arcengine enum pickle fix (collocated with the
only module that imports arcengine) and the ARC color palette.
"""

from __future__ import annotations

import asyncio
import copy
import threading
import time
import uuid
import warnings
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import arcengine
import numpy as np
import numpy.typing as npt

import taaf.support

#: How long ``finish_game`` waits for the action lock before finishing without
#: it: long enough for any RESET that will return, short enough that a stuck
#: one cannot hold teardown for longer than this.
FINISH_LOCK_TIMEOUT_S = 30.0

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.colors import ListedColormap

    import taaf.trace

    # Served by the module ``__getattr__`` below; declared so callers
    # (``taaf.diagnostics``) type-check against the real colormap.
    ARC_CMAP: ListedColormap
else:
    # Runtime stand-ins for the two matplotlib types the annotations name,
    # so ``typing.get_type_hints`` on this module and its methods resolves
    # without importing matplotlib — importing it is what this module
    # avoids, and a name that exists only for the type checker turns every
    # runtime consumer of the annotations (typeguard, schema and doc
    # tooling) into a NameError.
    Axes = Any
    ListedColormap = Any

T = TypeVar("T")

# arcengine's GameAction / GameState enums use composite keys in
# ``_value2member_map_``, so the default ``Enum(value)`` reconstruction
# fails on unpickle. Reduce by name instead.


def _reconstruct_enum(cls: type, name: str) -> Any:
    return cls[name]  # type: ignore[index]


def _enum_reduce(self: Any, protocol: int = 0) -> Any:
    return (_reconstruct_enum, (self.__class__, self._name_))


arcengine.enums.GameAction.__reduce_ex__ = _enum_reduce  # type: ignore[attr-defined]
arcengine.enums.GameState.__reduce_ex__ = _enum_reduce  # type: ignore[attr-defined]


# --- ARC palette -------------------------------------------------------------

ARC_COLORS: dict[int, tuple[float, float, float]] = {
    0: (1.0, 1.0, 1.0),
    1: (0.8, 0.8, 0.8),
    2: (0.6, 0.6, 0.6),
    3: (0.4, 0.4, 0.4),
    4: (0.2, 0.2, 0.2),
    5: (0.0, 0.0, 0.0),
    6: (0.898, 0.227, 0.639),
    7: (1.0, 0.482, 0.8),
    8: (0.976, 0.235, 0.192),
    9: (0.118, 0.576, 1.0),
    10: (0.533, 0.847, 0.945),
    11: (1.0, 0.863, 0.0),
    12: (1.0, 0.522, 0.106),
    13: (0.573, 0.071, 0.192),
    14: (0.310, 0.800, 0.188),
    15: (0.639, 0.337, 0.839),
}


def _arc_cmap() -> ListedColormap:
    """The palette as a matplotlib colormap, built on first use and then
    bound as the ordinary module global ``ARC_CMAP``.

    ``ARC_COLORS`` is the palette itself and costs nothing; the colormap
    is the matplotlib-shaped view of it that only a plotting caller
    wants, and importing matplotlib for it costs ~30 MB of private heap
    in every process that imports this module. Once built it is a
    normal global, so ``dir()``, star import and repeated access all see
    the same object.
    """
    cmap: ListedColormap | None = globals().get("ARC_CMAP")
    if cmap is None:
        import matplotlib.colors

        cmap = matplotlib.colors.ListedColormap([ARC_COLORS[i] for i in range(16)])
        globals()["ARC_CMAP"] = cmap
    return cmap


def __getattr__(name: str) -> ListedColormap:
    """``ARC_CMAP`` on first attribute access (PEP 562); bound after that."""
    if name == "ARC_CMAP":
        return _arc_cmap()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """The public surface whether or not ``ARC_CMAP`` has been built yet."""
    return sorted({*globals(), "ARC_CMAP"})


# The module's own API. Listed so ``from taaf.game import *`` reaches
# ``ARC_CMAP`` through ``__getattr__`` — a star import consults ``__all__``
# name by name — rather than only what happens to be bound already.
__all__ = [
    "ARC_CMAP",
    "ARC_COLORS",
    "ActionRecord",
    "CompressedStates",
    "Frame",
    "Game",
    "GameRun",
    "GameRunState",
    "GameState",
    "RunSession",
]


# --- Frame -------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Frame:
    """A single visible frame as a 2D int8 numpy array, values 0-15 (ARC palette)."""

    data: npt.NDArray[np.int8]

    def __post_init__(self) -> None:
        assert self.data.dtype == np.int8, f"Frame data must be int8, got {self.data.dtype}"
        assert self.data.ndim == 2, f"Frame data must be 2D, got shape {self.data.shape}"
        assert np.all((self.data >= 0) & (self.data <= 15)), "Frame data must contain values 0-15 only"
        # arcengine keeps writing into the buffer it hands back, so a Frame
        # built on it would change under any holder. Note np.asarray would
        # not do: on an already-int8 array it returns the same object.
        object.__setattr__(self, "data", np.array(self.data, dtype=np.int8, copy=True))

    def __eq__(self, other: object) -> bool:
        # Hand-written because the generated __eq__ compares the arrays with
        # ==, which yields an array rather than a bool.
        if not isinstance(other, Frame):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def draw(self, ax: Axes | None = None) -> Axes:
        """Draw this frame on ``ax``, or create a fresh axes if None.

        Returns the axes for chaining/inspection.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            h, w = self.data.shape
            scale = 4.0 / max(h, w)
            _fig, ax = plt.subplots(figsize=(w * scale, h * scale))
        ax.imshow(self.data, cmap=_arc_cmap(), vmin=0, vmax=15, interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax


# --- RunSession --------------------------------------------------------------


@dataclass
class RunSession:
    """Per-``Benchmark.run()`` resource cache. ``Game`` subclasses with
    shared resources (notably ``GameAPI``'s ``arc_agi.Arcade``) call
    ``get_or_make`` from ``_start_game``. Owned by ``Benchmark.run``'s
    frame, never pickled. Standalone callers (tests, notebooks) get a
    fresh session per ``Game.start_game()`` call.

    The lock is forward-compat insurance — ``Benchmark.run`` calls
    ``start_game`` serially today.
    """

    record_intermediate_states: bool = True
    # Effective trace root for this run (R5.02). Threaded onto each GameRun at
    # start_game so solvers' register_trace knows where to write.
    trace_root: Path | None = None

    _resources: dict[Hashable, Any] = field(default_factory=lambda: dict[Hashable, Any](), repr=False)
    _closeables: list[Any] = field(default_factory=lambda: list[Any](), repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Separate from ``_lock`` so a factory running inside ``get_or_make``
    # (which holds ``_lock``) can call ``register_closeable`` without
    # deadlocking — GameAPI's competition-sim build does exactly that.
    _closeables_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def get_or_make(self, key: Hashable, factory: Callable[[], T]) -> T:
        with self._lock:
            if key not in self._resources:
                self._resources[key] = factory()
            return self._resources[key]  # type: ignore[no-any-return]

    def register_closeable(self, resource: Any) -> None:
        """Register a resource exposing ``stop()`` to be torn down by
        ``close()`` at end of run (e.g. a local competition-arcade server)."""
        with self._closeables_lock:
            self._closeables.append(resource)

    def close(self) -> None:
        """Stop every registered closeable, most-recent first. Called by
        ``Benchmark.run`` at teardown, so it must not raise."""
        with self._closeables_lock:
            closeables = self._closeables[::-1]
            self._closeables = []
        for resource in closeables:
            try:
                resource.stop()
            except Exception as exc:  # noqa: BLE001 — teardown must not raise
                warnings.warn(f"RunSession.close: failed to stop {resource!r}: {exc!r}", stacklevel=2)


# --- GameState (R11.01) ------------------------------------------------------


@dataclass(frozen=True)
class GameState:
    """Wraps ``arcengine.FrameDataRaw`` with helper properties (R11.01).

    Fields:

    - ``raw``: the underlying arcengine frame data.
    - ``previous_action``: the ``ActionInput`` that led here, or ``None``
      for the initial state.
    - ``just_won_level``: True if this state crossed a level boundary.
    """

    raw: arcengine.FrameDataRaw
    previous_action: arcengine.ActionInput | None = None
    just_won_level: bool = False

    @property
    def frame(self) -> Frame:
        """The final visible frame (last in the raw frame list)."""
        return Frame(data=self.raw.frame[-1])

    @property
    def animation_frames(self) -> list[Frame]:
        """All frames except the final visible one."""
        return [Frame(data=f) for f in self.raw.frame[:-1]]

    @property
    def all_frames(self) -> list[Frame]:
        """All frames (animation + final)."""
        return [Frame(data=f) for f in self.raw.frame]

    @property
    def available_actions(self) -> list[int]:
        """Legal action ids, with RESET (0) always present."""
        raw = list(self.raw.available_actions)
        if 0 in raw:
            return raw
        return [0, *raw]

    @property
    def levels_completed(self) -> int:
        return self.raw.levels_completed

    @property
    def game_over(self) -> bool:
        """True if the game ended, won or lost (R11.01).

        arcengine's ``GAME_OVER`` is a lost attempt — recoverable via
        ``RESET`` — so unlike ``won`` it is not a terminal state for the run.
        """
        return self.raw.state in (arcengine.GameState.WIN, arcengine.GameState.GAME_OVER)

    @property
    def won(self) -> bool:
        """True if the game ended in a win."""
        return self.raw.state == arcengine.GameState.WIN


# --- ActionRecord ------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One executed action with token + wallclock cost (R11.02).

    Fields:

    - ``action``: the executed ``ActionInput``.
    - ``generated_tokens`` / ``uncached_input_tokens``: solver-reported
      token counts, or 0 if the solver doesn't track them (R11.03).
    - ``wallclock_seconds``: monotonic-clock seconds since
      ``Game.start_game()``.
    """

    action: arcengine.ActionInput
    generated_tokens: int
    uncached_input_tokens: int
    wallclock_seconds: float


class CompressedStates(taaf.support.CompressedList[GameState]):
    """``GameRun.intermediate_states``: every step's ``GameState``, held
    compressed (``taaf.support.CompressedList``). A long run keeps hundreds
    of thousands of them in the host process, for diagnostics alone. No
    live tail: the engine writes into the frame buffers it hands back, so
    every recorded state has to be the snapshot taken at ``append``.
    """


# --- GameRun (R11.02) --------------------------------------------------------

GameRunState = Literal["not_started", "playing", "won", "gave_up", "cancelled", "crashed"]


@dataclass
class GameRun:
    """Per-game run state (R11.02).

    State machine::

        not_started → playing               on Game.start_game()
        playing     → won                   on execute_action returning won=True
        playing     → cancelled             on finish_game with asyncio cancel outstanding
        playing     → gave_up               on finish_game otherwise
        playing     → crashed               Benchmark teardown, finish_game never called

    Fields:

    - ``game_id`` / ``number_of_levels`` / ``base_actions_per_level`` /
      ``hint``: basic info populated by ``Game.start_game()``.
      ``base_actions_per_level`` is ``None`` when the engine hides
      baselines (e.g. submission mode), and carries ``Game``'s
      ``baseline_multiplier`` (R11.15) already applied.
    - ``baseline_multiplier``: the factor those baselines were scaled by,
      recorded so a reader knows what this run's score was computed
      against. 1.0 unless the game asked for another. Keyword-only, so the
      fields around it keep their positional places.
    - ``run_id``: globally unique, filename-safe id (R5.11). The trace
      identity and join key back to this run from ``benchmark.json``;
      assigned to every run whether or not a trace is recorded.
    - ``pass_index``: which pass copy of the game this run is (R2.2),
      0-based; 0 for a game played outside a multi-pass benchmark.
      Assigned by ``Benchmark.run``, which owns the layout — a game copy
      cannot know it. Solvers may read it to tell the copies apart, and
      diagnostics groups passes by it.
    - ``state`` / ``levels_completed`` / ``final_score``: progress
      summary. ``final_score`` is set by ``finish_game()``.
    - ``history`` / ``actions_per_level``: per-action record and
      per-level counter. Invariant: ``sum(actions_per_level) == len(history)``.
    - ``final_generated_tokens`` / ``final_uncached_input_tokens``: tokens
      the solver spent on its final turn *without* producing a move — the
      give-up / cancellation decision — reported via ``finish_game(...)``
      (R11.03). 0 when the solver doesn't report them. Counted in token
      totals but not as an action (``actions_per_level`` is unchanged).
    - ``final_wallclock_seconds``: monotonic-clock seconds since
      ``start_game()`` at the moment ``finish_game()`` ran — the same
      reference and duration semantics as ``ActionRecord.wallclock_seconds``
      (so it sits on the diagnostics wallclock axis, and survives JSON as a
      duration). Lets the no-move ``final_*_tokens`` be placed at the time
      they were actually spent, after the last move.
    - ``intermediate_states``: full per-step ``GameState`` sequence, held
      compressed (``CompressedStates``). Kept out of JSON per R11.02;
      persisted to ``intermediate_states.pkl`` sidecar.
    - ``record_intermediate_states``: when false, only history and score
      bookkeeping are kept. Kaggle submission uses this to avoid storing
      frames in memory.
    - ``keepalive_resets``: RESETs sent to the engine by the benchmark's
      keepalive (R2.63) rather than by the solver. Deliberately not in
      ``history`` or ``actions_per_level``: each landed on a level no action
      had touched, where it changes nothing the solver sees. The engine
      counts them all the same, which is why R11.12 skips a run that has
      any.
    - ``solver_note`` / ``solver_analysis_html``: solver-provided
      diagnostics. Either can be re-set during play; the periodic save
      loop picks up the latest values. ``solver_analysis_html`` is a
      path **relative to job_dir**, linked from the movie wrapper page.
    - ``started_at`` / ``started_at_monotonic``: wall-clock start stamps,
      both populated at ``Game.start_game()``. ``started_at`` is a
      local-naive ``datetime`` that survives JSON — diagnostics offsets
      it against ``Benchmark.start_time`` to place each action on a
      true job-wallclock axis (games queued behind a pool semaphore
      don't collapse to benchmark t=0). ``started_at_monotonic`` is a
      ``time.monotonic()`` reading used purely as the reference for
      ``ActionRecord.wallclock_seconds``; immune to NTP / DST jumps
      mid-run, but not persisted (monotonic epochs don't survive a
      process boundary).
    """

    game_id: str
    number_of_levels: int
    base_actions_per_level: list[int] | None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    pass_index: int = 0
    baseline_multiplier: float = field(default=1.0, kw_only=True)
    hint: str | None = None
    state: GameRunState = "not_started"
    history: list[ActionRecord] = field(default_factory=lambda: list[ActionRecord]())
    intermediate_states: CompressedStates = field(default_factory=CompressedStates, repr=False)
    record_intermediate_states: bool = field(default=True, repr=False)
    actions_per_level: list[int] = field(default_factory=lambda: list[int]())
    levels_completed: int = 0
    started_at: datetime | None = None
    started_at_monotonic: float | None = field(default=None, repr=False)
    # Where register_trace writes this run's trace (R5.02). In-memory only,
    # copied from RunSession at start_game; not persisted to JSON.
    trace_root: Path | None = field(default=None, repr=False)
    final_score: float | None = None
    solver_note: str | None = None
    solver_analysis_html: str | None = None
    final_generated_tokens: int = 0
    final_uncached_input_tokens: int = 0
    final_wallclock_seconds: float = 0.0
    keepalive_resets: int = 0

    def __post_init__(self) -> None:
        if not self.actions_per_level:
            self.actions_per_level = [0] * self.number_of_levels

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict. ``intermediate_states`` is
        dropped per R11.02 (persisted to a sidecar pickle instead)."""
        return {
            "game_id": self.game_id,
            "number_of_levels": self.number_of_levels,
            "base_actions_per_level": self.base_actions_per_level,
            "run_id": self.run_id,
            "pass_index": self.pass_index,
            "baseline_multiplier": self.baseline_multiplier,
            "hint": self.hint,
            "state": self.state,
            "history": [
                {
                    "action": {"id": rec.action.id.name, "data": dict(rec.action.data)},
                    "generated_tokens": rec.generated_tokens,
                    "uncached_input_tokens": rec.uncached_input_tokens,
                    "wallclock_seconds": rec.wallclock_seconds,
                }
                for rec in self.history
            ],
            "record_intermediate_states": self.record_intermediate_states,
            "actions_per_level": list(self.actions_per_level),
            "levels_completed": self.levels_completed,
            "final_score": self.final_score,
            "solver_note": self.solver_note,
            "solver_analysis_html": self.solver_analysis_html,
            "final_generated_tokens": self.final_generated_tokens,
            "final_uncached_input_tokens": self.final_uncached_input_tokens,
            "final_wallclock_seconds": self.final_wallclock_seconds,
            "keepalive_resets": self.keepalive_resets,
            "started_at": self.started_at.isoformat() if self.started_at is not None else None,
        }

    @classmethod
    def from_json_dict(cls, d: dict[str, Any]) -> GameRun:
        started_at_raw = d.get("started_at")
        run = cls(
            game_id=d["game_id"],
            number_of_levels=d["number_of_levels"],
            base_actions_per_level=d["base_actions_per_level"],
            run_id=d.get("run_id") or uuid.uuid4().hex,
            # Absent in runs recorded before pass_index existed; Benchmark.from_json
            # backfills those from the list layout, which is all they can offer.
            pass_index=d.get("pass_index", 0),
            # Absent in runs recorded before the multiplier existed, which by
            # definition played at the baselines the engine reported.
            baseline_multiplier=float(d.get("baseline_multiplier", 1.0)),
            hint=d.get("hint"),
            state=d["state"],
            actions_per_level=list(d.get("actions_per_level") or []),
            record_intermediate_states=bool(d.get("record_intermediate_states", True)),
            levels_completed=d.get("levels_completed", 0),
            final_score=d.get("final_score"),
            solver_note=d.get("solver_note"),
            solver_analysis_html=d.get("solver_analysis_html"),
            final_generated_tokens=d.get("final_generated_tokens", 0),
            final_uncached_input_tokens=d.get("final_uncached_input_tokens", 0),
            final_wallclock_seconds=d.get("final_wallclock_seconds", 0.0),
            keepalive_resets=d.get("keepalive_resets", 0),
            started_at=datetime.fromisoformat(started_at_raw) if started_at_raw else None,
        )
        for entry in d["history"]:
            action = arcengine.ActionInput(
                id=arcengine.GameAction[entry["action"]["id"]],
                data=dict(entry["action"]["data"]),
            )
            run.history.append(
                ActionRecord(
                    action=action,
                    generated_tokens=entry["generated_tokens"],
                    uncached_input_tokens=entry["uncached_input_tokens"],
                    wallclock_seconds=entry["wallclock_seconds"],
                )
            )
        return run

    def register_trace(self, trace: taaf.trace.Trace) -> Path:
        """Validate and persist one trace for this run (R5.61). The trace's
        ``run_id`` must equal this run's. Writes ``<trace_root>/<run_id>/``
        with ``_SUCCESS`` last and returns the folder; raises (no ``_SUCCESS``)
        on failure. ``trace_root`` is set by ``Game.start_game`` from the
        ``RunSession`` — registering requires a benchmark/standalone run that
        supplied one.
        """
        from taaf import trace as trace_mod  # noqa: PLC0415 — avoids import cycle

        assert self.trace_root is not None, (
            "register_trace requires a trace_root; it is set by Game.start_game from RunSession.trace_root"
        )
        trace_mod.validate_trace(trace, self)
        return trace_mod.write_trace(trace, self, self.trace_root)

    def _compute_final_score(self) -> float:
        """ARC-AGI3 score formula. Mirrors
        ``arc_agi.scorecard.EnvironmentScoreCalculator`` (v0.9.8).

        Per level: ``min(115, (baseline / actions)² × 100)`` if completed
        with ≥ 1 action, else 0. Weights are 1-indexed (level 0 → 1).
        Final score is ``total_score / total_weights`` capped at
        ``max_weights / total_weights × 100``, where ``max_weights``
        sums only levels that scored > 0.
        """
        if self.base_actions_per_level is None or self.number_of_levels == 0:
            return 0.0
        total_score = 0.0
        total_weights = 0
        max_weights = 0
        for level_idx in range(self.number_of_levels):
            weight = level_idx + 1
            total_weights += weight
            completed = level_idx < self.levels_completed
            actions = self.actions_per_level[level_idx] if level_idx < len(self.actions_per_level) else 0
            baseline = self.base_actions_per_level[level_idx]
            if completed and actions > 0:
                level_score = min(115.0, (baseline / actions) ** 2 * 100)
            else:
                level_score = 0.0
            if level_score > 0:
                max_weights += weight
            total_score += level_score * weight
        if total_weights == 0:
            return 0.0
        score = total_score / total_weights
        max_score = max_weights / total_weights * 100
        return min(score, max_score)


# --- Game (R11.03 / R11.04 / R11.05) -----------------------------------------


@dataclass
class Game:
    """Abstract game interface (R11.03 / R11.04). Subclasses implement
    ``_start_game`` / ``_execute_action`` / ``_finish_game``; the public
    methods on this class handle ``GameRun`` bookkeeping, action
    validation, token logging, and the state machine.

    R11.05 pickle / deepcopy contract: pre-``start_game`` the default
    walk over ``__dict__`` runs as usual. Post-``start_game``,
    ``__getstate__`` raises and ``__deepcopy__`` raises unless
    ``allow_deepcopy=True`` is set on the instance.

    Fields:

    - ``allow_deepcopy``: opt-in for post-start deepcopy. Default
      ``False`` so subclasses that *can* support it still require an
      explicit instance-level toggle.
    - ``start_at_level`` / ``stop_at_level``: an optional level window to
      play, both **0-indexed** (same units as ``levels_completed`` /
      arcengine's ``level_index``) and **half-open** ``[start, stop)``:
        - ``start_at_level=N`` begins play on level index ``N`` (indices
          ``0..N-1`` are skipped — the subclass fast-forwards inside
          ``_start_game``; honoring this is mandatory, the base verifies
          the initial state reports ``levels_completed == N``).
        - ``stop_at_level=N`` treats the game as won the instant
          ``levels_completed`` reaches ``N`` (i.e. right after level index
          ``N-1`` is completed; level index ``N`` is *not* played). This is
          enforced generically here in ``execute_action``.
      So to play *only* level index ``k``: ``start_at_level=k,
      stop_at_level=k+1``. Scoring is untouched — skipped and not-reached
      levels simply score 0, which caps the achievable score below 100%.
      ``None`` (default) means no clamp on that end.
    - ``baseline_multiplier``: scales the baseline actions the subclass
      reports (R11.15), for games whose baselines were authored on a
      different footing than the rest of a benchmark's — an author's
      optimal solve against human play, say. ``start_game`` applies it to
      ``base_actions_per_level``, so every score derived from the run
      reads the scaled numbers; the run records the factor. A factor
      other than 1 also skips ``GameAPI``'s R11.12 reconciliation, which
      compares against an engine scoring by its own baselines.
    - ``game_id`` / ``number_of_levels`` / ``base_actions_per_level`` /
      ``hint`` / ``grid_size``: populated by ``_start_game()``.
      ``grid_size`` is ``(width, height)`` to match the (x, y) order
      of CLICK / ACTION6 data. ``base_actions_per_level`` is ``None``
      when the engine hides baselines (submission mode).
    - ``game_run``: the live ``GameRun``, or ``None`` before
      ``start_game``.
    """

    allow_deepcopy: bool = field(default=False, kw_only=True)
    baseline_multiplier: float = field(default=1.0, kw_only=True)
    start_at_level: int | None = field(default=None, kw_only=True)
    stop_at_level: int | None = field(default=None, kw_only=True)
    game_id: str = field(default="", init=False)
    number_of_levels: int = field(default=0, init=False)
    base_actions_per_level: list[int] | None = field(default=None, init=False)
    hint: str | None = field(default=None, init=False)
    grid_size: tuple[int, int] = field(default=(64, 64), init=False)
    game_run: GameRun | None = field(default=None, init=False, repr=False)
    _current_state: GameState | None = field(default=None, init=False, repr=False)
    # Serialises this game's moves with the benchmark keepalive (R2.63), which
    # moves from its own thread. Reentrant, so a subclass hook calling back
    # into the game cannot deadlock it. Created at start; a copy gets its own.
    _action_lock: threading.RLock | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def current_state(self) -> GameState:
        assert self._current_state is not None, "current_state requires start_game()"
        return self._current_state

    def start_game(self, session: RunSession | None = None) -> GameState:
        """Start the game, populate basic info, return initial state (R11.03).

        ``session`` carries shared per-run resources for subclasses that
        need them. Standalone callers can omit it; a fresh session is
        constructed on the spot.
        """
        assert self.game_run is None, "start_game() already called"
        if self.baseline_multiplier <= 0:
            raise ValueError(f"baseline_multiplier must be positive; got {self.baseline_multiplier}")
        if session is None:
            session = RunSession()
        initial = self._start_game(session)
        try:
            # Verify the subclass populated the R11.03 basic info.
            # ``base_actions_per_level`` may legitimately be None (R11.02:
            # engine hides baselines in submission mode).
            assert self.game_id, f"_start_game() must populate game_id (got {self.game_id!r})"
            assert self.number_of_levels > 0, (
                f"_start_game() must populate number_of_levels > 0 (got {self.number_of_levels})"
            )
            if self.base_actions_per_level is not None:
                assert len(self.base_actions_per_level) == self.number_of_levels, (
                    f"base_actions_per_level has {len(self.base_actions_per_level)} entries; "
                    f"number_of_levels is {self.number_of_levels}"
                )
            # Scaled once, here, rather than at each score: a reading that
            # applied the factor itself would be one of several, and the
            # others would disagree with it. A level the game gave a baseline
            # keeps one of at least an action: rounded to zero, a factor below
            # 1 would make it worth nothing however it was played.
            if self.base_actions_per_level is not None and self.baseline_multiplier != 1.0:
                self.base_actions_per_level = [
                    max(1, round(actions * self.baseline_multiplier)) if actions > 0 else actions
                    for actions in self.base_actions_per_level
                ]
            self._validate_level_window()
            # start_at_level is honored by the subclass inside _start_game (the
            # base has no generic way to fast-forward). Verify it actually did.
            if self.start_at_level is not None and initial.levels_completed != self.start_at_level:
                raise RuntimeError(
                    f"{type(self).__name__}._start_game ignored start_at_level={self.start_at_level}: "
                    f"initial state reports levels_completed={initial.levels_completed}. A subclass "
                    f"must fast-forward to the start level inside _start_game."
                )
        except BaseException:
            # The subclass already opened resources in _start_game, but
            # game_run was never created so finish_game() can never run.
            # Give the subclass its teardown hook so scorecards / competition
            # run slots are released (R11.07 window errors must not leak them).
            self._finish_game()
            raise
        self.game_run = GameRun(
            game_id=self.game_id,
            number_of_levels=self.number_of_levels,
            base_actions_per_level=self.base_actions_per_level,
            baseline_multiplier=self.baseline_multiplier,
            hint=self.hint,
            state="playing",
            # Seed from the initial frame so a start_at_level run reports the
            # skipped levels as already completed (0 for a normal start).
            levels_completed=initial.levels_completed,
            record_intermediate_states=session.record_intermediate_states,
            started_at=datetime.now(),
            started_at_monotonic=time.monotonic(),
            trace_root=session.trace_root,
        )
        if self.game_run.record_intermediate_states:
            self.game_run.intermediate_states.append(initial)
        self._current_state = initial
        self._action_lock = threading.RLock()
        return initial

    def execute_action(
        self,
        action: arcengine.ActionInput,
        generated_tokens: int = 0,
        uncached_input_tokens: int = 0,
    ) -> GameState:
        """Execute a move (R11.03). Validates the action, logs an
        ``ActionRecord``, then delegates to ``_execute_action``. Raises
        ``ValueError`` for disallowed actions; ``_execute_action`` may
        raise its own exceptions (e.g. ``GameAPI`` raises on
        engine-rejected moves).
        """
        assert self.game_run is not None and self._action_lock is not None, "Call start_game() first"
        with self._action_lock:
            return self._execute_action_locked(action, generated_tokens, uncached_input_tokens)

    def _execute_action_locked(
        self,
        action: arcengine.ActionInput,
        generated_tokens: int,
        uncached_input_tokens: int,
    ) -> GameState:
        assert self.game_run is not None
        assert self.game_run.state == "playing", f"Cannot execute_action in state {self.game_run.state!r}"
        valid = self.current_state.available_actions
        if action.id.value not in valid:
            raise ValueError(f"Action {action.id.name} (id={action.id.value}) not in available_actions {valid}")
        action.id.validate_data(action.data)  # arcengine's own pydantic shape check
        if action.id.is_simple() and action.data:
            raise ValueError(f"Simple action {action.id.name} must not carry data; got {action.data}")
        if action.id.is_complex():
            x = action.data["x"]
            y = action.data["y"]
            w, h = self.grid_size
            if not (isinstance(x, int) and isinstance(y, int) and 0 <= x < w and 0 <= y < h):
                raise ValueError(
                    f"Action {action.id.name} coordinates ({x}, {y}) out of bounds for grid_size {self.grid_size}"
                )

        assert self.game_run.started_at_monotonic is not None
        elapsed = time.monotonic() - self.game_run.started_at_monotonic
        level_before = self.current_state.levels_completed

        # Run the subclass hook *before* any bookkeeping so a raise leaves
        # the run in a consistent state (no orphan history entry, no partial
        # actions_per_level increment, no missing intermediate_states).
        new_state_inner = self._execute_action(action)

        just_won_level = new_state_inner.levels_completed > level_before

        # Manual green frame inserted if a level transition produced only a
        # single frame — arcengine normally emits at least two on
        # transitions; this is a safety net for less-conformant games.
        if just_won_level and not new_state_inner.won and len(new_state_inner.raw.frame) <= 1:
            w, h = self.grid_size
            green_frame = np.full((h, w), 14, dtype=np.int8)
            new_state_inner.raw.frame = [green_frame, *new_state_inner.raw.frame]

        # stop_at_level (0-indexed, half-open [start, stop)): the instant the
        # engine reports levels_completed >= stop_at_level (i.e. level index
        # stop_at_level-1 was just completed), treat the game as won. We forge
        # WIN on the returned copy only — the subclass's own engine state is
        # untouched — so partial-credit scoring still runs over ALL levels
        # (indices >= stop_at_level stay uncompleted and score 0, which is why
        # stopping early caps the achievable score below 100%).
        if (
            self.stop_at_level is not None
            and new_state_inner.levels_completed >= self.stop_at_level
            and new_state_inner.raw.state != arcengine.GameState.WIN
        ):
            new_state_inner.raw.state = arcengine.GameState.WIN

        # Re-wrap to stamp previous_action / just_won_level — don't trust
        # the subclass to set them.
        new_state = GameState(
            raw=new_state_inner.raw,
            previous_action=action,
            just_won_level=just_won_level,
        )

        assert level_before < self.number_of_levels, (
            f"level_before={level_before} >= number_of_levels={self.number_of_levels}: "
            "subclass returned a state with levels_completed past number_of_levels "
            "without also setting arcengine.GameState.WIN"
        )

        # Commit all bookkeeping atomically.
        self.game_run.history.append(
            ActionRecord(
                action=action,
                generated_tokens=generated_tokens,
                uncached_input_tokens=uncached_input_tokens,
                wallclock_seconds=elapsed,
            )
        )
        if self.game_run.record_intermediate_states:
            self.game_run.intermediate_states.append(new_state)
        self._current_state = new_state
        self.game_run.actions_per_level[level_before] += 1

        if new_state.levels_completed > self.game_run.levels_completed:
            self.game_run.levels_completed = new_state.levels_completed

        if new_state.won:
            self.game_run.state = "won"

        # Bookkeeping invariants — every action produces exactly one
        # history entry, one increment to actions_per_level, and one
        # intermediate state. The score formula depends on these agreeing.
        assert sum(self.game_run.actions_per_level) == len(self.game_run.history), (
            f"sum(actions_per_level)={sum(self.game_run.actions_per_level)} "
            f"!= len(history)={len(self.game_run.history)}"
        )
        if self.game_run.record_intermediate_states:
            assert len(self.game_run.intermediate_states) == len(self.game_run.history) + 1, (
                f"len(intermediate_states)={len(self.game_run.intermediate_states)} "
                f"!= len(history)+1={len(self.game_run.history) + 1}"
            )
        assert self.game_run.levels_completed <= self.number_of_levels, (
            f"levels_completed={self.game_run.levels_completed} > number_of_levels={self.number_of_levels}"
        )
        return new_state

    def keepalive_reset(self) -> bool:
        """Send the engine a RESET the solver never sees (R2.63).

        Only on a level no action has touched: there a RESET returns the
        board the solver is already looking at, so nothing is recorded and
        ``current_state`` stands. Anywhere else it would wipe the solver's
        progress on the level, so this declines and returns ``False``. Held
        under the game's action lock, so a solver move cannot land between
        the check and the RESET.
        """
        run = self.game_run
        if run is None or self._action_lock is None:
            return False
        with self._action_lock:
            level = self.current_state.levels_completed
            if run.state != "playing" or level >= self.number_of_levels or run.actions_per_level[level]:
                return False
            self._execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}))
            run.keepalive_resets += 1
            return True

    def finish_game(self, generated_tokens: int = 0, uncached_input_tokens: int = 0) -> None:
        """Mark the game as finished (R11.03 / R11.02 state machine).

        If the game is still ``playing``, transitions to ``cancelled``
        when an asyncio cancel is outstanding on the current task, else
        ``gave_up``. ``Benchmark.run`` sets the state to ``crashed``
        before calling this in its teardown path. Idempotent.

        ``generated_tokens`` / ``uncached_input_tokens`` (R11.03) let the
        solver report tokens it spent on this final turn *without* making a
        move — the give-up / cancellation decision. They land on the run's
        ``final_*_tokens`` fields and count toward token totals, but not as
        an action. The finish time is stamped into ``final_wallclock_seconds``
        (monotonic elapsed since ``start_game``, matching
        ``ActionRecord.wallclock_seconds``) so those tokens land at the time
        they were spent on the wallclock axis. Recorded on the first
        (non-idempotent) call only.
        """
        assert self.game_run is not None, "finish_game() before start_game()"
        lock = self._action_lock
        if lock is None:
            self._finish_game_locked(generated_tokens, uncached_input_tokens)
            return
        # Bounded: by the time a game is finished the solver is done with it,
        # so the only other holder is a keepalive RESET (R2.63) still inside
        # its engine call. Waiting on that forever would hold teardown, and
        # everything after it, hostage to one stuck call; finishing without
        # the lock is safe because the RESET's return touches nothing but its
        # own counter.
        if not lock.acquire(timeout=FINISH_LOCK_TIMEOUT_S):
            print(
                f"[taaf.game] finishing {self.game_id} while a keepalive RESET is still in flight",
                flush=True,
            )
            self._finish_game_locked(generated_tokens, uncached_input_tokens)
            return
        try:
            self._finish_game_locked(generated_tokens, uncached_input_tokens)
        finally:
            lock.release()

    def _finish_game_locked(self, generated_tokens: int, uncached_input_tokens: int) -> None:
        assert self.game_run is not None
        if self.game_run.final_score is not None:
            return
        self.game_run.final_generated_tokens = generated_tokens
        self.game_run.final_uncached_input_tokens = uncached_input_tokens
        if self.game_run.started_at_monotonic is not None:
            self.game_run.final_wallclock_seconds = time.monotonic() - self.game_run.started_at_monotonic
        self._finish_game()
        if self.game_run.state == "playing":
            cancelling = False
            try:
                task = asyncio.current_task()
                if task is not None and task.cancelling() > 0:
                    cancelling = True
            except RuntimeError:
                # No running loop — sync caller, treat as gave_up.
                cancelling = False
            self.game_run.state = "cancelled" if cancelling else "gave_up"
        self.game_run.final_score = self.game_run._compute_final_score()
        # One-line per-game finish note to stdout — same fields as the
        # per-pass row in the diagnostics HTML, plus per-level
        # actions/baseline so the score is auditable from the log alone.
        run = self.game_run
        actions = sum(run.actions_per_level)
        tokens = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
        per_level = ",".join(
            f"{run.actions_per_level[i]}/"
            f"{run.base_actions_per_level[i] if run.base_actions_per_level is not None else '?'}"
            for i in range(run.number_of_levels)
        )
        note_suffix = f' note="{run.solver_note}"' if run.solver_note else ""
        print(
            f"[finished] {run.game_id} state={run.state} "
            f"level={run.levels_completed}/{run.number_of_levels} "
            f"score={run.final_score:.2f} actions={actions} tokens={tokens} "
            f"per-level={per_level}{note_suffix}"
        )

    def _validate_level_window(self) -> None:
        """Validate ``start_at_level`` / ``stop_at_level`` against
        ``number_of_levels``. Both are 0-indexed and define the half-open
        window ``[start_at_level, stop_at_level)``. Called from
        ``start_game`` once ``_start_game`` has populated
        ``number_of_levels``; subclasses that fast-forward may also call it
        earlier for a clean error before touching their engine."""
        n = self.number_of_levels
        if self.start_at_level is not None and not (0 <= self.start_at_level < n):
            raise ValueError(f"start_at_level={self.start_at_level} out of range [0, {n})")
        if self.stop_at_level is not None and not (0 < self.stop_at_level <= n):
            raise ValueError(f"stop_at_level={self.stop_at_level} out of range (0, {n}]")
        if (
            self.start_at_level is not None
            and self.stop_at_level is not None
            and self.start_at_level >= self.stop_at_level
        ):
            raise ValueError(
                f"start_at_level={self.start_at_level} must be < stop_at_level={self.stop_at_level} "
                "(the window is half-open [start, stop))"
            )

    # --- pickle / deepcopy contract (R11.05) --------------------------------

    def __getstate__(self) -> dict[str, Any]:
        if self.game_run is not None:
            raise RuntimeError(
                f"Cannot pickle {type(self).__name__} after start_game(). "
                "R11.05 contract: pickle is forbidden post-start."
            )
        return self.__dict__.copy()

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def __deepcopy__(self, memo: dict[int, Any]) -> Game:
        if self.game_run is not None and not self.allow_deepcopy:
            raise RuntimeError(
                f"Cannot deepcopy {type(self).__name__} after start_game() unless "
                "allow_deepcopy=True. R11.05: deepcopy is the only post-start "
                "lifecycle step that subclasses can opt in to."
            )
        cls = type(self)
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if k == "_action_lock":
                object.__setattr__(new, k, None if v is None else threading.RLock())
            else:
                object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new

    # --- subclass hooks (R11.04) --------------------------------------------

    def _start_game(self, session: RunSession) -> GameState:
        raise NotImplementedError("Subclasses must implement _start_game")

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        raise NotImplementedError("Subclasses must implement _execute_action")

    def _finish_game(self) -> None:
        """Optional teardown. Default: no-op."""
