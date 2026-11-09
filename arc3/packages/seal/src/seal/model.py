"""The world model: a frozen, picklable predictor over the game.

A ``WorldModel`` answers three questions about a hypothetical state — what
happens if I do this (:meth:`~WorldModel.step`), what would the board look
like (:meth:`~WorldModel.render`), and how close is this to winning
(:meth:`~WorldModel.distance_to_win`) — and nothing else. No LLM, no async,
no I/O beyond what the model's own code does.

That narrowness is what lets a search ship one into a subprocess. The
concrete class over model-written ``world_model.py`` source is
:class:`seal.py_model.PyWorldModel`; everything here is substrate-free,
which is also what makes a hand-written model a first-class implementation
rather than a mock.
"""

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Generator, Literal

import numpy as np
import numpy.typing as npt

from platypus.contract import Action
from seal.harness.game import _is_decline  # pyright: ignore[reportPrivateUsage]

#: Outcome of a *hypothetical* state, as the model's own code records it in
#: ``state['outcome']``. Deliberately not the contract's four-valued
#: ``platypus.contract.Outcome``: the model reasons about one level at a
#: time, so it knows "won" but not "this won the whole game", and the
#: level/game distinction is infra's to make.
Outcome = Literal["playing", "won", "lost"]

#: How much a prediction claims.
#:
#: - ``claim`` — the model states the result outright. There is no
#:   confidence scale: a prediction is a claim or it is not made.
#: - ``unknown`` — this move's effect has never been observed, so the model
#:   declines to predict (the model's code raised ``Unknown(reason)``).
#: - ``error`` — the model's code crashed, or broke its contract.
Kind = Literal["claim", "unknown", "error"]

#: Key the model's own code writes its outcome under. Part of the contract
#: shown to the model: ``initial_state`` seeds ``'playing'``, ``do_move``
#: writes ``'won'``/``'lost'`` on transitions.
OUTCOME_KEY = "outcome"
VALID_OUTCOMES: tuple[Outcome, ...] = ("playing", "won", "lost")

#: Cell value meaning "the model declines to predict what is here". Renders
#: carrying it are compared as wildcards, and a cell that becomes visible is
#: itself a reason to re-learn.
UNKNOWN_CELL = -1

#: Key of the per-state energy scalar. Part of the contract: ``initial_state``
#: must set it to a number and ``do_move`` must keep it one, which the
#: scaffolding enforces. A search reads it for Pareto dedup, hence a name
#: rather than a hard-coded string on either side.
ENERGY_KEY = "energy"

#: Optional module-level function the model may define. Present, it is the
#: best-first heuristic a search orders by (lower is closer to a win);
#: absent, search falls back to plain cost order. Scaffolding checks it
#: returns a number on every replayed state so the heap key stays comparable.
DISTANCE_FN_NAME = "distance_to_win"

#: Required module-level function: ``render(state) -> ndarray``, the board
#: itself.
RENDER_FN_NAME = "render"


@dataclass(frozen=True)
class Prediction:
    """What :meth:`WorldModel.step` concluded.

    ``outcome`` is populated for ``claim`` only; ``unknown_reason`` for
    ``unknown``; ``error``/``exception`` for ``error``.

    A failure carries **both** a string and, best-effort, the exception
    object, because the model crosses a process boundary into search: an
    exception raised by model-written code need not be picklable, so
    ``error`` (a formatted traceback) is what is guaranteed to arrive.
    ``exception`` rides along for its type, args and chained cause, and is
    simply dropped when it cannot be pickled.

    **``exception`` is diagnostics only.** Nothing branches on it, ever —
    anything load-bearing reads ``error``, or it would behave differently
    either side of a process boundary.
    """

    kind: Kind
    outcome: Outcome | None = None
    unknown_reason: str | None = None
    error: str | None = None
    exception: Exception | None = None


class ModelContractError(RuntimeError):
    """The model's code broke the contract it was given — a missing or
    invalid ``state['outcome']``, acting on a state already lost. Raised
    inside :meth:`WorldModel.step`, which converts it into an ``error``
    prediction like any other failure; it exists so the message says which
    rule was broken rather than surfacing as a bare ``KeyError``."""


def read_outcome(state: dict[str, Any]) -> Outcome:
    """``state['outcome']``, or a message naming the contract violation."""
    if OUTCOME_KEY not in state:
        raise ModelContractError(
            f"state['{OUTCOME_KEY}'] is missing — initial_state must set it to 'playing' and do_move must maintain it."
        )
    value = state[OUTCOME_KEY]
    if value not in VALID_OUTCOMES:
        raise ModelContractError(f"state['{OUTCOME_KEY}'] = {value!r} is not one of {VALID_OUTCOMES}.")
    return value


class ActiveScope:
    """The scope :meth:`WorldModel.active` hands out.

    A plain object rather than a generator: ``step`` and ``render`` enter it on
    every call and a search makes millions of them, where building a generator
    and a ``_GeneratorContextManager`` per call costs more than the bookkeeping
    inside. One instance per model, re-entered rather than rebuilt — the depth
    counter on the model is what makes nesting work, so re-entry is correct.
    """

    __slots__ = ("_model", "_outer")

    def __init__(self, model: "WorldModel") -> None:
        self._model = model
        #: The environment context, held only while this is the outermost entry.
        self._outer: Any = None

    def __enter__(self) -> None:
        model = self._model
        if model._active_depth == 0:
            # Entered before the depth rises, so a raising environment leaves
            # the model un-entered rather than stuck at depth 1.
            outer = model._environment()
            outer.__enter__()
            self._outer = outer
        model._active_depth += 1

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        # Returns None, never True: an environment does not get to swallow an
        # exception raised by the model code running inside it, and none tries.
        model = self._model
        model._active_depth -= 1
        if model._active_depth == 0 and self._outer is not None:
            outer, self._outer = self._outer, None
            outer.__exit__(exc_type, exc, tb)


class WorldModel(ABC):
    """A frozen snapshot of the game's rules, ready for prediction.

    Subclasses override the ``_``-prefixed methods; the public ones own the
    contract — catching what must be caught, stamping the outcome, and
    entering :meth:`active` when a caller has not.

    Must be picklable: a search ships one into a subprocess.
    """

    #: Told what the model's code is about to run, and None once it returns:
    #: a search child records it, so a search killed inside one call can say
    #: which. On the class, so a model unpickled without ``__init__`` has it.
    call_marker: Callable[[str | None], None] | None = None

    def __init__(self) -> None:
        self._active_depth = 0
        self._active_scope = ActiveScope(self)

    # --- execution environment -------------------------------------------

    def active(self) -> ActiveScope:
        """Set up the environment a batch of prediction calls runs in —
        principally the working directory, so the relative file I/O in
        model-written top-level code (loading a pickle it stashed, reading a
        helper module) resolves the way it did when the model was written.

        An **optimisation, not a precondition**: ``step`` / ``render`` enter
        it themselves when a caller has not, and it nests harmlessly, so a
        search can wrap itself in one and have the thousands of calls inside
        pay the setup once.
        """
        return self._active_scope

    @contextlib.contextmanager
    def _environment(self) -> Generator[None, None, None]:
        """Entered once per outermost :meth:`active`. Default: nothing."""
        yield

    @contextlib.contextmanager
    def _calling(self, what: Callable[[], str]) -> Generator[None, None, None]:
        """Around one call into the model's code: :attr:`call_marker` hears
        ``what()`` before it and None after. ``what`` is only rendered when
        someone listens."""
        marker = self.call_marker
        if marker is None:
            yield
            return
        marker(what())
        try:
            yield
        finally:
            marker(None)

    def _describe_move(self, action: Action) -> str:
        """``action`` as a call into the model's code names it."""
        return repr(action)

    # --- prediction -------------------------------------------------------

    def initial_state(self) -> dict[str, Any]:
        """The model's own account of the level's starting state.

        A model whose ``initial_state`` raises yields a *poisoned* state
        rather than propagating: the error is carried under ``'_error'`` and
        every downstream use of it fails, which routes the failure into the
        ordinary "model is broken, re-learn" path instead of crashing the
        game.
        """
        with self.active():
            try:
                with self._calling(lambda: "initial_state()"):
                    return self._initial_state()
            except Exception as e:
                return {"_error": e}

    def step(self, state: dict[str, Any], action: Action) -> Prediction:
        """Apply ``action`` to ``state`` **in place** and say what the model
        concluded about the result.

        RESET is intercepted here, so model-written ``do_move`` code never
        sees it: the state is replaced in place by a fresh
        :meth:`initial_state`. Acting on an already-lost state is a contract
        violation, since only RESET is legal there.

        ``Unknown`` raised by the model becomes an ``unknown`` prediction;
        anything else it raises becomes an ``error`` prediction, which a
        search treats as a boundary — anything except ``MemoryError``, which
        is the memory guard firing rather than a prediction at all.
        """
        with self.active():
            try:
                if action.name == "RESET":
                    fresh = self.initial_state()
                    state.clear()
                    state.update(fresh)
                    return Prediction(kind="claim", outcome="playing")
                if state.get(OUTCOME_KEY) == "lost":
                    raise ModelContractError(f"Action {action.name} attempted on a lost state; only RESET is allowed")
                with self._calling(lambda: f"do_move(state, {self._describe_move(action)})"):
                    prediction = self._step(state, action)
                # The outcome lives in the state — model code writes it on a
                # transition — so it is read back here rather than trusted
                # from the subclass. One source of truth.
                if prediction.kind == "claim":
                    prediction = replace_outcome(prediction, read_outcome(state))
                return prediction
            except MemoryError:
                # Not the model mispredicting — the resource guard firing.
                # ``RLIMIT_AS`` refusing an allocation is the search child
                # being told it may have no more memory, and the parent
                # reports that death. Recorded as an error prediction it
                # would read as an ordinary boundary, and the search would
                # return a plan as though nothing had happened.
                raise
            except Exception as e:
                if _is_decline(e):
                    return Prediction(kind="unknown", unknown_reason=str(e))
                return Prediction(kind="error", error=format_exception(e), exception=e)

    def render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        """The board this state would show, as int8.

        May contain :data:`UNKNOWN_CELL` for content the model declines to
        predict. Exceptions **propagate**: a board that cannot be produced
        is a real failure, and every caller wants to handle it its own way —
        the replay records it and re-learns, a search treats the node as an
        unknown boundary.

        Always a copy: a ``render`` that returns the state's own array would
        otherwise hand back a board the next ``do_move`` rewrites in place,
        and a caller that keeps each move's board would keep the last one.
        """
        with self.active(), self._calling(lambda: "render(state)"):
            return np.array(self._render(state), dtype=np.int8)

    def distance_to_win(self, state: dict[str, Any]) -> Any:
        """Model-defined best-first heuristic; lower is closer to winning.
        ``None`` means the model defines no heuristic.

        Deliberately **not** exception-catching. The only caller is a
        search, which disables best-first ordering on the first exception
        or non-numeric return and records why; catching here would hide that
        failure mode from the mechanism built to handle it.
        """
        with self.active(), self._calling(lambda: "distance_to_win(state)"):
            return self._distance_to_win(state)

    # --- subclass interface ----------------------------------------------

    @abstractmethod
    def _initial_state(self) -> dict[str, Any]:
        """The model's starting state. Subclasses implement."""
        ...

    @abstractmethod
    def _step(self, state: dict[str, Any], action: Action) -> Prediction:
        """Advance ``state`` by one action and say what that concluded.
        Subclasses implement; the state is mutated in place."""
        ...

    @abstractmethod
    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        """The board this state draws. Subclasses implement."""
        ...

    def _distance_to_win(self, state: dict[str, Any]) -> Any:
        """A best-first heuristic, or ``None`` for "this model defines none" —
        absence is a value here, not an exception, so a search's auto-disable
        keeps its one meaning: a heuristic that misbehaved.
        """
        return None


def replace_outcome(prediction: Prediction, outcome: Outcome) -> Prediction:
    """``dataclasses.replace`` for the one field :meth:`WorldModel.step`
    stamps, kept as a named function so the frozen dataclass stays frozen."""
    return Prediction(
        kind=prediction.kind,
        outcome=outcome,
        unknown_reason=prediction.unknown_reason,
        error=prediction.error,
        exception=prediction.exception,
    )


def format_exception(e: BaseException) -> str:
    """A formatted traceback, which is what survives the search process
    boundary when the exception object itself does not."""
    import traceback

    return "".join(traceback.format_exception(type(e), e, e.__traceback__))
