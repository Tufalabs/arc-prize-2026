"""Swept parameters: how a value is drawn, and where it is applied.

A knob is a *dedicated DoE declaration* rather than an entry in a registry
of every settable field: the experiment names what it varies and how, and
anything not named is not swept.

Two properties shape the design. Values must **pickle** — the agent template
travels into a game subprocess by pickle, so a knob built from a lambda or a
closure cannot make the trip; the helpers here are small frozen dataclasses
precisely so they can. And each knob draws from **its own stream**, seeded
from ``(seed, name)``, so adding or removing a knob leaves every other
knob's value for a given seed unchanged — which is what lets results
recorded before a knob existed pool with later ones (see ``Knob.missing``).
"""

from __future__ import annotations

import math
import pickle
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol


class Draw(Protocol):
    """Draws one value for a knob from the knob's own RNG stream."""

    def __call__(self, rng: random.Random) -> Any: ...


class Apply(Protocol):
    """Writes a drawn value onto the agent. ``name`` is the knob's name, so
    one implementation can serve many knobs."""

    def __call__(self, target: Any, name: str, value: Any) -> None: ...


@dataclass(frozen=True)
class Knob:
    """One swept parameter: how to draw it, where to put it, and what it was
    before it existed.

    ``missing`` is the value to assume for a run recorded before this knob
    was added — the declared baseline that lets old and new jobs sit in one
    regression. There is no way to infer it (the field default at that commit
    is not recoverable from the results), so it is stated.
    """

    draw: Draw
    apply: Apply
    missing: Any = None


@dataclass(frozen=True)
class Field:
    """Apply: set a dotted attribute path, relative to the sweep agent.

    ``Field("inner.max_moves")`` sets the wrapped agent's field;
    ``Field("run_config.max_concurrent")`` sets a hosting knob, which only a
    RUN knob may target. The path is data rather than a closure, which is what
    keeps a knob picklable and what lets the batch rules see where a knob
    points.

    A segment may index a mapping — ``inner.llm_config.sampling_profiles[play]
    .max_tokens`` — because configuration that lives under a name rather than
    on an attribute is otherwise unreachable by any picklable apply.
    """

    path: str

    def __call__(self, target: Any, name: str, value: Any) -> None:
        """Walk the path and set its last segment."""
        del name  # the path already says where this goes
        set_by_path(target, self.path, value)


def _walk(target: Any, segment: str) -> Any:
    """One path segment: an attribute, optionally indexed by a mapping key."""
    if segment.endswith("]") and "[" in segment:
        attribute, _, key = segment[:-1].partition("[")
        return getattr(target, attribute)[key]
    return getattr(target, segment)


def unresolved_segment(target: Any, path: str) -> str | None:
    """The first segment of ``path`` that ``target`` does not have, or None.

    A ``Field`` sets an attribute, and ``setattr`` invents one where none
    existed — so a path that names nothing does not fail, it succeeds against
    a stray attribute nothing reads. The knob then varies in every table and
    changes nothing, which looks exactly like a knob that did not matter.
    Resolving the path against the built agent is what tells those apart, and
    it can only be done where an agent exists.

    The final segment is resolved like the rest: ``set_by_path`` on a
    bracketed last segment inserts the key beside the real ones when it is
    missing, so an unchecked ``profiles[typo]`` would be the same silent
    null one segment deeper.
    """
    parts = path.split(".")
    for index, part in enumerate(parts):
        attribute = part.partition("[")[0] if part.endswith("]") and "[" in part else part
        if not hasattr(target, attribute):
            return ".".join(parts[: index + 1])
        try:
            target = _walk(target, part)
        except (KeyError, IndexError, TypeError):
            return ".".join(parts[: index + 1])
    return None


def set_by_path(target: Any, path: str, value: Any) -> None:
    """Set a dotted path, walking attributes and ``name[key]`` segments."""
    parts = path.split(".")
    for part in parts[:-1]:
        target = _walk(target, part)
    last = parts[-1]
    if last.endswith("]") and "[" in last:
        attribute, _, key = last[:-1].partition("[")
        getattr(target, attribute)[key] = value
    else:
        setattr(target, last, value)


@dataclass(frozen=True)
class Record:
    """Apply: nothing — the value is consumed by the batch's derivation.

    Explicit rather than ``None`` so a knob that writes no field of its own
    still says so. A design coordinate (half of a ratio, one term of a sum)
    has no field to point at, and the alternative reads exactly like a knob
    someone forgot to wire up.
    """

    def __call__(self, target: Any, name: str, value: Any) -> None:
        """Deliberately nothing."""
        del target, name, value


#: The one instance anyone needs.
RECORD = Record()


@dataclass(frozen=True)
class Choice:
    """Draw: one of a fixed set, uniformly."""

    values: tuple[Any, ...]

    def __call__(self, rng: random.Random) -> Any:
        """Uniform pick."""
        return rng.choice(self.values)


@dataclass(frozen=True)
class LogUniformInt:
    """Draw: an integer log-uniform over ``[low, high]``.

    Log-uniform because the interesting questions about a budget-shaped knob
    are ratios rather than differences: 1 vs 10 is the same size of question
    as 100 vs 1000, and a uniform draw over 1..1000 would spend 90% of its
    arms above 100 and answer none of it.
    """

    low: int
    high: int

    def __post_init__(self) -> None:
        """Refuse a range that cannot be drawn from."""
        if self.low < 1 or self.high < self.low:
            raise ValueError(f"LogUniformInt needs 1 <= low <= high; got low={self.low}, high={self.high}")

    def __call__(self, rng: random.Random) -> int:
        """A rounded draw, clamped so rounding cannot escape the range."""
        value = round(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return max(self.low, min(self.high, int(value)))


@dataclass(frozen=True)
class UniformInt:
    """Draw: an integer uniform over ``[low, high]``, both ends included.

    The linear sibling of `LogUniformInt`, for a knob where differences are
    the interesting question rather than ratios — a slope fitted against the
    raw value wants its draws spread evenly on that axis, not bunched at the
    low end.
    """

    low: int
    high: int

    def __post_init__(self) -> None:
        """Refuse a range that cannot be drawn from."""
        if self.high < self.low:
            raise ValueError(f"UniformInt needs low <= high; got low={self.low}, high={self.high}")

    def __call__(self, rng: random.Random) -> int:
        """Uniform integer draw."""
        return rng.randint(self.low, self.high)


@dataclass(frozen=True)
class Uniform:
    """Draw: a float uniform over ``[low, high]``."""

    low: float
    high: float

    def __call__(self, rng: random.Random) -> float:
        """Uniform draw."""
        return rng.uniform(self.low, self.high)


def knob_rng(seed: int, name: str) -> random.Random:
    """This knob's own stream.

    Per knob, not per batch: a shared stream would make every knob's value
    depend on which other knobs were declared, so adding one to a running
    experiment would silently redraw the rest and split the pool.
    """
    return random.Random(f"{seed}:{name}")


def draw_values(knobs: Mapping[str, Knob], seed: int) -> dict[str, Any]:
    """Draw every knob for one seed. Deterministic, and independent of the
    order the knobs were declared in."""
    return {name: knobs[name].draw(knob_rng(seed, name)) for name in sorted(knobs)}


def apply_values(knobs: Mapping[str, Knob], values: Mapping[str, Any], target: Any) -> None:
    """Apply drawn values in name order, so a pair of knobs that touch the
    same object resolve the same way every time."""
    for name in sorted(values):
        knobs[name].apply(target, name, values[name])


def check_picklable(knobs: Mapping[str, Knob], *, batch: str) -> None:
    """Fail at construction on a knob that cannot reach a game subprocess.

    The failure this prevents is otherwise a pickling error inside a spawned
    child, thrown from the solver's plumbing with no mention of the knob that
    caused it — and only in subprocess mode, so it survives every in-process
    test.
    """
    for name, knob in sorted(knobs.items()):
        try:
            pickle.dumps(knob)
        except Exception as error:
            raise ValueError(
                f"{batch} knob {name!r} does not pickle ({error}). Knobs travel into game "
                "subprocesses with the agent, so draw/apply must be module-level functions or "
                "picklable objects — a lambda or a function defined inside another function is not."
            ) from error


class Derive(Protocol):
    """Maps a batch's drawn values to field assignments.

    Returns ``{dotted path: value}`` rather than mutating the agent, so the
    sweeper applies the result through the same path machinery a knob uses —
    and can refuse an assignment aimed somewhere it cannot take effect. A
    derivation that wrote to the agent directly would slip past that check.
    """

    def __call__(self, values: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class Design:
    """A sweep's two batches, resolved per job.

    ``run`` knobs are drawn once from ``run_seed`` and applied before the run
    starts, so they may touch anything the solver reads off the template —
    engine settings, concurrency. ``arm`` knobs are drawn per pass and applied
    inside each game, so they may touch only what is read from the per-game
    copy.

    A batch may also declare a **derivation**, for when the coordinates worth
    varying independently are not the fields the agent happens to have. The
    knobs then record (``RECORD``) and the derivation maps them onto fields
    once, after the whole batch is drawn — which is how a constrained set of
    fields becomes an unconstrained set of knobs, without reshaping the agent
    to suit the experiment.
    """

    run: Mapping[str, Knob] = field(default_factory=lambda: dict[str, Knob]())
    arm: Mapping[str, Knob] = field(default_factory=lambda: dict[str, Knob]())
    derive_run: Derive | None = None
    derive_arm: Derive | None = None

    def derived(self, batch: str, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """This batch's derived assignments for one draw; empty if none."""
        derive = self.derive_arm if batch == "arm" else self.derive_run
        return {} if derive is None else derive(values)

    def arm_values_for(self, seed: int) -> dict[str, Any]:
        """The per-arm values a given arm seed produces."""
        return draw_values(self.arm, seed)

    def run_values_for(self, seed: int) -> dict[str, Any]:
        """The run-level values a given run seed produces."""
        return draw_values(self.run, seed)


__all__ = [
    "RECORD",
    "Apply",
    "Choice",
    "Derive",
    "Design",
    "Draw",
    "Field",
    "Knob",
    "LogUniformInt",
    "Record",
    "Uniform",
    "UniformInt",
    "apply_values",
    "check_picklable",
    "draw_values",
    "knob_rng",
    "set_by_path",
]
