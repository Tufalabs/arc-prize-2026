"""Yield a training segment at the last iteration boundary its budget allows.

A scheduled segment end reaches the trainer as SIGTERM with a kill grace of
tens of seconds at every layer (the launcher, torchrun and the scheduler).
That is enough to stop cleanly between iterations but never enough to make a
multi-hundred-GiB full-weight checkpoint durable. So the trainer learns the
segment deadline up front and stops itself, with a durable checkpoint and a
closed distributed runtime, before starting an iteration it could not finish.
SIGTERM remains the backstop for deadlines the trainer was not told about.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

DEADLINE_ENV = "ARC3_SEGMENT_DEADLINE_EPOCH"
MARGIN_ENV = "ARC3_SEGMENT_YIELD_MARGIN_SECONDS"
DEFAULT_MARGIN_SECONDS = 120.0
# Iterations vary with collection time; one more must fit with room to spare.
SAFETY_FACTOR = 1.25
_TRAILING_ITERATIONS = 3


@dataclass
class SegmentBudget:
    """Decide, before each iteration, whether the segment can still finish it."""

    deadline_epoch: float | None
    margin_seconds: float = DEFAULT_MARGIN_SECONDS
    safety_factor: float = SAFETY_FACTOR
    iteration_seconds: list[float] = field(default_factory=list)
    checkpoint_seconds: float = 0.0
    clock: Callable[[], float] = time.time

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> SegmentBudget:
        environ = os.environ if environ is None else environ
        raw_deadline = environ.get(DEADLINE_ENV, "").strip()
        deadline: float | None = None
        if raw_deadline:
            try:
                deadline = float(raw_deadline)
            except ValueError as error:
                raise ValueError(
                    f"{DEADLINE_ENV} must be a unix timestamp in seconds; "
                    f"got {raw_deadline!r}"
                ) from error
        raw_margin = environ.get(MARGIN_ENV, "").strip()
        margin = DEFAULT_MARGIN_SECONDS
        if raw_margin:
            try:
                margin = float(raw_margin)
            except ValueError as error:
                raise ValueError(
                    f"{MARGIN_ENV} must be a number of seconds; got {raw_margin!r}"
                ) from error
            if margin < 0:
                raise ValueError(f"{MARGIN_ENV} must not be negative; got {margin}")
        return cls(deadline_epoch=deadline, margin_seconds=margin, clock=clock)

    @property
    def enabled(self) -> bool:
        return self.deadline_epoch is not None

    def seed_iteration_estimate(self, seconds: Any) -> None:
        """Carry the previous segment's last iteration wall time into this one."""
        if type(seconds) in (int, float) and seconds > 0:
            self.iteration_seconds.append(float(seconds))

    def record_iteration(self, seconds: float) -> None:
        """Wall time from one iteration's start to the next, collection included."""
        self.iteration_seconds.append(max(0.0, float(seconds)))
        del self.iteration_seconds[:-_TRAILING_ITERATIONS]

    def record_checkpoint(self, seconds: Any) -> None:
        """Longest observed time for a checkpoint to become durable."""
        if type(seconds) in (int, float) and seconds > 0:
            self.checkpoint_seconds = max(self.checkpoint_seconds, float(seconds))

    def remaining_seconds(self) -> float | None:
        if self.deadline_epoch is None:
            return None
        return self.deadline_epoch - self.clock()

    def required_seconds(self) -> float | None:
        """Wall time one more iteration needs before its state is durable."""
        if not self.iteration_seconds:
            return None
        return (
            self.safety_factor * max(self.iteration_seconds)
            + self.checkpoint_seconds
            + self.margin_seconds
        )

    def should_yield(self) -> bool:
        remaining = self.remaining_seconds()
        if remaining is None:
            return False
        required = self.required_seconds()
        if required is None:
            # Nothing timed yet: only the teardown margin itself is known.
            return remaining <= self.margin_seconds
        return remaining < required

    def describe(self) -> dict[str, Any]:
        return {
            "deadline_epoch": self.deadline_epoch,
            "remaining_seconds": self.remaining_seconds(),
            "required_seconds": self.required_seconds(),
            "iteration_seconds": list(self.iteration_seconds),
            "checkpoint_seconds": self.checkpoint_seconds,
            "margin_seconds": self.margin_seconds,
            "safety_factor": self.safety_factor,
        }
