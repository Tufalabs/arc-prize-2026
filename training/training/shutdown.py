"""Honor scheduled termination only where the DDP command protocol is idle."""

from __future__ import annotations

import signal
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar


class TrainingShutdown(KeyboardInterrupt):
    """Scheduled termination: clean up without poisoning continuation."""


class SegmentYield(TrainingShutdown):
    """The segment budget ends before one more iteration could finish."""


# EX_TEMPFAIL: the segment stopped at a safe boundary and wants a continuation.
SEGMENT_YIELD_EXIT_CODE = 75


_shutdown_requested = False
_shutdown_deferred = False
_P = ParamSpec("_P")
_R = TypeVar("_R")


def _handle_training_sigterm(_signum, _frame) -> None:
    # Never raise from a signal handler: rank 0 may be between collectives,
    # initializing worker groups, or staging completed artifacts. Repeated
    # TERM signals are harmless, including during cleanup. This does not mask
    # signals or change the independent fatal-worker watchdog.
    global _shutdown_requested
    _shutdown_requested = True


def shutdown_requested() -> bool:
    """Whether a scheduled termination is pending for this process."""
    return _shutdown_requested


def raise_if_shutdown_requested() -> None:
    """Call only when workers are idle and owned resources can be cleaned up."""
    if _shutdown_requested and not _shutdown_deferred:
        raise TrainingShutdown("received SIGTERM")


@contextmanager
def defer_training_shutdown() -> Iterator[None]:
    """Keep TERM pending while a collective command finishes on every rank."""
    global _shutdown_deferred
    previous = _shutdown_deferred
    _shutdown_deferred = True
    try:
        yield
    finally:
        _shutdown_deferred = previous


@contextmanager
def training_sigterm_handler() -> Iterator[None]:
    """Scope the handler and request state to one main-thread CLI invocation."""
    global _shutdown_requested
    previous_request = _shutdown_requested
    _shutdown_requested = False
    previous_handler = signal.signal(signal.SIGTERM, _handle_training_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        _shutdown_requested = previous_request


def shutdown_at_safe_boundary(function: Callable[_P, _R]) -> Callable[_P, _R]:
    """Finish a complete worker protocol before honoring a pending TERM.

    No shutdown checks may run inside the wrapped protocol. Genuine errors
    still propagate immediately; a pending TERM must not hide a fatal error.
    """

    @wraps(function)
    def run(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        raise_if_shutdown_requested()
        result = function(*args, **kwargs)
        raise_if_shutdown_requested()
        return result

    return run


@contextmanager
def peer_sigterm_handler() -> Iterator[None]:
    """Keep a full-weight peer serving rank zero's protocol through a TERM.

    torchrun forwards a scheduled TERM to every rank at once. Rank zero
    defers it to a safe boundary and then broadcasts EXIT; a peer that
    unwound on its own would strand rank zero inside a collective. The
    scheduler's KILL still ends a peer that never receives EXIT.
    """
    previous_handler = signal.signal(signal.SIGTERM, lambda _signum, _frame: None)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


@contextmanager
def worker_sigterm_handler(parent_pid: int) -> Iterator[None]:
    """Ask rank zero to stop; keep serving its collective protocol until EXIT.

    Slurm can signal every process at once. Workers must not independently
    unwind a collective or raise the trainer's deferred-shutdown exception.
    A worker-only TERM also reaches rank zero. The parent watchdog still handles
    a vanished coordinator and the scheduler can still enforce SIGKILL.
    """
    import os

    forwarded = False

    def request_parent_shutdown(_signum, _frame):
        nonlocal forwarded
        if not forwarded:
            forwarded = True
            try:
                os.kill(parent_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The worker's parent watchdog owns this failure path.

    previous_handler = signal.signal(signal.SIGTERM, request_parent_shutdown)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
