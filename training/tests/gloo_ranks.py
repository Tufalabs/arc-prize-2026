"""Spawn test ranks that run real gloo process groups, and exit them safely.

A gloo worker thread completes a collective, then drops its reference to
the collective's tensor. For a tensor Python also holds, that release needs
the GIL, which the thread waits for while the rank's main thread runs on. A
rank that leaves its worker within that wait (a CPython switch interval,
5 ms) is already finalizing the interpreter when the thread gets the GIL;
CPython then ends the thread with pthread_exit, whose forced unwind cannot
leave ProcessGroupGloo::runLoop, and std::terminate aborts the rank with
"terminate called without an active exception" (SIGABRT). On some AMD
nodes that aborted the two-rank GRPO window test about one run in five.

``run_rank`` ends a rank that passed with os._exit, so no finalization races
the gloo threads; a rank that raises still exits through multiprocessing,
which reports its traceback (torch.multiprocessing hands it to the parent).
``spawn_ranks`` is mp.spawn over run_rank; a multiprocessing.Process can
take ``target=run_rank, args=(rank, worker, *args)``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any

import torch.multiprocessing as mp


def run_rank(rank: int, worker: Callable[..., None], *args: Any) -> None:
    """Run ``worker(rank, *args)``, then exit without finalization if it passed."""
    worker(rank, *args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def spawn_ranks(worker: Callable[..., None], *args: Any, nprocs: int = 2) -> None:
    """``mp.spawn(worker, args=args, nprocs=nprocs, join=True)``, exiting ranks
    that passed without interpreter finalization."""
    mp.spawn(run_rank, args=(worker, *args), nprocs=nprocs, join=True)
