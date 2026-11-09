"""How many games are waiting for a slot, over the job dir.

An agent that gives up on a game frees a slot, and whether that is worth
doing depends on whether anything is waiting to take it. Once nothing is,
a give-up buys nothing — the machine simply idles for the rest of the
run — so an agent with a spending rule wants to know.

**Waiting means wanting a slot without holding one**, which is two
populations: games the run has not started yet, and games that started,
offered their slot back (``platypus.priority``) and have not been
re-granted one. Both can take a freed slot, so both count — that is the
question the number is for. They are not interchangeable in every
respect, and where the difference matters (what a run is holding in
memory) the per-game report tells them apart; here it would only make
the answer wrong.

**The parent already knows exactly.** ``AgentSolver._host_game`` registers
each game as waiting *before* it acquires a slot and flips it to running
inside, and a game that parks is counted back, so the tally is a complete
description at every instant. What is missing is a way for a game to read
it: games run in their own subprocesses, and the control pipe is
reply-only — the parent cannot push, and the child is off running an
agent rather than sitting in ``recv``.

**So it goes through the job dir**, the way :mod:`platypus.live` does, and
for the same reason its docstring gives: a Unix socket is bound to the host
whose process created it, and files are the portable transport. The parent
rewrites the tally whenever a game takes or frees a slot; a reader opens
the file *when it asks*, which is the property that matters. A move can sit
in one search for ten minutes, and any push-based scheme would be stale by
however long that took; reading at the point of the question is current by
construction.

Not knowing is never mistaken for a drained queue: a missing or unreadable
file leaves :attr:`SlotView.waiting` at ``None``, and
:attr:`SlotView.drained` is false unless it has positively read a zero. A
run with no writer therefore behaves exactly as one with no feature.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Name in the job dir. One file for the whole run, not one per game.
FILENAME = "game_slots.json"

#: How long a read is reused before the file is consulted again. The
#: numbers move only when a game starts or ends, and every consumer is
#: deciding something at human timescales, so a second of staleness costs
#: nothing and bounds the read rate whatever an agent does.
DEFAULT_TTL_S = 1.0


def slots_path(job_dir: str | Path) -> Path:
    """Where the run's queue tally lives."""
    return Path(job_dir) / FILENAME


def write_slots(job_dir: str | Path, *, waiting: int, running: int, limit: int) -> None:
    """Publish the tally. Never raises: this is bookkeeping, not the run.

    Written through a same-directory temp file and renamed, so a reader
    never sees a half-written file — the rename is what makes it atomic,
    on NFS included.
    """
    path = slots_path(job_dir)
    payload = {"waiting": waiting, "running": running, "limit": limit, "t": time.time()}
    try:
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, path)
    except OSError:
        pass


@dataclass
class SlotView:
    """The run's game queue, read from the job dir when asked.

    Held on :class:`~platypus.context.MoveContext`. Every property is a
    read, so the answer is current at the moment of the question rather
    than as of whenever something last pushed — which matters because a
    single move can occupy a game for many minutes.
    """

    #: The run's job directory, or ``None`` when nothing published a tally.
    job_dir: Path | None = None
    ttl_s: float = DEFAULT_TTL_S
    _cached: dict[str, Any] | None = field(default=None, repr=False)
    _read_at: float = field(default=0.0, repr=False)

    def _tally(self) -> dict[str, Any] | None:
        """The published tally, re-read at most once per :attr:`ttl_s`."""
        if self.job_dir is None:
            return None
        now = time.monotonic()
        # Stamped per attempt, so "unknown" is cached like any other answer.
        # That is what bounds the read rate before anything is published:
        # there is no value to short-circuit on then, and `drained` alone
        # reads `waiting`.
        if self._read_at and now - self._read_at < self.ttl_s:
            return self._cached
        self._read_at = now
        try:
            loaded = json.loads(slots_path(self.job_dir).read_text())
        except (OSError, ValueError):
            return self._cached  # a torn or absent read keeps the last good answer
        self._cached = loaded if isinstance(loaded, dict) else None
        return self._cached

    def _number(self, key: str) -> int | None:
        """One field of the tally, or ``None`` when it is not known."""
        tally = self._tally()
        value = tally.get(key) if tally is not None else None
        return int(value) if isinstance(value, int) else None

    @property
    def waiting(self) -> int | None:
        """Games that want a slot and do not hold one — queued to start and
        parked alike; ``None`` if unknown."""
        return self._number("waiting")

    @property
    def running(self) -> int | None:
        """Games currently holding a slot; ``None`` if unknown."""
        return self._number("running")

    @property
    def limit(self) -> int | None:
        """The run's concurrency cap; ``None`` if unknown."""
        return self._number("limit")

    @property
    def drained(self) -> bool:
        """Whether nothing at all is waiting for a slot.

        Neither queued to start nor parked: giving this game up would free
        capacity with nothing to spend it on. One parked game is enough to
        make it false, because that game can use the slot.

        False when unknown, deliberately: a decision to spend more because
        no one is queued should require having read that no one is queued.
        """
        return self.waiting == 0
