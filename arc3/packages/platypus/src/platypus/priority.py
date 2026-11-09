"""Which games hold the machine's concurrency, by rank rather than by
arrival.

``asyncio.Semaphore`` hands a freed slot to whoever waited longest. Under
a deadline with more games than slots that is the wrong question: what
the run wants is the game most worth spending the next tokens on, and
"waited longest" says nothing about that. :class:`PrioritySlots` keeps
the same acquire/release shape and answers the other question — a freed
slot goes to the best-ranked waiter, ranked by ``rank_key`` over what
each game last reported about itself.

Three properties make it safe to swap in:

- **Rank never preempts.** It decides who wins the *next* grant and
  touches nothing that is running. A game gives its slot up only by
  finishing or by asking to (:meth:`yield_slot`), so no scheduler
  decision can cut a turn in half.
- **A yield by the top-ranked game is a no-op.** The yielder re-enters
  the queue *before* its slot is released, so the same pump that would
  hand the slot away hands it straight back when nothing outranks it.
  That is what lets an agent yield freely at every point its KV cache is
  gone anyway, without the host having to predict whether it is worth it.
- **With no ``rank_key`` it is a plain concurrency cap.** Every key
  then ties, arrival order breaks the tie, and grants are exactly FIFO.

Ranking is done at the moment of the grant, from the hints as they stand
then — never from a periodically-recomputed order. Comparing is free and
only *swaps* cost anything, so there is nothing to buy by deciding on
stale numbers, and plenty to lose: the events that most change a game's
standing are exactly the ones a cadence would miss. A game that has just
won a level is cheap again on its next comparison, not at the next tick.

The numbers themselves are exact where it matters: a waiter is not
spending, so what it reported on the way out is still true however long
it waits.
"""

from __future__ import annotations

import asyncio
import itertools
from typing import Any, Callable, Mapping

#: How a key sorts: its slot key and what it last reported, to a tuple,
#: best (smallest) first.
RankKey = Callable[[str, Mapping[str, Any]], "tuple[Any, ...]"]

#: Told of every grant: the key granted, and the key whose slot it takes —
#: ``None`` for a slot nobody held (a run's opening grants).
OnGrant = Callable[[str, "str | None"], None]


class PrioritySlots:
    """A concurrency cap of ``active`` slots, granted by rank.

    Keys are opaque strings owned by the caller (the solver uses the
    game's index in the run). Every method is called from one event loop;
    there is no locking because there is nothing to lock against.
    """

    def __init__(self, active: int, rank_key: RankKey | None = None, on_grant: OnGrant | None = None) -> None:
        """Cap the slots at ``active``, ranked by ``rank_key`` — or, with
        none, by arrival alone, which is the plain concurrency cap.
        ``on_grant`` hears which slot each grant hands over (a recorder of
        slot provenance); it must not touch the slots itself."""
        if active < 1:
            raise ValueError(f"active={active} must be at least 1")
        self._active = active
        self._rank_key: RankKey = rank_key if rank_key is not None else (lambda key, hints: ())
        self._held: set[str] = set()
        self._waiting: dict[str, asyncio.Future[None]] = {}
        self._arrival: dict[str, int] = {}
        self._ticket = itertools.count()
        self._hints: dict[str, dict[str, Any]] = {}
        self._on_grant = on_grant

    @property
    def active(self) -> int:
        """How many slots exist — the run's concurrency cap."""
        return self._active

    @property
    def running(self) -> int:
        """Keys currently holding a slot."""
        return len(self._held)

    @property
    def waiting(self) -> int:
        """Keys queued for one, parked yielders included."""
        return len(self._waiting)

    def hints(self, key: str) -> dict[str, Any]:
        """What ``key`` last reported about itself; empty if it never has."""
        return dict(self._hints.get(key, {}))

    def report(self, key: str, hints: Mapping[str, Any]) -> None:
        """Record what ``key`` says about itself without touching its slot.

        A game reports far more often than it changes hands, and the next
        comparison reads whatever stands here at the time — so reporting
        often is the whole mechanism for keeping grants current.
        """
        self._hints[key] = dict(hints)

    def live_keys(self) -> list[str]:
        """Every key the run still owes a slot to — holding or waiting.

        The population a ranking policy ranks: a retired key is gone from
        it, so the policy never has to know which games have finished.
        """
        return [*self._held, *self._waiting]

    def order(self) -> list[str]:
        """Every live key as it ranks right now, best first."""
        return sorted(self.live_keys(), key=self._sort_key)

    def admitted(self) -> list[str]:
        """The top ``active`` of the order — who the ranking says should be
        running.

        Not who *is* running: rank never preempts, so the two differ until
        enough slots change hands. That gap is what a host reports
        crossings of.
        """
        return self.order()[: self._active]

    async def acquire(self, key: str, hints: Mapping[str, Any] | None = None) -> None:
        """Take a slot, waiting for one when every slot is held.

        Returns without suspending when a slot is free, so a run whose
        games never yield never touches the event loop for a grant.
        """
        if key in self._held:
            raise RuntimeError(f"{key!r} already holds a slot")
        future = self._enqueue(key, hints)
        self._pump()
        await self._granted(key, future)

    def release(self, key: str) -> None:
        """Give a slot back; the best-ranked waiter takes it."""
        freed = key if key in self._held else None
        self._held.discard(key)
        self._pump(freed)

    def retire(self, key: str) -> None:
        """Give up ``key``'s whole claim on the run — slot, queue place and
        hints. It is done playing.

        A retiring key is not always the one holding a slot: a parked
        yielder waits in the queue while its own hosting can still unwind
        under it (a pause callback that raises, a host that gives up on
        it). Its queue entry has to go too, and before the pump — left
        there, the next freed slot is granted to a future nobody is
        waiting on, and the run loses that slot for the rest of its life.

        Dropping the hints as well keeps :meth:`live_keys` the exact
        population a policy should rank.
        """
        freed = key if key in self._held else None
        self._held.discard(key)
        queued = self._waiting.pop(key, None)
        self._arrival.pop(key, None)
        if queued is not None and not queued.done():
            queued.cancel()
        self._hints.pop(key, None)
        self._pump(freed)

    async def yield_slot(
        self,
        key: str,
        hints: Mapping[str, Any] | None = None,
        on_park: Callable[[], None] | None = None,
    ) -> bool:
        """Offer the slot back and take it again, reporting ``hints``.

        Returns whether it actually parked. ``on_park`` fires at the
        moment it does — before the wait, so a caller can say so while it
        is true rather than once it is over.

        Re-enqueues before releasing, so a key nothing outranks is handed
        its own slot straight back: an offer, never a surrender.
        """
        if key not in self._held:
            raise RuntimeError(f"{key!r} yielded a slot it does not hold")
        future = self._enqueue(key, hints)
        self._held.discard(key)
        self._pump(key)
        parked = not future.done()
        if parked and on_park is not None:
            on_park()
        await self._granted(key, future)
        return parked

    def _enqueue(self, key: str, hints: Mapping[str, Any] | None) -> asyncio.Future[None]:
        """Join the queue for a slot, recording what the key reports.

        A fresh arrival ticket each time: among equally-ranked keys a
        yielder goes behind those already waiting, which is what makes the
        unranked fallback a round-robin rather than a monopoly.
        """
        if hints is not None:
            self._hints[key] = dict(hints)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._waiting[key] = future
        self._arrival[key] = next(self._ticket)
        return future

    async def _granted(self, key: str, future: asyncio.Future[None]) -> None:
        """Wait for the slot, leaving nothing behind if the wait is cut.

        Cancellation can land either side of the grant, and the two need
        opposite cleanups: still queued, drop the queue entry; already
        granted, release the slot so it reaches someone who can use it.
        """
        try:
            await future
        except asyncio.CancelledError:
            if self._waiting.get(key) is future:
                del self._waiting[key]
                self._arrival.pop(key, None)
            else:
                self.release(key)
            raise

    def _pump(self, freed: str | None = None) -> None:
        """Fill every free slot with the best-ranked waiter. ``freed`` is the
        key whose slot just came free, if one did: the first grant takes
        that slot, any further ones slots nobody held."""
        while self._waiting and len(self._held) < self._active:
            key = min(self._waiting, key=self._sort_key)
            future = self._waiting.pop(key)
            self._arrival.pop(key, None)
            self._held.add(key)
            if not future.done():
                future.set_result(None)
            if self._on_grant is not None:
                self._on_grant(key, freed)
            freed = None

    def _sort_key(self, key: str) -> tuple[Any, ...]:
        """The policy's rank over what ``key`` last reported, arrival order
        breaking ties — evaluated at the moment of the comparison, never
        cached."""
        return (*self._rank_key(key, self._hints.get(key, {})), self._arrival.get(key, 0))


class StartPacer:
    """Spaces consecutive starts at least ``gap_s`` apart, in grant order.

    Callers await :meth:`wait` and proceed one per gap; a gap of zero (or
    less) lets everyone through at once.
    """

    def __init__(self, gap_s: float) -> None:
        self._gap_s = gap_s
        self._lock = asyncio.Lock()
        self._last: float | None = None

    async def wait(self) -> None:
        """Return once at least the gap has passed since the previous caller returned."""
        if self._gap_s <= 0:
            return
        async with self._lock:
            loop = asyncio.get_running_loop()
            if self._last is not None:
                delay = self._last + self._gap_s - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
            self._last = loop.time()
