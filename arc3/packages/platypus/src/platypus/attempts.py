"""A level's attempts, counted as they are played.

An attempt is the actions from the level's start, or from a RESET, up to and
including the RESET that ends it; a RESET straight after another is the
attempt ``[RESET]``. Each distinct attempt keeps its own count on the level,
so a model cycling through a few sequences is caught on whichever it repeats
most. Counting is constant time an action: the attempt in play is held as a
running hash, and the level's attempts as counts keyed by it. A level change
starts the counts over.
"""

from __future__ import annotations

from platypus.contract import Action

RESET = "RESET"


class Attempts:
    """One game's attempts on the level it is playing, and its actions there."""

    def __init__(self) -> None:
        """An empty record, before any action."""
        self._start(None)

    def _start(self, level: int | None) -> None:
        """Start the record of ``level``, nothing played on it yet."""
        self.level = level
        #: Actions played on the level, RESETs included.
        self.actions = 0
        #: The attempt the last RESET closed; empty before the level's first.
        self.closed: list[Action] = []
        #: How many times that attempt has now been played on the level.
        self.closed_count = 0
        self._current: list[Action] = []
        self._hash = 0
        self._counts: dict[int, int] = {}

    def record(self, action: Action, level: int) -> int:
        """Count ``action``, played on ``level``; when it is a RESET, return how
        many times the attempt it closes has now been played on the level,
        else 0."""
        if level != self.level:
            self._start(level)
        self.actions += 1
        self._current.append(action)
        self._hash = hash((self._hash, action.name, action.row, action.col))
        if action.name != RESET:
            return 0
        count = self._counts.get(self._hash, 0) + 1
        self._counts[self._hash] = count
        self.closed, self.closed_count = self._current, count
        self._current, self._hash = [], 0
        return count
