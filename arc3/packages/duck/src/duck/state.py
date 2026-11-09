"""Typed per-game state for the duck's bookkeeping spine.

The duck-side record of what the game did: the aligned timeline the
kernel's runtime views are built from, the drained-batch record and the
step summary it becomes, and the journal entries. Host-side values only —
the model-visible wire shapes (the kernel state payload, the compact
``last_action_result`` the model reads as a global) remain plain dicts
built at the serialization seam, because their key presence and insertion
order are part of what the model sees.

Shaped congruent with the whale's ``trace.Step`` / ``LevelTrace`` — one
entry per executed action carrying the settled frame and the animation
that produced it — so a later world-model replay substrate is an adapter
over this rather than a rewrite of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FramePayload:
    """One settled or animation frame, as serialized for the kernel."""

    grid_b64: str
    shape: list[int]
    step: int
    level: int

    def payload(self) -> dict[str, Any]:
        """The kernel-payload dict. Key order is model-observable (a dict's
        repr preserves it), so it is fixed here and nowhere else."""
        return {"grid_b64": self.grid_b64, "shape": self.shape, "step": self.step, "level": self.level}


@dataclass
class TimelineEntry:
    """One aligned-timeline step: the action's display name (empty for the
    game-initial seed entry), the settled frame it produced, and the
    animation ending in that frame."""

    action: str
    frame: FramePayload
    animation: list[FramePayload]


@dataclass
class PendingLevelWin:
    """A level-winning action awaiting its split new-level observation.

    The winning action's animation spans both observations, so the entry
    is held here until the new level's initial observation supplies its
    settled endpoint."""

    display: str
    step: int
    animation: list[FramePayload]


@dataclass
class PendingBatch:
    """The drained-but-unreported execution record one move accumulates,
    folded action by action and converted to a :class:`StepSummary` when
    the next turn reports it."""

    executed_actions: list[str] = field(default_factory=lambda: list[str]())
    executed_count: int = 0
    level_transition: bool = False
    run_complete: bool = False
    game_over: bool = False
    board_changed: bool = False
    stop_reason: str | None = None
    level: int | None = None


@dataclass
class StepSummary:
    """What the previous batch of actions did, held for the next user prompt.

    Exactly what that prompt renders, no wider: a field here is a field the
    model reads about. The batch's own bookkeeping — where a level win
    resolved, why it stopped short — stays on :class:`PendingBatch`, which
    is where it is used.
    """

    executed_count: int
    executed_actions: list[str]
    level: int | None
    level_transition: bool
    run_complete: bool
    game_over: bool


@dataclass
class MemoryEntry:
    """One host-timestamped journal entry."""

    index: int
    response: int
    step: int
    level: int
    last_action: str | None
    text: str
