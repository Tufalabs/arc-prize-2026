"""Typed per-game state for the seal's bookkeeping spine.

The seal-side record of what the game did: the aligned timeline the
kernel's runtime views are built from, the drained-batch record and the
step summary it becomes, and the journal entries. Host-side values only —
the model-visible wire shapes (the kernel state payload, the result
``submit_action`` returns) remain plain dicts
built at the serialization seam, because their key presence and insertion
order are part of what the model sees.

Shaped congruent with the whale's ``trace.Step`` / ``LevelTrace`` — one
entry per executed action carrying the settled frame and the animation
that produced it — so a later world-model replay substrate is an adapter
over this rather than a rewrite of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from platypus import Action

#: What one move changed on the settled board: how many cells, and the
#: colours those cells held before and after, each sorted.
MoveChange = tuple[int, tuple[int, ...], tuple[int, ...]]


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
    """One aligned-timeline step: the action (None for the level-initial
    seed entry), the settled frame it produced, the animation
    ending in that frame, and how the game stood after it.

    ``outcome`` is the field the duck's record has no equivalent for, and the
    one a world-model replay checks a prediction against: 'playing', 'won' or
    'lost', in the whale's vocabulary rather than the engine's."""

    action: Action | None
    frame: FramePayload
    animation: list[FramePayload]
    outcome: str = "playing"
    #: The level this turn was PLAYED on, which is not always its settled
    #: frame's level: a level-winning action's endpoint is the next level's
    #: opening board. The kernel slices its per-level history by this, so
    #: getting the two confused would file a win under the level it opened.
    level: int = 0
    #: The ``plan`` of the python call that played it (``plan_argument``).
    plan: str | None = None


@dataclass
class PendingLevelWin:
    """A level-winning action awaiting its split new-level observation.

    The winning action's animation spans both observations, so the entry
    is held here until the new level's initial observation supplies its
    settled endpoint."""

    action: Action
    step: int
    animation: list[FramePayload]
    level: int
    plan: str | None = None


@dataclass
class PendingBatch:
    """The drained-but-unreported execution record one move accumulates,
    folded action by action and converted to a :class:`StepSummary` when
    the next turn reports it."""

    #: Each move as the executed-actions line names it.
    executed_actions: list[str] = field(default_factory=lambda: list[str]())
    #: Each move's entry position in ``history_entries`` (None for a level
    #: win until the new level's first frame files its turn) and how many
    #: animation frames its turn keeps, in step with ``executed_actions``.
    turns: list[tuple[int | None, int]] = field(default_factory=lambda: list[tuple[int | None, int]]())
    #: What each move changed on the board, in step with
    #: ``executed_actions``: None for a level win, whose settled frame is the
    #: next level's.
    changes: list[MoveChange | None] = field(default_factory=lambda: list[MoveChange | None]())
    executed_count: int = 0
    level_transition: bool = False
    run_complete: bool = False
    game_over: bool = False
    board_changed: bool = False
    stop_reason: str | None = None
    level: int | None = None
    #: Set when the kernel's world model stopped the batch: ``"mismatch"``
    #: or ``"unknown"``.
    world_model_stop: str | None = None
    #: How the move's last batch that executed anything was sent
    #: (``seal.agent._Batch.route``).
    route: str | None = None
    #: The last attempt in the batch that repeated an earlier one on its
    #: level, RESET included, as the reminder names its moves
    #: (``SealAgent.replay_reminder``); None when none did.
    repeated: list[str] | None = None
    #: How many cells the batch's last RESET left differing from the board its
    #: level opened on, where a world model is in use; None when every RESET
    #: restored it.
    irregular_reset: int | None = None


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
    #: Why the world model stopped the batch, if it did: ``"mismatch"`` or
    #: ``"unknown"``.
    world_model_stop: str | None = None
    #: In step with ``executed_actions``: None for a move without animation
    #: frames, else how many its turn keeps and its index in the history the
    #: model has now (None when its turn left with a won level).
    animations: list[tuple[int, int | None] | None] = field(
        default_factory=lambda: list[tuple[int, int | None] | None]()
    )
    #: In step with ``executed_actions`` where moves are listed with what
    #: they changed (``SealAgent.move_changes``), else empty.
    changes: list[MoveChange | None] = field(default_factory=lambda: list[MoveChange | None]())
    #: Whether the last batch ran checked by a passing world model: None when
    #: it went neither checked nor unchecked (exploring, or no gate at all).
    checked: bool | None = None
    #: Whether the reflection turn after the batch's level win already gave
    #: its report: the next prompt then opens at the level change.
    reported: bool = False
    #: An attempt the batch repeated from the level's start and ended in
    #: RESET, as its moves are named (``SealAgent.replay_reminder``); None
    #: when it repeated none.
    repeated: list[str] | None = None
    #: How many cells a RESET in the batch left differing from the board its
    #: level opened on (``PendingBatch.irregular_reset``); None when none did.
    irregular_reset: int | None = None


@dataclass
class MemoryEntry:
    """One host-timestamped journal entry."""

    index: int
    response: int
    step: int
    level: int
    last_action: str | None
    text: str
