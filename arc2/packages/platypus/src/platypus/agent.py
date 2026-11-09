"""The ``Agent`` ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import cast

import taaf.game
from platypus.context import MoveContext
from platypus.contract import (
    ContractViolationError,
    GameOutcome,
    GiveUp,
    GiveUpError,
    RunConfig,
)


@dataclass(kw_only=True)
class Agent(ABC):
    """Abstract per-game player: pure state plus configuration — everything
    on the instance must pickle, and templates must deepcopy cheaply. Shared
    services arrive per call via the context and are never stored. ALL
    configuration lives on the agent; solver classes hold none.

    Subclasses implement the underscored hooks; the public ``do_move`` /
    ``on_finish`` are framework-owned.
    """

    run_config: RunConfig = field(default_factory=RunConfig)

    async def do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Play the game: any amount of work, answering through
        ``ctx.submit`` as often as it likes, then return.

        Called exactly once. Returning ends the game with whatever was last
        submitted standing; returning ``GiveUp`` ends it early and says why,
        which frees the slot without withdrawing the answer. Converts a
        raised ``GiveUpError`` and validates the return type.
        """
        try:
            # Cast to object: the annotation promises GiveUp | None, but
            # validating that promise at runtime is this wrapper's job.
            decision = cast(object, await self._do_move(ctx, view))
        except GiveUpError as e:
            return GiveUp(e.reason)
        if decision is None or isinstance(decision, GiveUp):
            return decision
        raise ContractViolationError(
            f"_do_move returned {type(decision).__name__}, expected GiveUp | None — "
            "answers go through ctx.submit, not the return value"
        )

    # Everything runs on one event loop today, so long CPU-bound work in
    # _do_move must yield (any await) regularly, or cancellation and other
    # games stall. Heavy compute belongs behind the ctx compute facility
    # once it exists.
    @abstractmethod
    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """The agent's play. Implemented by subclasses; the public
        ``do_move`` above owns the framework half (give-up conversion,
        validation)."""
        ...

    async def on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Resource cleanup / final bookkeeping only — no LLM calls, and
        ``ctx.submit`` is refused by then. Runs on every terminal state,
        best-effort after a crash."""
        await self._on_finish(ctx, outcome)

    async def _on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Optional end-of-game hook; does nothing unless a subclass needs it."""


__all__ = ["Agent"]
