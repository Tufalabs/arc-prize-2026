"""The ``Agent`` ABC (requirements.md §1, §3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

from platypus.context import MoveContext
from platypus.contract import (
    Action,
    ContractViolationError,
    GameResult,
    GiveUp,
    GiveUpError,
    Observation,
    RunConfig,
)


@dataclass(kw_only=True)
class Agent(ABC):
    """Abstract per-game player: pure state plus configuration (A1) —
    everything on the instance must pickle, and templates must deepcopy
    cheaply (A3). Shared services arrive per call via the context and are
    never stored. ALL configuration lives on the agent; solver classes
    hold none.

    Subclasses implement the underscored hooks; the public ``do_move`` /
    ``on_finish`` are framework-owned.
    """

    # Cadences the agent CLASS could ever run under. The cadence an
    # INSTANCE actually supports is reported by ``supported_checkpointing()``
    # below, which may narrow this from configuration.
    default_checkpointing_support: ClassVar[frozenset[str]] = frozenset({"move", "level", "off"})

    run_config: RunConfig = field(default_factory=RunConfig)

    def supported_checkpointing(self) -> frozenset[str]:
        """Snapshot cadences THIS agent can honor (A4: a kernel never survives
        a checkpoint boundary).

        Deliberately a method resolved at run start, not a class constant:
        whether an agent can be snapshotted at a move boundary can depend on
        its configuration — an agent that keeps a kernel warm across moves
        cannot be pickled at one. A class constant would be fixed before that
        configuration was even set, so infra would validate the wrong answer.

        Callers must consult this only after the run has started, and must
        not cache it across a configuration change."""
        return self.default_checkpointing_support

    async def do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        """One move (§2): any amount of work, then exactly one action to
        execute — or a give-up — or, after executing one or more actions
        in-move through ``ctx.execute``, ``None`` (the move loop verifies
        at least one action was executed). Converts a raised
        ``GiveUpError`` and validates the returned action. For a
        ``level_won`` observation this is a delivery-only call: the solver
        discards the returned action (a ``GiveUp`` is still honored)."""
        try:
            # Cast to object: the annotation promises Action | GiveUp | None,
            # but validating that promise at runtime is this wrapper's job.
            decision = cast(object, await self._do_move(ctx, obs))
        except GiveUpError as e:
            return GiveUp(e.reason)
        if decision is None or isinstance(decision, GiveUp):
            return decision
        if not isinstance(decision, Action):
            raise ContractViolationError(
                f"_do_move returned {type(decision).__name__}, expected Action | GiveUp | None"
            )
        if decision.name not in obs.available_actions:
            raise ContractViolationError(
                f"_do_move returned unavailable action {decision.name!r}; available: {obs.available_actions}"
            )
        return decision

    # Everything runs on one event loop today, so long CPU-bound work in
    # _do_move must yield (any await) at least every ~5 s — TAAF's R12.02 —
    # or cancellation and other games stall. Heavy compute belongs behind
    # the ctx compute facility once it exists.
    @abstractmethod
    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp | None:
        """The agent's move. Implemented by subclasses; the public `do_move`
        above owns the framework half (give-up conversion, validation)."""
        ...

    async def on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        """Resource cleanup / final bookkeeping only — no LLM calls (I5).
        Runs on every terminal state, best-effort after a crash."""
        await self._on_finish(ctx, result)

    async def _on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        """Optional end-of-game hook; does nothing unless a subclass needs it."""

    async def _host_setup(self, shared: dict[str, Any]) -> None:
        """Optional; runs ONCE on the host process before the per-game
        fan-out (§4). Not yet called by ``AgentSolver``."""
