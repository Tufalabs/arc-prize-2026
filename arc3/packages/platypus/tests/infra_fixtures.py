"""Fixture agents/games for infra tests. Deliberately misbehaving or
scripted — these test platypus, they are not reference agents."""

from __future__ import annotations

import asyncio
import os
import pathlib
import random
import signal
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import arcengine
import numpy as np
from taaf.game import GameState
from taaf.game_examples import ExampleGame

from platypus import Action, Agent, GameResult, GiveUp, Move, MoveContext, Observation, RunConfig


def _inprocess_config() -> RunConfig:
    return RunConfig(process_mode="inprocess")


@dataclass
class ProbeAgent(Agent):
    """Plays a fixed script of action names, then gives up. Records every
    observation it is handed and every on_finish into class-level logs
    (deepcopied per-game agent copies share the class, so tests can see
    inside the run)."""

    # ClassVar-log introspection can't cross a process boundary, so these
    # fixtures default to in-process hosting; subprocess tests assert on
    # engine-side records instead.
    run_config: RunConfig = field(default_factory=_inprocess_config)

    script: tuple[str, ...] = ()
    give_up_reason: str = "script exhausted"
    give_up_on_level_won: bool = False
    # Await before each decision. A scripted agent otherwise finishes a
    # whole game inside a millisecond, which is too fast for any test that
    # needs something else to happen WHILE a game is in flight.
    move_delay_s: float = 0.0

    # -- state --
    step: int = 0

    seen: ClassVar[list[Observation]] = []
    finishes: ClassVar[list[tuple[str, GameResult, tuple[Move, ...]]]] = []

    @classmethod
    def reset_log(cls) -> None:
        cls.seen.clear()
        cls.finishes.clear()

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        ProbeAgent.seen.append(obs)
        if self.move_delay_s:
            await asyncio.sleep(self.move_delay_s)
        if obs.outcome == "level_won":  # delivery-only call: ack with anything
            return GiveUp("saw level_won") if self.give_up_on_level_won else Action("RESET")
        if self.step >= len(self.script):
            return GiveUp(self.give_up_reason)
        name = self.script[self.step]
        self.step += 1
        return Action(name)

    async def _on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        ProbeAgent.finishes.append((ctx.game.game_id, result, tuple(ctx.history)))


@dataclass
class SlowAgent(ProbeAgent):
    """Blocks on an await forever — cancellation always lands mid-move."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        ProbeAgent.seen.append(obs)  # a test can wait for the move to be reached
        await asyncio.sleep(3600)
        return GiveUp("unreachable")


@dataclass
class CrashingAgent(ProbeAgent):
    """Raises after playing its script."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        if self.step >= len(self.script):
            raise RuntimeError("deliberate agent crash")
        return await super()._do_move(ctx, obs)


@dataclass
class BadReturnAgent(ProbeAgent):
    """Returns a plain string instead of an Action."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        return cast(Any, "ACTION1")


@dataclass
class UnavailableActionAgent(ProbeAgent):
    """Returns a well-formed Action the game does not offer."""

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        return Action("ACTION5")


@dataclass
class LosableExampleGame(ExampleGame):
    """ExampleGame plus a lose button: ACTION5 lands the run in
    arcengine.GameState.GAME_OVER (a recoverable dead-end, like the real
    engine); RESET recovers."""

    def _execute_action(self, action: arcengine.ActionInput) -> GameState:
        if action.id == arcengine.GameAction.ACTION5:
            return self._build_state(state=arcengine.GameState.GAME_OVER)
        return super()._execute_action(action)

    def _build_state(self, state: arcengine.GameState) -> GameState:
        game_state = super()._build_state(state=state)
        if arcengine.GameAction.ACTION5.value not in game_state.raw.available_actions:
            game_state.raw.available_actions.append(arcengine.GameAction.ACTION5.value)
        return game_state


@dataclass
class PrintingAgent(ProbeAgent):
    """Prints through ``print`` and writes to descriptor 1 directly, as native
    code would, then plays its script; ``die`` SIGKILLs it after writing.
    Only ever construct with subprocess hosting."""

    run_config: RunConfig = field(default_factory=RunConfig)
    die: bool = False

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        if self.step == 0:
            print("a line the child prints")
            print("a partial line the child flushes", end="", flush=True)
            os.write(1, b"a line native code writes\n")
            if self.die:
                os.kill(os.getpid(), signal.SIGKILL)
        return await super()._do_move(ctx, obs)


@dataclass
class SelfKillingAgent(ProbeAgent):
    """SIGKILLs its own process mid-move — the child-death-without-protocol
    case. Only ever construct with subprocess hosting (an in-process run
    would kill pytest)."""

    run_config: RunConfig = field(default_factory=RunConfig)

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        os.kill(os.getpid(), signal.SIGKILL)
        return GiveUp("unreachable")


@dataclass
class HungryAgent(ProbeAgent):
    """Allocates far past any sane memory limit (RLIMIT_AS breach path)."""

    run_config: RunConfig = field(default_factory=RunConfig)
    allocate_bytes: int = 32 * 2**30

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        block = np.ones(self.allocate_bytes // 8)
        return GiveUp(f"allocated {block.nbytes} bytes")


def hold_mapped_file(path: str, ready: Any, stop: Any) -> None:
    """Map ``path`` read-only, touch every page, then wait to be released.

    A spawn target for the memory tests: N of these resident at once share
    one set of physical pages, which is the situation the tree total has
    to see through. Module-level and in this file because spawn resolves
    targets by import."""
    import mmap

    with open(path, "rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        checksum = 0
        for offset in range(0, len(mapped), 4096):
            checksum += mapped[offset]
        ready.set()
        stop.wait(60)


@dataclass
class StatusPublishingAgent(ProbeAgent):
    """Holds one status label across a bounded await, then gives up — long
    enough for the monitor to tick while the label is live. Subprocess by
    default: the label crossing the control pipe is the part that can
    silently stop working."""

    run_config: RunConfig = field(default_factory=RunConfig)
    label: str = "waiting for LLM response"
    hold_s: float = 1.0

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        with ctx.status(self.label):
            await asyncio.sleep(self.hold_s)
        return GiveUp("held long enough")


@dataclass
class ResultDumpingSlowAgent(SlowAgent):
    """Marks readiness, blocks forever, and dumps its on_finish result to a
    per-game file — the only way a test can observe what a
    subprocess-hosted agent was told (and when it is safely cancellable)."""

    dump_dir: str = ""
    # Crash in this game only, once ``crash_after_ready`` games have reached
    # their move loop: lets one game fail while its siblings are provably
    # in flight, without racing on startup order.
    crash_on_game_id: str | None = None
    crash_after_ready: int = 0
    run_config: RunConfig = field(default_factory=RunConfig)

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        dump_dir = pathlib.Path(self.dump_dir)
        (dump_dir / f"{ctx.game.game_id}.ready").touch()
        if self.crash_on_game_id is not None and ctx.game.game_id == self.crash_on_game_id:
            for _ in range(600):
                if len(list(dump_dir.glob("*.ready"))) >= self.crash_after_ready:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("siblings never became ready")
            raise RuntimeError("deliberate agent crash")
        return await super()._do_move(ctx, obs)

    async def _on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        path = pathlib.Path(self.dump_dir) / f"{ctx.game.game_id}.result"
        path.write_text(f"{result.state} {result.levels_completed} {result.score}")


@dataclass
class LevelLearnerAgent(Agent):
    """Learns each level of an ``ExampleGame`` by trial and error, so games
    that reach the same level get there with different histories.

    Reads its in-level progress off the board (``ExampleGame`` fills it with
    ``level * 4 + progress``), remembers every step it has confirmed and
    every action that reset it, and explores at random among the rest. What
    it has learnt, its decision log and — under ``own_rng`` — its random
    stream are all instance state, which is what a checkpoint must carry. The
    action it learns from is read off the canonical history
    (``ctx.history``), not kept on itself, so a game started from a
    checkpoint plays only if the source's moves came with it.

    ``own_rng`` draws from a stream seeded once off ``ctx.rng`` and kept on
    the agent, so a game started from a checkpoint replays the source game's
    continuation exactly. Off, every move draws from ``ctx.rng``, and branches
    of one checkpoint diverge by their seeds."""

    run_config: RunConfig = field(default_factory=_inprocess_config)
    own_rng: bool = False

    # -- state --
    rng: Any = None
    known: dict[int, list[str]] = field(default_factory=lambda: dict[int, list[str]]())
    wrong: dict[tuple[int, int], set[str]] = field(default_factory=lambda: dict[tuple[int, int], set[str]]())
    last: tuple[int, int] | None = None  # (level, progress) of the last decision
    # (observation index, level, action) of every decision, in order
    log: list[tuple[int, int, str]] = field(default_factory=lambda: list[tuple[int, int, str]]())

    finishes: ClassVar[list[tuple[str, int, list[tuple[int, int, str]]]]] = []

    @staticmethod
    def _progress(obs: Observation) -> int:
        return (int(obs.frame.data[0, 0]) - 4 * obs.level) % 16

    def _learn(self, ctx: MoveContext, obs: Observation, progress: int) -> None:
        if self.last is None:
            return
        level, before = self.last
        action = ctx.history[-1].action.name
        steps = self.known.setdefault(level, [])
        if obs.level > level or progress == before + 1:
            if len(steps) == before:
                steps.append(action)
        else:
            self.wrong.setdefault((level, before), set()).add(action)

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        if obs.outcome == "level_won":  # delivery-only call: ack with anything
            return Action("RESET")
        if self.own_rng and self.rng is None:
            self.rng = random.Random(ctx.rng.getrandbits(64))
        rng = self.rng if self.own_rng else ctx.rng
        progress = self._progress(obs)
        self._learn(ctx, obs, progress)
        steps = self.known.get(obs.level, [])
        if progress < len(steps):
            name = steps[progress]
        else:
            ruled_out = self.wrong.get((obs.level, progress), set())
            candidates = [a for a in obs.available_actions if a != "RESET" and a not in ruled_out]
            name = rng.choice(sorted(candidates))
        self.last = (obs.level, progress)
        self.log.append((obs.index, obs.level, name))
        return Action(name)

    async def _on_finish(self, ctx: MoveContext, result: GameResult) -> None:
        LevelLearnerAgent.finishes.append((ctx.game.game_id, ctx.game.pass_index, list(self.log)))
