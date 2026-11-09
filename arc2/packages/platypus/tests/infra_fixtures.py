"""Fixture agents for infra tests. Deliberately misbehaving or scripted —
these test platypus, they are not reference agents."""

from __future__ import annotations

import asyncio
import os
import pathlib
import signal
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, cast

import numpy as np

import taaf.game
from platypus import Agent, GameOutcome, GiveUp, MoveContext, RunConfig, Submission
from taaf.grid import Grid


def _inprocess_config() -> RunConfig:
    """Hosting in this process, so ClassVar logs are visible to tests."""
    return RunConfig(process_mode="inprocess", status_interval_s=None)


def mirror(view: taaf.game.GameView) -> Submission:
    """The answer the example tasks are built to reward."""
    return Submission(attempts=tuple((Grid(np.fliplr(q.data)), q) for q in view.questions))


@dataclass
class ProbeAgent(Agent):
    """Submits the mirror ``n_answers`` times, then returns. Records every
    view it is handed and every on_finish into class-level logs (deepcopied
    per-game agent copies share the class, so tests can see inside the
    run)."""

    # ClassVar-log introspection can't cross a process boundary, so these
    # fixtures default to in-process hosting; subprocess tests assert on
    # game-side records instead.
    run_config: RunConfig = field(default_factory=_inprocess_config)

    n_answers: int = 1
    # Await before each answer. A scripted agent otherwise finishes a whole
    # game inside a millisecond, which is too fast for any test that needs
    # something else to happen WHILE a game is in flight.
    answer_delay_s: float = 0.0
    # State: how many times THIS copy was asked. Zero on the template after
    # a run is what proves the per-game copies are copies.
    calls: int = 0

    seen: ClassVar[list[taaf.game.GameView]] = []
    # (game_id, outcome, the finishing copy's ``calls``)
    finishes: ClassVar[list[tuple[str, GameOutcome, int]]] = []

    @classmethod
    def reset_log(cls) -> None:
        """Clear the shared logs between tests."""
        cls.seen.clear()
        cls.finishes.clear()

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Submit the mirror ``n_answers`` times."""
        self.calls += 1
        ProbeAgent.seen.append(view)
        for _ in range(self.n_answers):
            await asyncio.sleep(self.answer_delay_s)
            ctx.submit(mirror(view))
        return None

    async def _on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Record what the agent was told about the end of its game."""
        ProbeAgent.finishes.append((ctx.game_id, outcome, self.calls))


@dataclass
class RngDumpingAgent(ProbeAgent):
    """Writes the first draw off ``ctx.rng`` to a file under ``dump_dir``
    named by game id plus a nonce — the one observation of a game's seed
    that works from either process mode and across pass copies."""

    dump_dir: str = ""
    run_config: RunConfig = field(default_factory=RunConfig)

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Record one draw, answer, and return."""
        path = pathlib.Path(self.dump_dir) / f"{ctx.game_id}.{uuid.uuid4().hex}.rng"
        path.write_text(repr(ctx.rng.random()))
        ctx.submit(mirror(view))
        return None


@dataclass
class SlowAgent(ProbeAgent):
    """Blocks on an await forever — cancellation always lands mid-call."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Answer once, then block until something cancels this."""
        for _ in range(self.n_answers):
            ctx.submit(mirror(view))
        ProbeAgent.seen.append(view)  # a test can wait for the call to be reached
        await asyncio.sleep(3600)
        return GiveUp("unreachable")


@dataclass
class CrashingAgent(ProbeAgent):
    """Raises after submitting its answers."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Answer, then fail."""
        await super()._do_move(ctx, view)
        raise RuntimeError("deliberate agent crash")


@dataclass
class BadReturnAgent(ProbeAgent):
    """Returns a submission instead of routing it through ``ctx.submit``."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Return the wrong type entirely."""
        return cast(Any, mirror(view))


@dataclass
class MisShapedAgent(ProbeAgent):
    """Submits an answer covering the wrong number of pairs."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Submit one attempt-pair too many."""
        attempts = tuple((Grid(np.fliplr(q.data)), q) for q in view.questions)
        ctx.submit(Submission(attempts=(*attempts, attempts[0])))
        return None


@dataclass
class LateSubmitAgent(ProbeAgent):
    """Keeps ``ctx.submit`` and calls it from ``on_finish``, after the call
    that armed it has ended."""

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Answer normally, then stash the view for the late attempt."""
        ctx.submit(mirror(view))
        self._view = view
        return None

    async def _on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Submit when the contract says the channel is closed."""
        ctx.submit(mirror(self._view))


@dataclass
class PrintingAgent(ProbeAgent):
    """Prints through ``print`` and writes to descriptor 1 directly, as native
    code would, then answers; ``die`` SIGKILLs it after writing. Only ever
    construct with subprocess hosting."""

    run_config: RunConfig = field(default_factory=RunConfig)
    die: bool = False

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Write both ways first."""
        print("a line the child prints")
        print("a partial line the child flushes", end="", flush=True)
        os.write(1, b"a line native code writes\n")
        if self.die:
            os.kill(os.getpid(), signal.SIGKILL)
        return await super()._do_move(ctx, view)


@dataclass
class SelfKillingAgent(ProbeAgent):
    """SIGKILLs its own process mid-call — the child-death-without-protocol
    case. Only ever construct with subprocess hosting (an in-process run
    would kill pytest)."""

    run_config: RunConfig = field(default_factory=RunConfig)

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Die without warning."""
        os.kill(os.getpid(), signal.SIGKILL)
        return GiveUp("unreachable")


@dataclass
class HungryAgent(ProbeAgent):
    """Allocates far past any sane memory limit (RLIMIT_AS breach path)."""

    run_config: RunConfig = field(default_factory=RunConfig)
    allocate_bytes: int = 32 * 2**30

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Ask for more address space than the cap allows."""
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

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Publish a label, hold it, then stop."""
        with ctx.status(self.label):
            await asyncio.sleep(self.hold_s)
        return GiveUp("held long enough")


@dataclass
class OutcomeDumpingSlowAgent(SlowAgent):
    """Marks readiness, blocks forever, and dumps its on_finish outcome to a
    per-game file — the only way a test can observe what a
    subprocess-hosted agent was told (and when it is safely cancellable)."""

    dump_dir: str = ""
    # Crash in this game only, once ``crash_after_ready`` games have reached
    # their call: lets one game fail while its siblings are provably in
    # flight, without racing on startup order.
    crash_on_game_id: str | None = None
    crash_after_ready: int = 0
    run_config: RunConfig = field(default_factory=RunConfig)

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Announce readiness, optionally crash, else block."""
        dump_dir = pathlib.Path(self.dump_dir)
        (dump_dir / f"{ctx.game_id}.ready").touch()
        if self.crash_on_game_id is not None and ctx.game_id == self.crash_on_game_id:
            for _ in range(600):
                if len(list(dump_dir.glob("*.ready"))) >= self.crash_after_ready:
                    break
                await asyncio.sleep(0.05)
            else:
                raise AssertionError("siblings never became ready")
            raise RuntimeError("deliberate agent crash")
        return await super()._do_move(ctx, view)

    async def _on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Write what the agent was told, where the parent can read it."""
        path = pathlib.Path(self.dump_dir) / f"{ctx.game_id}.outcome"
        path.write_text(f"{outcome.state} {outcome.n_submitted}")
