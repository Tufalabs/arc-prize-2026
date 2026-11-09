"""One game sacrificed to prove the run is alive.

Kaggle's arena ends a run that goes too long without any move. A real
agent's first turn can exceed that window on its own — a model load plus
a long opening deliberation is minutes of silence — and the run dies
before it has played anything.

The answer is to give up one game. Infra takes it away from the agent
before the fan-out and plays it here, in the host process, one uniformly
random action every ``interval_s``. No subprocess, no LLM, no agent
involvement: the agent is simply handed a shorter list and never learns
the game existed. That makes this work for any agent, including ones
that have not been written yet.

The cost is honest and bounded: one game scores as a random walk. The
sacrifice is deliberate — a run that dies at minute fifteen scores
nothing at all.

Action choice is owned here rather than delegated, so the loop depends on
nothing outside this package. The awkward cases it therefore has to
handle itself are RESET after GAME_OVER and picking coordinates for a
CLICK.
"""

from __future__ import annotations

import asyncio
import contextlib
import random
import traceback
from typing import Optional

import arcengine
import taaf.game

from platypus.contract import COORDINATE_ACTIONS, Action, action_names

# A failing move is worth reporting and retrying; a persistently failing
# one means the game is unusable and the loop should stop rather than
# spend the run logging. At the default interval this is five minutes of
# trying.
_MAX_CONSECUTIVE_FAILURES = 30


class HeartbeatGame:
    """Plays ``game`` with random moves until stopped.

    Owned by the solver, started before any slow setup and stopped in its
    teardown. ``start`` is idempotent so a subclass can start the
    heartbeat early — covering its own engine build, which is exactly the
    silence the watchdog would otherwise catch — without the base class
    having to know whether it already ran.
    """

    def __init__(self, game: taaf.game.Game, interval_s: float):
        """Take over ``game`` as the sacrificed liveness game."""
        self.game = game
        self.interval_s = interval_s
        self._task: Optional[asyncio.Task[None]] = None

    @property
    def running(self) -> bool:
        """Whether the move loop is still going."""
        return self._task is not None

    def start(self) -> None:
        """Begin playing, idempotently — callers race deliberately (the LLM
        solver starts it before its slow engine build, the base solver again
        before the fan-out)."""
        if self._task is not None:
            return
        print(
            f"[heartbeat] game {self.game.game_id} plays random moves every "
            f"{self.interval_s:g}s in this process; the agent will not see it",
            flush=True,
        )
        self._task = asyncio.create_task(self._loop(), name="platypus-heartbeat")

    async def stop(self) -> None:
        """Stop the loop and close the game out.

        ``finish_game`` here is what makes the sacrificed game land in a
        terminal state (``gave_up``) instead of being stamped ``crashed``
        by the benchmark's teardown, which would misreport a game that did
        exactly what was asked of it."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        run = self.game.game_run
        if run is not None and run.state == "playing":
            try:
                self.game.finish_game()
            except Exception:
                print("[heartbeat] finish_game failed:", flush=True)
                traceback.print_exc()

    def _random_action(self, rng: random.Random) -> arcengine.ActionInput:
        """One legal action, chosen at random, for the liveness move.

        Owned here rather than delegated to taaf's example solver: this
        loop's entire job is to keep making moves, so it must not depend on
        a name from another package that nobody owes us. RESET is skipped
        while anything else is legal — always available, it would otherwise
        undo the walk about as often as it advances it.

        The exception is a lost board, where RESET is the only action that
        does anything: the engine still offers the others and answers them
        with an empty frame, which taaf raises on. Reading
        ``available_actions`` cannot see this — it does not shrink when the
        game ends."""
        state = self.game.current_state
        if state.raw.state == arcengine.GameState.GAME_OVER:
            return Action("RESET").to_action_input()
        names = [name for name in action_names(state.available_actions) if name != "RESET"]
        if not names:
            names = list(action_names(state.available_actions))
        name = rng.choice(names)
        if name in COORDINATE_ACTIONS:
            width, height = self.game.grid_size
            return Action(name, row=rng.randrange(height), col=rng.randrange(width)).to_action_input()
        return Action(name).to_action_input()

    async def _loop(self) -> None:
        """Play one random move every ``interval_s`` until the game ends, the
        task is cancelled, or moves keep failing."""
        rng = random.Random()
        failures = 0
        while True:
            await asyncio.sleep(self.interval_s)
            run = self.game.game_run
            if run is None or run.state != "playing":
                # A lucky random win, or a cancel that already finished it.
                print(
                    f"[heartbeat] game {self.game.game_id} is {None if run is None else run.state}; heartbeat stops",
                    flush=True,
                )
                return
            try:
                self.game.execute_action(self._random_action(rng))
                failures = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                print(f"[heartbeat] move on {self.game.game_id} failed ({failures} in a row):", flush=True)
                traceback.print_exc()
                if failures >= _MAX_CONSECUTIVE_FAILURES:
                    print(
                        f"[heartbeat] giving up on {self.game.game_id} after {failures} consecutive failures",
                        flush=True,
                    )
                    return
