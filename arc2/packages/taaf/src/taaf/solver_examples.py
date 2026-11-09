"""Example solvers: the reference agent, and one that exercises the
solver-provided diagnostics seams."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from html import escape
from typing import Any

import numpy as np

import taaf.game
import taaf.trace
from taaf.game import N_ATTEMPTS, Game, GameView, Submission
from taaf.grid import Grid
from taaf.solver import Solver


def flip_lr(view: GameView) -> Submission:
    """Answer every question with its left-right mirror.

    Attempt 2 is the question unchanged rather than a copy of attempt 1, so
    the two-attempt machinery is genuinely exercised end to end instead of
    being duplicated past.
    """
    return Submission(
        attempts=tuple((Grid(np.fliplr(q.data)), q) for q in view.questions),
    )


def all_black(view: GameView) -> Submission:
    """Answer every question with an all-zero grid of its shape.

    A deliberately wrong answer that is still well-formed, so the demo can
    show a solver replacing one answer with another without either being a
    shape error.
    """
    return Submission(
        attempts=tuple((Grid(np.zeros_like(q.data)),) * N_ATTEMPTS for q in view.questions),
    )


#: The answers the demo solver knows how to build, by name.
ANSWERS: dict[str, Callable[[GameView], Submission]] = {"flip": flip_lr, "black": all_black}

#: What the demo solver submits, in order. It changes its mind twice before
#: settling on the mirror, so on a task the mirror solves the lane's score
#: reads 0 -> 100 -> 0 -> 100 as it spends. A score that goes *down* is the
#: case a curve reader has to be able to see, and nothing produces it unless a
#: solver replaces a right answer with a wrong one.
DEMO_ANSWERS = ("black", "flip", "black", "flip")


@dataclass
class SolverFlipLR(Solver):
    """The reference solver: submit each question mirrored left-right.

    Deterministic and free, which makes it the end-to-end check for every
    deployment path — the answer a run should produce is known before it
    starts. Across ARC-AGI-2's public training set exactly one task
    (``67a3c6ac``) is a left-right flip throughout, and none of the public
    evaluation set is.

    It submits ``answers`` in order, replacing its own answer each time and
    settling on the mirror. One answer would exercise the harness just as
    well; several exercise the part of it that only a solver changing its
    mind can reach.

    ``delay_s`` sleeps *before* each answer, which is what makes a run
    observable in flight and cancellable between two answers; it defaults to
    not sleeping at all. ``report_tokens`` is charged per answer — leave it at
    zero and every lane's cost is zero, which is a curve with no x-axis.
    """

    label: str = "flip_lr"
    delay_s: float = 0.0
    report_tokens: int = 0
    answers: tuple[str, ...] = DEMO_ANSWERS

    async def _run_games(self, games: list[Game]) -> None:
        """Answer each game, in order, settling on the mirror."""
        for game in games:
            try:
                await self._answer(game)
            except asyncio.CancelledError:
                # The contract: finish every still-playing game before
                # re-raising, so it lands `cancelled` rather than `crashed`
                # and the answer it had reached is the one scored.
                for pending in games:
                    if pending.game_run is not None and pending.game_run.state == "playing":
                        pending.finish_game()
                raise
            game.finish_game()

    async def _answer(self, game: Game) -> None:
        """Submit every answer in ``answers`` for one game."""
        view = game.query()
        for name in self.answers:
            # Yield before answering: solvers must hand control back
            # regularly, and a cancel must be able to land between two
            # answers, on a game that already has one standing.
            await asyncio.sleep(self.delay_s)
            game.execute_action(ANSWERS[name](view), generated_tokens=self.report_tokens)


@dataclass
class SolverAnalysing(SolverFlipLR):
    """``SolverFlipLR`` plus the solver-owned diagnostics seams — a note per
    game and one HTML analysis page — so those paths have a user in tests and
    in the demo."""

    label: str = "flip_lr_analysing"
    _seen: list[str] = field(default_factory=lambda: list[str](), init=False)

    async def _run_games(self, games: list[Game]) -> None:
        """As :class:`SolverFlipLR`, plus a per-game note."""
        for game in games:
            await self._answer(game)
            run = game.game_run
            assert run is not None
            # Written before finishing, so it reaches the finish line — and
            # phrased in what the solver knows. Whether it was right is not
            # that: scoring happens in finish_game and is never handed back.
            run.solver_note = f"settled after {len(run.history)} answer(s)"
            self._seen.append(run.game_id)
            game.finish_game()

    def _teardown(self) -> None:
        """Write the solver's own analysis page."""
        if self.job_dir is None:
            return
        rows = "\n".join(f"<tr><td><code>{escape(g)}</code></td></tr>" for g in self._seen)
        path = self.job_dir / "solver_analysis" / "flip_lr.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"<!doctype html><title>flip_lr</title><h1>Games played</h1><table>{rows}</table>",
            encoding="utf-8",
        )


@dataclass
class SolverTracing(SolverFlipLR):
    """``SolverFlipLR`` that also registers one synthetic training trace per
    game, so the trace-collection path has a user outside its own tests."""

    label: str = "flip_lr_tracing"

    async def _run_games(self, games: list[Game]) -> None:
        """As :class:`SolverFlipLR`, plus one registered trace per game."""
        for game in games:
            await self._answer(game)
            if game.game_run is not None and game.game_run.trace_root is not None:
                game.game_run.register_trace(build_demo_trace(game.game_run))
            game.finish_game()


def build_demo_trace(run: taaf.game.GameRun) -> taaf.trace.Trace:
    """One synthetic ``TraceStep`` per submitted answer, each linked to its
    own history entry — the smallest thing that satisfies trace validation."""
    pv: dict[str, Any] = {
        "provider_name": "example",
        "provider_version": "0",
        "api_kind": "chat.completions",
        "provider_model_id": "example-model",
        "request_args": {},
        "validation_method": "solver-example-v1",
        "original_payload_sha256": "0" * 64,
        "reconstructed_payload_sha256": "0" * 64,
        "input_token_reconstruction_status": "unavailable",
        "output_token_reconstruction_status": "not_applicable",
        "media_reconstruction_status": "not_applicable",
    }
    steps = [
        taaf.trace.TraceStep(
            run_id=run.run_id,
            step_index=i,
            messages=[{"role": "user", "content": f"task {run.game_id}"}],
            output_text="mirrored",
            provider_validation=dict(pv),
            action_index=i,
            action=rec.submission,
            uncached_input_tokens=rec.uncached_input_tokens,
        )
        for i, rec in enumerate(run.history)
    ]
    return taaf.trace.Trace(
        run_id=run.run_id,
        source_kind="solver_demo",
        model_id="example-model",
        tokenizer_id="example-tokenizer",
        steps=steps,
        total_uncached_input_tokens=sum(rec.uncached_input_tokens for rec in run.history),
    )


__all__ = [
    "ANSWERS",
    "DEMO_ANSWERS",
    "N_ATTEMPTS",
    "SolverAnalysing",
    "SolverFlipLR",
    "SolverTracing",
    "all_black",
    "build_demo_trace",
    "flip_lr",
]
