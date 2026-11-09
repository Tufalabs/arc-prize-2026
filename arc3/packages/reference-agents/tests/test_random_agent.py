"""RandomAgent: seeded reproducibility (I11/A2 — all randomness from
ctx.rng) and the lost-state reset decision."""

from __future__ import annotations

import random

import arcengine
import numpy as np
from taaf.benchmark import Benchmark
from taaf.game import GameState
from taaf.game_examples import ExampleGame

from platypus import Action, AgentSolver, GameInfo, HistoryView, MoveContext, Observation, RunConfig
from reference_agents import RandomAgent


def _fake_observation(available: list[int], frame: np.ndarray, outcome: str = "playing") -> Observation:
    """An observation over a synthetic frame, for the per-move tests that
    need no engine."""
    raw = arcengine.FrameDataRaw(
        game_id="fake",
        state=arcengine.GameState.GAME_OVER if outcome == "lost" else arcengine.GameState.NOT_FINISHED,
        levels_completed=0,
        win_levels=1,
        available_actions=available,
    )
    raw.frame = [frame]
    state = GameState(raw=raw)
    return Observation.from_state(state, outcome=outcome, level=0, index=1, caused_by=Action("ACTION1"))  # type: ignore[arg-type]


async def _action_names_per_run(seed: int) -> list[list[str]]:
    """The action names each pass played, for one run at ``seed``."""
    agent = RandomAgent(max_moves=25, run_config=RunConfig(rng_seed=seed, process_mode="inprocess"))
    benchmark = Benchmark(games=[ExampleGame()], solver=AgentSolver(agent=agent), n_passes=2)
    await benchmark.run()
    return [[record.action.id.name for record in run.history] for run in benchmark.game_runs]


async def test_fixed_seed_reproduces_run_and_passes_differ() -> None:
    """A fixed seed reproduces a run exactly (A2/I11), and the per-game seed
    offset keeps the two passes from replaying each other."""
    first = await _action_names_per_run(seed=123)
    second = await _action_names_per_run(seed=123)
    assert first == second
    # Per-game seed offset: the two passes draw different sequences.
    assert first[0] != first[1]
    other_seed = await _action_names_per_run(seed=124)
    assert other_seed != first


def _ctx(rng_seed: int = 0) -> MoveContext:
    """A minimal context: no history, no diagnostics, one seeded RNG."""
    return MoveContext(
        history=HistoryView([]),
        game=GameInfo(game_id="fake", hint=None, action_space=("RESET", "ACTION6"), number_of_levels=1),
        rng=random.Random(rng_seed),
        shared={},
    )


async def test_resets_after_loss() -> None:
    """A loss is an ordinary observation the agent answers with RESET —
    LLM-free, and the agent's own decision (ledger item 8)."""
    lost_observation = _fake_observation([1, 2], np.zeros((64, 64), dtype=np.int8), outcome="lost")
    decision = await RandomAgent().do_move(_ctx(), lost_observation)
    assert decision == Action("RESET")


async def test_click_on_color_targets_small_blobs() -> None:
    # 4 pixels of color 7 in a 64x64 field of 0. Color-weighted clicking
    # hits the blob ~half the time; uniform-pixel clicking would hit it
    # with probability 4/4096 per draw — 20 seeded draws separate the two.
    """Colour-weighted clicking must actually favour small interactable
    blobs, which is the whole reason the mode exists."""
    frame = np.zeros((64, 64), dtype=np.int8)
    frame[10:12, 20:22] = 7
    observation = _fake_observation([6], frame)
    agent = RandomAgent(click_on_color=True)
    blob_hits = 0
    for seed in range(20):
        decision = await agent.do_move(_ctx(rng_seed=seed), observation)
        assert isinstance(decision, Action) and decision.name == "ACTION6"
        assert decision.row is not None and decision.col is not None
        assert frame[decision.row, decision.col] in (0, 7)
        if frame[decision.row, decision.col] == 7:
            blob_hits += 1
    assert blob_hits >= 5
