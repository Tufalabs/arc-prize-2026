"""Per-game seed resolution: the seed a game's RNG is built from is
explicit, overridable, and recorded faithfully enough to replay the run it
came from — in either process mode."""

from __future__ import annotations

from pathlib import Path

import pytest
from infra_fixtures import RngDumpingAgent

import taaf.benchmark
from platypus import AgentSolver, RunConfig
from taaf.game_examples import constant_game, flip_game

Draws = dict[str, list[float]]  # game_id -> that game's first draws, one per pass, sorted


async def _draws(tmp_path: Path, config: RunConfig, *, n_passes: int = 1) -> Draws:
    """Run two games under ``config`` and collect what each drew."""
    dump = tmp_path / f"dump{len(list(tmp_path.iterdir()))}"
    dump.mkdir()
    agent = RngDumpingAgent(dump_dir=str(dump), run_config=config)
    benchmark = taaf.benchmark.Benchmark(
        label="seeds", games=[flip_game("g0"), constant_game("g1")], solver=AgentSolver(agent=agent), n_passes=n_passes
    )
    await benchmark.run()
    out: Draws = {}
    for path in dump.iterdir():
        out.setdefault(path.name.split(".")[0], []).append(float(path.read_text()))
    return {game_id: sorted(draws) for game_id, draws in out.items()}


def _config(mode: str, **kwargs: object) -> RunConfig:
    """A quiet config in the given process mode."""
    return RunConfig(process_mode=mode, status_interval_s=None, **kwargs)  # type: ignore[arg-type]


async def test_a_fixed_seed_reproduces_a_run_and_its_passes_differ(tmp_path: Path) -> None:
    """A fixed seed reproduces a run exactly, and the per-game offset keeps
    the passes of one game from replaying each other."""
    first = await _draws(tmp_path, _config("inprocess", rng_seed=123), n_passes=2)
    second = await _draws(tmp_path, _config("inprocess", rng_seed=123), n_passes=2)
    assert first == second
    assert all(len(draws) == 2 and draws[0] != draws[1] for draws in first.values())
    assert await _draws(tmp_path, _config("inprocess", rng_seed=124), n_passes=2) != first


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_per_game_override_replaces_the_derived_seed(tmp_path: Path, mode: str) -> None:
    """An override is exactly the derived seed it replaces, and the
    run-level ``rng_seed`` no longer has a say."""
    derived = await _draws(tmp_path, _config(mode, rng_seed=777))
    # Derived seeds are rng_seed + game index; these name the same integers
    # by the other route, under a run seed that would otherwise differ.
    overridden = await _draws(tmp_path, _config(mode, rng_seed=999, rng_seed_per_game={"g0": 777, "g1": 778}))
    assert overridden == derived


@pytest.mark.parametrize("mode", ["inprocess", "subprocess"])
async def test_override_covers_only_the_games_it_names(tmp_path: Path, mode: str) -> None:
    """A partial override leaves the other games on their derived seeds, so
    one recorded game can be pinned without disturbing the rest."""
    derived = await _draws(tmp_path, _config(mode, rng_seed=777))
    partial = await _draws(tmp_path, _config(mode, rng_seed=777, rng_seed_per_game={"g0": 777}))
    assert partial == derived
