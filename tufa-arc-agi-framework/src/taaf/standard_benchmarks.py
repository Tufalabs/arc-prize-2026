"""Team-agreed standard benchmarks: shared game lists + weights so
results are directly comparable across runs. Each benchmark is frozen
once published; new ones land as new functions.
"""

from __future__ import annotations

import collections

import taaf.benchmark
import taaf.competition_arcade
import taaf.game
import taaf.game_api
import taaf.solver


def make_benchmark_all_games(solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    """All-game benchmark: ``re_arc``'s ``train`` + ``eval`` datasets.
    Pinning to those JSON files keeps the benchmark stable as games are
    added.

    Weights: ``20`` for ``official``-tagged games regardless of level
    count; ``min(number_of_levels, 8)`` otherwise. Callers are free to
    bump ``n_passes`` — the games and weights are the benchmark identity.
    """
    import re_arc

    train_ids = re_arc.list_game_ids(datasets="train")
    eval_ids = re_arc.list_game_ids(datasets="eval")
    game_ids = list(train_ids) + list(eval_ids)
    official_ids = set(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))

    # number_of_levels requires starting each game.
    session = taaf.game.RunSession()
    games: list[taaf.game.Game] = []
    weights: list[float] = []
    for gid in game_ids:
        probe = taaf.game_api.GameAPI(env_name=gid)
        probe.start_game(session=session)
        n_levels = probe.number_of_levels
        if gid in official_ids:
            weights.append(20.0)
        else:
            weights.append(float(min(n_levels, 8)))
        games.append(taaf.game_api.GameAPI(env_name=gid))

    return taaf.benchmark.Benchmark(
        label="all_games",
        games=games,
        solver=solver,
        n_passes=1,
        game_weights=weights,
    )


def make_benchmark_official_25(solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    """The 25 ``official``-tagged games, one pass each, unweighted."""
    import re_arc

    official_ids = list(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))
    if len(official_ids) != 25:
        raise RuntimeError(f"Expected 25 official games, got {len(official_ids)}.")
    return taaf.benchmark.Benchmark(
        label="official_25",
        games=[taaf.game_api.GameAPI(env_name=gid) for gid in official_ids],
        solver=solver,
        n_passes=1,
    )


#: Seal sweep 3's game mix as 110 whole entries, by game (arc3-experiments
#: ``seal_sweep3/design.py``, 2026-10-14). The mix weighs a game by its
#: set -- easy, medium or hard, the set weights fitted so the seal's defaults
#: score what its 5- and 9-hour leaderboard submissions did -- and leaves the
#: trivial official games (ar25, ft09, lp85, m0r0) out. Each set's entries are
#: its weight total rounded by largest remainder, the spare ones going to its
#: games in alphabetical order, as the official 110's round-robin spreads its own.
SEAL_SWEEP3_110_ENTRIES: dict[str, int] = {
    # Easy.
    "cd82": 3, "cn04": 3, "lantern_moths": 3, "paint_snakes": 3, "r11l": 3,
    "re86": 3, "sb26": 2, "tr87": 2, "tu93": 2, "vc33": 2,
    # Medium.
    "anchor_drift": 5, "g50t": 5, "glass_orchard": 5, "js26": 5, "ka59": 5,
    "ls20": 5, "sc25": 5, "su15": 4, "tn36": 4, "wa30": 4,
    # Hard.
    "ball_bounce": 4, "bp35": 3, "dc22": 3, "en18": 3, "escape_velocity": 3, "jc26": 3,
    "lf52": 3, "marble_loom": 3, "s5i5": 3, "sk48": 3, "sp80": 3, "talos_beams": 3,
}  # fmt: skip

#: The re-arc-3 games' baseline multipliers in seal sweep 3 (R11.15): their
#: per-level action counts are optimal solves where the official games' are
#: human play, and ball_bounce's is set so the seal's defaults score it about a
#: quarter given time, where the common one leaves it near zero in every arm.
SEAL_SWEEP3_BASELINE_MULTIPLIERS: dict[str, float] = {
    "anchor_drift": 3.0, "ball_bounce": 9.0, "en18": 3.0, "escape_velocity": 3.0,
    "glass_orchard": 3.0, "jc26": 3.0, "js26": 3.0, "lantern_moths": 3.0,
    "marble_loom": 3.0, "paint_snakes": 3.0, "talos_beams": 3.0,
}  # fmt: skip


def make_benchmark_seal_sweep3_110(solver: taaf.solver.Solver | None = None) -> taaf.benchmark.Benchmark:
    """Seal sweep 3's game mix as a 110-entry run (`SEAL_SWEEP3_110_ENTRIES`),
    one pass, each entry weighing the same: a Kaggle run of it is what the
    sweep's box replay simulates. The official games are the ``official``-tagged
    ones; the re-arc-3 games play their ``-0001`` variants, scored against
    their scaled baselines (`SEAL_SWEEP3_BASELINE_MULTIPLIERS`).

    Entries go round-robin, every game's first copy before any second, and
    each carries a public ``game_id`` of ``{env}_{i}``, unique as
    ``Benchmark.run`` requires, as ``make_benchmark_kaggle_official_110``'s do.
    """
    import re_arc

    official = {
        gid.split("-")[0]: gid for gid in re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official")
    }
    env_names: dict[str, str] = {}
    for name in SEAL_SWEEP3_110_ENTRIES:
        if name in SEAL_SWEEP3_BASELINE_MULTIPLIERS:
            env_names[name] = f"{name}-0001"
        elif name in official:
            env_names[name] = official[name]
        else:
            raise RuntimeError(f"{name} is neither an official game nor a re-arc-3 one in seal sweep 3's mix.")
    games: list[taaf.game.Game] = []
    for copy in range(max(SEAL_SWEEP3_110_ENTRIES.values())):
        for name, entries in SEAL_SWEEP3_110_ENTRIES.items():
            if copy < entries:
                games.append(
                    taaf.game_api.GameAPI(
                        env_name=env_names[name],
                        external_game_id=f"{env_names[name]}_{copy}",
                        baseline_multiplier=SEAL_SWEEP3_BASELINE_MULTIPLIERS.get(name, 1.0),
                    )
                )
    if len(games) != 110:
        raise RuntimeError(f"Expected 110 entries in seal sweep 3's mix, got {len(games)}.")
    return taaf.benchmark.Benchmark(
        label="seal_sweep3_110",
        games=games,
        solver=solver,
        n_passes=1,
    )


def make_benchmark_kaggle_official_110(
    solver: taaf.solver.Solver | None = None,
    *,
    arcade_spec: taaf.game_api.ArcadeSpec | None = None,
    competition_clone_ids: bool = False,
    competition_sim: bool = False,
) -> taaf.benchmark.Benchmark:
    """Kaggle submission-shaped public benchmark (R2.57).

    The 25 official games are repeated round-robin to 110 independent
    ``GameAPI`` entries, with ``n_passes=1``. When
    ``competition_clone_ids=True``, the repeated entries use the same
    deterministic clone IDs exposed by
    ``CompetitionArcadeServer.official_110()``, making the benchmark
    compatible with the competition-style Arcade that does not support
    multiple runs for the same game ID.

    ``competition_sim=True`` targets a local competition-arcade simulator
    (R11.13) started lazily on whichever process runs the benchmark: it
    sets a ``competition_sim`` ``ArcadeSpec`` and forces
    ``competition_clone_ids`` so the game IDs line up with the server's
    official-110 clone set. This lets a non-submission run exercise the
    submission-shaped arcade without an actual leaderboard submission.
    """
    import re_arc

    if competition_sim:
        if arcade_spec is not None:
            raise ValueError("Pass either competition_sim=True or an explicit arcade_spec, not both.")
        arcade_spec = taaf.game_api.ArcadeSpec(competition_sim=True)
        competition_clone_ids = True
    official_ids = list(re_arc.list_game_ids(datasets=["train", "eval"], include_tags="official"))
    if len(official_ids) != 25:
        raise RuntimeError(f"Expected 25 official games for Kaggle benchmark, got {len(official_ids)}.")
    if competition_clone_ids:
        # Competition arcade registers each clone as a distinct EnvironmentInfo,
        # so the natural ``info.game_id`` (``p005`` etc.) is already unique — no
        # external override needed.
        game_ids = taaf.competition_arcade.clone_game_ids(official_ids, total_runs=110)
        external_ids: list[str | None] = [None] * len(game_ids)
    else:
        # No competition arcade to synthesize unique env IDs — re_arc only knows
        # the 25 base IDs. Override each instance's public ``game_id`` with
        # ``{base}_{i}`` where ``i`` is the per-base occurrence index, so
        # ``Benchmark.game_runs`` has unique IDs (Benchmark.run asserts this)
        # while each instance still plays its base re_arc env underneath.
        counts: collections.Counter[str] = collections.Counter()
        game_ids = []
        external_ids = []
        for i in range(110):
            base = official_ids[i % len(official_ids)]
            game_ids.append(base)
            external_ids.append(f"{base}_{counts[base]}")
            counts[base] += 1
    games: list[taaf.game.Game]
    if arcade_spec is None:
        games = [taaf.game_api.GameAPI(env_name=g, external_game_id=ext) for g, ext in zip(game_ids, external_ids)]
    else:
        games = [
            taaf.game_api.GameAPI(env_name=g, external_game_id=ext, arcade_spec=arcade_spec)
            for g, ext in zip(game_ids, external_ids)
        ]
    return taaf.benchmark.Benchmark(
        label="kaggle_official_110",
        games=games,
        solver=solver,
        n_passes=1,
    )
