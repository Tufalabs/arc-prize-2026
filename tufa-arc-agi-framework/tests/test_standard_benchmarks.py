# type: ignore
"""Tests for ``taaf.standard_benchmarks``."""

import pytest
import re_arc

import taaf.game_api
import taaf.standard_benchmarks


@pytest.mark.slow
def test_all_games_benchmark_constructs_with_unequal_weights():
    """End-to-end probe: ``make_benchmark_all_games`` starts each of the available games
    once to read ``number_of_levels``, so this test exercises the
    full re-arc-3 / arcengine stack. Marked ``slow`` for that
    reason — kept out of the default suite."""
    bm = taaf.standard_benchmarks.make_benchmark_all_games()
    assert bm.label == "all_games"
    assert bm.game_weights is not None
    # Mixed weights (level-count ∪ official-override) ⇒ unequal.
    assert len(set(bm.game_weights)) > 1, f"expected unequal weights, got {set(bm.game_weights)}"
    # The 25 official games each pin to weight 20 regardless of level
    # count; anything else means the official-tag override regressed.
    assert bm.game_weights.count(20.0) == 25


def test_official_25_plays_each_official_game_once(monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    bm = taaf.standard_benchmarks.make_benchmark_official_25()

    assert bm.label == "official_25"
    assert bm.n_passes == 1
    assert [game.env_name for game in bm.games] == official_ids
    assert bm.game_weights is None


def test_official_25_rejects_wrong_official_count(monkeypatch):
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: ["official_00"])

    with pytest.raises(RuntimeError, match="Expected 25 official games"):
        taaf.standard_benchmarks.make_benchmark_official_25()


def test_kaggle_official_110_repeats_official_games(monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    bm = taaf.standard_benchmarks.make_benchmark_kaggle_official_110()

    assert bm.label == "kaggle_official_110"
    assert bm.n_passes == 1
    assert len(bm.games) == 110
    counts = {game_id: 0 for game_id in official_ids}
    for game in bm.games:
        counts[game.env_name] += 1
    assert sorted(counts.values()) == [4] * 15 + [5] * 10


def test_kaggle_official_110_can_use_competition_clone_ids(monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    bm = taaf.standard_benchmarks.make_benchmark_kaggle_official_110(competition_clone_ids=True)

    assert bm.label == "kaggle_official_110"
    assert bm.n_passes == 1
    assert len(bm.games) == 110
    assert [game.env_name for game in bm.games[:3]] == ["k000", "k001", "k002"]
    assert bm.games[-1].env_name == "k109"
    assert len({game.env_name for game in bm.games}) == 110


def test_kaggle_official_110_competition_sim_sets_spec_and_clone_ids(monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    bm = taaf.standard_benchmarks.make_benchmark_kaggle_official_110(competition_sim=True)

    assert len(bm.games) == 110
    # competition_sim forces cloned IDs so they line up with the sim's official-110 set.
    assert [game.env_name for game in bm.games[:3]] == ["k000", "k001", "k002"]
    # every game targets the competition simulator.
    assert all(game.arcade_spec.competition_sim for game in bm.games)


def test_kaggle_official_110_competition_sim_rejects_explicit_arcade_spec(monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    with pytest.raises(ValueError, match="competition_sim"):
        taaf.standard_benchmarks.make_benchmark_kaggle_official_110(
            competition_sim=True,
            arcade_spec=taaf.game_api.ArcadeSpec(),
        )


#: The official games' base names, as ``re_arc`` tags them, with stand-in hashes.
_OFFICIAL_NAMES = (
    "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30"
).split()


def test_seal_sweep3_110_plays_the_mix_in_110_unique_entries(monkeypatch):
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: [f"{name}-0000" for name in _OFFICIAL_NAMES])

    bm = taaf.standard_benchmarks.make_benchmark_seal_sweep3_110()

    assert bm.label == "seal_sweep3_110"
    assert bm.n_passes == 1
    assert bm.game_weights is None
    assert len({game.external_game_id for game in bm.games}) == 110
    counts: dict[str, int] = {}
    for game in bm.games:
        counts[game.env_name.split("-")[0]] = counts.get(game.env_name.split("-")[0], 0) + 1
    assert counts == taaf.standard_benchmarks.SEAL_SWEEP3_110_ENTRIES
    # Every game's first copy plays before any game's second.
    first_round = {game.env_name for game in bm.games[: len(counts)]}
    assert len(first_round) == len(counts)
    # The trivial official games are out; the re-arc-3 ones play their -0001
    # variants at their scaled baselines, the official ones at their own.
    assert not {"ar25", "ft09", "lp85", "m0r0"} & set(counts)
    for game in bm.games:
        name = game.env_name.split("-")[0]
        if name in taaf.standard_benchmarks.SEAL_SWEEP3_BASELINE_MULTIPLIERS:
            assert game.env_name == f"{name}-0001"
            assert game.baseline_multiplier == taaf.standard_benchmarks.SEAL_SWEEP3_BASELINE_MULTIPLIERS[name]
        else:
            assert game.env_name == f"{name}-0000"
            assert game.baseline_multiplier == 1.0


def test_seal_sweep3_110_rejects_a_missing_official_game(monkeypatch):
    monkeypatch.setattr(
        re_arc, "list_game_ids", lambda **_kwargs: [f"{name}-0000" for name in _OFFICIAL_NAMES if name != "cd82"]
    )

    with pytest.raises(RuntimeError, match="cd82"):
        taaf.standard_benchmarks.make_benchmark_seal_sweep3_110()
