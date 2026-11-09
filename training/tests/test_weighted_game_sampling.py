import random
from collections import Counter, deque

import pytest

from training.cli import _game_nonzero_return_rates
from training.config import SamplingConfig
from training.game_sampling import (
    configure_game_sampling,
    game_sampling_weights,
    publish_game_nonzero_rates,
    reset_game_sampling,
    sample_training_games,
)


@pytest.fixture(autouse=True)
def _reset_process_wide_sampling_state():
    reset_game_sampling()
    yield
    reset_game_sampling()


def _legacy_balanced_window(
    games: list[str], *, slot_count: int, seed: int, schedule_index: int
) -> list[str]:
    """The pre-weighted-mode sampler, verbatim, as the byte-identical oracle."""
    normalized = sorted({str(game).strip() for game in games if str(game).strip()})
    if slot_count <= 0 or not normalized:
        return []
    rng = random.Random(
        f"train-game-sample:{seed}:{schedule_index}:{slot_count}:{'|'.join(normalized)}"
    )
    window = [normalized[index % len(normalized)] for index in range(slot_count)]
    rng.shuffle(window)
    return window


def test_game_sampling_weights_normalize_and_respect_floor() -> None:
    weights = game_sampling_weights(
        ["game-a", "game-b", "game-c"],
        weight_floor=0.2,
        nonzero_rates={"game-a": 0.0, "game-b": 0.5, "game-c": 1.0},
    )

    # Un-normalized: floor + (1 - floor) * rate = 0.2, 0.6, 1.0 (total 1.8).
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["game-a"] == pytest.approx(0.2 / 1.8)
    assert weights["game-b"] == pytest.approx(0.6 / 1.8)
    assert weights["game-c"] == pytest.approx(1.0 / 1.8)
    # The floor keeps a never-scoring game strictly in the draw.
    assert weights["game-a"] > 0.0


def test_game_sampling_weights_cold_start_is_uniform() -> None:
    weights = game_sampling_weights(
        ["game-a", "game-b", "game-c", "game-d"],
        weight_floor=0.15,
        nonzero_rates={},
    )

    assert set(weights) == {"game-a", "game-b", "game-c", "game-d"}
    for weight in weights.values():
        assert weight == pytest.approx(0.25)


def test_game_sampling_weights_unseen_game_draws_at_full_weight() -> None:
    # A game with no recorded returns must weigh exactly like a perfectly
    # scoring one (rate 1.0), never like a zero-rate one.
    weights = game_sampling_weights(
        ["seen-perfect", "seen-zero", "unseen"],
        weight_floor=0.15,
        nonzero_rates={"seen-perfect": 1.0, "seen-zero": 0.0},
    )

    assert weights["unseen"] == pytest.approx(weights["seen-perfect"])
    assert weights["unseen"] > weights["seen-zero"]


def test_game_sampling_weights_reject_out_of_range_floor() -> None:
    with pytest.raises(ValueError):
        game_sampling_weights(["game-a"], weight_floor=0.0, nonzero_rates={})
    with pytest.raises(ValueError):
        game_sampling_weights(["game-a"], weight_floor=1.5, nonzero_rates={})


def test_balanced_default_is_byte_identical_to_legacy_sampler() -> None:
    games = [f"official-{index:02d}" for index in range(25)]

    # Divisible windows are byte-identical to the legacy sampler; a window
    # with a remainder shuffles the game order first (repeat sampling) and
    # is covered by test_window_with_one_repeated_game_splits_into_unique_child_windows.
    for seed in (7, 42):
        for schedule_index in range(6):
            for slot_count in (25, 50):
                assert sample_training_games(
                    games,
                    slot_count=slot_count,
                    seed=seed,
                    schedule_index=schedule_index,
                ) == _legacy_balanced_window(
                    games,
                    slot_count=slot_count,
                    seed=seed,
                    schedule_index=schedule_index,
                )


def test_nonzero_rate_sampling_is_deterministic_for_fixed_seed() -> None:
    games = ["game-a", "game-b", "game-c"]
    rates = {"game-a": 1.0, "game-b": 0.25, "game-c": 0.0}

    kwargs = dict(
        slot_count=32,
        seed=7,
        mode="nonzero_rate",
        weight_floor=0.15,
        games_per_window=2,
        nonzero_rates=rates,
    )
    first = sample_training_games(games, schedule_index=5, **kwargs)
    second = sample_training_games(games, schedule_index=5, **kwargs)

    assert first == second
    assert len(first) == 32
    assert set(first) <= set(games)

    windows = {
        tuple(sample_training_games(games, schedule_index=index, **kwargs))
        for index in range(8)
    }
    assert len(windows) > 1


def test_nonzero_rate_windows_are_uniform_multisets() -> None:
    games = [f"official-{index:02d}" for index in range(25)]

    for schedule_index in range(6):
        window = sample_training_games(
            games,
            slot_count=25,
            seed=13,
            schedule_index=schedule_index,
            mode="nonzero_rate",
            weight_floor=0.15,
            games_per_window=5,
            nonzero_rates={games[0]: 1.0},
        )
        counts = Counter(window)
        # 5 distinct games, each with exactly 25/5 passes: the platypus
        # worker's equal-multiset contract holds for every weighted window.
        assert len(counts) == 5
        assert set(counts.values()) == {5}


def test_nonzero_rate_cold_start_selects_games_uniformly() -> None:
    games = ["game-a", "game-b", "game-c"]

    counts: Counter[str] = Counter()
    for schedule_index in range(600):
        window = sample_training_games(
            games,
            slot_count=3,
            seed=11,
            schedule_index=schedule_index,
            mode="nonzero_rate",
            weight_floor=0.15,
            games_per_window=1,
            nonzero_rates={},
        )
        assert len(set(window)) == 1
        counts[window[0]] += 1

    assert set(counts) == set(games)
    for game_id in games:
        assert 150 <= counts[game_id] <= 250


def test_nonzero_rate_sampling_skews_toward_scoring_games_but_keeps_floor() -> None:
    games = ["cold", "hot"]

    counts: Counter[str] = Counter()
    for schedule_index in range(1000):
        window = sample_training_games(
            games,
            slot_count=2,
            seed=3,
            schedule_index=schedule_index,
            mode="nonzero_rate",
            weight_floor=0.15,
            games_per_window=1,
            nonzero_rates={"hot": 1.0, "cold": 0.0},
        )
        counts[window[0]] += 1

    # Expected window shares: hot 1.0/1.15 ~ 0.87, cold 0.15/1.15 ~ 0.13.
    assert counts["hot"] > 800
    assert 50 <= counts["cold"] <= 200


def test_nonzero_rate_rejects_impossible_window_shapes() -> None:
    games = ["game-a", "game-b", "game-c"]
    kwargs = dict(seed=1, schedule_index=0, mode="nonzero_rate", weight_floor=0.15)

    with pytest.raises(ValueError, match="not divisible"):
        sample_training_games(
            [f"game-{index}" for index in range(6)],
            slot_count=25,
            games_per_window=4,
            nonzero_rates={},
            **kwargs,
        )
    with pytest.raises(ValueError, match="exceeds"):
        sample_training_games(
            games, slot_count=8, games_per_window=4, nonzero_rates={}, **kwargs
        )
    with pytest.raises(ValueError, match="games_per_window"):
        sample_training_games(
            games, slot_count=8, games_per_window=0, nonzero_rates={}, **kwargs
        )


def test_process_wide_configuration_drives_bare_call_sites() -> None:
    # training.inference calls the sampler with only the window shape; the
    # curriculum must arrive through the process-wide configuration.
    games = ["game-a", "game-b"]
    configure_game_sampling(mode="nonzero_rate", weight_floor=0.15, games_per_window=1)
    publish_game_nonzero_rates({"game-a": 1.0, "game-b": 0.0})

    chosen: Counter[str] = Counter()
    for schedule_index in range(200):
        window = sample_training_games(
            games, slot_count=2, seed=3, schedule_index=schedule_index
        )
        chosen[window[0]] += 1
    assert chosen["game-a"] > 140

    reset_game_sampling()
    balanced = Counter(
        sample_training_games(games, slot_count=1000, seed=3, schedule_index=1)
    )
    assert balanced == {"game-a": 500, "game-b": 500}


def test_configure_game_sampling_rejects_unknown_mode_and_bad_floor() -> None:
    with pytest.raises(ValueError):
        configure_game_sampling(mode="adaptive", weight_floor=0.15)
    with pytest.raises(ValueError):
        configure_game_sampling(mode="nonzero_rate", weight_floor=0.0)
    with pytest.raises(ValueError):
        # nonzero_rate without a positive games_per_window is unusable.
        configure_game_sampling(mode="nonzero_rate", weight_floor=0.15)
    with pytest.raises(ValueError):
        sample_training_games(
            ["game-a"], slot_count=2, seed=1, schedule_index=0, mode="bogus"
        )


@pytest.mark.parametrize(
    "payload",
    [
        None,
        {},
        {"return_history_size": 16},
        {"game_sampling_mode": None},
        {"game_sampling_mode": ""},
    ],
)
def test_sampling_config_defaults_to_balanced(payload) -> None:
    assert SamplingConfig().game_sampling_mode == "balanced"
    defaults = SamplingConfig.from_dict(payload)
    assert defaults.game_sampling_mode == "balanced"
    assert defaults.game_sampling_weight_floor == pytest.approx(0.15)
    assert defaults.game_sampling_games_per_window == 0


def test_sampling_config_parses_weighted_mode_knobs() -> None:
    parsed = SamplingConfig.from_dict(
        {
            "game_sampling_mode": "nonzero_rate",
            "game_sampling_weight_floor": 0.3,
            "game_sampling_games_per_window": 8,
        }
    )
    assert parsed.game_sampling_mode == "nonzero_rate"
    assert parsed.game_sampling_weight_floor == pytest.approx(0.3)
    assert parsed.game_sampling_games_per_window == 8


def test_sampling_config_rejects_bad_mode_and_floor() -> None:
    with pytest.raises(ValueError):
        SamplingConfig.from_dict({"game_sampling_mode": "adaptive"})
    with pytest.raises(ValueError):
        SamplingConfig.from_dict({"game_sampling_weight_floor": 0.0})
    with pytest.raises(ValueError):
        SamplingConfig.from_dict({"game_sampling_weight_floor": 1.5})


def test_game_nonzero_return_rates_fraction_of_trailing_returns() -> None:
    history = {
        "game-a": deque([0.0, 10.0, 0.0, 5.0], maxlen=8),
        "game-b": deque([0.0, 0.0], maxlen=8),
        "game-c": deque(maxlen=8),
    }

    rates = _game_nonzero_return_rates(history)

    assert rates["game-a"] == pytest.approx(0.5)
    assert rates["game-b"] == pytest.approx(0.0)
    # An empty history stays absent so the sampler treats the game
    # optimistically rather than as a confirmed zero.
    assert "game-c" not in rates
