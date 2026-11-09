from collections import Counter

from training.game_sampling import sample_training_games


def test_sample_training_games_builds_balanced_multiset_when_divisible() -> None:
    games = ["game-b", "game-a", "game-c"]

    window = sample_training_games(games, slot_count=6, seed=7, schedule_index=0)

    assert len(window) == 6
    assert Counter(window) == {"game-a": 2, "game-b": 2, "game-c": 2}


def test_sample_training_games_keeps_counts_within_one_when_not_divisible() -> None:
    games = ["game-a", "game-b", "game-c"]

    window = sample_training_games(games, slot_count=4, seed=7, schedule_index=3)

    counts = Counter(window)
    assert len(window) == 4
    assert set(counts) <= set(games)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_sample_training_games_is_deterministic_for_fixed_seed_and_index() -> None:
    games = ["game-a", "game-b", "game-c"]

    first = sample_training_games(games, slot_count=6, seed=7, schedule_index=5)
    second = sample_training_games(games, slot_count=6, seed=7, schedule_index=5)

    assert first == second


def test_sample_training_games_ignores_input_order_and_duplicates() -> None:
    ordered = sample_training_games(
        ["game-a", "game-b"], slot_count=4, seed=7, schedule_index=1
    )
    shuffled = sample_training_games(
        ["game-b", "game-a", " game-a "], slot_count=4, seed=7, schedule_index=1
    )

    assert ordered == shuffled


def test_sample_training_games_varies_across_schedule_indices() -> None:
    games = [f"game-{index:02d}" for index in range(8)]

    windows = {
        tuple(
            sample_training_games(
                games, slot_count=8, seed=7, schedule_index=schedule_index
            )
        )
        for schedule_index in range(12)
    }

    assert len(windows) > 1


def test_sample_training_games_empty_cases() -> None:
    assert sample_training_games([], slot_count=4, seed=7, schedule_index=0) == []
    assert (
        sample_training_games(["game-a"], slot_count=0, seed=7, schedule_index=0) == []
    )
    assert sample_training_games(["  "], slot_count=4, seed=7, schedule_index=0) == []


def test_balanced_mode_draws_a_uniform_subset_when_games_exceed_the_window() -> None:
    from training.game_sampling import sample_training_games

    games = [f"g{i:02d}" for i in range(49)]
    window = sample_training_games(
        games,
        slot_count=25,
        seed=7,
        schedule_index=3,
        mode="balanced",
        games_per_window=25,
    )
    assert len(window) == 25 and len(set(window)) == 25
    assert set(window) <= set(games)
    # Different windows draw different subsets; the same window is reproducible.
    other = sample_training_games(
        games,
        slot_count=25,
        seed=7,
        schedule_index=4,
        mode="balanced",
        games_per_window=25,
    )
    assert set(other) != set(window)
    again = sample_training_games(
        games,
        slot_count=25,
        seed=7,
        schedule_index=3,
        mode="balanced",
        games_per_window=25,
    )
    assert again == window
    # Two passes per game when the window is larger than the subset.
    doubled = sample_training_games(
        games,
        slot_count=50,
        seed=7,
        schedule_index=3,
        mode="balanced",
        games_per_window=25,
    )
    assert len(doubled) == 50 and all(doubled.count(g) == 2 for g in set(doubled))
    # Coverage over many windows reaches every game.
    seen: set[str] = set()
    for index in range(40):
        seen.update(
            sample_training_games(
                games,
                slot_count=25,
                seed=7,
                schedule_index=index,
                mode="balanced",
                games_per_window=25,
            )
        )
    assert seen == set(games)
