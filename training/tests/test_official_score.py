import pytest

from harness_support.arc3.scoring import compute_game_score


def test_compute_game_score_weights_levels_and_requires_missing_levels_as_zero() -> (
    None
):
    score = compute_game_score(
        [
            (1, 100.0),
            (2, 100.0),
            (3, 100.0),
            (4, 0.0),
            (5, 0.0),
        ]
    )

    assert score == pytest.approx(40.0)


def test_compute_game_score_caps_faster_levels_to_completed_weight() -> None:
    score = compute_game_score(
        [
            (1, 115.0),
            (2, 115.0),
            (3, 0.0),
        ]
    )

    assert score == pytest.approx(50.0)
