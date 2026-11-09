"""Repo-wide test fixtures."""

from pathlib import Path

import pytest
from re_arc import official_games


@pytest.fixture(autouse=True)
def _no_installed_tufa_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """R2.35: the installed-package half of source bundling reads the
    launcher's own venv, so a deploy test that doesn't stub it snapshots
    whatever Tufa wheels happen to be installed. The vLLM fork's built
    extensions alone are hundreds of MB, copied into ``tmp_path`` once
    per test — tens of GB of ``/tmp`` across a parallel run, retained
    for three runs.

    Empty by default; the tests that cover installed-package bundling
    stub it with their own fixtures. The editable half is left alone —
    ``test_inline_deploy_snapshots_tufalabs_sources`` deploys for real
    and expects to find this repo's own checkout in the bundle.
    """
    from taaf import deploy

    def _none() -> list[tuple[str, list[Path]]]:
        return []

    monkeypatch.setattr(deploy, "_discover_noneditable_tufa_packages", _none)


#: Tests that play an official game, whose code is fetched and not shipped.
_PLAY_AN_OFFICIAL_GAME = (
    "test_competition_arcade.py::test_competition_arcade_runs_cloned_games_with_shared_scorecard",
    "test_competition_arcade.py::test_competition_arcade_rejects_reusing_same_game_id_in_one_scorecard",
    "test_competition_arcade.py::test_competition_sim_arcade_spec_autostarts_and_tears_down_server",
    "test_game_api.py::test_game_api_full_play_through",
    "test_game_api.py::test_game_api_session_caches_arcade",
    "test_game_api.py::test_game_api_score_matches_engine_scorecard",
    "test_game_api.py::test_baseline_multiplier_scales_baselines_and_skips_reconciliation",
    "test_game_api.py::test_game_api_keepalive_reset_leaves_play_identical",
    "test_game_api.py::test_game_api_start_at_level_jumps_engine",
    "test_game_api.py::test_game_api_stop_at_level_wins_and_reconciles",
    "test_game_api.py::test_game_api_pickle_after_start_raises",
    "test_startup_stealth.py::test_startup_stealth_sends_the_gateway_nothing_until_the_solver_is_up",
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip what plays an official game when its code is not installed."""
    if not official_games.missing_sources():
        return
    skip = pytest.mark.skip(reason="official game code not installed: run python -m re_arc.official_games")
    for item in items:
        if any(name in item.nodeid for name in _PLAY_AN_OFFICIAL_GAME):
            item.add_marker(skip)
