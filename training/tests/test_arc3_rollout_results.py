"""Harness-side ARC reward semantics at the callback boundary."""

from types import SimpleNamespace

import pytest

from harness_support.arc3.results import enrich_arc3_result, progress_snapshot


def _payload():
    return {
        "run_id": "run",
        "level_replay_entries": [],
        "game_id": "game",
        "state": "won",
        "levels_completed": 2,
        "number_of_levels": 2,
        "start_level": 0,
        "actions_per_level": [2, 2],
        "base_actions_per_level": [2, 4],
        "final_wallclock_seconds": 40.0,
        "action_wallclock_seconds": [10.0, 20.0, 30.0, 40.0],
        "steps": [
            {"nonce": "a", "action_index": 0, "output_len": 1, "output_crc32": 1},
            {
                "nonce": "memory",
                "action_index": None,
                "output_len": 1,
                "output_crc32": 1,
            },
            {"nonce": "b", "action_index": 2, "output_len": 1, "output_crc32": 1},
        ],
    }


def test_batch_rewards_use_official_game_score_and_memory_earns_zero():
    raw = _payload()
    result = enrich_arc3_result(raw, max_runtime_seconds=300.0)
    assert [step["reward"] for step in result["steps"]] == pytest.approx(
        [100.0 / 3.0, 0.0, 200.0 / 3.0]
    )
    assert result["episode_return"] == result["final_arc_score"] == 100.0
    assert result["level_scores"] == {"1": 100.0, "2": 115.0}
    assert result["level_rewards"] == pytest.approx({"1": 100 / 3, "2": 200 / 3})
    assert result["steps"][1]["stage"]["actions_before_decision"] == 2
    assert result["steps"][1]["stage"]["time_remaining_seconds"] == 280.0
    assert "reward" not in raw["steps"][0]
    # One response can author a batch crossing both completion boundaries.
    raw["steps"] = raw["steps"][:1]
    result = enrich_arc3_result(raw, max_runtime_seconds=None)
    assert result["steps"][0]["reward"] == result["episode_return"] == 100.0


def test_inefficient_success_weights_squared_raw_score():
    raw = _payload()
    raw.update(levels_completed=1, actions_per_level=[4], base_actions_per_level=[2, 4])
    result = enrich_arc3_result(raw, max_runtime_seconds=300.0)
    assert result["episode_return"] == pytest.approx(25.0 / 3.0)
    assert [step["reward"] for step in result["steps"]] == pytest.approx(
        [0.0, 0.0, 25.0 / 3.0]
    )
    assert result["level_scores"] == {"1": 25.0}
    assert result["final_arc_score"] == pytest.approx(25.0 / 3.0)


def test_reward_delta_uses_the_whole_game_cap_including_prior_excess():
    raw = _payload()
    raw["base_actions_per_level"] = [4, 1]
    # Raw level scores115 and25 yield official prefixes100/3 and55.
    # The second reward includes previously capped excess from level1.
    result = enrich_arc3_result(raw, max_runtime_seconds=300)
    assert result["level_scores"] == {"1": 115.0, "2": 25.0}
    assert result["level_rewards"] == pytest.approx({"1": 100 / 3, "2": 55 - 100 / 3})
    assert result["episode_return"] == result["final_arc_score"] == 55
    assert sum(step["reward"] for step in result["steps"]) == pytest.approx(55)


@pytest.mark.parametrize("replayed_actions", [0, 2])
def test_live_progress_excludes_initial_prefix_score(replayed_actions):
    run = SimpleNamespace(
        levels_completed=1,
        number_of_levels=3,
        actions_per_level=[replayed_actions, 0, 0],
        base_actions_per_level=[2, 1, 1],
    )
    initial = progress_snapshot(run, start_level=1)
    assert initial["score"] == 0
    assert initial["data"]["arc_score"] == pytest.approx(
        100 / 6 if replayed_actions else 0
    )
    run.levels_completed = 2
    run.actions_per_level[1] = 2
    final = progress_snapshot(run, start_level=1)
    assert final["score"] == pytest.approx(50 / 6)
    assert final["score"] == pytest.approx(
        final["data"]["arc_score"] - initial["data"]["arc_score"]
    )


@pytest.mark.parametrize("start_level", [0, 1])
def test_live_and_terminal_official_scores_reconcile_each_completed_level(start_level):
    raw = _payload()
    raw.update(
        number_of_levels=3,
        levels_completed=3,
        start_level=start_level,
        actions_per_level=[2 if start_level == 0 else 0, 2, 2],
        base_actions_per_level=[2, 1, 2],
        action_wallclock_seconds=list(range(1, 7 if start_level == 0 else 5)),
    )
    raw["steps"] = [
        {
            "nonce": str(index),
            "action_index": 2 * index,
            "output_len": 1,
            "output_crc32": 1,
        }
        for index in range(3 - start_level)
    ]
    result = enrich_arc3_result(raw, max_runtime_seconds=300)
    run = SimpleNamespace(**raw)
    prior = 0.0
    for completed in range(start_level, 4):
        run.levels_completed = completed
        progress = progress_snapshot(run, start_level=start_level)
        if completed > start_level:
            reward = result["steps"][completed - start_level - 1]["reward"]
            assert progress["score"] - prior == pytest.approx(reward)
        prior = progress["score"]
    assert result["episode_return"] == prior == result["final_arc_score"]
    assert result["episode_return"] == pytest.approx(
        75 if start_level == 0 else 350 / 6
    )


@pytest.mark.parametrize("elapsed", [0.0, 90.0, 300.0, 305.0])
def test_zero_call_loss_reports_zero_return_and_actual_terminal_clock(elapsed):
    raw = _payload()
    raw.update(
        state="gave_up",
        levels_completed=0,
        actions_per_level=[],
        action_wallclock_seconds=[],
        steps=[],
        final_wallclock_seconds=elapsed,
    )
    result = enrich_arc3_result(raw, max_runtime_seconds=300.0)
    assert result["episode_return"] == result["final_arc_score"] == 0.0
    assert result["level_scores"] == result["level_rewards"] == {}
    assert result["final_stage"]["actions_before_decision"] == 0
    assert result["final_stage"]["level"] == 1
    # No actions leaves the game state unchanged, but still consumes time.
    assert result["final_stage"]["run_elapsed_seconds"] == elapsed
    remaining = result["final_stage"]["time_remaining_seconds"]
    assert remaining == max(0.0, 300.0 - elapsed)
    assert 0.0 <= remaining <= 300.0


def test_a_crashed_game_keeps_the_level_rewards_it_earned():
    """on_agent_error="crash_game" (2026-10-11): a game whose agent raised
    after one level is scored like any other run on what it reached."""
    raw = _payload()
    raw.update(
        levels_completed=1,
        actions_per_level=[2, 1],
        action_wallclock_seconds=[10.0, 20.0, 30.0],
        steps=raw["steps"],
    )
    won = enrich_arc3_result(dict(raw), max_runtime_seconds=300.0)
    crashed = enrich_arc3_result(
        {**raw, "state": "crashed", "solver_note": "agent error:\nboom"},
        max_runtime_seconds=300.0,
    )
    assert crashed["episode_return"] == won["episode_return"] == pytest.approx(100.0 / 3.0)
    assert [step["reward"] for step in crashed["steps"]] == [
        step["reward"] for step in won["steps"]
    ]
    assert crashed["level_rewards"] == won["level_rewards"]
    assert crashed["solver_note"] == "agent error:\nboom"
