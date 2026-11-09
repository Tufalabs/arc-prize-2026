"""A rollout the infrastructure truncated fails the run loudly, naming the
episode (owner ruling). A game whose agent raised is not one: since
2026-10-11 the harness contains it (``on_agent_error="crash_game"``) and it
trains as a CRASHED terminal, scored on what it reached."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.cli import _require_no_infra_error_rollouts


def _rollout(
    reason: str, task: str = "game-x", trajectory: str = "t-1"
) -> SimpleNamespace:
    return SimpleNamespace(
        termination_reason=reason, task_id=task, trajectory_id=trajectory, metadata={}
    )


def test_clean_window_passes() -> None:
    _require_no_infra_error_rollouts(
        [_rollout("COMPLETED_RUN"), _rollout("TIMEOUT"), _rollout("")],
        iteration_index=4,
    )


def test_crashed_rollout_fails_the_run_and_names_the_episode() -> None:
    window = [
        _rollout("COMPLETED_RUN"),
        _rollout("ERROR", task="tu93-0768757b", trajectory="abc123"),
        _rollout(
            "error", task="ar25-0c556536", trajectory="def456"
        ),  # case-insensitive
    ]
    with pytest.raises(RuntimeError) as excinfo:
        _require_no_infra_error_rollouts(window, iteration_index=6)
    message = str(excinfo.value)
    assert "FATAL: 2 of 3" in message
    assert "iteration 7" in message
    assert "tu93-0768757b" in message and "abc123" in message
    assert "ar25-0c556536" in message and "def456" in message


def test_a_game_crashed_by_its_agent_is_a_terminal_not_an_infra_error() -> None:
    _require_no_infra_error_rollouts(
        [_rollout("COMPLETED_RUN"), _rollout("CRASHED"), _rollout("crashed")],
        iteration_index=2,
    )
