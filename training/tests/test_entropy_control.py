"""Tests for the adaptive entropy-coefficient controller (nats-band targeting)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.cli import (
    _initial_entropy_control_state,
    _update_entropy_control_state,
)
from training.config import EntropyControlConfig


def _config(**overrides) -> SimpleNamespace:
    control = EntropyControlConfig.from_dict({"enabled": True, **overrides})
    return SimpleNamespace(training=SimpleNamespace(entropy_control=control))


def _result(
    *, mean_entropy: float, sample_count: int = 100, optimizer_steps: int = 8
) -> SimpleNamespace:
    return SimpleNamespace(
        mean_entropy=mean_entropy,
        sample_count=sample_count,
        optimizer_steps=optimizer_steps,
    )


# -- config parsing ----------------------------------------------------------


def test_config_defaults_are_disabled_with_run_calibrated_band() -> None:
    control = EntropyControlConfig.from_dict(None)
    assert control.enabled is False
    assert control.target_low == pytest.approx(0.30)
    assert control.target_high == pytest.approx(0.40)
    assert control.coef_max == pytest.approx(0.002)
    assert control.coef_init == pytest.approx(0.001)


def test_config_rejects_inverted_band_and_bounds() -> None:
    with pytest.raises(ValueError, match="target_low < target_high"):
        EntropyControlConfig.from_dict({"target_low": 0.5, "target_high": 0.4})
    with pytest.raises(ValueError, match="coef_max must exceed"):
        EntropyControlConfig.from_dict({"coef_min": 0.01, "coef_max": 0.002})
    with pytest.raises(ValueError, match="coef_step must be positive"):
        EntropyControlConfig.from_dict({"coef_step": 0.0})


def test_config_clamps_initial_coefficient_into_bounds() -> None:
    control = EntropyControlConfig.from_dict({"coef_init": 1.0})
    assert control.coef_init == pytest.approx(control.coef_max)


# -- controller state machine ------------------------------------------------


def test_initial_state_uses_coef_init_and_resume_restores() -> None:
    config = _config()
    state = _initial_entropy_control_state(config, None)
    assert state["current_coef"] == pytest.approx(0.001)
    assert state["last_action"] == "initial"

    resumed = _initial_entropy_control_state(
        config, {"entropy_control_state": {"current_coef": 0.0016}}
    )
    assert resumed["current_coef"] == pytest.approx(0.0016)

    # A resumed value outside the configured bounds is clamped, so tightening
    # the caps in frozen_config takes effect on the next leg.
    clamped = _initial_entropy_control_state(
        config, {"entropy_control_state": {"current_coef": 0.5}}
    )
    assert clamped["current_coef"] == pytest.approx(0.002)


def test_steps_up_below_band_until_capped() -> None:
    config = _config()
    state = _initial_entropy_control_state(config, None)
    for _ in range(20):  # far more steps than (max - init) / step
        state = _update_entropy_control_state(
            config=config, state=state, train_result=_result(mean_entropy=0.21)
        )
    assert state["current_coef"] == pytest.approx(0.002)  # hard cap, no windup
    assert state["last_action"] == "step_up"


def test_resume_preserves_active_entropy_bonus_through_skipped_window() -> None:
    config = _config()
    active = _update_entropy_control_state(
        config=config,
        state=_initial_entropy_control_state(config, None),
        train_result=_result(mean_entropy=0.21),
    )
    resumed = _initial_entropy_control_state(config, {"entropy_control_state": active})
    assert resumed["applied_coef"] == pytest.approx(active["applied_coef"])
    skipped = _update_entropy_control_state(
        config=config,
        state=resumed,
        train_result=_result(mean_entropy=0.0, optimizer_steps=0),
    )
    assert skipped["applied_coef"] == pytest.approx(active["applied_coef"])


def test_resume_clamps_saved_bonus_and_disables_it_when_configured_off() -> None:
    state = {"entropy_control_state": {"current_coef": 0.5, "applied_coef": 0.5}}
    config = _config()
    resumed = _initial_entropy_control_state(config, state)
    assert resumed["applied_coef"] == pytest.approx(
        config.training.entropy_control.coef_max
    )
    config.training.entropy_control.enabled = False
    assert _initial_entropy_control_state(config, state)["applied_coef"] == 0.0


def test_one_sided_floor_gates_bonus_off_at_or_above_floor() -> None:
    # At/above the floor the APPLIED coefficient is exactly zero immediately
    # (one-sided constraint: never push entropy down), while the internal
    # coefficient relaxes stepwise toward the floor value.
    config = _config()
    state = _initial_entropy_control_state(config, None)
    updated = _update_entropy_control_state(
        config=config, state=state, train_result=_result(mean_entropy=0.35)
    )
    assert updated["applied_coef"] == pytest.approx(0.0)
    assert updated["last_action"] == "floor_ok"
    for _ in range(20):
        updated = _update_entropy_control_state(
            config=config, state=updated, train_result=_result(mean_entropy=0.35)
        )
    assert updated["current_coef"] == pytest.approx(0.0)


def test_below_floor_applies_the_stepped_coefficient() -> None:
    config = _config()
    state = _initial_entropy_control_state(config, None)
    assert state["applied_coef"] == pytest.approx(0.0)  # off until proven low
    updated = _update_entropy_control_state(
        config=config, state=state, train_result=_result(mean_entropy=0.21)
    )
    assert updated["applied_coef"] == pytest.approx(updated["current_coef"])
    assert updated["applied_coef"] > 0.0


def test_above_alert_threshold_logs_and_stays_gated_off() -> None:
    config = _config()
    state = _initial_entropy_control_state(config, None)
    updated = _update_entropy_control_state(
        config=config, state=state, train_result=_result(mean_entropy=0.55)
    )
    assert updated["applied_coef"] == pytest.approx(0.0)
    assert updated["last_action"] == "alert_high"


def test_skipped_optimizer_window_leaves_coefficient_untouched() -> None:
    # A window with no optimizer step reports mean_entropy=0.0 -- without the
    # optimizer_steps guard that would read as collapse and ratchet the
    # coefficient up on pure noise.
    config = _config()
    state = _initial_entropy_control_state(config, None)
    updated = _update_entropy_control_state(
        config=config,
        state=state,
        train_result=_result(mean_entropy=0.0, sample_count=0, optimizer_steps=0),
    )
    assert updated["current_coef"] == pytest.approx(state["current_coef"])
    assert updated["last_action"] == "no_signal"
    assert updated["last_mean_entropy"] is None


def test_parity_skipped_window_with_samples_is_no_signal() -> None:
    # The REAL parity-skip shape from train_window: the window still reports
    # sample_count=len(samples) (the samples arrived and were tokenized) but
    # optimizer_steps=0 and mean_entropy=0.0 because optimization was skipped
    # wholesale. Gating on sample_count alone misreads this as entropy
    # collapse and steps the coefficient up (observed 0.001 -> 0.0012 live).
    config = _config()
    state = _initial_entropy_control_state(config, None)
    updated = _update_entropy_control_state(
        config=config,
        state=state,
        train_result=_result(mean_entropy=0.0, sample_count=97, optimizer_steps=0),
    )
    assert updated["current_coef"] == pytest.approx(state["current_coef"])
    assert updated["last_action"] == "no_signal"
    assert updated["last_mean_entropy"] is None


def test_disabled_mode_never_moves() -> None:
    control = EntropyControlConfig.from_dict({"enabled": False})
    config = SimpleNamespace(training=SimpleNamespace(entropy_control=control))
    state = _initial_entropy_control_state(config, None)
    updated = _update_entropy_control_state(
        config=config, state=state, train_result=_result(mean_entropy=0.05)
    )
    assert updated["current_coef"] == pytest.approx(state["current_coef"])
    assert updated["last_action"] == "disabled"
