"""The segment budget yields only when one more iteration cannot become durable."""

from __future__ import annotations

import pytest

from training.segment_budget import (
    DEADLINE_ENV,
    DEFAULT_MARGIN_SECONDS,
    MARGIN_ENV,
    SegmentBudget,
)


def _budget(deadline: float | None, now: float, **kwargs) -> SegmentBudget:  # noqa: ANN003
    return SegmentBudget(deadline_epoch=deadline, clock=lambda: now, **kwargs)


def test_without_a_deadline_the_budget_never_yields() -> None:
    budget = SegmentBudget.from_environment({}, clock=lambda: 1_000.0)
    assert not budget.enabled
    budget.record_iteration(10_000.0)
    assert budget.remaining_seconds() is None
    assert not budget.should_yield()


def test_environment_parsing_rejects_malformed_values() -> None:
    budget = SegmentBudget.from_environment(
        {DEADLINE_ENV: " 2000 ", MARGIN_ENV: "30"}, clock=lambda: 1_500.0
    )
    assert budget.enabled
    assert budget.remaining_seconds() == 500.0
    assert budget.margin_seconds == 30.0
    default = SegmentBudget.from_environment({DEADLINE_ENV: "2000"})
    assert default.margin_seconds == DEFAULT_MARGIN_SECONDS
    with pytest.raises(ValueError, match=DEADLINE_ENV):
        SegmentBudget.from_environment({DEADLINE_ENV: "noon"})
    with pytest.raises(ValueError, match=MARGIN_ENV):
        SegmentBudget.from_environment({DEADLINE_ENV: "2000", MARGIN_ENV: "-1"})


def test_untimed_segment_yields_only_inside_the_teardown_margin() -> None:
    # Nothing has been timed: the only known cost is the teardown margin.
    assert not _budget(2000.0, 1000.0, margin_seconds=120.0).should_yield()
    assert _budget(2000.0, 1900.0, margin_seconds=120.0).should_yield()


def test_yields_when_one_safety_scaled_iteration_plus_checkpoint_does_not_fit() -> None:
    budget = _budget(10_000.0, 0.0, margin_seconds=100.0, safety_factor=1.25)
    budget.record_iteration(1_000.0)
    budget.record_checkpoint(600.0)
    assert budget.required_seconds() == pytest.approx(1_250.0 + 600.0 + 100.0)
    assert not budget.should_yield()
    budget.clock = lambda: 10_000.0 - 1_951.0
    assert not budget.should_yield()
    budget.clock = lambda: 10_000.0 - 1_949.0
    assert budget.should_yield()


def test_estimate_tracks_the_slowest_trailing_iterations_and_longest_checkpoint() -> (
    None
):
    budget = _budget(1.0e9, 0.0, margin_seconds=0.0, safety_factor=1.0)
    for seconds in (500.0, 900.0, 300.0, 400.0):
        budget.record_iteration(seconds)
    # Only the trailing three iterations count: the 900 s one is retained.
    assert budget.iteration_seconds == [900.0, 300.0, 400.0]
    budget.record_checkpoint(200.0)
    budget.record_checkpoint(50.0)
    budget.record_checkpoint(None)
    assert budget.checkpoint_seconds == 200.0
    assert budget.required_seconds() == 1_100.0


def test_resumed_segment_seeds_its_estimate_from_the_previous_segment() -> None:
    budget = _budget(3_000.0, 0.0, margin_seconds=0.0, safety_factor=1.0)
    budget.seed_iteration_estimate(None)
    budget.seed_iteration_estimate("fast")
    assert budget.required_seconds() is None
    budget.seed_iteration_estimate(2_500.0)
    assert budget.required_seconds() == 2_500.0
    assert not budget.should_yield()
    budget.clock = lambda: 600.0
    assert budget.should_yield()
    description = budget.describe()
    assert description["remaining_seconds"] == 2_400.0
    assert description["required_seconds"] == 2_500.0
