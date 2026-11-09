"""Task-opaque feedback ordering, boundary proof and final reward reconciliation."""

from dataclasses import replace
import math

import pytest
import requests

from training.gateway_progress import ProgressError, ProgressLedger, validate_progress
from training.rollout_gateway import GatewayError, token_ids_crc32
from harness_support.arc3.results import enrich_arc3_result
from test_rollout_gateway import _generate as _completion, _lineage, _StubVllm
from test_rollout_gateway import gateway as _gateway_fixture

gateway = _gateway_fixture


def snapshot(after=None, score=0.0, stage=0, elapsed=0.0):
    return {
        "version": 1,
        "after_nonce": after,
        "score": score,
        "stage": stage,
        "data": {"decision_elapsed_seconds": elapsed},
    }


@pytest.mark.parametrize(
    "change",
    [
        {"version": True},
        {"score": True},
        {"score": float("nan")},
        {"stage": {}},
        {"stage": False},
        {"after_nonce": ""},
        {"data": {"x": float("inf")}},
        {"data": {"x": "x" * 16384}},
        {"extra": 1},
    ],
)
def test_progress_is_explicit_bounded_finite_json(change):
    with pytest.raises(ProgressError):
        validate_progress({**snapshot(), **change})


def test_progress_links_to_latest_response_and_does_not_share_mutable_payloads():
    ledger = ProgressLedger()
    first = snapshot()
    ledger.admit("r", "a", first, previous_nonce=None)
    first["score"] = 100
    assert ledger.snapshots["r"]["a"]["score"] == 0
    with pytest.raises(ProgressError, match="immediately preceding"):
        ledger.admit("r", "b", snapshot(), previous_nonce="a")
    ledger.admit("r", "b", snapshot("a", 2, 1), previous_nonce="a", level_boundary=True)
    with pytest.raises(ProgressError, match="newly reported stage"):
        ledger.admit(
            "r", "c", snapshot("b", 2, 1), previous_nonce="b", level_boundary=True
        )


@pytest.mark.parametrize("stage", [0, 1, -1, "next"])
def test_boundary_progress_cannot_decrease_repeat_or_use_opaque_stage(stage):
    ledger = ProgressLedger()
    ledger.admit("r", "a", snapshot(stage=1), previous_nonce=None)
    with pytest.raises(ProgressError, match="higher completed level"):
        ledger.admit(
            "r",
            "b",
            snapshot("a", stage=stage),
            previous_nonce="a",
            level_boundary=True,
        )


def test_terminal_reward_reconciliation_handles_compaction_and_unprocessed_deadline_draw():
    ledger = ProgressLedger()
    ledger.admit("r", "a", snapshot(), previous_nonce=None)
    ledger.admit(
        "r", "summary", snapshot("a", 115, 1), previous_nonce="a", level_boundary=True
    )
    ledger.admit("r", "reset", snapshot("summary", 115, 1), previous_nonce="summary")
    ledger.admit("r", "deadline", snapshot("reset", 120, 2), previous_nonce="reset")
    steps = [
        {"nonce": "a", "reward": 115},
        {"nonce": "summary", "reward": 0},
        {"nonce": "reset", "reward": 5},
    ]
    ledger.reconcile("r", snapshot("reset", 120, 2), steps, 120)
    assert ledger.close("r", [s["nonce"] for s in steps]) == {
        "captured_after_nonce": "deadline",
        "unprocessed_nonces": ["deadline"],
    }


@pytest.mark.parametrize("failure", ["intermediate", "terminal", "return", "anchor"])
def test_final_report_cannot_silently_rewrite_prior_reward_evidence(failure):
    ledger = ProgressLedger()
    ledger.admit("r", "a", snapshot(), previous_nonce=None)
    ledger.admit("r", "b", snapshot("a", 10, 1), previous_nonce="a")
    steps = [{"nonce": "a", "reward": 10}, {"nonce": "b", "reward": 0}]
    final, total = snapshot("b", 10, 1), 10
    if failure == "intermediate":
        steps[0]["reward"], steps[1]["reward"] = 0, 10
    elif failure == "terminal":
        final["score"] = 9
    elif failure == "return":
        total = 9
    else:
        final["after_nonce"] = "a"
    with pytest.raises(ProgressError):
        ledger.reconcile("r", final, steps, total)


def test_no_generation_episode_has_explicit_zero_reward_terminal_state():
    ledger = ProgressLedger()
    ledger.reconcile("empty", snapshot(stage=4), [], 0)
    assert ledger.close("empty", []) == {
        "captured_after_nonce": None,
        "unprocessed_nonces": [],
    }
    with pytest.raises(ProgressError):
        ledger.reconcile("empty", snapshot(score=1), [], 1)


def _terminal(gateway, nonces, *, score=0, deadline=False, include_progress=True):
    raw = {
        "run_id": "run-9",
        "game_id": "game",
        "state": "gave_up",
        "levels_completed": 0,
        "number_of_levels": 2,
        "actions_per_level": [0, 0],
        "base_actions_per_level": [2, 4],
        "final_wallclock_seconds": 12.5 if deadline else 1,
        "action_wallclock_seconds": [],
        "level_replay_entries": [],
        "steps": [
            {
                "nonce": nonce,
                "decision_elapsed_seconds": 0.0,
                "action_index": None,
                "exact_history": True,
                "continued": nonce != nonces[0],
                "history_epoch": 0,
                "output_len": 2,
                "output_crc32": token_ids_crc32([3, 4]),
            }
            for nonce in nonces
        ],
    }
    run = enrich_arc3_result(raw, max_runtime_seconds=12.5)
    run["terminal_progress"] = snapshot(nonces[-1] if nonces else None, score)
    if not include_progress:
        del run["terminal_progress"]
    return requests.post(gateway.results_url, json={"runs": [run]}, timeout=5)


def test_http_progress_is_not_forwarded_to_engine_and_terminal_flush_is_captured(
    gateway,
):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    assert _completion(gateway, arc3_progress=snapshot()).status_code == 200
    assert "arc3_progress" not in _StubVllm.seen[0]
    assert (
        _completion(
            gateway,
            user="run-9#tail",
            prompt=[11, 12, 13, 3, 4, 14],
            arc3_lineage=_lineage("continue"),
            arc3_progress=snapshot("tok"),
        ).status_code
        == 200
    )
    assert _terminal(gateway, ["tok"], deadline=True).status_code == 200
    window = gateway.drain_window()
    assert window.captures["run-9", "tail"].progress == snapshot("tok")
    assert window.results["run-9"].progress_reconciliation == {
        "captured_after_nonce": "tail",
        "unprocessed_nonces": ["tail"],
    }


@pytest.mark.parametrize("failure", ["missing", "wrong_parent", "bad_final"])
def test_progress_violations_stop_upstream_and_latch_fatal_even_if_harness_catches_them(
    gateway, failure
):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    if failure == "missing":
        reply = _completion(gateway)
        assert not _StubVllm.seen
    else:
        assert _completion(gateway, arc3_progress=snapshot()).status_code == 200
        if failure == "bad_final":
            reply = _terminal(gateway, ["tok"], score=1)
        else:
            reply = _completion(
                gateway,
                user="run-9#next",
                prompt=[11, 12, 13, 3, 4, 14],
                arc3_progress=snapshot("wrong"),
                arc3_lineage=_lineage("continue"),
            )
        assert len(_StubVllm.seen) == 1
    assert reply.status_code == 400
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()
    with pytest.raises(GatewayError):
        gateway.drain_window()


def test_terminal_progress_is_mandatory(gateway):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    assert _completion(gateway, arc3_progress=snapshot()).status_code == 200
    assert _terminal(gateway, ["tok"], include_progress=False).status_code == 400
    with pytest.raises(GatewayError, match="snapshot is required"):
        gateway.raise_if_invalid()


def test_unexpected_callback_handler_errors_also_latch_fatal(gateway, monkeypatch):
    def failed_callback(runs):
        raise RuntimeError("callback decoding failed")

    monkeypatch.setattr(gateway, "_record_results", failed_callback)
    assert _terminal(gateway, []).status_code == 400
    with pytest.raises(GatewayError, match="callback decoding failed"):
        gateway.raise_if_invalid()


@pytest.mark.parametrize("clock", [None, True, -1, "1", float("inf"), float("nan")])
def test_required_decision_clock_rejected_before_generation(gateway, clock):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    progress = snapshot(elapsed=clock)
    if clock is None:
        progress["data"] = {}
    # NaN/Infinity cannot be serialized by requests, so check those at the
    # admission boundary directly; the HTTP decoder uses the same validation.
    if isinstance(clock, float) and not math.isfinite(clock):
        with pytest.raises(ProgressError, match="decision_elapsed_seconds"):
            validate_progress(progress, require_decision_clock=True)
        return
    assert _completion(gateway, arc3_progress=progress).status_code == 400
    assert not _StubVllm.seen
    assert gateway._progress.snapshots == {}
    with pytest.raises(GatewayError, match="decision_elapsed_seconds"):
        gateway.raise_if_invalid()


def test_decreasing_decision_clock_leaves_ledger_unchanged(gateway):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    assert _completion(gateway, arc3_progress=snapshot(elapsed=2)).status_code == 200
    response = _completion(
        gateway,
        user="run-9#later",
        prompt=[11, 12, 13, 3, 4, 14],
        arc3_lineage=_lineage("continue"),
        arc3_progress=snapshot("tok", elapsed=1),
    )
    assert response.status_code == 400
    assert len(_StubVllm.seen) == 1
    assert list(gateway._progress.snapshots["run-9"]) == ["tok"]
    with pytest.raises(GatewayError, match="must not decrease"):
        gateway.raise_if_invalid()


def test_lineage_rejection_does_not_commit_valid_progress(gateway):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=12.5)
    )
    assert _completion(gateway, arc3_progress=snapshot()).status_code == 200
    response = _completion(
        gateway,
        user="run-9#bad-prefix",
        prompt=[99, 12, 13, 3, 4, 14],
        arc3_lineage=_lineage("continue"),
        arc3_progress=snapshot("tok", elapsed=1),
    )
    assert response.status_code == 400
    assert len(_StubVllm.seen) == 1
    assert gateway._progress.snapshots == {"run-9": {"tok": snapshot()}}
    with pytest.raises(GatewayError):
        gateway.raise_if_invalid()


def test_generic_progress_can_omit_decision_clock():
    progress = snapshot()
    progress["data"] = {}
    assert validate_progress(progress) == progress
    with pytest.raises(ProgressError, match="decision_elapsed_seconds"):
        validate_progress(progress, require_decision_clock=True)
