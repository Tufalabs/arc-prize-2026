"""Standing ARC-2 outcomes keep one unfinished draw at the runtime deadline."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import threading

import pytest

from harness_support.arc2.worker import add_response_reports
from test_arc2_integration import result_payload
import test_rollout_gateway as fixtures
from training.platypus_rollout import build_window_trajectories


gateway = fixtures.gateway


def _progress(after):
    return dict(
        version=1,
        after_nonce=after,
        score=0.0,
        stage="episode",
        data={"decision_elapsed_seconds": 0.0},
    )


def _start(gateway, state):
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=900.0)
    )
    response = fixtures._generate(
        gateway, user="abc#first", prompt=[1, 2], arc3_progress=_progress(None)
    )
    assert response.status_code == 200, response.text
    payload = result_payload(success=(state == "won", False), malformed=2, state=state)
    payload["serving_adapter"] = fixtures._adapter()
    payload["steps"][0]["decision_elapsed_seconds"] = 0.0
    return payload


def _next(gateway, nonce="final", after="first", prompt=None):
    return fixtures._generate(
        gateway,
        user=f"abc#{nonce}",
        prompt=prompt or [1, 2, 3, 4, 5],
        arc3_lineage=fixtures._lineage("continue"),
        arc3_progress=_progress(after),
    )


def _trace_report(nonce):
    return {
        **fixtures._step(),
        "nonce": nonce,
        "continued": True,
        "decision_elapsed_seconds": 0.0,
    }


def _processed_journal(tmp_path):
    directory = tmp_path / "processed_responses"
    directory.mkdir()
    (directory / "abc.jsonl").write_text(
        json.dumps({"nonce": "first", "malformed_call_count": 2, "empty_reply": False})
        + "\n"
    )


@pytest.mark.parametrize("state", ["submitted", "won"])
@pytest.mark.parametrize("pending", [False, True])
def test_standing_deadline_retains_reward_and_one_unprocessed_draw(
    gateway, monkeypatch, tmp_path, state, pending
):
    payload = _start(gateway, state)
    entered, release = threading.Event(), threading.Event()
    upstream = gateway._upstream

    async def delayed(method, url, **kwargs):
        if pending and url.endswith("/generate"):
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.01)
        return await upstream(method, url, **kwargs)

    monkeypatch.setattr(gateway, "_upstream", delayed)
    _processed_journal(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        generation = executor.submit(_next, gateway)
        try:
            if pending:
                assert entered.wait(timeout=5)
            else:
                assert generation.result(timeout=5).status_code == 200
                # The host completed the draw, but the child did not consume it.
                payload["steps"].append(_trace_report("final"))
            add_response_reports(payload, job_dir=tmp_path, max_runtime_seconds=900.0)
            assert [step["nonce"] for step in payload["steps"]] == ["first"]
            response = fixtures._report(gateway, payload)
            assert response.status_code == 200, response.text
            drained = executor.submit(gateway.drain_window, timeout_seconds=5)
            if pending:
                assert not drained.done()
        finally:
            release.set()
        assert generation.result(timeout=5).status_code == 200
        window = drained.result(timeout=5)
    assert len(fixtures._StubVllm.seen) == len(window.captures) == 2
    assert window.results["abc"].state == state
    assert window.results["abc"].progress_reconciliation["unprocessed_nonces"] == [
        "final"
    ]
    trajectory = build_window_trajectories(
        window, policy_iteration=0, max_runtime_seconds=900.0
    )[0]
    correctness = 0.5 if state == "won" else 0.0
    assert trajectory.trajectory_reward == pytest.approx(correctness - 0.1)
    assert trajectory.metadata["final_arc_score"] == correctness
    assert trajectory.metadata["malformed_call_count"] == 2
    assert trajectory.done and len(trajectory.steps) == 1
    assert trajectory.steps[0].completion_token_ids == [3, 4]
    assert trajectory.steps[0].reward == pytest.approx(correctness - 0.1)


@pytest.mark.parametrize("invalid", ["before_deadline", "multiple", "interior"])
def test_standing_outcome_does_not_relax_deadline_or_nonce_order(gateway, invalid):
    payload = _start(gateway, "submitted")
    assert _next(gateway).status_code == 200
    if invalid == "before_deadline":
        payload["final_wallclock_seconds"] = 899.999
    else:
        assert (
            _next(gateway, "third", "final", [1, 2, 3, 4, 5, 3, 4, 6]).status_code
            == 200
        )
        if invalid == "interior":
            payload["steps"].append(
                {
                    **_trace_report("third"),
                    "malformed_call_count": 0,
                    "empty_reply": False,
                    "reward": 0.0,
                }
            )
            payload["terminal_progress"]["after_nonce"] = "third"
    response = fixtures._report(gateway, payload)
    assert response.status_code == 400
    assert (
        "omitted an interior generation" if invalid == "interior" else "exactly one"
    ) in response.text


@pytest.mark.parametrize("invalid", ["before_deadline", "multiple", "action"])
def test_response_adapter_discards_only_one_final_deadline_draw(tmp_path, invalid):
    payload = result_payload(malformed=2, state="submitted")
    _processed_journal(tmp_path)
    payload["steps"].append(_trace_report("final"))
    if invalid == "before_deadline":
        payload["final_wallclock_seconds"] = 899.999
    elif invalid == "multiple":
        payload["steps"].append(_trace_report("third"))
    else:
        payload["steps"][-1]["action_index"] = 0
    with pytest.raises(ValueError, match="lacks processed-response penalty evidence"):
        add_response_reports(payload, job_dir=tmp_path, max_runtime_seconds=900.0)
