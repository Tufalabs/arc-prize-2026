"""A game whose agent raises mid-game ends ``crashed`` and the window goes on.

Since 2026-10-11 the rollout workers run the harness's own
``on_agent_error="crash_game"`` (harness_support/arc2/worker.py): platypus
leaves the game unfinished, Benchmark teardown marks it ``crashed`` and scores
its standing answer, and ``benchmark.json`` carries it with ``finished`` and
its ``solver_note`` (the traceback). The worker reports it in the window's one
result batch, scored like ``gave_up``/``cancelled`` (correctness of what it
submitted, less its malformed-call penalty, at most the cap); the trainer
re-derives that and trains it as a ``CRASHED`` terminal with the note's head
and tail in its metadata. The Duck-environment test of the same flow, with a
real Benchmark, is tests/test_arc2_crashed_game_duck.py.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_support.agent_settings import record_agent_settings
from harness_support.arc2 import worker as arc2_worker
from harness_support.arc2.results import enrich_arc2_result
from test_arc2_integration import AGENT_SETTINGS, _trajectory, result_payload
from test_full_game_slots import _worker
from training.inference import _platypus_window_outcomes
from training.platypus_rollout import CRASH_NOTE_CHARS, build_window_trajectories
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)

# The 2026-10-11 crash as platypus notes it, with enough frames that the
# note's head and tail differ.
NOTE = (
    "agent error:\nTraceback (most recent call last):\n"
    + '  File "duck/python_tool.py", line 556, in run\n'
    '    notices = await self._housekeeping() if self.file_tools else ""\n'
    * 6
    + "RuntimeError: kernel housekeeping failed: Traceback (most recent call last):\n"
    "TypeError: 'NoneType' object is not subscriptable\n"
)
SNAPSHOT = {"name": "policy", "id": 1, "path": "/adapter"}


def _crashed(success=(True, False), malformed=0, *, penalty=0.05, cap=None, note=NOTE):
    payload = result_payload(success, malformed, "crashed", penalty=penalty, cap=cap)
    payload["solver_note"] = note
    return payload


@pytest.mark.parametrize(
    ("success", "malformed", "cap", "reward"),
    [
        ((True, False), 0, None, 0.5),
        ((True,), 0, None, 1.0),
        ((True, False), 2, None, 0.5 - 0.2),
        # Penalties and the cap apply as to any episode.
        ((True,), 30, 0.5, 0.5),
        ((False,), 3, None, -0.3),
    ],
)
def test_a_crashed_run_is_scored_on_its_standing_answer(
    success, malformed, cap, reward
):
    payload = _crashed(success, malformed, penalty=0.1, cap=cap)
    outcome = payload["arc2_outcome"]
    assert outcome["termination_category"] == "crashed"
    assert outcome["per_output_success"] == list(success)
    assert payload["terminal_reward"] == pytest.approx(reward)
    # The trainer re-derives it with its own coefficient and cap.
    run = RunResult.from_payload(
        payload, malformed_call_penalty=0.1, max_episode_penalty=cap
    )
    trajectory = _trajectory(run)
    assert trajectory.termination_reason == "CRASHED"
    assert trajectory.metadata["status"] == "CRASHED"
    assert trajectory.done is True
    assert trajectory.trajectory_reward == pytest.approx(reward)
    assert trajectory.steps[-1].reward == pytest.approx(reward)
    assert trajectory.metadata["final_arc_score"] == sum(success) / len(success)


def test_a_crashed_trajectory_carries_the_head_and_tail_of_its_note():
    run = RunResult.from_payload(_crashed())
    metadata = _trajectory(run).metadata
    assert len(NOTE) > 2 * CRASH_NOTE_CHARS
    assert metadata["crash_note"] == NOTE[:CRASH_NOTE_CHARS]
    assert metadata["crash_note"].startswith("agent error:\nTraceback")
    assert metadata["crash_note_tail"] == NOTE[-CRASH_NOTE_CHARS:]
    assert "kernel housekeeping failed" in metadata["crash_note_tail"]
    assert "'NoneType' object is not subscriptable" in metadata["crash_note_tail"]
    # No note: nothing to carry. Any other state: never carried.
    run = RunResult.from_payload(_crashed(note=None))
    assert _trajectory(run).metadata["crash_note"] is None
    payload = result_payload()
    payload["solver_note"] = "gave up: budget"
    metadata = _trajectory(RunResult.from_payload(payload)).metadata
    assert (metadata["crash_note"], metadata["crash_note_tail"]) == (None, None)
    assert metadata["status"] == "COMPLETED_RUN"


def test_a_note_must_be_text():
    payload = _crashed()
    payload["solver_note"] = ["agent error"]
    with pytest.raises(GatewayError, match="solver_note must be a string"):
        RunResult.from_payload(payload)


def _never_answered(*, steps, result):
    """A run that crashed before it submitted anything, as teardown scored it."""
    payload = dict(
        run_id="abc",
        game_id="puzzle",
        state="crashed",
        levels_completed=0,
        number_of_levels=1,
        actions_per_level=[0],
        base_actions_per_level=[],
        final_wallclock_seconds=12.0,
        action_wallclock_seconds=[],
        rollout_slot_id="puzzle#s0:abc",
        rollout_seed=17,
        behavior_policy_snapshot=SNAPSHOT,
        serving_adapter=SNAPSHOT,
        steps=steps,
        solver_note=NOTE,
    )
    enriched = enrich_arc2_result(
        payload,
        benchmark_run={
            "finished": True,
            "history": [],
            "result": result,
            "final_score": 0.0,
        },
        test_output_count=2,
        malformed_call_penalty=0.05,
        max_episode_penalty=None,
    )
    enriched["agent_settings"] = dict(AGENT_SETTINGS)
    return enriched


@pytest.mark.parametrize(
    "result", [{"pairs": [{"correct": False}, {"correct": False}]}, None]
)
def test_a_crash_before_the_first_model_call_is_a_zero_step_outcome(result):
    payload = _never_answered(steps=[], result=result)
    assert payload["arc2_outcome"]["per_output_success"] == [False, False]
    assert payload["terminal_reward"] == 0.0
    assert payload["terminal_progress"]["after_nonce"] is None
    run = RunResult.from_payload(payload)
    (trajectory,) = build_window_trajectories(
        WindowCapture(results={run.run_id: run}),
        policy_iteration=0,
        max_runtime_seconds=900.0,
    )
    assert trajectory.steps == []
    assert trajectory.termination_reason == "CRASHED"
    assert trajectory.trajectory_reward == 0.0
    assert trajectory.metadata["crash_note"] == NOTE[:CRASH_NOTE_CHARS]


def test_a_crash_after_malformed_draws_without_an_answer_is_charged():
    step = dict(
        nonce="first",
        action_index=None,
        output_len=2,
        output_crc32=token_ids_crc32([3, 4]),
        continued=False,
        history_epoch=0,
        exact_history=True,
        malformed_call_count=2,
        empty_reply=False,
    )
    payload = _never_answered(steps=[step], result=None)
    assert payload["terminal_reward"] == pytest.approx(-0.1)
    assert _trajectory(
        RunResult.from_payload(payload)
    ).trajectory_reward == pytest.approx(-0.1)


@pytest.mark.parametrize("state", ["playing", "not_started", ""])
def test_a_run_the_harness_never_finished_is_still_refused(state):
    with pytest.raises((RuntimeError, ValueError)):
        result_payload(state=state)
    payload = result_payload()
    payload["state"] = state or "playing"
    with pytest.raises(GatewayError, match="terminal state"):
        RunResult.from_payload(payload)


def test_the_outcome_must_name_the_state_the_run_ended_in():
    payload = result_payload(state="submitted")
    payload["state"] = "crashed"
    with pytest.raises(GatewayError, match="termination category disagrees"):
        RunResult.from_payload(payload)


def test_an_unfinished_benchmark_record_is_still_refused():
    with pytest.raises(ValueError, match="before final scoring"):
        enrich_arc2_result(
            {"state": "crashed", "steps": []},
            benchmark_run={"finished": False, "history": [], "result": None},
            test_output_count=1,
            malformed_call_penalty=0.05,
            max_episode_penalty=None,
        )


# --- the worker's report, from the files the harness leaves -------------------


def _journal(job_dir: Path, run_id: str, nonces: list[str], malformed: int = 0) -> None:
    root = job_dir / "processed_responses"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{run_id}.jsonl").open("a", encoding="utf-8") as stream:
        for nonce in nonces:
            record = {
                "nonce": nonce,
                "malformed_call_count": malformed,
                "truncation_cap_source": None,
                "stop_reason": "tool_use",
                "has_malformed_error": False,
                "has_tool_calls": True,
                "empty_reply": False,
            }
            stream.write(json.dumps(record) + "\n")


def _trace(job_dir: Path, run_id: str, actions: list[int | None]) -> None:
    """The host trace of RUN_ID: one step per draw, each [3, 4], continuing."""
    folder = job_dir / "traces" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "provider_validation": {
                    "request_args": {"request_id": f"{run_id}#n{index}"}
                },
                "action_index": action,
                "output_token_ids": [3, 4],
                "extra": {
                    "continued": index > 0,
                    "history_epoch": 0,
                    "exact_history": True,
                },
            }
        )
        for index, action in enumerate(actions)
    ]
    (folder / "steps.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _benchmark_run(run_id, slot, state, *, answers, pairs, note=None, elapsed=40.0):
    return {
        "run_id": run_id,
        "game_id": slot,
        "state": state,
        "finished": True,
        "history": [{"wallclock_seconds": 5.0 * (i + 1)} for i in range(answers)],
        "result": {"pairs": [{"correct": value} for value in pairs]},
        "final_score": 100.0 * sum(pairs) / len(pairs),
        "levels_completed": int(any(pairs)),
        "number_of_levels": 1,
        "actions_per_level": [answers],
        "base_actions_per_level": None,
        "final_wallclock_seconds": elapsed,
        "solver_note": note,
    }


def _arc2_spec(job_dir: Path) -> dict:
    return {
        "environment": "arc2",
        "dataset": "arc2-eval",
        "games": ["steady", "crashy"],
        "slot_seeds": [11, 22],
        "job_dir": str(job_dir),
        "results_url": "http://unused.invalid/results",
        "max_runtime_s_per_game": 2400.0,
        "behavior_policy_snapshot": SNAPSHOT,
        "serving_adapter": SNAPSHOT,
        "malformed_call_penalty": 0.05,
        "max_episode_penalty": 0.5,
    }


def _post(worker, spec, monkeypatch) -> list[dict]:
    captured: list[dict] = []

    def capture(request, timeout):
        captured.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"accepted": len(captured[-1]["runs"])}).encode())

    monkeypatch.setattr(worker.urllib.request, "urlopen", capture)
    worker._post_results(spec)
    (batch,) = captured
    return batch["runs"]


def _captures(run_id: str, count: int) -> dict:
    """The gateway's captures of RUN_ID's draws: each prompt extends the last."""
    captures = {}
    prompt = [1, 2]
    for index in range(count):
        captures[(run_id, f"n{index}")] = StepCapture(
            run_id,
            f"n{index}",
            list(prompt),
            [3, 4],
            [-0.1, -0.2],
            "stop",
            index,
            native_token_prompt=True,
            adapter=SNAPSHOT,
        )
        prompt = prompt + [3, 4, 5]
    return captures


def test_the_worker_reports_a_crashed_run_in_the_windows_one_batch(
    tmp_path, monkeypatch
):
    """One of two games crashed after its first answer, one draw after its
    last processed one: the window's results carry both, the crashed one
    scored on its standing answer; the draw its agent never processed is not
    reported, and the trainer trains the rest as a CRASHED terminal."""
    worker = _worker()
    monkeypatch.setattr(
        arc2_worker,
        "load_tasks",
        lambda dataset: {
            "steady": SimpleNamespace(to_answer=[None]),
            "crashy": SimpleNamespace(to_answer=[None, None]),
        },
    )
    benchmark = {
        "game_runs": [
            _benchmark_run("runA", "steady#s0", "won", answers=1, pairs=[True]),
            _benchmark_run(
                "runB",
                "crashy#s1",
                "crashed",
                answers=1,
                pairs=[True, False],
                note=NOTE,
                # Teardown finishes a crashed run when the benchmark ends.
                elapsed=2500.0,
            ),
        ]
    }
    (tmp_path / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")
    _trace(tmp_path, "runA", [0, None])
    _journal(tmp_path, "runA", ["n0", "n1"])
    # runB: answered at its first draw, malformed at its second, and crashed
    # after the host recorded its third, before its agent processed it.
    _trace(tmp_path, "runB", [0, None, None])
    _journal(tmp_path, "runB", ["n0"])
    _journal(tmp_path, "runB", ["n1"], malformed=1)
    record_agent_settings(tmp_path, AGENT_SETTINGS)
    runs = _post(worker, _arc2_spec(tmp_path), monkeypatch)

    steady, crashy = runs
    assert (steady["state"], steady["game_id"]) == ("won", "steady")
    assert (crashy["state"], crashy["game_id"]) == ("crashed", "crashy")
    assert crashy["solver_note"] == NOTE
    assert [step["nonce"] for step in crashy["steps"]] == ["n0", "n1"]
    assert crashy["arc2_outcome"]["termination_category"] == "crashed"
    assert crashy["arc2_outcome"]["per_output_success"] == [True, False]
    assert crashy["terminal_reward"] == pytest.approx(0.5 - 0.05)

    results = {
        run["run_id"]: RunResult.from_payload(
            run, malformed_call_penalty=0.05, max_episode_penalty=0.5
        )
        for run in runs
    }
    window = WindowCapture(
        captures={**_captures("runA", 2), **_captures("runB", 3)}, results=results
    )
    by_run = {
        rollout.metadata["run_id"]: rollout
        for rollout in build_window_trajectories(
            window, policy_iteration=4, max_runtime_seconds=2400.0
        )
    }
    assert by_run["runA"].termination_reason == "COMPLETED_RUN"
    crashed = by_run["runB"]
    assert crashed.termination_reason == "CRASHED"
    assert len(crashed.steps) == 2
    assert crashed.trajectory_reward == pytest.approx(0.45)
    assert crashed.metadata["crash_note_tail"].endswith("not subscriptable\n")
    # The window completes, counting the crash.
    outcomes = _platypus_window_outcomes(
        "w", list(by_run.values()), log_path=tmp_path / "worker.log"
    )
    assert outcomes["crashed"] == 1 and outcomes["errors"] == 0


@pytest.mark.parametrize(
    ("state", "actions", "journaled", "accepted"),
    [
        # The crashed run's last draw, never processed: dropped.
        ("crashed", [0, None], ["n0"], True),
        # Only the last draw, and only one that authorized no action.
        ("crashed", [0, None, None], ["n0", "n2"], False),
        ("crashed", [None, 0], ["n0"], False),
        # A finished run below the deadline has evidence for every draw.
        ("submitted", [0, None], ["n0"], False),
    ],
)
def test_only_a_crashed_runs_last_unprocessed_draw_may_lack_its_journal(
    tmp_path, state, actions, journaled, accepted
):
    _journal(tmp_path, "run", journaled)
    run = {
        "run_id": "run",
        "state": state,
        "final_wallclock_seconds": 10.0,
        "steps": [
            {"nonce": f"n{index}", "action_index": action}
            for index, action in enumerate(actions)
        ],
    }
    if not accepted:
        with pytest.raises(ValueError):
            arc2_worker.add_response_reports(
                run, job_dir=tmp_path, max_runtime_seconds=2400.0
            )
        return
    arc2_worker.add_response_reports(run, job_dir=tmp_path, max_runtime_seconds=2400.0)
    assert [step["nonce"] for step in run["steps"]] == journaled


def test_the_results_are_posted_before_the_window_is_reported_done(
    tmp_path, monkeypatch
):
    """Teardown marks a crashed run only when the benchmark ends; the worker
    posts the window's results after that and before ``<window>.done.json``,
    which is what the trainer waits on before it drains the results."""
    worker = _worker()
    events: list[tuple[str, bool]] = []
    done = tmp_path / "w1.done.json"

    async def run(spec):
        events.append(("benchmark ended", done.exists()))

    def post(spec):
        events.append(("results posted", done.exists()))

    monkeypatch.setattr(worker, "_run", run)
    monkeypatch.setattr(worker, "_post_results", post)
    spec = {"window_id": "w1", "results_url": "http://unused.invalid/results"}
    (tmp_path / "w1.spec.json").write_text(json.dumps(spec), encoding="utf-8")
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    assert worker.serve(tmp_path, poll_seconds=0.0) == 0
    assert events == [("benchmark ended", False), ("results posted", False)]
    assert json.loads(done.read_text(encoding="utf-8")) == {"status": "ok"}
