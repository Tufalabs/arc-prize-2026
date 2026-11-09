"""The callback must not silently change counts, clocks, or reward linkage."""

import asyncio
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from harness_support.agent_settings import record_agent_settings

from training.context_contract import resolve_context_contract
from training.platypus_rollout import build_window_trajectories
from training.rollout_gateway import (
    CaptureGateway,
    GatewayError,
    RunResult,
    SamplingPolicy,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)
from training.run_failure import (
    FatalTrainingError,
    raise_if_run_failed,
    record_run_failure,
)


def _payload():
    return {
        "run_id": "run-1",
        "game_id": "ar25-0c556536",
        "state": "gave_up",
        "levels_completed": 1,
        "number_of_levels": 2,
        "actions_per_level": [2, 0],
        "base_actions_per_level": [2, 4],
        "final_wallclock_seconds": 900.0,
        "action_wallclock_seconds": [1.0, 2.0],
        "steps": [
            {"nonce": "act", "action_index": 0, "output_len": 2, "output_crc32": 1},
            {
                "nonce": "inspect",
                "action_index": None,
                "output_len": 1,
                "output_crc32": 2,
            },
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("run_id", 1),
        ("game_id", ""),
        ("state", None),
        ("levels_completed", True),
        ("levels_completed", 1.5),
        ("levels_completed", "1"),
        ("levels_completed", -1),
        ("levels_completed", 3),
        ("number_of_levels", False),
        ("number_of_levels", "2"),
        ("actions_per_level", [1.5, 0.5]),
        ("actions_per_level", [True, True]),
        ("actions_per_level", [1]),
        ("actions_per_level", "20"),
        ("base_actions_per_level", [-1]),
        ("base_actions_per_level", []),
        ("base_actions_per_level", ["2"]),
        ("final_wallclock_seconds", float("nan")),
        ("final_wallclock_seconds", float("inf")),
        ("final_wallclock_seconds", -1.0),
        ("final_wallclock_seconds", "900.0"),
        ("final_wallclock_seconds", 1.5),
        ("action_wallclock_seconds", [float("nan"), 2.0]),
        ("action_wallclock_seconds", [-1.0, 2.0]),
        ("action_wallclock_seconds", [True, 2.0]),
        ("action_wallclock_seconds", [2.0, 1.0]),
        ("action_wallclock_seconds", None),
        ("steps", {}),
        ("steps", [None]),
        ("steps", []),
    ],
)
def test_malformed_outcome_is_rejected_without_coercion(field, value):
    payload = _payload()
    payload[field] = value
    with pytest.raises(GatewayError):
        RunResult.from_payload(payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("nonce", ""),
        ("nonce", 1),
        ("nonce", "act"),
        ("output_len", 1.5),
        ("output_len", True),
        ("output_crc32", "2"),
        ("output_crc32", -1),
        ("action_index", True),
        ("action_index", 0.5),
        ("action_index", -1),
        ("action_index", 0),
        ("action_index", 2),
    ],
)
def test_malformed_step_cannot_duplicate_tokens_or_misattribute_actions(field, value):
    payload = _payload()
    payload["steps"][1][field] = value
    with pytest.raises(GatewayError):
        RunResult.from_payload(payload)


def test_action_indices_cannot_run_backwards_or_omit_initial_actions():
    payload = _payload()
    payload["steps"][0]["action_index"] = 1
    payload["steps"][1]["action_index"] = 0
    with pytest.raises(GatewayError, match="increase"):
        RunResult.from_payload(payload)
    payload["steps"][1]["action_index"] = None
    with pytest.raises(GatewayError, match="linkage"):
        RunResult.from_payload(payload)
    del payload["steps"][0]["action_index"]
    with pytest.raises(GatewayError, match="lacks action_index"):
        RunResult.from_payload(payload)


def test_completed_level_cannot_lack_action_evidence():
    payload = _payload()
    payload.update(levels_completed=2, actions_per_level=[2])
    with pytest.raises(GatewayError, match="completed-level evidence"):
        RunResult.from_payload(payload)


def test_valid_deadline_inspection_and_zero_action_failure_remain_reportable():
    assert RunResult.from_payload(_payload()).steps[-1]["action_index"] is None
    payload = _payload()
    payload.update(
        state="crashed",
        levels_completed=0,
        number_of_levels=0,
        actions_per_level=[],
        base_actions_per_level=[],
        final_wallclock_seconds=None,
        action_wallclock_seconds=[],
        steps=[],
    )
    run = RunResult.from_payload(payload)
    (trajectory,) = build_window_trajectories(
        WindowCapture(results={run.run_id: run}),
        policy_iteration=0,
        max_runtime_seconds=900.0,
    )
    # A crashed run is its own terminal, scored on what it reached; ERROR is
    # left for runs the harness never finished.
    assert trajectory.termination_reason == "CRASHED"
    assert trajectory.metadata["status"] == "CRASHED"


def test_assembly_also_rejects_invalid_in_memory_reports():
    run = RunResult.from_payload(_payload())
    run.steps[1]["action_index"] = 0
    with pytest.raises(GatewayError, match="increase"):
        build_window_trajectories(
            WindowCapture(results={run.run_id: run}),
            policy_iteration=0,
            max_runtime_seconds=900.0,
        )


class _Request:
    def __init__(self, runs):
        self.runs = runs

    async def json(self):
        return {"runs": self.runs}


def test_identical_callback_retry_is_idempotent_but_conflicting_batch_is_fatal(
    tmp_path,
):
    gateway = CaptureGateway(
        upstream_base_url="http://unused",
        port=0,
        on_contract_violation=lambda error: record_run_failure(tmp_path, error),
    )
    payload = _payload()
    gateway.set_policy(
        SamplingPolicy(
            model="model",
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            context=resolve_context_contract(32768, 4096, 16384, 0),
            max_runtime_seconds=900.0,
        )
    )
    for index, (step, prompt, output) in enumerate(
        zip(payload["steps"], [[1, 2], [1, 2, 3, 4, 5]], [[3, 4], [6]], strict=True)
    ):
        gateway._admit_generation(
            {
                "user": f"run-1#{step['nonce']}",
                "prompt": prompt,
                "max_tokens": 4096,
                "arc3_lineage": {
                    "version": 1,
                    "epoch": 0,
                    "kind": "continue" if index else "start",
                    "compaction_prompt_token_ids": None,
                },
            },
        )
        gateway._record_capture(
            StepCapture(
                run_id="run-1",
                nonce=step["nonce"],
                prompt_token_ids=prompt,
                completion_token_ids=output,
                completion_logprobs=[-0.1] * len(output),
                finish_reason="stop",
                order=index,
                native_token_prompt=True,
            )
        )
        step.update(
            output_len=len(output),
            output_crc32=token_ids_crc32(output),
            continued=index > 0,
            history_epoch=0,
            exact_history=True,
        )
    for _ in range(2):
        response = asyncio.run(
            gateway._handle_results(_Request([copy.deepcopy(payload)]))
        )
        assert response.status == 200
        assert len(gateway._window.results) == 1
    second = copy.deepcopy(payload)
    second["run_id"] = "run-2"
    conflicting = copy.deepcopy(payload)
    conflicting["state"] = "lost"
    response = asyncio.run(gateway._handle_results(_Request([second, conflicting])))
    assert response.status == 400
    assert set(gateway._window.results) == {"run-1"}
    assert gateway._window.results["run-1"].state == "gave_up"
    with pytest.raises(FatalTrainingError, match="conflicting result evidence"):
        raise_if_run_failed(tmp_path)
    assert (tmp_path / "STOP_AUTO_RELAUNCH").exists()


def test_worker_callback_remains_compatible_with_validated_full_game_results(
    tmp_path, monkeypatch
):
    path = Path(__file__).resolve().parents[1] / "scripts/platypus_rollout_worker.py"
    spec = importlib.util.spec_from_file_location("run_results_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    game = "ar25-0c556536"
    payload = _payload()
    run = {
        key: value
        for key, value in payload.items()
        if key not in ("steps", "action_wallclock_seconds")
    }
    run["game_id"] = worker._slot_game_ids([game])[0]
    run["history"] = [
        {"wallclock_seconds": value} for value in payload["action_wallclock_seconds"]
    ]
    (tmp_path / "benchmark.json").write_text(json.dumps({"game_runs": [run]}))
    trace_dir = tmp_path / "traces" / run["run_id"]
    trace_dir.mkdir(parents=True)
    (trace_dir / "steps.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "provider_validation": {
                        "request_args": {
                            "request_id": f"run-1#{step['nonce']}",
                            "request_context": {
                                "arc3_progress": {
                                    "data": {"decision_elapsed_seconds": 0.0}
                                },
                            },
                        }
                    },
                    "action_index": step["action_index"],
                    "output_token_ids": [3, 4],
                    "extra": {
                        "continued": index > 0,
                        "history_epoch": 0,
                        "exact_history": True,
                    },
                }
            )
            for index, step in enumerate(payload["steps"])
        )
    )
    received = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def read(self):
            return b'{"accepted": 1}'

    def receive(request, timeout):
        received.extend(
            RunResult.from_payload(run) for run in json.loads(request.data)["runs"]
        )
        return Response()

    monkeypatch.setattr(worker.urllib.request, "urlopen", receive)
    record_agent_settings(tmp_path, {"max_tokens_per_level": None})
    worker._post_results(
        {
            "job_dir": str(tmp_path),
            "games": [game],
            "results_url": "http://unused",
            "max_runtime_s_per_game": 60,
        }
    )
    assert len(received) == 1
    assert received[0].game_id == game
    assert received[0].actions_per_level == payload["actions_per_level"]
    assert [step["action_index"] for step in received[0].steps] == [0, None]
    assert received[0].agent_settings == {"max_tokens_per_level": None}
