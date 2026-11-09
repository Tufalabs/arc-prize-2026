"""ARC-2 admission, terminal reward and cross-server policy identity regressions."""

from dataclasses import replace

import pytest
import requests

from engine_wire import generation
from test_arc2_integration import result_payload
import test_rollout_gateway as gateway_fixtures
from training.contracts import RolloutTrajectory
from training.grpo import compute_grpo_step_signals
from training.inference import _behavior_policy_snapshot
from training.platypus_rollout import build_window_trajectories


gateway = gateway_fixtures.gateway


@pytest.mark.parametrize("penalty", [0.0, 0.05, 0.4])
@pytest.mark.parametrize("worker_agrees", [True, False])
def test_gateway_verifies_configured_penalty_and_preserves_it_in_trajectory(
    gateway, penalty, worker_agrees
):
    gateway.set_policy(replace(gateway._policy, malformed_call_penalty=penalty))
    assert (
        gateway_fixtures._generate(
            gateway,
            user="abc#first",
            prompt=[1, 2],
            arc3_progress={
                "version": 1,
                "after_nonce": None,
                "score": 0.0,
                "stage": "episode",
                "data": {},
            },
        ).status_code
        == 200
    )
    payload = result_payload(
        malformed=2, penalty=penalty if worker_agrees else penalty + 1
    )
    payload["serving_adapter"] = gateway_fixtures._adapter()
    # A worker cannot choose the coefficient by adding it to its own report.
    payload["malformed_call_penalty"] = penalty + 1
    reply = requests.post(gateway.results_url, json={"runs": [payload]}, timeout=5)
    if not worker_agrees:
        assert reply.status_code == 400
        assert "malformed_penalty disagrees" in reply.text
        return
    assert reply.status_code == 200, reply.text
    window = gateway.drain_window()
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=0, max_runtime_seconds=900.0
    )
    assert trajectory.trajectory_reward == pytest.approx(0.5 - 2 * penalty)
    assert trajectory.steps[-1].reward == pytest.approx(0.5 - 2 * penalty)


@pytest.mark.parametrize("worker_cap", [0.1, None, 0.2])
def test_gateway_verifies_its_episode_penalty_cap(gateway, worker_cap):
    """The window's cap rides the sampling policy; a worker that scored with
    another (or none) is refused, and a binding cap is what the reward pays."""
    gateway.set_policy(replace(gateway._policy, max_episode_penalty=0.1))
    assert (
        gateway_fixtures._generate(
            gateway,
            user="abc#first",
            prompt=[1, 2],
            arc3_progress={
                "version": 1,
                "after_nonce": None,
                "score": 0.0,
                "stage": "episode",
                "data": {},
            },
        ).status_code
        == 200
    )
    payload = result_payload(malformed=3, cap=worker_cap)
    payload["serving_adapter"] = gateway_fixtures._adapter()
    reply = requests.post(gateway.results_url, json={"runs": [payload]}, timeout=5)
    if worker_cap != 0.1:
        assert reply.status_code == 400
        assert "max_episode_penalty disagrees" in reply.text
        return
    assert reply.status_code == 200, reply.text
    (trajectory,) = build_window_trajectories(
        gateway.drain_window(), policy_iteration=0, max_runtime_seconds=900.0
    )
    assert trajectory.trajectory_reward == pytest.approx(0.5 - 0.1)
    assert trajectory.metadata["penalty_capped"] is True
    assert trajectory.metadata["malformed_penalty"] == pytest.approx(0.15)


def _image_request():
    """Use the real image contract while keeping private imports optional in CI."""
    from test_multimodal_training import fixture_media

    _, ids, media, evidence = fixture_media()
    wire = generation(
        dict(
            model="stub-model",
            user="abc#first",
            prompt=ids,
            max_tokens=4,
            arc3_lineage=gateway_fixtures._lineage(),
            skip_special_tokens=False,
            **gateway_fixtures._sampling(),
        )
    )
    wire["prompt"]["media"] = media
    return wire, evidence


def test_image_prompt_passes_real_sampling_admission_and_capture(gateway, monkeypatch):
    """The HTTP policy accepts media and retains the exact consumed image evidence."""
    wire, evidence = _image_request()
    render_output = gateway_fixtures.output
    monkeypatch.setattr(
        gateway_fixtures,
        "output",
        lambda body: {**render_output(body), "media_evidence": evidence},
    )
    reply = requests.post(f"{gateway.base_url}/generate", json=wire, timeout=5)
    assert reply.status_code == 200, reply.text
    assert len(gateway_fixtures._StubVllm.seen) == 1
    assert gateway_fixtures._StubVllm.seen[0]["media"] == wire["prompt"]["media"]
    capture = gateway.drain_window().captures["abc", "first"]
    assert capture.prompt_token_ids == wire["prompt"]["prompt_token_ids"]
    assert capture.media == wire["prompt"]["media"]
    assert capture.media_evidence == evidence


@pytest.mark.parametrize("with_prompt_media", [False, True])
def test_sampling_parameter_cannot_shadow_prompt_media(gateway, with_prompt_media):
    """Media belongs only to the token prompt, even when a sampler value is equal."""
    wire, _ = _image_request()
    wire["sampling_params"]["media"] = wire["prompt"]["media"]
    if not with_prompt_media:
        del wire["prompt"]["media"]
    reply = requests.post(f"{gateway.base_url}/generate", json=wire, timeout=5)
    assert reply.status_code == 400
    assert "cannot shadow" in reply.text
    assert not gateway_fixtures._StubVllm.seen


def test_terminal_standing_reward_survives_unprocessed_deadline_draw(gateway):
    """The actual callback reconciles terminal reward after zero-valued progress."""
    gateway.set_policy(
        replace(gateway._policy, require_progress=True, max_runtime_seconds=900.0)
    )

    def progress(after):
        return dict(
            version=1,
            after_nonce=after,
            score=0.0,
            stage="episode",
            data={"decision_elapsed_seconds": 0.0},
        )

    first = gateway_fixtures._generate(
        gateway, user="abc#first", prompt=[1, 2], arc3_progress=progress(None)
    )
    assert first.status_code == 200, first.text
    pending = gateway_fixtures._generate(
        gateway,
        user="abc#pending",
        prompt=[1, 2, 3, 4, 5],
        arc3_lineage=gateway_fixtures._lineage("continue"),
        arc3_progress=progress("first"),
    )
    assert pending.status_code == 200, pending.text
    payload = result_payload(malformed=2, state="cancelled")
    payload["serving_adapter"] = gateway_fixtures._adapter()
    payload["steps"][0]["decision_elapsed_seconds"] = 0.0
    reply = requests.post(gateway.results_url, json={"runs": [payload]}, timeout=5)
    assert reply.status_code == 200, reply.text
    window = gateway.drain_window()
    assert len(gateway_fixtures._StubVllm.seen) == len(window.captures) == 2
    assert all(capture.progress["score"] == 0 for capture in window.captures.values())
    assert window.results["abc"].progress_reconciliation == {
        "captured_after_nonce": "pending",
        "unprocessed_nonces": ["pending"],
    }
    trajectory = build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=900.0
    )[0]
    assert trajectory.trajectory_reward == pytest.approx(0.4)
    assert trajectory.metadata["final_arc_score"] == 0.5
    assert trajectory.metadata["malformed_call_count"] == 2
    assert trajectory.done and trajectory.termination_reason == "TIMEOUT"
    assert len(trajectory.steps) == 1
    assert trajectory.steps[0].completion_token_ids == [3, 4]
    assert trajectory.steps[0].reward == pytest.approx(0.4)


def test_identical_adapter_weights_group_across_serving_processes(tmp_path):
    """Server-local IDs and paths differ while canonical weights define a group."""
    adapters = []
    for index in range(2):
        directory = tmp_path / f"server-{index}"
        directory.mkdir()
        (directory / "adapter_config.json").write_text('{"r": 16}')
        (directory / "adapter_model.safetensors").write_bytes(b"identical-weights")
        adapters.append(
            dict(
                name="policy",
                id=index + 1,
                path=str(directory),
                server_instance_id=f"instance-{index}",
            )
        )
    snapshots = [
        _behavior_policy_snapshot(adapter, model="base", revision="sha")
        for adapter in adapters
    ]
    assert adapters[0] != adapters[1]
    assert snapshots[0] == snapshots[1]
    episodes = [
        RolloutTrajectory(
            trajectory_id=f"run-{index}",
            task_id=f"puzzle-{index // 8}",
            steps=[],
            done=True,
            trajectory_reward=float(index % 2),
            metadata={
                "rollout_slot_id": f"slot-{index}",
                "rollout_seed": 1000 + index,
                "serving_adapter": adapters[index % 2],
                "behavior_policy_snapshot": snapshots[index % 2],
            },
        )
        for index in range(32)
    ]
    _, stats = compute_grpo_step_signals(episodes, group_size=8, puzzles_per_window=4)
    assert stats.rollout_count == 32 and stats.mean_reward == 0.5
    (tmp_path / "server-1/adapter_model.safetensors").write_bytes(b"changed-weights")
    episodes[1].metadata["behavior_policy_snapshot"] = _behavior_policy_snapshot(
        adapters[1], model="base", revision="sha"
    )
    with pytest.raises(ValueError, match="snapshots"):
        compute_grpo_step_signals(episodes, group_size=8, puzzles_per_window=4)
