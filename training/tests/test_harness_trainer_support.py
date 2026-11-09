"""Both harnesses reach the shared trainer through the same adapter surface.

The adapters hold only harness differences (datasets, identity, outcome
semantics, reward translation); the gateway contract, media wiring and
processor selection are exercised here through the shared entry points.
"""

from __future__ import annotations

import copy
import inspect
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from harness_support.arc2 import trainer as arc2_support
from harness_support.arc3 import trainer as arc3_support
from test_arc2_integration import AGENT_SETTINGS, result_payload
from test_platypus_rollout import _window as arc3_window
from training.config import ExperimentConfig
from training.harness import ENVIRONMENTS, TrainerHarnessSupport, get_trainer_support
from training.metrics import summarize_rollouts
from training.multimodal import build_checkpoint_identity
from training.platypus_rollout import build_window_trajectories
from training.ppo import PPOTrainer
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
)

ROOT = Path(__file__).resolve().parents[1]
ARC3_RECIPE = ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json"
ARC2_RECIPE = ROOT / "configs/grpo_3gpu_arc2_overfit.json"
FLASH_ARC2_RECIPE = ROOT / "configs/flash_8gpu_arc2_grpo_tokmean.json"


@pytest.mark.parametrize("environment", ENVIRONMENTS)
def test_every_harness_implements_the_trainer_surface_without_harness_imports(
    environment,
):
    support = get_trainer_support(environment)
    for name in (
        member for member in dir(TrainerHarnessSupport) if not member.startswith("_")
    ):
        assert hasattr(support, name), f"{environment} adapter lacks {name}"
    source = inspect.getsource(support)
    for package in ("duck", "taaf", "platypus", "ullm"):
        assert f"import {package}" not in source and f"from {package}" not in source
    with pytest.raises(ValueError, match="Unsupported harness"):
        get_trainer_support("arc4")


def test_harness_config_rules_are_applied_through_the_adapters(tmp_path):
    arc2 = json.loads(ARC2_RECIPE.read_text())
    arc2["training"]["advantage_estimator"] = "state_critic"
    path = tmp_path / "arc2.json"
    path.write_text(json.dumps(arc2))
    with pytest.raises(ValueError, match="advantage_estimator='grpo'"):
        ExperimentConfig.load(path)

    arc3 = json.loads(ARC3_RECIPE.read_text())
    arc3.setdefault("official_eval", {})["dataset"] = "arc2-eval"
    path = tmp_path / "arc3.json"
    path.write_text(json.dumps(arc3))
    with pytest.raises(ValueError, match="supported only for ARC-2"):
        ExperimentConfig.load(path)


def test_media_capabilities_decide_the_pinned_processing_stack(monkeypatch):
    """The image stack follows the declared media, not the harness name."""
    resolved = []
    monkeypatch.setattr(
        "training.multimodal.resolve_processing_identity",
        lambda model, revision: (
            resolved.append((model, revision))
            or {"model": model, "revision": revision, "sha256": "stack"}
        ),
    )
    arc3_config = ExperimentConfig.load(ARC3_RECIPE)
    assert arc3_support.MEDIA == {"images": False}
    assert arc3_support.resolve_datasets(arc3_config) is None
    assert (
        build_checkpoint_identity(arc3_config, support=arc3_support, manifest=None)
        is None
    )
    assert resolved == []

    arc2_config = ExperimentConfig.load(ARC2_RECIPE)
    manifest = {"name": "arc2-train", "sha256": "d", "puzzle_ids": ["p"]}
    identity = build_checkpoint_identity(
        arc2_config, support=arc2_support, manifest=manifest
    )
    assert resolved == [
        (arc2_config.training.model_name, arc2_config.training.model_revision)
    ]
    assert identity == {
        "environment": "arc2",
        "advantage_estimator": "grpo",
        "dataset_manifest": manifest,
        "processing_stack": {
            "model": arc2_config.training.model_name,
            "revision": arc2_config.training.model_revision,
            "sha256": "stack",
        },
    }


def _cpu_trainer(monkeypatch, identity, config=None):
    if config is None:
        config = ExperimentConfig.load(ARC2_RECIPE).training
    config = replace(
        config, device="cpu", device_map_gpus=[], bf16=False, episode_packing=False
    )
    monkeypatch.setattr(
        "training.ppo.AutoTokenizer.from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(pad_token_id=0),
    )
    monkeypatch.setattr(
        PPOTrainer, "_load_policy_model", lambda *_a, **_kw: torch.nn.Linear(2, 2)
    )
    return PPOTrainer(
        config,
        grpo_group_size=8,
        grpo_puzzles_per_window=4,
        checkpoint_identity=identity,
    )


@pytest.mark.parametrize("images", [False, True])
def test_arc3_media_capability_initializes_processor_without_harness_identity(
    monkeypatch, images
):
    config = ExperimentConfig.load(ARC3_RECIPE)
    assert arc3_support.MEDIA == {"images": False}
    assert arc3_support.training_identity(config, None) is None
    monkeypatch.setattr(arc3_support, "MEDIA", {"images": images})
    resolved = []
    loaded = []
    stack = {"sha256": "arc3-processor"}
    monkeypatch.setattr(
        "training.multimodal.resolve_processing_identity",
        lambda *args: resolved.append(args) or stack,
    )
    monkeypatch.setattr(
        "training.multimodal.load_processing_stack",
        lambda identity, **kwargs: loaded.append(identity) or "processor",
    )
    identity = build_checkpoint_identity(config, support=arc3_support, manifest=None)
    if images:
        assert identity == {
            "environment": "arc3",
            "advantage_estimator": config.training.advantage_estimator,
            "games": sorted(config.inference.games or [config.inference.game]),
            "processing_stack": stack,
        }
        assert resolved == [
            (config.training.model_name, config.training.model_revision)
        ]
    else:
        assert identity is None
        assert resolved == []
    trainer = _cpu_trainer(
        monkeypatch, identity, replace(config.training, value_baseline="group_mean")
    )
    assert trainer.processor == ("processor" if images else None)
    assert loaded == ([stack] if images else [])


def test_configured_penalty_changes_checkpoint_identity_without_breaking_legacy_default():
    config = ExperimentConfig.load(ARC2_RECIPE)
    manifest = {"name": "arc2-train", "sha256": "dataset"}
    original = arc2_support.training_identity(config, manifest)
    assert "malformed_call_penalty" not in original
    config.training.malformed_call_penalty = 0.0
    disabled = arc2_support.training_identity(config, manifest)
    assert disabled == {**original, "malformed_call_penalty": 0.0}
    config.training.malformed_call_penalty = 0.2
    assert arc2_support.training_identity(config, manifest) == {
        **original,
        "malformed_call_penalty": 0.2,
    }


def test_a_checkpoint_resumes_only_under_the_penalty_cap_it_trained_with(
    monkeypatch, tmp_path
):
    """A checkpoint from before the cap (no key) trained uncapped and resumes
    only uncapped; a capped one only under its own cap; either refusal names
    the cap."""
    config = ExperimentConfig.load(ARC2_RECIPE)
    manifest = {"sha256": "dataset"}

    def identity(cap):
        config.training.max_episode_penalty = cap
        return arc2_support.training_identity(config, manifest)

    def saved(checkpoint_identity):
        trainer = _cpu_trainer(monkeypatch, checkpoint_identity)
        path = tmp_path / "trainer_state.pt"
        torch.save(
            trainer._training_state_payload(
                iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
            ),
            path,
        )
        return path

    # What the code before the cap wrote at the default malformed coefficient.
    legacy = {"environment": "arc2", "advantage_estimator": "grpo", "dataset_manifest": manifest}
    path = saved(legacy)
    resumed = _cpu_trainer(monkeypatch, identity(None))
    resumed.load_training_state(path)
    assert resumed._checkpoint_identity == legacy
    with pytest.raises(RuntimeError, match="differs in max_episode_penalty"):
        _cpu_trainer(monkeypatch, identity(0.5)).load_training_state(path)
    path = saved(identity(0.5))
    assert torch.load(path, weights_only=False)["checkpoint_identity"]["max_episode_penalty"] == 0.5
    _cpu_trainer(monkeypatch, identity(0.5)).load_training_state(path)
    for cap in (None, 1.0):
        with pytest.raises(RuntimeError, match="differs in max_episode_penalty"):
            _cpu_trainer(monkeypatch, identity(cap)).load_training_state(path)


@pytest.mark.parametrize("recipe", [ARC2_RECIPE, FLASH_ARC2_RECIPE])
def test_the_recipe_does_not_decide_the_prompt_injection(recipe):
    """The harness's engine preset does: the identity a config builds has none."""
    config = ExperimentConfig.load(recipe)
    identity = arc2_support.training_identity(config, {"sha256": "dataset"})
    assert "prompt_injection" not in identity and "harness_agent" not in identity


def _arc2_window_rollouts(**settings):
    """One ARC-2 rollout per reported agent settings."""
    rollouts = []
    for index, changes in enumerate(settings.get("runs", [{}])):
        payload = result_payload()
        payload["agent_settings"] = {**payload["agent_settings"], **changes}
        run = RunResult.from_payload(payload)
        capture = StepCapture(
            "abc", "first", [1, 2], [3, 4], [-0.1, -0.2], "stop", 0,
            native_token_prompt=True, adapter=run.behavior_policy_snapshot,
        )
        window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
        (trajectory,) = build_window_trajectories(
            window, policy_iteration=1, max_runtime_seconds=900.0
        )
        rollouts.append(replace(trajectory, trajectory_id=f"abc-{index}"))
    return rollouts


def test_every_arc2_run_reports_the_prompt_injection_its_agent_served():
    rollouts = _arc2_window_rollouts()
    assert rollouts[0].metadata["agent_settings"] == AGENT_SETTINGS
    assert arc2_support.rollout_identity(rollouts) == AGENT_SETTINGS
    for settings in (
        None,
        {},
        {**AGENT_SETTINGS, "extra": "x"},
        {**AGENT_SETTINGS, "prompt_injection_schedule": "turn"},
        {**AGENT_SETTINGS, "inject_text_on_later_prompts": None},
    ):
        payload = result_payload()
        payload["agent_settings"] = settings
        with pytest.raises(GatewayError, match="prompt injection as agent_settings"):
            RunResult.from_payload(payload)
    # One window's rollouts serve one injection.
    mixed = _arc2_window_rollouts(runs=[{}, {"inject_text_on_later_prompts": ""}])
    with pytest.raises(ValueError, match="different inject_text_on_later_prompts"):
        arc2_support.rollout_identity(mixed)
    assert arc3_support.rollout_identity(mixed) is None


def test_the_first_window_pins_the_injection_and_a_change_refuses(monkeypatch, tmp_path):
    """The checkpoint identity holds the first window's injection; a later
    window, or a resumed run's next window, reporting another one stops
    training naming the setting."""
    config = ExperimentConfig.load(ARC2_RECIPE)
    identity = arc2_support.training_identity(config, {"sha256": "dataset"})
    trainer = _cpu_trainer(monkeypatch, identity)
    reported = arc2_support.rollout_identity(_arc2_window_rollouts())
    trainer.pin_checkpoint_identity(reported)
    assert trainer._checkpoint_identity == {**identity, "harness_agent": AGENT_SETTINGS}
    trainer.pin_checkpoint_identity(dict(reported))
    changed = {**reported, "inject_text_on_later_prompts": ""}
    with pytest.raises(RuntimeError, match="inject_text_on_later_prompts changed"):
        trainer.pin_checkpoint_identity(changed)
    state = trainer._training_state_payload(
        iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
    )
    assert state["checkpoint_identity"]["harness_agent"] == AGENT_SETTINGS
    path = tmp_path / "trainer_state.pt"
    torch.save(state, path)

    # A resumed trainer starts from the config's identity and adopts the pin.
    resumed = _cpu_trainer(monkeypatch, identity)
    resumed.load_training_state(path)
    assert resumed._checkpoint_identity == {**identity, "harness_agent": AGENT_SETTINGS}
    resumed.pin_checkpoint_identity(dict(reported))
    resumed = _cpu_trainer(monkeypatch, identity)
    resumed.load_training_state(path)
    with pytest.raises(RuntimeError, match="prompt_injection_schedule changed"):
        resumed.pin_checkpoint_identity({**reported, "prompt_injection_schedule": "context"})

    # A checkpoint from before the first window pins nothing yet.
    fresh = _cpu_trainer(monkeypatch, identity)
    unpinned = _cpu_trainer(monkeypatch, identity)
    torch.save(
        fresh._training_state_payload(
            iteration_index=-1, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
        ),
        path,
    )
    unpinned.load_training_state(path)
    assert "harness_agent" not in unpinned._checkpoint_identity
    unpinned.pin_checkpoint_identity(changed)

    # The config's own facts still refuse a resume.
    other = _cpu_trainer(monkeypatch, {**identity, "dataset_manifest": {"sha256": "other"}})
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="differs in dataset_manifest"):
        other.load_training_state(path)
    # A checkpoint from before the harness decided the injection recorded the
    # trainer's own table; it does not resume.
    earlier = copy.deepcopy(state)
    earlier["checkpoint_identity"] = {
        **identity,
        "prompt_injection": {"schedule": "context", "reasoning_prefill": "We need solve"},
    }
    torch.save(earlier, path)
    with pytest.raises(RuntimeError, match="differs in prompt_injection"):
        _cpu_trainer(monkeypatch, identity).load_training_state(path)


@pytest.mark.parametrize(
    ("identity", "expect_processor"),
    [
        (
            {
                "environment": "arc2",
                "advantage_estimator": "grpo",
                "dataset_manifest": {"sha256": "d"},
            },
            False,
        ),
        (
            {
                "environment": "arc3",
                "advantage_estimator": "grpo",
                "dataset_manifest": {"sha256": "d"},
                "processing_stack": {"sha256": "processor"},
            },
            True,
        ),
        (
            {
                "environment": "arc2",
                "advantage_estimator": "grpo",
                "dataset_manifest": {"sha256": "d"},
                "processing_stack": {"sha256": "processor"},
            },
            True,
        ),
    ],
)
def test_processor_selection_keys_on_the_pinned_stack_for_either_harness(
    monkeypatch, identity, expect_processor
):
    loaded = []
    monkeypatch.setattr(
        "training.multimodal.load_processing_stack",
        lambda stack, **_kw: loaded.append(stack) or "processor",
    )
    trainer = _cpu_trainer(monkeypatch, identity)
    assert (trainer.processor == "processor") is expect_processor
    assert loaded == ([identity["processing_stack"]] if expect_processor else [])


def test_arc3_conversion_preserves_witnessed_media(monkeypatch):
    window = arc3_window()
    run = next(iter(window.results.values()))
    media = {"processing": {"sha256": "stack"}, "images": ["img-0"], "layout": [1]}
    evidence = {"consumed": ["img-0"]}
    # Step 2 opens a new segment, so its media is not a continuation check.
    window.captures[(run.run_id, "n2")].media = media
    window.captures[(run.run_id, "n2")].media_evidence = evidence
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=300.0, game_ids=["pathfinder"]
    )
    assert [step.media for step in trajectory.steps] == [None, None, media]
    assert [step.media_evidence for step in trajectory.steps] == [None, None, evidence]
    assert "environment" not in trajectory.metadata
    assert trajectory.metadata.get("score_aggregation") is None


def test_arc3_default_recipe_stays_text_only():
    config = ExperimentConfig.load(ARC3_RECIPE)
    assert config.inference.environment == "arc3"
    assert config.inference.dataset is None
    # The policy loads through the image-text auto-class, yet no image stack
    # is pinned and no step carries media: the inputs stay text-only.
    assert (
        build_checkpoint_identity(config, support=arc3_support, manifest=None) is None
    )
    (trajectory,) = build_window_trajectories(
        arc3_window(), policy_iteration=1, max_runtime_seconds=300.0
    )
    assert all(step.media is None for step in trajectory.steps)
    assert all(step.media_evidence is None for step in trajectory.steps)


@pytest.mark.parametrize("saved_mode", [None, "level_start", "current_grid"])
def test_arc3_image_resume_requires_matching_media_identity(monkeypatch, tmp_path, saved_mode):
    config = ExperimentConfig.load(ARC3_RECIPE).training
    identity = {
        "environment": "arc3",
        "advantage_estimator": config.advantage_estimator,
        "multimodal_context": "current_grid",
        "processing_stack": {"sha256": "processor"},
    }
    monkeypatch.setattr("training.multimodal.load_processing_stack", lambda *a, **kw: "processor")
    trainer = _cpu_trainer(monkeypatch, identity, replace(config, value_baseline="group_mean"))
    state = trainer._training_state_payload(
        iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
    )
    state["checkpoint_identity"] = None if saved_mode is None else {
        **identity, "multimodal_context": saved_mode,
    }
    path = tmp_path / "state.pt"
    torch.save(state, path)
    if saved_mode == "current_grid":
        trainer.load_training_state(path)
    else:
        with pytest.raises(RuntimeError, match="processing stack mismatch"):
            trainer.load_training_state(path)


def test_gateway_applies_each_harness_outcome_validation_itself():
    payload = result_payload()
    payload["arc2_outcome"]["correctness"] = 0.75
    with pytest.raises(GatewayError, match="ARC-2 correctness disagrees"):
        RunResult.from_payload(payload)
    payload = result_payload()
    payload["behavior_policy_snapshot"] = None
    with pytest.raises(GatewayError, match="behavior-policy snapshot identity"):
        RunResult.from_payload(payload)

    arc3_run = next(iter(arc3_window().results.values()))
    arc3_run.terminal_reward = 1.0
    with pytest.raises(GatewayError, match="ARC-3 uses ordered action rewards"):
        arc3_run.validate()
    arc3_run.terminal_reward = 0.0
    arc3_run.actions_per_level = [4]
    with pytest.raises(GatewayError, match="lacks completed-level evidence"):
        arc3_run.validate()

    unknown = next(iter(arc3_window().results.values()))
    unknown.environment = "arc4"
    with pytest.raises(GatewayError, match="unsupported environment"):
        unknown.validate()


def test_harness_deadline_states_differ_only_by_standing_submissions():
    assert arc3_support.DEADLINE_TERMINAL_STATES == {"gave_up", "cancelled"}
    assert arc2_support.DEADLINE_TERMINAL_STATES == {
        "gave_up",
        "cancelled",
        "won",
        "submitted",
    }
    assert arc2_support.REQUIRES_BEHAVIOR_POLICY_SNAPSHOT
    assert not arc3_support.REQUIRES_BEHAVIOR_POLICY_SNAPSHOT


def test_arc2_trajectories_declare_per_task_score_aggregation():
    run = RunResult.from_payload(result_payload(success=(True, False)))
    capture = StepCapture(
        "abc",
        "first",
        [1, 2],
        [3, 4],
        [-0.1, -0.2],
        "stop",
        0,
        native_token_prompt=True,
        adapter=run.behavior_policy_snapshot,
    )
    window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=900.0
    )
    assert trajectory.metadata["environment"] == "arc2"
    assert trajectory.metadata["score_aggregation"] == "per_task_mean"
    # Two rollouts of one puzzle and one of another: the puzzle mean weighs
    # each puzzle equally rather than each rollout.
    second = replace(trajectory, trajectory_id="abc-2")
    other = replace(
        trajectory,
        trajectory_id="def-1",
        task_id="other",
        metadata={**trajectory.metadata, "final_arc_score": 1.0},
    )
    metrics = summarize_rollouts([trajectory, second, other])
    assert metrics.avg_arc_score == pytest.approx((0.5 + 1.0) / 2)


def test_arc2_penalty_metrics_count_empty_replies_among_malformed_calls():
    """rollout/avg_malformed_calls counts every malformed call, an empty reply
    as one; avg_empty_replies_per_rollout counts the empty ones among them;
    both are format breaks; penalty_cap_fraction is the share charged the cap."""
    from training.cli import _format_break_fraction

    def trajectory(payload, **coefficients):
        run = RunResult.from_payload(payload, **coefficients)
        capture = StepCapture(
            "abc",
            "first",
            [1, 2],
            [3, 4],
            [-0.1, -0.2],
            "stop",
            0,
            native_token_prompt=True,
            adapter=run.behavior_policy_snapshot,
        )
        window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
        (built,) = build_window_trajectories(
            window, policy_iteration=1, max_runtime_seconds=900.0
        )
        return built

    empty = trajectory(result_payload(malformed=1, empty_reply=True))
    assert empty.trajectory_reward == pytest.approx(0.5 - 0.05)
    malformed = trajectory(result_payload(malformed=2))
    clean = trajectory(result_payload())
    metrics = summarize_rollouts([empty, malformed, clean])
    assert metrics.avg_malformed_calls == pytest.approx(3 / 3)
    assert metrics.avg_malformed_penalty == pytest.approx(0.15 / 3)
    assert metrics.avg_empty_replies_per_rollout == pytest.approx(1 / 3)
    scalars = metrics.as_scalars()
    assert scalars["rollout/avg_malformed_calls"] == pytest.approx(1.0)
    assert scalars["rollout/avg_empty_replies_per_rollout"] == pytest.approx(1 / 3)
    assert "rollout/avg_empty_reply_penalty" not in scalars
    assert _format_break_fraction([empty, malformed, clean]) == pytest.approx(2 / 3)
    assert scalars["rollout/penalty_cap_fraction"] == 0.0
    # One of four episodes charged the cap (0.15 of penalties, cap 0.1).
    capped = trajectory(result_payload(malformed=3, cap=0.1), max_episode_penalty=0.1)
    assert capped.trajectory_reward == pytest.approx(0.4)
    metrics = summarize_rollouts([empty, malformed, clean, capped])
    assert metrics.penalty_cap_fraction == pytest.approx(0.25)
    assert metrics.as_scalars()["rollout/penalty_cap_fraction"] == pytest.approx(0.25)
    # The penalty average stays uncapped.
    assert metrics.avg_malformed_penalty == pytest.approx((0.05 + 0.1 + 0.15) / 4)


@pytest.mark.parametrize(("recipe", "cap"), [(ARC2_RECIPE, None), (FLASH_ARC2_RECIPE, 0.5)])
def test_the_identity_records_a_set_cap(recipe, cap):
    config = ExperimentConfig.load(recipe)
    assert config.training.max_episode_penalty == cap
    identity = arc2_support.training_identity(config, {"sha256": "dataset"})
    assert identity.get("max_episode_penalty") == cap
    assert ("max_episode_penalty" in identity) is (cap is not None)


def test_arc3_trajectories_keep_the_per_level_budget_the_worker_reported():
    window = arc3_window()
    run = next(iter(window.results.values()))
    run.agent_settings = {"max_tokens_per_level": [90_000]}
    arc3_support.validate_run_result(run)
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=300.0
    )
    assert trajectory.metadata["agent_settings"] == {"max_tokens_per_level": [90_000]}
    for malformed in ([], [0], "90000", [90_000.0]):
        run.agent_settings = {"max_tokens_per_level": malformed}
        with pytest.raises(ValueError, match="max_tokens_per_level"):
            arc3_support.validate_run_result(run)
    run.agent_settings = {"max_tokens_per_level": None}
    arc3_support.validate_run_result(run)


def test_full_weight_checkpoints_carry_the_pinned_injection(monkeypatch):
    """Full-weight checkpoints record the runtime's identity fields: the pin
    goes there (and to the trainer), and a resume adopts the saved one."""
    from training.full_weight_distributed import FullWeightDistributedRuntime

    monkeypatch.setattr(FullWeightDistributedRuntime, "_restore_rng_state", staticmethod(lambda state: None))
    config = ExperimentConfig.load(ARC2_RECIPE)
    identity = arc2_support.training_identity(config, {"sha256": "dataset"})

    def runtime():
        built = object.__new__(FullWeightDistributedRuntime)
        built.rank, built.world_size = 0, 1
        built._trainer = SimpleNamespace(_checkpoint_identity=dict(identity), state_critic=None)
        built._checkpoint_identity_fields = {
            "checkpoint_identity": dict(identity),
            "grpo_group_size": 8,
            "grpo_puzzles_per_window": 4,
        }
        return built

    first = runtime()
    first.pin_checkpoint_identity(dict(AGENT_SETTINGS))
    pinned = {**identity, "harness_agent": AGENT_SETTINGS}
    assert first._checkpoint_identity_fields["checkpoint_identity"] == pinned
    assert first.trainer._checkpoint_identity == pinned
    with pytest.raises(RuntimeError, match="inject_text_on_first_prompt changed"):
        first.pin_checkpoint_identity({**AGENT_SETTINGS, "inject_text_on_first_prompt": ""})

    resumed = runtime()
    expected = {"iteration_index": 3, **resumed._checkpoint_identity_fields}
    metadata = SimpleNamespace(
        trainer_payload={"iteration_index": 3, "rng_by_rank": [{}]},
        metadata_fields=dict(first._checkpoint_identity_fields),
        optimizer_step=4,
    )
    resumed._restore_checkpoint_payload(metadata, expected=expected)
    assert resumed._checkpoint_identity_fields["checkpoint_identity"] == pinned
    assert resumed.trainer._checkpoint_identity == pinned
    with pytest.raises(RuntimeError, match="prompt_injection_schedule changed"):
        resumed.pin_checkpoint_identity({**AGENT_SETTINGS, "prompt_injection_schedule": "context"})
