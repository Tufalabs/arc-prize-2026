"""Regressions for critic vocabulary, packed metrics and complete-window drift."""

from copy import deepcopy
import json
from types import SimpleNamespace as NS

import pytest
import torch

from test_episode_packing import _step_sample
from test_ppo import _make_tiny_ppo_trainer, _single_step_rollout, _tiny_action_sample
from training import ppo
from training.episode_packing import pack_episode_samples


def _prepare_window(monkeypatch, tmp_path, *, learning_rate):
    trainer = _make_tiny_ppo_trainer(advantage=1, learning_rate=learning_rate)
    trainer.config.max_window_clip_fraction = 0.25
    trainer.config.extra_train_cycles_until_window = True
    sample = _tiny_action_sample(advantage=1)
    monkeypatch.setattr(ppo, "build_samples", lambda *a, **k: [sample])
    exports = []
    monkeypatch.setattr(
        trainer,
        "_export_policy_snapshot",
        lambda *a: (
            exports.append(trainer.model.logits.detach().clone())
            or (tmp_path, tmp_path, 0.0)
        ),
    )
    monkeypatch.setattr(
        trainer, "_save_training_state", lambda *a, **k: tmp_path / "state.pt"
    )
    return trainer, exports


def _one_extra_window(trainer, tmp_path):
    remaining = iter([1000.0, 1000.0, 0.0])
    return trainer.train_window(
        [_single_step_rollout()],
        output_dir=tmp_path,
        iteration_index=0,
        next_window_ready=lambda: False,
        next_window_seconds_remaining=lambda: next(remaining),
    )


def _assert_state_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_state_equal(actual[key], expected[key])
    elif isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_state_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("learning_rate,rejected", [(0.07, True), (1e-5, False)])
def test_final_extra_guard_checks_window_start_and_restores_only_extras(
    monkeypatch,
    tmp_path,
    learning_rate,
    rejected,
):
    trainer, exports = _prepare_window(
        monkeypatch, tmp_path, learning_rate=learning_rate
    )
    snapshots, scored_ids, scored_values = [], [], []
    original_snapshot = trainer._snapshot_update_state
    original_score = trainer._trust_region_token_logprobs

    def snapshot():
        result = original_snapshot()
        snapshots.append(result)
        return result

    def score(samples):
        scored_ids.append([sample.input_ids[:] for sample in samples])
        values = original_score(samples)
        scored_values.append(values.clone())
        return values

    monkeypatch.setattr(trainer, "_snapshot_update_state", snapshot)
    monkeypatch.setattr(trainer, "_trust_region_token_logprobs", score)
    result = _one_extra_window(trainer, tmp_path)
    assert len(snapshots) == 2
    assert len(scored_ids) == 3  # before, post-primary, post-extra; same bounded probe
    assert scored_ids[0] == scored_ids[1] == scored_ids[2]
    assert torch.exp(scored_values[1] - scored_values[0]).max() <= 1.2
    assert (
        bool((torch.exp(scored_values[2] - scored_values[0]) > 1.2).any()) is rejected
    )
    assert not result.window_update_rejected  # primary update is retained
    assert len(exports) == 1
    assert result.post_update_clip_fraction <= trainer.config.max_window_clip_fraction
    assert getattr(trainer, "_trust_region_lr_scale", 1.0) == 1.0
    if rejected:
        assert result.extra_stop_reason == "window_clip_guard"
        assert result.extra_cycles == result.extra_optimizer_steps == 0
        assert result.optimizer_steps == trainer.optimizer_step_count == 1
        for parameter, expected in snapshots[1]["parameters"]:
            assert torch.equal(parameter, expected)
        _assert_state_equal(
            trainer.optimizer.state_dict(), snapshots[1]["optimizer_state"]
        )
        assert torch.equal(exports[0], snapshots[1]["parameters"][0][1])
        assert not torch.equal(exports[0], snapshots[0]["parameters"][0][1])
        # Rejected-extra forward statistics do not contaminate retained metrics.
        assert result.mean_ratio == pytest.approx(1.0)
    else:
        assert result.extra_stop_reason == "insufficient_rollout_slack"
        assert result.extra_cycles == result.extra_optimizer_steps == 1
        assert result.optimizer_steps == trainer.optimizer_step_count == 2


@pytest.mark.parametrize("missing_call", [1, 2, 3])
def test_enabled_guard_missing_probe_is_fatal_before_export(
    monkeypatch, tmp_path, missing_call
):
    trainer, exports = _prepare_window(monkeypatch, tmp_path, learning_rate=1e-5)
    original = trainer._trust_region_token_logprobs
    calls = 0

    def score(samples):
        nonlocal calls
        calls += 1
        return torch.empty(0) if calls == missing_call else original(samples)

    monkeypatch.setattr(trainer, "_trust_region_token_logprobs", score)
    with pytest.raises(RuntimeError, match="requires sampled probe tokens"):
        _one_extra_window(trainer, tmp_path)
    assert not exports
    assert trainer.optimizer_step_count == missing_call - 1


def test_extra_snapshot_rechecks_rollout_deadline_before_updating(
    monkeypatch, tmp_path
):
    trainer, exports = _prepare_window(monkeypatch, tmp_path, learning_rate=1e-5)
    remaining = iter([1000.0, 0.0])  # slack expires during the post-primary snapshot
    result = trainer.train_window(
        [_single_step_rollout()],
        output_dir=tmp_path,
        iteration_index=0,
        next_window_ready=lambda: False,
        next_window_seconds_remaining=lambda: next(remaining),
    )
    assert result.extra_stop_reason == "insufficient_rollout_slack"
    assert result.extra_optimizer_steps == 0
    assert result.optimizer_steps == trainer.optimizer_step_count == 1
    assert len(exports) == 1


@pytest.mark.parametrize("scores", [torch.tensor([float("nan")]), torch.empty(2)])
def test_trust_probe_requires_matching_finite_evidence(scores):
    trainer = _make_tiny_ppo_trainer(advantage=1)
    with pytest.raises(RuntimeError, match="matching finite token scores"):
        trainer._post_update_clip_fraction([_tiny_action_sample(advantage=1)], scores)


@pytest.mark.parametrize("compacted", [False, True])
def test_advantage_reporting_preserves_mixed_signs_and_levels_across_packing(compacted):
    first = _step_sample("episode", 0, [1, 2], [3], advantage=3)
    second = _step_sample("episode", 1, [1, 2, 3, 4], [5, 6, 7], advantage=-1)
    third = _step_sample("episode", 2, [1, 2, 3, 4, 5, 6, 7, 8], [9, 10], advantage=0)
    first.level, second.level, third.level = 1, 2, 2
    third.segment_start = compacted
    samples = [first, second, third]
    packed, _ = pack_episode_samples(samples, max_sequence_tokens=32)
    before, after = map(ppo._sample_advantage_summary, (samples, packed))
    assert len(packed) == (2 if compacted else 1)
    assert before == after
    assert after.mean == pytest.approx(0)
    assert after.std == pytest.approx(2**0.5)
    assert after.positive_fraction == pytest.approx(1 / 6)
    assert after.nonzero_fraction == pytest.approx(4 / 6)
    assert after.mean_by_level == pytest.approx({1: 3, 2: -0.6})
    assert packed[0].advantage == 3  # zero-signal sentinel remains independent
    prefix = ppo._parity_prefix(packed[0], 1)
    assert ppo._sample_advantage_summary([prefix]).mean_by_level == {1: 3}


def test_train_window_reports_target_advantages_instead_of_packed_sentinel(
    monkeypatch, tmp_path
):
    trainer, _ = _prepare_window(monkeypatch, tmp_path, learning_rate=1e-5)
    sample = _tiny_action_sample(advantage=3)
    sample.token_advantages = [-1]
    sample.span_ids = [1]
    sample.token_levels = [2]
    monkeypatch.setattr(ppo, "build_samples", lambda *a, **k: [sample])
    result = trainer.train_window(
        [_single_step_rollout()], output_dir=tmp_path, iteration_index=0
    )
    assert result.mean_advantage == -1
    assert result.std_advantage == 0
    assert result.advantage_positive_fraction_kept == 0
    assert result.advantage_nonzero_fraction_kept == 1
    assert result.mean_advantage_by_level == {2: -1}


def test_ppo_constructor_and_resume_use_configured_game_vocabulary(
    monkeypatch, tmp_path
):
    template = _make_tiny_ppo_trainer(advantage=1)
    config = deepcopy(template.config)
    config.value_baseline = "hidden_critic"
    config.worker_device_map_gpus = [[1]]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        ppo.AutoTokenizer, "from_pretrained", lambda *a, **k: NS(pad_token_id=0)
    )
    monkeypatch.setattr(
        ppo.PPOTrainer, "_load_policy_model", lambda *a, **k: deepcopy(template.model)
    )
    monkeypatch.setattr(ppo.PPOTrainer, "_policy_hidden_size", lambda self: 4)
    worker_vocabs = []
    monkeypatch.setattr(
        ppo, "_DdpWorkerGroup", lambda **k: worker_vocabs.append(k["game_vocab"])
    )
    trainer = ppo.PPOTrainer(config, game_vocab=["b", "a"])
    assert trainer.state_critic.game_vocab == {"a": 1, "b": 2}
    assert worker_vocabs == [("b", "a")]
    saved = trainer._training_state_payload(
        iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
    )
    assert saved["state_critic_state"]["game_vocab"] == {"a": 1, "b": 2}
    checkpoint = tmp_path / "trainer.pt"
    torch.save(saved, checkpoint)
    resumed = ppo.PPOTrainer(
        config, game_vocab=["a", "b"], resume_state_path=checkpoint
    )
    assert resumed.state_critic.game_vocab == trainer.state_critic.game_vocab
    with pytest.raises(RuntimeError, match="game vocabulary"):
        ppo.PPOTrainer(config, game_vocab=["a", "c"], resume_state_path=checkpoint)


def test_ddp_spec_and_worker_constructor_preserve_game_vocabulary(
    monkeypatch, tmp_path
):
    from training import ddp_worker

    config = _make_tiny_ppo_trainer(advantage=1).config
    config.worker_device_map_gpus = [[1]]
    monkeypatch.setenv("ARC3_RUN_DIR", str(tmp_path))
    monkeypatch.setattr("training.inference.find_available_local_port", lambda: 12345)
    monkeypatch.setattr(ppo.subprocess, "Popen", lambda *a, **k: NS(pid=123))
    monkeypatch.setattr(ppo.threading, "Thread", lambda **k: NS(start=lambda: None))
    monkeypatch.setattr(ppo.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(ppo.dist, "new_group", lambda *a, **k: None)
    monkeypatch.setattr(ppo.dist, "broadcast", lambda *a, **k: None)
    ppo._DdpWorkerGroup(
        config=config,
        parameters=[torch.nn.Parameter(torch.ones(1))],
        resume_adapter_dir=None,
        game_vocab=["a", "b"],
    )
    spec = json.loads((tmp_path / "ddp_worker_1.spec.json").read_text())
    assert spec["game_vocab"] == ["a", "b"]

    class CapturedConstruction(Exception):
        pass

    def trainer(config, **kwargs):
        assert kwargs["game_vocab"] == ["a", "b"]
        raise CapturedConstruction

    monkeypatch.setattr(ppo, "PPOTrainer", trainer)
    with pytest.raises(CapturedConstruction):
        ddp_worker.run_worker(spec)
