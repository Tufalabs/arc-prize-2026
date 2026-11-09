"""Rollout augmentation is part of a checkpoint's identity, for both harnesses."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from harness_support.arc2 import trainer as arc2_support
from harness_support.arc3 import trainer as arc3_support
from test_harness_trainer_support import ARC2_RECIPE, ARC3_RECIPE, _cpu_trainer
from training.config import ExperimentConfig, RolloutAugmentationConfig
from training.multimodal import build_checkpoint_identity

AUGMENTED = {"version": 1, "dihedral": True, "colors": True}


def _augment(config, **settings):
    config.inference.augmentation = RolloutAugmentationConfig.from_dict(
        {"enabled": True, **settings}, environment=config.inference.environment
    )
    return config


def test_arc2_identity_records_the_augmentation(monkeypatch):
    monkeypatch.setattr(
        "training.multimodal.resolve_processing_identity",
        lambda model, revision: {"sha256": "stack"},
    )
    manifest = {"name": "arc2-train", "sha256": "d", "puzzle_ids": ["p"]}
    plain = build_checkpoint_identity(
        ExperimentConfig.load(ARC2_RECIPE), support=arc2_support, manifest=manifest
    )
    assert "rollout_augmentation" not in plain
    augmented = build_checkpoint_identity(
        _augment(ExperimentConfig.load(ARC2_RECIPE)),
        support=arc2_support,
        manifest=manifest,
    )
    assert augmented == {**plain, "rollout_augmentation": AUGMENTED}
    flat = build_checkpoint_identity(
        _augment(ExperimentConfig.load(ARC2_RECIPE), dihedral=False),
        support=arc2_support,
        manifest=manifest,
    )
    assert flat["rollout_augmentation"] == {**AUGMENTED, "dihedral": False}


def test_a_text_arc3_run_gains_an_identity_only_when_augmenting():
    config = ExperimentConfig.load(ARC3_RECIPE)
    assert build_checkpoint_identity(config, support=arc3_support, manifest=None) is None
    identity = build_checkpoint_identity(
        _augment(config), support=arc3_support, manifest=None
    )
    assert identity == {
        "environment": "arc3",
        "advantage_estimator": config.training.advantage_estimator,
        "games": sorted(config.inference.games),
        "rollout_augmentation": {"version": 1, "dihedral": False, "colors": True},
    }


def _saved_state(trainer, tmp_path, saved_identity):
    state = trainer._training_state_payload(
        iteration_index=0, adapter_dir=tmp_path, runtime_adapter_dir=tmp_path
    )
    state["checkpoint_identity"] = saved_identity
    path = tmp_path / "state.pt"
    torch.save(state, path)
    return path


@pytest.mark.parametrize("saved", ["plain", "augmented", "other"])
@pytest.mark.parametrize("expected", ["plain", "augmented"])
def test_resume_requires_the_same_augmentation(monkeypatch, tmp_path, saved, expected):
    base = {
        "environment": "arc2",
        "advantage_estimator": "grpo",
        "dataset_manifest": {"name": "arc2-train", "sha256": "d"},
    }
    identities = {
        "plain": base,
        "augmented": {**base, "rollout_augmentation": AUGMENTED},
        "other": {**base, "rollout_augmentation": {**AUGMENTED, "dihedral": False}},
    }
    trainer = _cpu_trainer(monkeypatch, identities[expected])
    path = _saved_state(trainer, tmp_path, identities[saved])
    if saved == expected:
        trainer.load_training_state(path)
    else:
        with pytest.raises(RuntimeError, match="rollout augmentation"):
            trainer.load_training_state(path)


def test_an_identityless_arc3_checkpoint_does_not_resume_under_augmentation(
    monkeypatch, tmp_path
):
    """The legacy ARC-3 allowance (no saved identity) covers unaugmented runs only."""
    config = ExperimentConfig.load(ARC3_RECIPE)
    training = replace(config.training, value_baseline="group_mean")
    assert training.advantage_estimator == "state_critic"
    plain = _cpu_trainer(monkeypatch, {"environment": "arc3", "advantage_estimator": "state_critic"}, training)
    plain.load_training_state(_saved_state(plain, tmp_path, None))
    identity = build_checkpoint_identity(
        _augment(config), support=arc3_support, manifest=None
    )
    augmented = _cpu_trainer(monkeypatch, identity, training)
    with pytest.raises(RuntimeError, match="rollout augmentation"):
        augmented.load_training_state(_saved_state(augmented, tmp_path, None))
