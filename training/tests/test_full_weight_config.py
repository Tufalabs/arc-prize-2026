from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.config import ExperimentConfig


def _payload() -> dict[str, object]:
    return {
        "num_iterations": 1,
        "parallel_rollouts": 2,
        "inference": {
            "base_url": "http://127.0.0.1:1234",
            "model_id": "Qwen/Qwen3.6-27B",
            "games": ["pathfinder-0001"],
            "max_runtime_minutes": 1,
            "agent_max_completion_tokens": 128,
            "platypus_venv": "/tmp/platypus-venv",
            "server_gpus": ["0"],
        },
        "training": {
            "model_name": "Qwen/Qwen3.6-27B",
            "output_dir": "runs",
            "device": "cuda",
            "seed": 7,
            "max_sequence_tokens": 512,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "epochs_per_window": 1,
            "learning_rate": 1e-5,
            "weight_decay": 0.0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "max_grad_norm": 1.0,
            "bf16": True,
            "gradient_checkpointing": False,
            "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
        },
    }


def _load(tmp_path: Path, payload: dict[str, object]) -> ExperimentConfig:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return ExperimentConfig.load(path)


def _full_weight_payload() -> dict[str, object]:
    payload = _payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training.update(
        {
            "weight_mode": "full_weight",
            "model_revision": "a" * 40,
            "full_weight": {"device_ids": [1, 2]},
        }
    )
    return payload


def test_weight_mode_defaults_to_lora(tmp_path: Path) -> None:
    config = _load(tmp_path, _payload())

    assert config.training.weight_mode == "lora"
    assert config.training.full_weight.device_ids == []
    assert config.training.full_weight.checkpoint_every_iterations == 20
    assert config.training.full_weight.checkpoint_keep_last == 6


def test_lora_mode_rejects_full_weight_devices(tmp_path: Path) -> None:
    payload = _payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["full_weight"] = {"device_ids": [1]}

    with pytest.raises(ValueError, match="only valid when"):
        _load(tmp_path, payload)


def test_full_weight_config_accepts_same_model_and_disjoint_devices(
    tmp_path: Path,
) -> None:
    config = _load(tmp_path, _full_weight_payload())

    assert config.training.weight_mode == "full_weight"
    assert config.training.full_weight.device_ids == [1, 2]
    assert config.training.full_weight.checkpoint_every_iterations == 20
    assert config.training.full_weight.checkpoint_keep_last == 6
    assert config.inference.model_id == config.training.model_name


def test_full_weight_checkpoint_cadence_is_configurable(tmp_path: Path) -> None:
    payload = _full_weight_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["full_weight"] = {
        "device_ids": [1, 2],
        "checkpoint_every_iterations": 3,
    }

    config = _load(tmp_path, payload)

    assert config.training.full_weight.checkpoint_every_iterations == 3


def test_full_weight_checkpoint_retention_is_configurable(tmp_path: Path) -> None:
    payload = _full_weight_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["full_weight"] = {
        "device_ids": [1, 2],
        "checkpoint_keep_last": 4,
    }

    config = _load(tmp_path, payload)

    assert config.training.full_weight.checkpoint_keep_last == 4


@pytest.mark.parametrize("revision", ["", "main", "abc123"])
def test_full_weight_model_revision_must_be_an_immutable_commit(
    tmp_path: Path,
    revision: str,
) -> None:
    payload = _full_weight_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["model_revision"] = revision

    with pytest.raises(ValueError, match="immutable 40-character commit SHA"):
        _load(tmp_path, payload)


@pytest.mark.parametrize("device_ids", ["12", [1.5], [True], {"gpu": 1}])
def test_full_weight_device_ids_require_an_integer_json_list(
    tmp_path: Path, device_ids: object
) -> None:
    payload = _full_weight_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["full_weight"] = {"device_ids": device_ids}

    with pytest.raises(ValueError, match="JSON list of integers"):
        _load(tmp_path, payload)


@pytest.mark.parametrize(
    ("section", "value", "message"),
    [
        ("training", {"weight_mode": "unknown"}, "training.weight_mode"),
        (
            "training",
            {"full_weight": {"device_ids": [1], "mystery": True}},
            "training.full_weight.mystery",
        ),
        (
            "training",
            {
                "full_weight": {
                    "device_ids": [1],
                    "checkpoint_every_iterations": 0,
                }
            },
            "checkpoint_every_iterations must be positive",
        ),
        (
            "training",
            {
                "full_weight": {
                    "device_ids": [1],
                    "checkpoint_every_iterations": True,
                }
            },
            "checkpoint_every_iterations must be an integer",
        ),
        (
            "training",
            {
                "full_weight": {
                    "device_ids": [1],
                    "checkpoint_keep_last": 0,
                }
            },
            "checkpoint_keep_last must be positive",
        ),
        (
            "training",
            {
                "full_weight": {
                    "device_ids": [1],
                    "checkpoint_keep_last": True,
                }
            },
            "checkpoint_keep_last must be an integer",
        ),
        (
            "training",
            {"full_weight": {"device_ids": []}},
            "device_ids must be non-empty",
        ),
        (
            "training",
            {"full_weight": {"device_ids": [1, 1]}},
            "must not repeat",
        ),
        (
            "training",
            {"full_weight": {"device_ids": [-1]}},
            "non-negative GPU ids",
        ),
        ("training", {"reference_kl_coef": 0.1}, "disabled LoRA adapters"),
        (
            "training",
            {"device_map_gpus": [1]},
            "device_map_gpus and training.worker_device_map_gpus are LoRA-only",
        ),
        (
            "training",
            {"init_adapter_dir": "adapter", "init_trainer_state_path": "state.pt"},
            "init_adapter_dir and training.init_trainer_state_path are LoRA-only",
        ),
        ("training", {"bf16": False}, "requires training.bf16=true"),
        ("training", {"device": "cpu"}, "requires training.device='cuda'"),
        ("training", {"device": "cuda:"}, "requires training.device='cuda'"),
        ("training", {"device": "cuda:-1"}, "requires training.device='cuda'"),
        ("training", {"device": "cuda:gpu"}, "requires training.device='cuda'"),
    ],
)
def test_full_weight_config_rejects_incompatible_options(
    tmp_path: Path,
    section: str,
    value: dict[str, object],
    message: str,
) -> None:
    payload = _full_weight_payload()
    target = payload[section]
    assert isinstance(target, dict)
    target.update(value)

    with pytest.raises(ValueError, match=message):
        _load(tmp_path, payload)


def test_full_weight_devices_must_not_overlap_local_servers(tmp_path: Path) -> None:
    payload = _full_weight_payload()
    training = payload["training"]
    assert isinstance(training, dict)
    training["full_weight"] = {"device_ids": [0, 1]}

    with pytest.raises(ValueError, match="must claim disjoint GPUs"):
        _load(tmp_path, payload)


def test_full_weight_devices_must_not_overlap_implicit_server_gpu(
    tmp_path: Path,
) -> None:
    payload = _full_weight_payload()
    inference = payload["inference"]
    training = payload["training"]
    assert isinstance(inference, dict)
    assert isinstance(training, dict)
    inference["server_gpus"] = []
    training["full_weight"] = {"device_ids": [0]}

    with pytest.raises(ValueError, match="must claim disjoint GPUs"):
        _load(tmp_path, payload)


def test_full_weight_config_accepts_official_eval_on_disjoint_gpu(
    tmp_path: Path,
) -> None:
    payload = _full_weight_payload()
    payload["official_eval"] = {
        "enabled": True,
        "games": ["eval-game"],
        "server_gpus": ["3"],
    }

    config = _load(tmp_path, payload)

    assert config.training.weight_mode == "full_weight"
    assert config.official_eval.enabled is True
    assert config.official_eval.server_gpus == ["3"]


def test_full_weight_preserves_window_guard_and_extra_cycles(tmp_path: Path) -> None:
    payload = _full_weight_payload()
    payload["training"]["extra_train_cycles_until_window"] = True
    config = _load(tmp_path, payload)
    assert config.training.max_window_clip_fraction == 0.25
    assert config.training.extra_train_cycles_until_window
