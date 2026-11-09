"""Checkpoint CUDA RNG topology changes preserve available generator streams."""

import logging

import pytest
import torch

from test_ppo import _make_runtime_export_test_trainer


def _checkpoint(monkeypatch, tmp_path, saved_count):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    trainer = _make_runtime_export_test_trainer(monkeypatch, learning_rate=0.05)
    trainer._optimizer_step_count = 7
    payload = trainer._training_state_payload(
        iteration_index=3,
        adapter_dir=tmp_path / "adapter",
        runtime_adapter_dir=tmp_path / "adapter" / "runtime_vllm",
    )
    # Use real CPU generator state machines as stand-ins for CUDA generators.
    # Native set_state still validates the entire binary format; no test
    # replaces that validation with a success-returning mock.
    payload["torch_cuda_rng_state_all"] = [
        torch.Generator().manual_seed(30 + index).get_state()
        for index in range(saved_count)
    ]
    path = tmp_path / "state.pt"
    torch.save(payload, path)
    return trainer, payload, path


def _mock_visible_cuda(monkeypatch, count):
    real_generator = torch.Generator
    generators = [real_generator().manual_seed(100 + index) for index in range(count)]
    installed = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch, "Generator", lambda **kwargs: real_generator())

    def install(state, device):
        installed.append(device)
        generators[device].set_state(state)

    monkeypatch.setattr(torch.cuda, "set_rng_state", install)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda states: pytest.fail("unbounded all-device restore must not run"),
    )
    return generators, installed


@pytest.mark.parametrize("saved_count,visible_count", [(4, 2), (2, 4), (2, 2), (0, 2)])
def test_resume_restores_matching_ordinals_and_preserves_new_device_streams(
    monkeypatch, tmp_path, caplog, saved_count, visible_count
):
    trainer, payload, path = _checkpoint(monkeypatch, tmp_path, saved_count)
    generators, installed = _mock_visible_cuda(monkeypatch, visible_count)
    initial_states = [generator.get_state() for generator in generators]
    with caplog.at_level(logging.WARNING):
        trainer.load_training_state(path)
    assert trainer.optimizer_step_count == 7
    assert torch.equal(torch.get_rng_state(), payload["torch_rng_state"])
    assert installed == list(range(min(saved_count, visible_count)))
    for index, generator in enumerate(generators):
        expected = (
            payload["torch_cuda_rng_state_all"][index]
            if index < saved_count
            else initial_states[index]
        )
        assert torch.equal(generator.get_state(), expected)
    if saved_count != visible_count:
        assert f"saved={saved_count} visible={visible_count}" in caplog.text
        assert "matching visible-device ordinals" in caplog.text
    else:
        assert "topology changed" not in caplog.text


@pytest.mark.parametrize(
    "corruption", ["container", "dtype", "shape", "empty", "binary_format"]
)
def test_corrupt_saved_rng_is_fatal_even_for_a_now_absent_device(
    monkeypatch, tmp_path, corruption
):
    trainer, payload, path = _checkpoint(monkeypatch, tmp_path, 4)
    if corruption == "container":
        payload["torch_cuda_rng_state_all"] = "invalid"
    else:
        payload["torch_cuda_rng_state_all"][3] = {
            "dtype": torch.zeros(16),
            "shape": torch.zeros((4, 4), dtype=torch.uint8),
            "empty": torch.zeros(0, dtype=torch.uint8),
            "binary_format": torch.zeros(7, dtype=torch.uint8),
        }[corruption]
    torch.save(payload, path)
    generators, installed = _mock_visible_cuda(monkeypatch, 2)
    before = [generator.get_state() for generator in generators]
    with pytest.raises(RuntimeError, match="Invalid CUDA RNG state"):
        trainer.load_training_state(path)
    assert not installed
    assert all(
        torch.equal(generator.get_state(), old)
        for generator, old in zip(generators, before)
    )


def test_device_rng_restore_failure_is_not_silently_swallowed(monkeypatch, tmp_path):
    trainer, _payload, path = _checkpoint(monkeypatch, tmp_path, 2)
    _mock_visible_cuda(monkeypatch, 2)

    def fail_restore(*args, **kwargs):
        raise RuntimeError("device rejected its RNG state")

    monkeypatch.setattr(torch.cuda, "set_rng_state", fail_restore)
    with pytest.raises(RuntimeError, match="device rejected"):
        trainer.load_training_state(path)


def test_cpu_resume_reports_unrestored_cuda_states(monkeypatch, tmp_path, caplog):
    trainer, _payload, path = _checkpoint(monkeypatch, tmp_path, 4)
    with caplog.at_level(logging.WARNING):
        trainer.load_training_state(path)
    assert "not restoring 4 CUDA generator states" in caplog.text
    assert trainer.optimizer_step_count == 7
