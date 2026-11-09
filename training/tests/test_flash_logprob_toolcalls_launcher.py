"""The tool-call study launcher's resumable pieces, run outside Slurm: offload-candidate requests and the identity a
continuation resumes under (review of PR 125)."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

LAUNCHER = Path(__file__).resolve().parents[1] / "scripts/flash_logprob_toolcalls_20261030/train.py"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.setenv("TOOLCALLS_EXPERIMENT", "reap307-kimi-mixed-toolcalls")
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    spec = importlib.util.spec_from_file_location("toolcalls_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    try:
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_blank_candidate_variable_falls_back_to_the_request_file(launcher, tmp_path):
    request = tmp_path / "OFFLOAD_CANDIDATES"
    request.write_text("4,2\n1\n")
    # A continuation exports the variable blank; the request file placed while it was pending is still honored.
    assert launcher.offload_candidates({"TOOLCALLS_OFFLOAD_CANDIDATES": ""}, request, "7") == [4, 2, 1]
    assert not request.exists() and (tmp_path / "OFFLOAD_CANDIDATES.used-job-7").read_text() == "4,2\n1\n"
    # An absent variable behaves the same; a set one wins and leaves the file alone.
    request.write_text("2")
    assert launcher.offload_candidates({}, request, "8") == [2]
    request.write_text("2")
    assert launcher.offload_candidates({"TOOLCALLS_OFFLOAD_CANDIDATES": "4, 1"}, request, "9") == [4, 1]
    assert request.exists()
    assert launcher.offload_candidates({"TOOLCALLS_OFFLOAD_CANDIDATES": " "}, tmp_path / "absent", "10") == []


def checkpoint(production, config, update):
    directory = production / "checkpoints" / f"update-{update:06d}"
    directory.mkdir(parents=True)
    (directory / "state.json").write_text(json.dumps({"update": update, "config_sha256": config_sha(config)}))
    (production / "checkpoints/latest.json").write_text(json.dumps({"path": directory.name, "update": update}))


def config_sha(config):
    import hashlib

    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def test_resume_identity_follows_the_latest_checkpoint_across_an_interrupted_placement_change(launcher, tmp_path):
    public, production = tmp_path / "experiment", tmp_path / "experiment/training"
    frozen = {"offload_every": 1, "max_length": 192000, "optimizer": {"lr": 1e-4}}
    config_path = public / "train-config.json"
    launcher.save(config_path, frozen)
    # A plain continuation: the checkpoint hashes train-config.json itself.
    checkpoint(production, frozen, 500)
    assert launcher.resume_identity(production, public, config_path) == frozen
    # A placement probe remembered its variant, trained to a new checkpoint under it, then was preempted before
    # the launcher could update train-config.json: the next continuation still finds the identity.
    variant = dict(frozen, offload_every=2)
    remembered = launcher.remember_identity(public, variant)
    assert remembered == public / "identities" / (config_sha(variant) + ".json")
    checkpoint(production, variant, 520)
    assert launcher.resume_identity(production, public, config_path) == variant
    assert launcher.placement_free(variant) == launcher.placement_free(frozen)
    # A probe that failed before any checkpoint leaves the earlier identity in force.
    launcher.remember_identity(public, dict(frozen, offload_every=4))
    assert launcher.resume_identity(production, public, config_path) == variant
    # Remembering is idempotent and never rewrites an identity.
    before = remembered.read_text()
    assert launcher.remember_identity(public, variant) == remembered and remembered.read_text() == before
    # An earlier frozen config kept beside train-config.json also qualifies.
    older = dict(frozen, offload_every=3)
    launcher.save(public / "train-config.before-job-1.json", older)
    checkpoint(production, older, 540)
    assert launcher.resume_identity(production, public, config_path) == older
    # Nothing hashes to the checkpoint: refuse rather than let the trainer reject the resume.
    checkpoint(production, dict(frozen, offload_every=5), 560)
    with pytest.raises(RuntimeError, match="no known configuration"):
        launcher.resume_identity(production, public, config_path)
