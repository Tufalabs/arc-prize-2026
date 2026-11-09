"""Run the actual observer hooks without importing the GPU worker stack."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from context_observer_sequences import compaction_stamps, expected_stamps


class TokenCount(int):
    def __new__(cls, value, prompt_token_ids, media=None):
        count = super().__new__(cls, value)
        count.prompt_token_ids = prompt_token_ids
        count.media = media
        return count


@pytest.fixture
def observer(monkeypatch):
    backend = ModuleType("ullm.backend")
    backend.TokenCount = TokenCount
    training = ModuleType("training_backend")
    training.TrainingLLMAgentSolver = object
    training.validate_training_backend = lambda agent: None
    monkeypatch.setitem(sys.modules, "ullm.backend", backend)
    monkeypatch.setitem(sys.modules, "training_backend", training)
    spec = importlib.util.spec_from_file_location(
        "_observer_under_test",
        Path(__file__).resolve().parents[1] / "scripts/serving_contract.py",
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module.TrainingContextObserver()


@pytest.mark.parametrize("served", [False, True])
def test_exact_hook_sequence(observer, served):
    assert compaction_stamps(observer, TokenCount, served=served) == expected_stamps(
        served=served
    )
    params = SimpleNamespace(request_context={})
    epoch = 2 if served else 1
    observer.prepare_request(params)
    observer.request_prepared(params, epoch=epoch, prompt_tokens=4, target=8)
    assert params.request_context["arc3_lineage"] == dict(
        version=1, epoch=epoch, kind="continue", compaction_prompt_token_ids=None
    )
    with pytest.raises(RuntimeError, match="only during a verified"):
        observer.history_rewritten(epoch=epoch)


def start(observer):
    params = SimpleNamespace(request_context={})
    observer.request_prepared(params, epoch=0, prompt_tokens=2, target=3)
    return params


@pytest.mark.parametrize("served", [False, True])
@pytest.mark.parametrize(
    "damage", ["same_epoch", "too_large", "wrong_declaration", "repeat"]
)
def test_invalid_reset_transitions(observer, served, damage):
    params = start(observer)
    proof = TokenCount(9, list(range(9)))
    observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
    if served:
        observer.request_prepared(params, epoch=0, prompt_tokens=proof, target=8)
    if damage == "wrong_declaration":
        with pytest.raises(RuntimeError, match="cannot reuse"):
            observer.history_rewritten(epoch=1)
        return
    observer.history_rewritten(epoch=0)
    if damage == "repeat":
        with pytest.raises(RuntimeError, match="cannot reuse"):
            observer.history_rewritten(epoch=0)
    else:
        with pytest.raises(RuntimeError, match="advance|configured context target"):
            observer.request_prepared(
                params,
                epoch=0 if damage == "same_epoch" else 2,
                prompt_tokens=9,
                target=8,
            )


def test_first_epoch_and_undeclared_rewrites(observer):
    params = SimpleNamespace(request_context={})
    with pytest.raises(RuntimeError, match="first training generation"):
        observer.request_prepared(params, epoch=1, prompt_tokens=2, target=3)
    start(observer)
    with pytest.raises(RuntimeError, match="undeclared"):
        observer.request_prepared(params, epoch=2, prompt_tokens=2, target=3)
    with pytest.raises(RuntimeError, match="exact pre-trim"):
        observer.prompt_measured(9, epoch=0, trigger=8, compacting=True)


def test_no_stale_proof_across_a_prepared_request(observer):
    params = start(observer)
    proof = TokenCount(9, list(range(9)))
    observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
    observer.request_prepared(params, epoch=0, prompt_tokens=proof, target=8)
    observer.prepare_request(params)
    observer.prompt_measured(3, epoch=0, trigger=8, compacting=False)
    observer.request_prepared(params, epoch=0, prompt_tokens=3, target=8)
    with pytest.raises(RuntimeError, match="only during a verified"):
        observer.history_rewritten(epoch=0)


def test_trigger_is_refreshed_for_each_preparation(observer):
    params = start(observer)
    observer.prompt_measured(3, epoch=0, trigger=2, compacting=False)
    observer.prepare_request(params)
    observer.prompt_measured(3, epoch=0, trigger=8, compacting=False)
    observer.request_prepared(params, epoch=0, prompt_tokens=3, target=8)
    assert observer.served_proof is None


def test_measured_reset_keeps_media_and_cannot_be_declared_twice(observer):
    params = start(observer)
    media = {"images": ["identity"]}
    proof = TokenCount(9, list(range(9)), media=media)
    observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
    observer.history_rewritten(epoch=0)
    with pytest.raises(RuntimeError, match="cannot reuse"):
        observer.history_rewritten(epoch=1)
    observer.request_prepared(params, epoch=1, prompt_tokens=3, target=3)
    assert params.request_context["arc3_lineage"] == {
        **expected_stamps(served=False)[-1],
        "compaction_media": media,
    }


def test_measured_reset_cannot_skip_an_epoch(observer):
    params = start(observer)
    proof = TokenCount(9, list(range(9)))
    observer.prompt_measured(proof, epoch=0, trigger=8, compacting=True)
    observer.history_rewritten(epoch=0)
    with pytest.raises(RuntimeError, match="exactly one epoch"):
        observer.request_prepared(params, epoch=2, prompt_tokens=3, target=3)
