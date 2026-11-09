"""The shared engine wire cannot weaken trainer-owned admission evidence."""

from copy import deepcopy

import pytest

from engine_wire import generation, output
from training.engine_protocol import (
    EngineProtocolError,
    unpack_generation,
    unpack_output,
)


ADAPTER = {
    "name": "policy",
    "id": 1,
    "path": "/policy/one",
    "server_instance_id": "engine-one",
}


def _request():
    """A generic envelope with separate sampling and application metadata."""
    return generation(
        {
            "model": "policy",
            "user": "game#draw",
            "prompt": [1],
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 4,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0,
            "logprobs": 0,
            "skip_special_tokens": False,
            "arc3_lineage": {"kind": "start"},
        },
        ADAPTER,
    )


@pytest.mark.parametrize(
    "extra_args",
    [None, {}, {"prefix_cache_context": "game/thread", "prefix_cache_epoch": 2}],
)
def test_prefix_cache_hints_are_removed_before_both_admission_layers(extra_args):
    from training.context_contract import resolve_context_contract
    from training.native_vllm_server import validate_generation
    from training.rollout_gateway import SamplingPolicy

    wire = _request()
    wire["sampling_params"]["extra_args"] = extra_args
    original = deepcopy(wire)
    body = unpack_generation(wire, model="policy", adapter=ADAPTER)

    assert "extra_args" not in body
    assert wire == original
    SamplingPolicy(
        model="policy",
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        context=resolve_context_contract(32768, 4096, 16384, 0),
    ).validate(body)
    body.pop("arc3_lineage")
    assert (
        validate_generation(body, max_model_len=32768, vocab_size=100)["max_tokens"]
        == 4
    )


@pytest.mark.parametrize(
    "extra_args",
    [
        [],
        "metadata",
        {"unknown": 1},
        {"prefix_cache_epoch": 0, "logits_processor": "hidden"},
        # Only the whole tag, bound to this request's run id ("game#draw").
        {"prefix_cache_context": "game/0f3a"},
        {"prefix_cache_context": "other/0f3a", "prefix_cache_epoch": 0},
        {"prefix_cache_context": "game/", "prefix_cache_epoch": 0},
        {"prefix_cache_context": "game/0f3a", "prefix_cache_epoch": True},
        {"prefix_cache_context": "game/0f3a", "prefix_cache_epoch": -1},
    ],
)
def test_sampling_extras_cannot_hide_unapproved_engine_controls(extra_args):
    wire = _request()
    wire["sampling_params"]["extra_args"] = extra_args
    with pytest.raises(EngineProtocolError, match="extra_args"):
        unpack_generation(wire, model="policy", adapter=ADAPTER)


@pytest.mark.parametrize(
    "fault", ["legacy", "alias", "prompt", "context", "shadow", "lora", "boolean_lora"]
)
def test_request_envelope_rejects_changes_before_admission(fault):
    body = _request()
    if fault == "legacy":
        body = {"model": "policy", "user": "game#draw", "prompt": [1]}
    elif fault == "alias":
        body["model"] = "other"
    elif fault == "prompt":
        body["prompt"]["multi_modal_data"] = {}
    elif fault == "context":
        body["request_context"]["retry"] = True
    elif fault == "shadow":
        body["sampling_params"]["arc3_lineage"] = {"kind": "reset"}
    elif fault == "lora":
        body["lora_request"]["lora_path"] = "/other"
    else:
        body["lora_request"]["lora_int_id"] = True
    with pytest.raises(EngineProtocolError):
        unpack_generation(body, model="policy", adapter=ADAPTER)


@pytest.mark.parametrize(
    "fault",
    [
        "unfinished",
        "multiple",
        "index",
        "missing_logprob",
        "boolean_lora",
        "changed_lora",
    ],
)
def test_response_envelope_rejects_conflicting_engine_evidence(fault):
    body = output(
        {
            "request_id": "game#draw",
            "prompt_token_ids": [1],
            "output_token_ids": [2],
            "output_logprobs": [-0.3],
            "text": "answer",
            "finish_reason": "length",
            "stop_reason": None,
            "num_cached_tokens": 0,
            "adapter": ADAPTER,
        }
    )
    if fault == "unfinished":
        body["finished"] = False
    elif fault == "multiple":
        body["outputs"].append(deepcopy(body["outputs"][0]))
    elif fault == "index":
        body["outputs"][0]["index"] = True
    elif fault == "missing_logprob":
        body["outputs"][0]["logprobs"] = [{}]
    elif fault == "boolean_lora":
        body["lora_request"]["lora_int_id"] = True
    else:
        body["lora_request"]["lora_path"] = "/other"
    with pytest.raises(EngineProtocolError):
        unpack_output(body)
