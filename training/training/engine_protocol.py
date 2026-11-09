"""Trainer admission over the shared engine protocol's opaque request context."""

from __future__ import annotations

import math
from typing import Any


class EngineProtocolError(ValueError):
    """The remote engine envelope cannot represent a valid training draw."""


def normalize_logit_bias(value: Any) -> dict[int, float]:
    """Normalize JSON token keys without coercing invalid IDs or bias values."""
    if not isinstance(value, dict):
        raise EngineProtocolError("logit_bias must be a token-to-bias object")
    result = {}
    for key, bias in value.items():
        if type(key) is int:
            token = key
        elif isinstance(key, str) and key.isascii() and key.isdecimal():
            token = int(key)
        else:
            raise EngineProtocolError(
                "logit_bias keys must be nonnegative integer token IDs"
            )
        if token < 0 or token in result:
            raise EngineProtocolError(
                "logit_bias token IDs must be nonnegative and unique"
            )
        if (
            type(bias) not in (int, float)
            or not -100 <= bias <= 100
            or not math.isfinite(bias)
        ):
            raise EngineProtocolError(
                "logit_bias values must be finite numbers in [-100, 100]"
            )
        result[token] = float(bias)
    return result


def _is_prefix_cache_tag(value: Any, request_id: Any) -> bool:
    """Exactly the harness's two keys, the context under this request's run id."""
    run_id = request_id.rpartition("#")[0] if isinstance(request_id, str) else ""
    if not run_id or not isinstance(value, dict) or set(value) != {
        "prefix_cache_context",
        "prefix_cache_epoch",
    }:
        return False
    context, epoch = value["prefix_cache_context"], value["prefix_cache_epoch"]
    return (
        isinstance(context, str)
        and context.startswith(run_id + "/")
        and len(context) > len(run_id) + 1
        and type(epoch) is int
        and epoch >= 0
    )


def unpack_generation(body: Any, *, model: str, adapter: dict | None) -> dict:
    """Validate the generic envelope before exposing task fields to admission."""
    if not isinstance(body, dict) or set(body) != {
        "model",
        "request_id",
        "prompt",
        "sampling_params",
        "lora_request",
        "request_context",
    }:
        raise EngineProtocolError(
            "generation requires the exact shared engine envelope"
        )
    if body["model"] != model:
        raise EngineProtocolError(
            "engine request model differs from the trainer policy"
        )
    prompt = body["prompt"]
    if (
        not isinstance(prompt, dict)
        or set(prompt) - {"type", "prompt_token_ids", "media"}
        or not {"type", "prompt_token_ids"} <= set(prompt)
        or prompt["type"] != "token"
    ):
        raise EngineProtocolError("training requires an integer-token engine prompt")
    if "media" in prompt:
        from ullm.media import validate_media

        try:
            validate_media(prompt["media"], prompt["prompt_token_ids"])
        except (ValueError, TypeError, KeyError) as error:
            raise EngineProtocolError(str(error)) from error
    params = body["sampling_params"]
    context = body["request_context"]
    if not isinstance(params, dict) or not isinstance(context, dict):
        raise EngineProtocolError("sampling_params and request_context must be objects")
    params = dict(params)
    extra_args = params.pop("extra_args", None)
    # The harness's cache hints target its fork; the trainer owns cache behavior
    # on its qualified serving stack. Never discard arbitrary engine controls:
    # only ullm's prefix-cache tag (vllmx: prefix_cache_context =
    # "<run_id>/<conversation>", prefix_cache_epoch = the history epoch), bound
    # to this request's run id, is dropped; an empty one is nothing to drop.
    if extra_args not in (None, {}) and not _is_prefix_cache_tag(
        extra_args, body["request_id"]
    ):
        raise EngineProtocolError("unapproved sampling extra_args")
    if set(context) - {"arc3_lineage", "arc3_progress"}:
        raise EngineProtocolError("unapproved training request context")
    reserved = {"model", "user", "prompt", "media", "arc3_lineage", "arc3_progress"}
    if set(params) & reserved:
        raise EngineProtocolError(
            "sampling parameters cannot shadow training request metadata"
        )
    selected = body["lora_request"]
    expected = (
        None
        if adapter is None or adapter["id"] == 0
        else {
            "lora_name": adapter["name"],
            "lora_int_id": adapter["id"],
            "lora_path": adapter["path"],
        }
    )
    if selected != expected or (
        selected is not None
        and (
            not isinstance(selected, dict)
            or type(selected.get("lora_int_id")) is not int
        )
    ):
        raise EngineProtocolError(
            "engine request LoRA identity differs from the trainer policy"
        )
    return {
        **params,
        "model": model,
        "user": body["request_id"],
        "prompt": prompt["prompt_token_ids"],
        **({"media": prompt["media"]} if "media" in prompt else {}),
        **context,
    }


def unpack_output(body: Any) -> dict:
    """Extract the sampled-token evidence without changing the served reply."""
    if not isinstance(body, dict) or body.get("finished") is not True:
        raise EngineProtocolError("engine response must be finished")
    outputs = body.get("outputs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], dict)
    ):
        raise EngineProtocolError("training requires exactly one engine output")
    output = outputs[0]
    if type(output.get("index")) is not int or output["index"] != 0:
        raise EngineProtocolError("engine response has a different sequence index")
    tokens, rows = output.get("token_ids"), output.get("logprobs")
    if (
        not isinstance(tokens, list)
        or not isinstance(rows, list)
        or len(tokens) != len(rows)
    ):
        raise EngineProtocolError("engine sampled token/logprob lengths differ")
    if any(type(token) is not int or token < 0 for token in tokens):
        raise EngineProtocolError(
            "engine response token IDs must be exact nonnegative integers"
        )
    sampled = []
    for token, row in zip(tokens, rows, strict=True):
        if (
            type(token) is not int
            or not isinstance(row, dict)
            or not isinstance(row.get(str(token)), dict)
        ):
            raise EngineProtocolError("engine omitted a sampled token logprob")
        sampled.append(row[str(token)].get("logprob"))
    adapter = body.get("adapter")
    if isinstance(adapter, dict):
        expected_lora = (
            None
            if adapter.get("id") == 0
            else {
                "lora_name": adapter.get("name"),
                "lora_int_id": adapter.get("id"),
                "lora_path": adapter.get("path"),
            }
        )
        actual_lora = body.get("lora_request")
        if actual_lora != expected_lora or (
            actual_lora is not None
            and (
                not isinstance(actual_lora, dict)
                or type(actual_lora.get("lora_int_id")) is not int
            )
        ):
            raise EngineProtocolError(
                "engine response LoRA identity conflicts with its adapter evidence"
            )
    return {
        "request_id": body.get("request_id"),
        "model": adapter.get("name") if isinstance(adapter, dict) else None,
        "adapter": adapter,
        "prompt_token_ids": body.get("prompt_token_ids"),
        "output_token_ids": tokens,
        "output_logprobs": sampled,
        "text": output.get("text"),
        "finish_reason": output.get("finish_reason"),
        "stop_reason": output.get("stop_reason"),
        "num_cached_tokens": body.get("num_cached_tokens"),
        **(
            {"media_evidence": body["media_evidence"]}
            if "media_evidence" in body
            else {}
        ),
    }
