"""Model-mechanics validation shared by engine owners and remote consumers."""

from __future__ import annotations

import types
from typing import Any

import ullm.tracing


def validate_output_text(
    tokenizer: Any,
    prompt_ids: list[int],
    output_ids: list[int],
    text: str,
    finish_reason: str,
    sampling_params: Any,
) -> None:
    """Reproduce engine text, including prompt-dependent decoding and stop handling.

    Call before forwarding or capturing a completion: a caller may stop waiting
    while the engine finishes, so consumer-side validation alone cannot cover
    every generated reply.
    """
    import vllm.v1.engine.detokenizer

    if finish_reason not in ("stop", "length"):
        raise ullm.tracing.TraceReconstructionError("engine response has no completed generation")
    request: Any = types.SimpleNamespace(
        request_id="output-text-validation",
        prompt_token_ids=prompt_ids,
        prompt_embeds=None,
        sampling_params=sampling_params,
    )
    decoder = vllm.v1.engine.detokenizer.IncrementalDetokenizer.from_new_request(tokenizer, request)
    decoder.update(output_ids, stop_terminated=finish_reason == "stop")
    expected = decoder.get_next_output_text(finished=True, delta=False)
    if not isinstance(text, str) or text != expected:
        raise ullm.tracing.TraceReconstructionError("engine text does not match sampled token IDs")
