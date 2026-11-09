"""Shared-engine wire fixtures for trainer gateway tests."""


def generation(body, adapter=None):
    """Render a task admission fixture into the generic engine envelope."""
    body = dict(body)
    model = body.pop("model", "stub-model")
    if adapter is None and model not in {"stub-model", "base"}:
        adapter = {"name": model, "id": 1, "path": "/unexpected-policy"}
    return {
        "model": model,
        "request_id": body.pop("user", None),
        "prompt": {"type": "token", "prompt_token_ids": body.pop("prompt", None)},
        "request_context": {
            key: body.pop(key)
            for key in ("arc3_lineage", "arc3_progress")
            if key in body
        },
        "lora_request": None
        if not adapter or adapter["id"] == 0
        else {
            "lora_name": adapter["name"],
            "lora_int_id": adapter["id"],
            "lora_path": adapter["path"],
        },
        "sampling_params": body,
    }


def output(body):
    """Render captured evidence as the engine protocol's single output."""
    return {
        "request_id": body["request_id"],
        "prompt_token_ids": body["prompt_token_ids"],
        "finished": True,
        "num_cached_tokens": body["num_cached_tokens"],
        "lora_request": None
        if not body["adapter"] or body["adapter"]["id"] == 0
        else {
            "lora_name": body["adapter"]["name"],
            "lora_int_id": body["adapter"]["id"],
            "lora_path": body["adapter"]["path"],
        },
        "adapter": body["adapter"],
        "outputs": [
            {
                "index": 0,
                "text": body["text"],
                "token_ids": body["output_token_ids"],
                "finish_reason": body["finish_reason"],
                "stop_reason": body.get("stop_reason"),
                "cumulative_logprob": 0.0,
                "logprobs": [
                    {str(token): {"logprob": lp, "rank": None, "decoded_token": None}}
                    for token, lp in zip(
                        body["output_token_ids"], body["output_logprobs"]
                    )
                ],
            }
        ],
    }
