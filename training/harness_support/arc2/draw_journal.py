"""Completed native draws for failure diagnostics, never finalized training traces."""

import json
from pathlib import Path
from typing import Any


def append_native_draw(job_dir: str | Path, record: Any) -> None:
    """Close each append before returning so worker termination retains the draw."""
    run_id = record.run_id
    if not isinstance(run_id, str) or not run_id.isascii() or not run_id.isalnum():
        raise ValueError("native draw journal requires an alphanumeric run ID")
    payload = {
        "version": 1,
        "run_id": run_id,
        "input_token_ids": record.input_token_ids,
        "output_token_ids": record.output_token_ids,
        "output_text": record.output_text,
        "logprobs": record.logprobs,
        "sampling_params": record.sampling_params,
        "request_payload": record.request_payload,
        "extra": record.extra,
    }
    line = json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n"
    root = Path(job_dir) / "native_draws"
    root.mkdir(parents=True, exist_ok=True)
    with (root / f"{run_id}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(line)
