"""Completed draw evidence survives abrupt worker loss without finalizing a trace."""

import base64
import hashlib
import io
import json
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from harness_support.arc2.draw_journal import append_native_draw


def test_completed_draws_survive_sigterm_with_exact_native_evidence(tmp_path):
    image_bytes = io.BytesIO()
    Image.new("RGB", (1, 1), (3, 7, 11)).save(image_bytes, format="PNG")
    png = image_bytes.getvalue()
    record = {
        "run_id": "abc123",
        "input_token_ids": [151644, 3, 151655, 151645],
        "output_token_ids": [7, 19, 151645],
        "output_text": "sampled text",
        "logprobs": [{"token_id": token, "logprob": -0.2} for token in (7, 19, 151645)],
        "sampling_params": {"temperature": 1.0, "top_p": 1.0, "seed": 71},
        "request_payload": {
            "request_id": "abc123#first",
            "media": {
                "images": [
                    {
                        "png_base64": base64.b64encode(png).decode(),
                        "sha256": hashlib.sha256(png).hexdigest(),
                    }
                ],
                "image_hashes": [hashlib.sha256(png).hexdigest()],
                "processing": {"model_revision": "pinned", "transformers": "5.14.1"},
            },
        },
        "extra": {
            "adapter": {"name": "policy", "id": 4},
            "media_evidence": {"pixel_values": "tensor-hash"},
        },
    }
    program = """
import json, os, signal, sys
from types import SimpleNamespace
from harness_support.arc2.draw_journal import append_native_draw
record = SimpleNamespace(**json.load(sys.stdin))
append_native_draw(sys.argv[1], record)
record.request_payload['request_id'] = 'abc123#second'
record.output_token_ids = [5, 151645]
record.logprobs = [{'token_id': token, 'logprob': -0.1} for token in record.output_token_ids]
append_native_draw(sys.argv[1], record)
os.kill(os.getpid(), signal.SIGTERM)
"""
    child = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path)],
        input=json.dumps(record),
        text=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert child.returncode == -signal.SIGTERM, child.stderr
    journal = tmp_path / "native_draws" / "abc123.jsonl"
    first, second = [json.loads(line) for line in journal.read_text().splitlines()]
    assert first == {"version": 1, **record}
    assert second["request_payload"]["request_id"] == "abc123#second"
    assert second["output_token_ids"] == [5, 151645]
    assert second["extra"] == first["extra"]
    assert list(tmp_path.rglob("*")) == [tmp_path / "native_draws", journal]


@pytest.mark.parametrize("run_id", ["", "../escape", "nested/name", "nonasciié", None])
def test_journal_rejects_run_ids_that_are_not_safe_file_names(tmp_path, run_id):
    with pytest.raises(ValueError, match="run ID"):
        append_native_draw(tmp_path, SimpleNamespace(run_id=run_id))
    assert not list(tmp_path.iterdir())
