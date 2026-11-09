"""The ARC-2 worker refuses a server whose image processor would shrink a
Duck render (harness_support/arc2/worker.py, check_served_image_ceiling)."""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from harness_support.arc2 import worker


@pytest.fixture
def metadata_server():
    """A /metadata endpoint answering with the body the test sets."""
    seen = []
    state = {"body": {}}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - http.server's name
            seen.append((self.path, self.headers.get("Authorization")))
            payload = json.dumps(state["body"]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", state, seen
    finally:
        server.shutdown()
        server.server_close()


def test_the_worker_reads_the_served_ceiling_from_metadata(metadata_server, monkeypatch):
    base_url, state, seen = metadata_server
    spec = {"base_url": base_url + "/", "served_model": "policy"}
    monkeypatch.setenv("SERVER_API_KEY", "secret")
    state["body"] = {"transport": "x", "image_max_pixels": 16_777_216}
    assert worker.served_image_max_pixels(spec) == 16_777_216
    assert seen[-1] == ("/metadata?model=policy", "Bearer secret")
    state["body"] = {"image_max_pixels": None}
    assert worker.served_image_max_pixels(spec) is None
    for body in ({}, {"image_max_pixels": 0}, {"image_max_pixels": "16777216"}):
        state["body"] = body
        with pytest.raises(RuntimeError, match="image pixel ceiling"):
            worker.served_image_max_pixels(spec)


def test_the_ducks_largest_render_must_fit_the_served_ceiling():
    """The Duck's own ceiling (images.largest_image_pixels) is accepted, one
    pixel less or a server without images is refused."""
    pytest.importorskip("vllm")
    root = Path(__file__).resolve().parents[1]
    program = """
from duck import images
from duck.agent import DuckAgent
from ullm.backends.engine_proxy import EngineProxy
import harness_support.arc2.worker as worker
from training_backend import qwen_engine_factory

agent = DuckAgent.build(
    engine='qwen3.8-flash-next', environment='slurm-b200',
    backend_factory=qwen_engine_factory(
        EngineProxy(base_url='http://127.0.0.1:1', model='policy'), engine='qwen3.8-flash-next'
    ),
    derive_max_model_len=False, derive_max_num_seqs=False, derive_max_pixels=False,
    reply_cap_tokens=4096,
)
needed = images.largest_image_pixels(scale=agent._image_scale(), gridlines=agent.image_gridlines)
assert agent._images_enabled() and 0 < needed < 16_777_216
for served, refused in ((16_777_216, None), (needed, None), (needed - 1, 'shrinks'), (None, 'disabled')):
    worker.served_image_max_pixels = lambda spec, served=served: served
    try:
        worker.check_served_image_ceiling(agent, {})
    except ValueError as error:
        assert refused is not None and refused in str(error), (served, error)
    else:
        assert refused is None, served
print('largest render', needed)
"""
    source_paths = [
        root / "arc2-platypus/packages" / name / "src"
        for name in ("duck", "platypus", "taaf")
    ] + [root / "platypus/packages/ullm/src", root / "scripts", root]
    result = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, source_paths))},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
