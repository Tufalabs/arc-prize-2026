"""ullm.engine_output: a child process started inside ``capture`` writes
through to the log as well as to the original stdout and stderr; one started
after the block does not reach the log."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest
import ullm.engine_output as engine_output


def _wait_for(path: Path, text: str, timeout_s: float = 5.0) -> str:
    deadline = time.monotonic() + timeout_s
    content = ""
    while time.monotonic() < deadline:
        content = path.read_text() if path.exists() else ""
        if text in content:
            return content
        time.sleep(0.02)
    return content


def _child(code: str) -> subprocess.Popen[bytes]:
    return subprocess.Popen([sys.executable, "-c", code])


def test_a_child_started_inside_writes_through_to_the_log_and_the_console(
    tmp_path: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "engine_output.log"
    with engine_output.capture(log):
        child = _child(
            "import sys; print('worker says hi', flush=True); print('worker trace', file=sys.stderr, flush=True)"
        )
    child.wait()
    content = _wait_for(log, "worker trace")
    assert "worker says hi" in content and "worker trace" in content
    time.sleep(0.1)  # the copy threads' writes to the original descriptors
    out, err = capfd.readouterr()
    assert "worker says hi" in out
    assert "worker trace" in err


def test_a_child_started_after_the_block_stays_out_of_the_log(tmp_path: Path) -> None:
    log = tmp_path / "engine_output.log"
    with engine_output.capture(log):
        inside = _child("print('inside', flush=True)")
    inside.wait()
    _wait_for(log, "inside")
    after = _child("print('after', flush=True)")
    after.wait()
    time.sleep(0.2)
    assert "after" not in log.read_text()
