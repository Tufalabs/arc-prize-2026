from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


SCRIPT = Path("scripts/stop_tensorboard.sh")


def _fake_tensorboard(port: int) -> subprocess.Popen:
    return subprocess.Popen(
        [
            "bash",
            "-c",
            f'exec -a tensorboard "{sys.executable}" -c '
            f"'import time; time.sleep(60)' --port {port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def test_stop_tensorboard_only_stops_requested_port() -> None:
    target = _fake_tensorboard(54321)
    unrelated = _fake_tensorboard(54322)
    try:
        time.sleep(0.1)
        result = subprocess.run(
            [str(SCRIPT), "--port", "54321"],
            check=True,
            capture_output=True,
            text=True,
        )

        target.wait(timeout=3)
        assert unrelated.poll() is None
        assert "port 54321" in result.stdout
    finally:
        for process in (target, unrelated):
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=3)


def test_stop_tensorboard_requires_valid_port() -> None:
    result = subprocess.run(
        [str(SCRIPT), "--port", "not-a-port"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid TensorBoard port" in result.stderr
