"""The launcher lets an inference host drain its engine before group cleanup."""

import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def test_shutdown_signals_host_before_engine_children(tmp_path: Path) -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/platypus_debug_smoke.sbatch"
    ).read_text()
    start = source.index("__arc3_stop_servers() {")
    stop = source.index("\n}\n", start) + 2
    child = tmp_path / "engine.py"
    child.write_text(
        "import pathlib, signal, sys, time\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "def interrupted(*_):\n"
        "    (root / 'interrupted').touch()\n"
        "    sys.exit(3)\n"
        "signal.signal(signal.SIGTERM, interrupted)\n"
        "(root / 'ready').touch()\n"
        "while not (root / 'drain').exists(): time.sleep(0.01)\n"
        "time.sleep(0.1)\n"
        "(root / 'completed').touch()\n"
    )
    host = tmp_path / "host.py"
    host.write_text(
        "import pathlib, signal, subprocess, sys\n"
        "root = pathlib.Path(sys.argv[1])\n"
        "signal.signal(signal.SIGTERM, lambda *_: (root / 'drain').touch())\n"
        "sys.exit(subprocess.call([sys.executable, str(root / 'engine.py'), str(root)]))\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(host), str(tmp_path)], start_new_session=True
    )
    reaper = threading.Thread(target=process.wait)
    reaper.start()
    try:
        deadline = time.monotonic() + 5
        while not (tmp_path / "ready").exists():
            assert time.monotonic() < deadline, "engine child did not start"
            time.sleep(0.01)
        result = subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                f"SERVER_PIDS=({process.pid})\n{source[start:stop]}\n__arc3_stop_servers",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        reaper.join(timeout=5)
        assert result.returncode == 0, result.stderr
        assert process.returncode == 0
        assert (tmp_path / "completed").exists()
        assert not (tmp_path / "interrupted").exists()
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        reaper.join(timeout=5)
