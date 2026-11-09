"""Preserve a successful segment yield across torchrun's process boundary."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from training.shutdown import SEGMENT_YIELD_EXIT_CODE


SEGMENT_RESULT_ENV = "ARC3_SEGMENT_RESULT_PATH"


def run(arguments: list[str]) -> int:
    # A fresh, private result file cannot be confused with a previous segment.
    # Ranks exit successfully after a yield; torchrun otherwise converts their
    # nonzero exit codes to 1. Never translate a real torchrun failure.
    with tempfile.TemporaryDirectory(prefix="arc3-segment-") as temporary:
        result_path = Path(temporary) / "result"
        process = subprocess.Popen(
            [sys.executable, "-m", "torch.distributed.run", *arguments],
            env={**os.environ, SEGMENT_RESULT_ENV: str(result_path)},
        )
        previous_handlers = {}

        def forward_signal(signum, _frame):
            process.send_signal(signum)

        try:
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.signal(signum, forward_signal)
            returncode = process.wait()
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
        if returncode == 0 and result_path.exists():
            if result_path.read_text() != str(SEGMENT_YIELD_EXIT_CODE):
                raise RuntimeError("Invalid full-weight segment result")
            return SEGMENT_YIELD_EXIT_CODE
        return returncode if returncode >= 0 else 128 - returncode


def main() -> None:
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
