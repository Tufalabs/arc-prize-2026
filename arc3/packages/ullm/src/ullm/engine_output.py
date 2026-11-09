"""Every line the engine's processes write, in a file of the run's own.

vLLM's engine core and workers are child processes: they write to the file
descriptors they inherit, not to the host's ``sys.stdout``, so a log that tees
the host's Python streams (TAAF's ``stdout.log`` on Kaggle) never sees them,
and a worker's traceback reaches only the console. ``capture`` points this
process's descriptors 1 and 2 at pipes while the engine starts, so the
children inherit the pipes, and a thread per pipe copies it both to where the
descriptor pointed before and to the log file. The host's own descriptors are
handed back when the block ends; the children keep writing into the pipes
until they exit.
"""

from __future__ import annotations

import collections.abc
import contextlib
import os
import pathlib
import sys
import threading
from typing import BinaryIO


class _Log:
    """The log file the copy threads share, closed when the last one ends."""

    def __init__(self, path: pathlib.Path, writers: int) -> None:
        self._file: BinaryIO = open(path, "ab", buffering=0)  # noqa: SIM115 -- outlives the block
        self._lock = threading.Lock()
        self._writers = writers

    def write(self, chunk: bytes) -> None:
        """Append a chunk; the two copy threads never interleave inside one."""
        with self._lock:
            self._file.write(chunk)

    def writer_done(self) -> None:
        """One copy thread has seen its pipe close; the last one closes the file."""
        with self._lock:
            self._writers -= 1
            if self._writers == 0:
                self._file.close()


def _copy(read_end: int, original: int, log: _Log) -> None:
    """Copy a pipe to its descriptor's old target and to the log until every
    writer has closed it. It never stops reading early: a pipe nobody drains
    blocks every process writing into it once its buffer fills."""
    while chunk := os.read(read_end, 65536):
        with contextlib.suppress(OSError):
            os.write(original, chunk)
        with contextlib.suppress(Exception):
            log.write(chunk)
    os.close(read_end)
    os.close(original)
    log.writer_done()


@contextlib.contextmanager
def capture(path: pathlib.Path) -> collections.abc.Generator[None, None, None]:
    """Inside the block, processes started from here write through to ``path``
    as well as to this process's own stdout and stderr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    log = _Log(path, writers=2)
    saved: dict[int, int] = {}
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.flush()
    for fd in (1, 2):
        read_end, write_end = os.pipe()
        saved[fd] = os.dup(fd)
        os.dup2(write_end, fd)
        os.close(write_end)
        threading.Thread(
            target=_copy, args=(read_end, os.dup(saved[fd]), log), daemon=True, name=f"engine-output-{fd}"
        ).start()
    try:
        yield
    finally:
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        for fd, original in saved.items():
            os.dup2(original, fd)
            os.close(original)
