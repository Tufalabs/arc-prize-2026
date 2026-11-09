"""Generic support utilities for TAAF."""

import json
import os
import pickle
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileWrite:
    """One rendered file, not yet written.

    Rendering and writing are split so a caller can do them in different
    places: render where the state being rendered is consistent, write where
    blocking on the filesystem costs nothing (``write_files`` does no work but
    the I/O, so it is safe on a worker thread). ``atomic`` writes via a temp
    file and ``os.replace``, as ``atomic_json_dump`` does.
    """

    path: Path
    data: str | bytes
    atomic: bool = False


def write_files(writes: Iterable[FileWrite]) -> None:
    """Write each file, in order, stopping at the first that fails.

    The order is the caller's contract: a file that records what the others
    hold (an index of rendered pages) goes after them, so a failed write
    leaves the previous record, and the next save redoes what it lost.
    """
    for write in writes:
        write.path.parent.mkdir(parents=True, exist_ok=True)
        target = write.path.with_suffix(write.path.suffix + ".tmp") if write.atomic else write.path
        if isinstance(write.data, bytes):
            target.write_bytes(write.data)
        else:
            target.write_text(write.data)
        if write.atomic:
            os.replace(target, write.path)


def atomic_json_dump(obj: Any, path: Path) -> None:
    """Write ``obj`` as JSON to ``path`` via tempfile + os.replace.

    A crash mid-write leaves either the previous file or nothing — never a
    partial one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def atomic_pickle_dump(obj: Any, path: Path) -> None:
    """Write ``obj`` as a pickle to ``path`` via tempfile + os.replace.

    Same crash-safety guarantee as ``atomic_json_dump``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
