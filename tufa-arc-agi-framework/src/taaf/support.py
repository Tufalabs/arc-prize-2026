"""Generic support utilities for TAAF."""

import json
import os
import pickle
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar, cast, overload

T = TypeVar("T")


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


class CompressedList(Sequence[T], Generic[T]):
    """An append-only list whose items are held as zlib-compressed pickles
    and rebuilt on access.

    For long per-step records that are written every step and read rarely:
    a game's states or moves are mostly 64x64 frames of 16 colours, which
    compress about 12x. Compressing at ``append`` snapshots the item, and
    each access returns a fresh object, so mutating one does not change the
    record.

    ``keep_last`` also keeps the newest item as the object that was
    appended, for a record whose readers want the latest item every step:
    reading it then costs nothing and returns that same object, so the
    tail alone is not a snapshot.
    """

    def __init__(self, items: Iterable[T] = (), *, keep_last: bool = False) -> None:
        self._blobs: list[bytes] = []
        self._keep_last = keep_last
        self._last: T | None = None
        if isinstance(items, CompressedList):
            self._blobs = list(items._blobs)
        else:
            for item in items:
                self.append(item)

    def append(self, item: T) -> None:
        # Level 1: ~35 us an ARC state against ~70 at the default, for a
        # quarter less space.
        self._blobs.append(zlib.compress(pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL), 1))
        if self._keep_last:
            self._last = item

    def clear(self) -> None:
        self._blobs.clear()
        self._last = None

    def __len__(self) -> int:
        return len(self._blobs)

    @overload
    def __getitem__(self, index: int) -> T: ...
    @overload
    def __getitem__(self, index: slice) -> list[T]: ...

    def __getitem__(self, index: int | slice) -> T | list[T]:
        if isinstance(index, slice):
            return [self._decode(blob) for blob in self._blobs[index]]
        if self._last is not None and index in (-1, len(self._blobs) - 1):
            return self._last
        return self._decode(self._blobs[index])

    def __eq__(self, other: object) -> bool:
        # Item by item, as a reader sees them: equal items can pickle to
        # different bytes (a dict's insertion order), and a live tail mutated
        # after ``append`` no longer matches its own blob. Identical stored
        # bytes count as the same item, as identity does in a list's
        # comparison, so a record of items whose own ``==`` cannot compare
        # two copies (a ``GameState``'s arrays) still compares with a copy.
        if not isinstance(other, Sequence):
            return NotImplemented
        others = cast("Sequence[object]", other)
        if len(self) != len(others):
            return False
        that = cast("CompressedList[object]", other) if isinstance(other, CompressedList) else None
        for index in range(len(self)):
            stored_alike = (
                that is not None
                and self._blobs[index] == that._blobs[index]
                and not (index == len(self) - 1 and (self._last is not None or that._last is not None))
            )
            if not stored_alike and self[index] != others[index]:
                return False
        return True

    def __getstate__(self) -> dict[str, Any]:
        # The live tail is a convenience for this process; its blob is the record.
        return {**self.__dict__, "_last": None}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} items, {sum(map(len, self._blobs))} bytes)"

    @staticmethod
    def _decode(blob: bytes) -> T:
        item: T = pickle.loads(zlib.decompress(blob))
        return item
