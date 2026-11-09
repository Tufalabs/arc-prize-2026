"""Directory listings the LLM can reach must not depend on the filesystem.

``readdir`` order on ext4 with ``dir_index`` is a filename hash carrying a
per-filesystem seed, so the same tmpdir lists differently on a different
machine. The LLM calls ``os.listdir()`` in the REPL, which puts that order
straight into its context — and the committed references replay on other
machines, so an unsorted listing is unmatchable by construction. The
whale's tests, over the seal's copy of the patch.
"""

from __future__ import annotations

import glob
import os
import pathlib
from typing import Any, Iterator

import pytest

import seal.harness.game


@pytest.fixture
def unsorted_dir(tmp_path: pathlib.Path) -> Iterator[pathlib.Path]:
    """A directory whose readdir order is not name order, with the patch
    applied over it.

    One fixture rather than two on purpose: the raw order has to be sampled
    *before* patching, and two fixtures would leave that ordering to the
    order the test happens to name them in — which silently turns the
    rawness check into a no-op and skips every test here.

    The patch is global, so ``os`` is restored on the way out; leaking it
    would quietly reorder every later test in the session.
    """
    for name in ("solver.py", "mid.py", "alpha.py", "zebra.py", "notes.txt"):
        (tmp_path / name).touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "inner.py").touch()

    raw = os.listdir(tmp_path)
    saved: tuple[Any, Any] = (os.listdir, os.scandir)
    seal.harness.game._sort_directory_listings()
    try:
        if raw == sorted(raw):
            pytest.skip("this filesystem returns readdir in name order; nothing to sort")
        yield tmp_path
    finally:
        os.listdir, os.scandir = saved


def test_listdir_is_sorted(unsorted_dir: pathlib.Path):
    assert os.listdir(unsorted_dir) == sorted(os.listdir(unsorted_dir))


def test_scandir_is_sorted_and_still_a_context_manager(unsorted_dir: pathlib.Path):
    """``os.walk`` and ``glob`` both do ``with scandir(...) as it``, so a
    replacement that returns a bare iterator breaks them."""
    assert [e.name for e in os.scandir(unsorted_dir)] == sorted(os.listdir(unsorted_dir))
    with os.scandir(unsorted_dir) as entries:
        assert [e.name for e in entries] == sorted(os.listdir(unsorted_dir))


def test_the_paths_built_on_scandir_are_sorted_too(unsorted_dir: pathlib.Path):
    """``glob``, ``os.walk`` and ``Path.iterdir`` never call ``listdir`` —
    patching that alone would leave three unsorted routes to the same
    directory."""
    names = [os.path.basename(p) for p in glob.glob(str(unsorted_dir / "*.py"))]
    assert names == sorted(names)

    walked = [files for _, _, files in os.walk(unsorted_dir)]
    assert all(files == sorted(files) for files in walked)

    entries = [p.name for p in pathlib.Path(unsorted_dir).iterdir()]
    assert entries == sorted(entries)


def test_patching_twice_does_not_stack(unsorted_dir: pathlib.Path):
    """The REPL kernel runs ``initial_code`` once per kernel, but a reused
    kernel or a re-import must not wrap the wrapper — each layer would
    re-open and re-drain the directory."""
    inner = os.listdir
    seal.harness.game._sort_directory_listings()
    assert os.listdir is inner
    assert os.listdir(unsorted_dir) == sorted(os.listdir(unsorted_dir))
