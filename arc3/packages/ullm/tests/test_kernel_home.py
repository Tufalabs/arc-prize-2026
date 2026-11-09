"""The scratch HOME kernel subprocesses run with.

It lives under the system temp directory, which is shared, and model-written
code runs with it — so who owns it and who can write it are the whole point.
The caches inside are read back by later kernels, which is what makes write
access to them equivalent to choosing what a model's imports return.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from ullm.tools.python import _kernel_home


def test_the_home_is_per_user_and_private(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Named per uid and created 0700.

    A shared name is first-creator-wins: the first user's umask decides the
    mode, and everyone after that inherits a HOME they cannot write.
    """
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))

    home = Path(_kernel_home())

    assert home.name.endswith(f"-{os.getuid()}")
    assert stat.S_IMODE(home.lstat().st_mode) == 0o700


def test_a_planted_symlink_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A path under a world-writable directory can be created in advance.

    Following a symlink would put every kernel's cache wherever the planter
    chose, so the entry is checked rather than trusted.
    """
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, tmp_path / f"ullm-kernel-home-{os.getuid()}")

    with pytest.raises(RuntimeError, match="not a directory"):
        _kernel_home()


def test_a_loose_mode_is_tightened_rather_than_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Our own directory, left group- or world-readable, is ours to fix.

    Refusing would strand a user whose earlier run created it under a looser
    umask; the caches inside are the only thing the mode protects.
    """
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    home = tmp_path / f"ullm-kernel-home-{os.getuid()}"
    home.mkdir(mode=0o755)

    assert _kernel_home() == str(home)
    assert stat.S_IMODE(home.lstat().st_mode) == 0o700
