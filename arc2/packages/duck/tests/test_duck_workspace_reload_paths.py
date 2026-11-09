"""Workspace-module detection containment and filesystem cost in the kernel."""

from __future__ import annotations

import os
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from duck import kernel_workspace


def _module(origin: str) -> types.ModuleType:
    """A module whose ``__file__`` is ``origin``."""
    module = types.ModuleType("m")
    module.__file__ = origin
    return module


def _probe(monkeypatch: Any, workspace: Path) -> tuple[Callable[[str], bool], Mock]:
    """``_is_workspace_module`` pointed at ``workspace``, with counted resolution."""
    monkeypatch.setattr(kernel_workspace, "tmpdir", str(workspace))
    realpath = Mock(wraps=os.path.realpath)
    monkeypatch.setattr(os.path, "realpath", realpath)
    return (lambda origin: kernel_workspace._is_workspace_module(_module(origin))), realpath


def test_installed_modules_never_resolve_paths(tmp_path: Path, monkeypatch: Any) -> None:
    probe, realpath = _probe(monkeypatch, tmp_path / "workspace")
    for index in range(1000):
        assert not probe(f"/nfs/python/lib/python3.12/library{index}.py")
    assert not probe(str(tmp_path / "workspace_other" / "helper.py"))
    assert not probe(str(tmp_path / "workspace" / "nested" / "helper.py"))
    realpath.assert_not_called()


def test_workspace_paths_and_symlink_targets_are_checked_live(tmp_path: Path, monkeypatch: Any) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    local = workspace / "helper.py"
    local.write_text("X = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("X = 2\n")
    link = workspace / "alias.py"
    link.symlink_to(local)
    probe, _ = _probe(monkeypatch, workspace)
    assert probe(str(local))
    assert probe(str(link))
    link.unlink()
    link.symlink_to(outside)
    assert not probe(str(link))
    local.unlink()
    assert probe(str(local)), "Deleted workspace modules must still be evicted"
