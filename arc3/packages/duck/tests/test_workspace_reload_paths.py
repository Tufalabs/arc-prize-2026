"""Workspace reload containment and filesystem cost with file tools enabled."""

from __future__ import annotations

import ast
import os
from pathlib import Path
from typing import Any
from unittest.mock import Mock

from duck.kernel_code import _PRELUDE_TEMPLATE


def _probe(workspace: Path) -> tuple[Any, Mock]:
    """Run the production prelude's probe factory with counted path resolution."""
    factory = next(
        node
        for node in ast.parse(_PRELUDE_TEMPLATE).body
        if isinstance(node, ast.FunctionDef) and node.name == "_make_workspace_probes"
    )
    namespace: dict[str, Any] = {"_DUCK_WORKSPACE": str(workspace.resolve())}
    exec(compile(ast.Module(body=[factory], type_ignores=[]), "<prelude>", "exec"), namespace)
    realpath = Mock(wraps=os.path.realpath)
    probe, _ = namespace["_make_workspace_probes"](
        realpath, os.path.abspath, os.path.dirname, os.path.join, os.path.isfile
    )
    return probe, realpath


def test_installed_modules_never_resolve_paths(tmp_path: Path) -> None:
    probe, realpath = _probe(tmp_path / "workspace")
    for _ in range(3):
        for index in range(1000):
            assert not probe(f"/nfs/python/lib/python3.12/library{index}.py")
        assert not probe(str(tmp_path / "workspace_other" / "helper.py"))
        assert not probe(str(tmp_path / "workspace" / "nested" / "helper.py"))
    realpath.assert_not_called()


def test_workspace_paths_and_symlink_targets_are_checked_live(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    local = workspace / "helper.py"
    local.write_text("X = 1\n")
    outside = tmp_path / "outside.py"
    outside.write_text("X = 2\n")
    link = workspace / "alias.py"
    link.symlink_to(local)
    probe, _ = _probe(workspace)
    assert probe(str(local))
    assert probe(str(link))
    link.unlink()
    link.symlink_to(outside)
    assert not probe(str(link))
    link.unlink()
    link.symlink_to(local)
    assert probe(str(link))
    local.unlink()
    assert probe(str(local)), "Deleted workspace modules must still be evicted"


def test_relative_workspace_paths_are_supported(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.chdir(tmp_path)
    probe, _ = _probe(tmp_path)
    assert probe("helper.py")
    assert probe("./helper.py")
    assert not probe("../outside.py")
