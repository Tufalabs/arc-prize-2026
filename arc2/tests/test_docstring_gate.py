"""AGENTS.md's docstring rule, enforced.

"Every function and class gets at least a short docstring" is an aspiration
until something checks it, so this makes it checkable with the exceptions
written down rather than implied.

The structural exception is a property of what the code IS: **tests**
document per module. Their module docstring names what the file covers and
which requirement, which is the useful granularity for a file of scenarios;
per-function prose there mostly restates the test name. Enforced in the
other direction below — every test module must have that docstring.

Two mechanical exceptions: dunder methods (``__repr__`` and friends restate
themselves) and functions nested inside functions (closures are read in the
context of their parent).

Everything else is in. There is no grandfather list, so a failure here is
always a missing docstring and never a number to bump.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent


def _undocumented(path: pathlib.Path) -> list[str]:
    """Names of definitions in ``path`` that owe a docstring, after the
    dunder and nested-function exceptions."""
    tree = ast.parse(path.read_text())
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    owed: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if ast.get_docstring(node) is not None:
            continue
        if node.name.startswith("__") and node.name.endswith("__"):
            continue
        parent = parents.get(node)
        while parent is not None:
            if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                break
            parent = parents.get(parent)
        else:
            owed.append(f"{node.name} (line {node.lineno})")
    return owed


def _gated_sources() -> list[pathlib.Path]:
    """Every source file the rule applies to: package sources plus the root
    ``tools/``."""
    roots = sorted((REPO / "packages").glob("*/src")) + [REPO / "tools"]
    files: list[pathlib.Path] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            files.append(path)
    return files


@pytest.mark.parametrize("path", _gated_sources(), ids=lambda p: str(p.relative_to(REPO)))
def test_definitions_are_documented(path: pathlib.Path) -> None:
    """Every function and class carries a docstring."""
    owed = _undocumented(path)
    listing = "\n".join(f"    {name}" for name in owed)
    assert not owed, (
        f"{path.relative_to(REPO)}: {len(owed)} definitions without a docstring.\n{listing}\n"
        f"See AGENTS.md, Coding guidelines — a short one is enough, and restating the name is not."
    )


@pytest.mark.parametrize(
    "path",
    sorted(p for p in REPO.rglob("test_*.py") if ".venv" not in str(p) and "tests/data" not in str(p)),
    ids=lambda p: str(p.relative_to(REPO)),
)
def test_test_modules_document_themselves(path: pathlib.Path) -> None:
    """Tests are excused per-function only because they document per module —
    so the module docstring is the part that has to be there."""
    assert ast.get_docstring(ast.parse(path.read_text())), (
        f"{path.relative_to(REPO)}: no module docstring. Test files are exempt from the "
        f"per-function rule because this line carries the same information — say what the "
        f"file covers and, where there is one, which requirement."
    )
