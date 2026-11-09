"""Run the jc26 game's standalone level tests under pytest.

The per-level puzzle tests live next to the game
(``re_arc/environment_files/jc26/0001/test_level*.py``) and are normally run
directly with ``uv run``. This wires each of them into the ``tests/`` suite so
``make prepare`` exercises them too: every ``test_*`` function in each file is
loaded by path and invoked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_JC26 = Path(__file__).resolve().parents[1] / "re_arc" / "environment_files" / "jc26" / "0001"
_LEVEL_TESTS = sorted(p.name for p in _JC26.glob("test_level*.py"))


def _load(fname: str):
    spec = importlib.util.spec_from_file_location(f"jc26_{fname[:-3]}", _JC26 / fname)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("fname", _LEVEL_TESTS)
def test_jc26_level(fname: str) -> None:
    module = _load(fname)
    cases = [v for k, v in vars(module).items() if k.startswith("test_") and callable(v)]
    assert cases, f"{fname} defines no test_ functions"
    for case in cases:
        case()
