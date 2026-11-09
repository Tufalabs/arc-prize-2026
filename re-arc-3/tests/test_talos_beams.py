"""Pytest entry point for the talos_beams game's in-folder test suites.

The game lives under re_arc/environment_files/, outside pytest's testpaths
(`tests/`), so its test_resolver.py / test_doors.py — handy as a standalone
`uv run python <path>` dev harness — aren't collected by CI on their own. This
module path-loads them and re-exposes their `test_*` functions so `make prepare`
runs them too. Broad integration coverage (reset / render / random-action smoke)
already comes from the all-games tests, so nothing new is added here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_GAME_DIR = Path(__file__).resolve().parents[1] / "re_arc" / "environment_files" / "talos_beams" / "0001"


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _GAME_DIR / filename)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _adopt_tests(module) -> None:
    for name in dir(module):
        if name.startswith("test_"):
            assert name not in globals(), f"duplicate test name {name!r} across talos_beams suites"
            globals()[name] = getattr(module, name)


_adopt_tests(_load("talos_beams_resolver_tests", "test_resolver.py"))
_adopt_tests(_load("talos_beams_door_tests", "test_doors.py"))
