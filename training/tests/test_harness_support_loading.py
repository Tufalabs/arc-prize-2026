"""Harness selection must work without loading colliding harness packages."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("environment,other", [("arc2", "arc3"), ("arc3", "arc2")])
def test_adapter_selection_does_not_load_either_harness(environment, other):
    root = Path(__file__).resolve().parents[1]
    program = """
import sys
from training.harness import get_harness
adapter = get_harness(sys.argv[1])
assert callable(adapter.build_benchmark)
assert callable(adapter.enrich_results)
assert callable(adapter.progress_snapshot)
assert not any(name in sys.modules for name in ('duck', 'platypus', 'taaf'))
assert not any(name.startswith('harness_support.' + sys.argv[2]) for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", program, environment, other],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
