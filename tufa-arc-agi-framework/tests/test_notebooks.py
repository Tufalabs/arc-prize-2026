# type: ignore
"""Smoke test that the demo notebook executes end-to-end.

Marked ``slow``: spawns nbconvert + a Jupyter kernel, and the demo hits
the real arc_agi API for the ``ls20`` env. Skipped by ``make test``;
run by ``make test-all`` (which is what ``make prepare-ci`` uses).

``deploy_workflow.ipynb`` is intentionally NOT executed here — its
Slurm cells call ``sbatch`` for real, so it's a manual reference doc
run from a submit host (``ssh tufaslurm``), not a CI fixture.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def test_demo_workflow_executes(tmp_path: Path):
    nb = Path(__file__).parent.parent / "notebooks" / "demo_workflow.ipynb"
    assert nb.exists(), nb
    # Force the kernel to use the venv's python by prepending the venv's bin
    # to PATH. The default kernel.json uses a bare `python`, which would
    # otherwise resolve to whatever's first on PATH (often the system python,
    # which doesn't have `taaf` installed).
    venv_bin = Path(sys.executable).parent
    env = {**os.environ, "PATH": f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"}
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            str(tmp_path / "taaf-demo-smoke.ipynb"),
            str(nb),
        ],
        check=True,
        env=env,
    )
