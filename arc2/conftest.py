"""Fixtures for every suite in the repo."""

import pytest


@pytest.fixture(autouse=True)
def diagnostics_in_the_job_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run writes its diagnostics to node-local scratch when TAAF's Slurm
    preamble has set ``TAAF_LOCAL_JOB_ROOT`` (``RunConfig.diagnostics_scratch``),
    and the suites read them in the job dir. A suite run inside such a job
    would inherit the variable; a test that wants scratch names it on its
    ``RunConfig``."""
    monkeypatch.delenv("TAAF_LOCAL_JOB_ROOT", raising=False)
