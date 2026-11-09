"""Fixtures for every suite in the repo."""

import pytest
from re_arc import official_games


@pytest.fixture(autouse=True)
def diagnostics_in_the_job_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """A run writes its diagnostics to node-local scratch when TAAF's Slurm
    preamble has set ``TAAF_LOCAL_JOB_ROOT`` (``RunConfig.diagnostics_scratch``),
    and the suites read them in the job dir. A suite run inside such a job
    would inherit the variable; a test that wants scratch names it on its
    ``RunConfig``."""
    monkeypatch.delenv("TAAF_LOCAL_JOB_ROOT", raising=False)


#: Tests that load an official game, whose code is fetched and not shipped.
_NEED_AN_OFFICIAL_GAME = ("test_a_line_takes_the_cells_r11l_draws", "test_random_agent_multipass_on_gameapi")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip what loads an official game when its code is not installed."""
    if not official_games.missing_sources():
        return
    skip = pytest.mark.skip(reason="official game code not installed: run python -m re_arc.official_games")
    for item in items:
        if item.name in _NEED_AN_OFFICIAL_GAME:
            item.add_marker(skip)
