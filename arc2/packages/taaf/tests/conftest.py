"""Repo-wide test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_installed_tufa_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """The installed-package half of source bundling reads the
    launcher's own venv, so a deploy test that doesn't stub it snapshots
    whatever Tufa wheels happen to be installed. The vLLM fork's built
    extensions alone are hundreds of MB, copied into ``tmp_path`` once
    per test — tens of GB of ``/tmp`` across a parallel run, retained
    for three runs.

    Empty by default; the tests that cover installed-package bundling
    stub it with their own fixtures. The editable half is left alone —
    ``test_inline_deploy_snapshots_tufalabs_sources`` deploys for real
    and expects to find this repo's own checkout in the bundle.
    """
    from taaf import deploy

    def _none() -> list[tuple[str, list[Path]]]:
        return []

    monkeypatch.setattr(deploy, "_discover_noneditable_tufa_packages", _none)
