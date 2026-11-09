"""``ullm.kaggle_site``: a kernel installs with pip, untouched, when the
record beside its wheelhouse does not match it, and ends when a replay that
matched fails, since the files the replay removed cannot be put back.

The record here matches the interpreter running the test: its start state is
this venv's own installed distributions.
"""

from __future__ import annotations

import json
import pathlib
import sys
from collections.abc import Sequence

import pytest
import ullm.kaggle
import ullm.kaggle_site as kaggle_site

REQUIREMENTS = ("vllm==0.29.0",)


def _matching_record(tmp_path: pathlib.Path) -> pathlib.Path:
    """A wheelhouse with a record this interpreter matches; its wheels directory."""
    wheels = tmp_path / "site" / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "vllm-0.29.0-py3-none-any.whl").write_bytes(b"wheel")
    archive = wheels.parent / "site.tar.gz"
    archive.write_bytes(b"archive")
    digest, lines = kaggle_site.state_digest(kaggle_site.snapshot())
    manifest = {
        "format": kaggle_site.FORMAT,
        "requirements": list(REQUIREMENTS),
        "python": sys.version,
        "wheelhouse_files": [[p.name, p.stat().st_size] for p in sorted(wheels.iterdir())],
        "before": {"sha256": digest, "distributions": lines},
        "removed": [],
        "archive": {"file": archive.name, "bytes": archive.stat().st_size, "members": 1},
        "after_sha256": digest,
    }
    (wheels.parent / kaggle_site.MANIFEST).write_text(json.dumps(manifest))
    return wheels


@pytest.fixture
def kernel(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> tuple[pathlib.Path, list[tuple[str, ...]]]:
    """A kernel whose wheelhouse holds a matching record; its outputs
    directory, and the pip installs it runs."""
    wheels = _matching_record(tmp_path)
    out = tmp_path / "working"
    out.mkdir()
    pip_runs: list[tuple[str, ...]] = []

    def mounted(ref: str) -> pathlib.Path:
        """The wheelhouse, wherever the ref names."""
        return wheels

    def pip(wheels: pathlib.Path, requirements: Sequence[str]) -> None:
        """Record the install instead of running it."""
        pip_runs.append(tuple(requirements))

    monkeypatch.setattr(ullm.kaggle, "resolve_kaggle_wheel_dir", mounted)
    monkeypatch.setattr(kaggle_site, "pip_install", pip)
    monkeypatch.setenv("TAAF_KAGGLE_WORKING_DIR", str(out))
    return out, pip_runs


def test_a_failed_replay_ends_the_install_with_its_cause(
    kernel: tuple[pathlib.Path, list[tuple[str, ...]]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A replay that stops part-way (here the unpack) raises after writing
    the report, and runs no pip over the half-replayed state."""
    out, pip_runs = kernel

    def interrupted(archive: pathlib.Path, root: str | pathlib.Path = "/") -> None:
        """The unpack, stopped part-way."""
        raise OSError("unpack interrupted")

    monkeypatch.setattr(kaggle_site, "_unpack", interrupted)
    with pytest.raises(OSError, match="unpack interrupted"):
        kaggle_site.install("owner/site", REQUIREMENTS)

    assert pip_runs == []
    report = json.loads((out / "site_install.json").read_text())
    assert report["path"] == "replay failed"
    assert "unpack interrupted" in report["reason"]


def test_a_record_that_does_not_match_installs_with_pip(kernel: tuple[pathlib.Path, list[tuple[str, ...]]]) -> None:
    """Asked for other requirements than the record installed, the kernel
    runs pip from the same wheels and replays nothing."""
    out, pip_runs = kernel

    assert kaggle_site.install("owner/site", ("vllm==0.30.0",)) == "pip"

    assert pip_runs == [("vllm==0.30.0",)]
    report = json.loads((out / "site_install.json").read_text())
    assert report["path"] == "pip"
    assert "different requirement list" in report["reason"]
