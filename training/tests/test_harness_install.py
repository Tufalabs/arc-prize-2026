"""Fresh and upgraded installs use the committed harness pin, not branch HEAD."""

from pathlib import Path
import shutil
import subprocess

import pytest


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _commit(repo: Path) -> str:
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def harness_checkout(tmp_path: Path) -> tuple[Path, str, str]:
    upstream = tmp_path / "harness"
    upstream.mkdir()
    _git(upstream, "init", "-q")
    (upstream / "pyproject.toml").write_text('version = "old"\n')
    old = _commit(upstream)

    repo = tmp_path / "trainer"
    repo.mkdir()
    _git(repo, "init", "-q")
    for harness in ("platypus", "arc2-platypus"):
        _git(
            repo,
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            str(upstream),
            harness,
        )
    scripts = repo / "scripts"
    scripts.mkdir()
    source = Path(__file__).resolve().parents[1]
    shutil.copyfile(source / "scripts/sync_platypus.sh", scripts / "sync_platypus.sh")
    shutil.copyfile(source / "Makefile", repo / "Makefile")
    _commit(repo)

    (upstream / "pyproject.toml").write_text('version = "pinned"\n')
    pinned = _commit(upstream)
    for harness in ("platypus", "arc2-platypus"):
        _git(repo / harness, "fetch", "origin")
        _git(repo / harness, "checkout", "--detach", pinned)
    _commit(repo)
    # A newer remote commit must not replace the trainer's recorded pin.
    (upstream / "pyproject.toml").write_text('version = "unpinned"\n')
    _commit(upstream)
    return repo, old, pinned
