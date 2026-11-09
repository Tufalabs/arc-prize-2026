"""The official games' code is fetched, cached and installed, never shipped."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from re_arc import official_games


@pytest.fixture
def games(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A package tree with one official game, one of our own, and an empty cache."""
    environments = tmp_path / "environment_files"
    for game, tags in (("ls20", ["official"]), ("ours", ["arc-like"])):
        version_dir = environments / game / "0001"
        version_dir.mkdir(parents=True)
        (version_dir / "metadata.json").write_text(json.dumps({"game_id": f"{game}-0001", "tags": tags}))
    monkeypatch.setattr(official_games, "ENVIRONMENTS_DIR", environments)
    monkeypatch.setenv("RE_ARC_OFFICIAL_GAMES_DIR", str(tmp_path / "cache"))
    return environments


def _fake_download(calls: list[Path]):
    def download(relative: Path, destination: Path) -> None:
        calls.append(relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("# game\n")

    return download


def test_only_official_games_are_fetched(games: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(official_games, "_download", _fake_download(calls))

    assert official_games.install() == 1

    assert calls == [Path("ls20/0001/ls20.py")]
    assert (games / "ls20" / "0001" / "ls20.py").read_text() == "# game\n"
    assert not official_games.missing_sources()


def test_the_cache_spares_a_second_download(games: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(official_games, "_download", _fake_download(calls))
    official_games.install()
    (games / "ls20" / "0001" / "ls20.py").unlink()

    assert official_games.install() == 1

    assert len(calls) == 1
    assert not official_games.missing_sources()


@pytest.mark.usefixtures("games")
def test_optional_turns_a_failed_fetch_into_a_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object) -> None:
        raise RuntimeError("no credentials")

    monkeypatch.setattr(official_games, "_download", refuse)

    assert official_games.main(["--optional"]) == 0
    with pytest.raises(SystemExit, match="no credentials"):
        official_games.main([])
