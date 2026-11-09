"""Fetch the code of the 25 public ARC-AGI-3 games, which this package does not ship.

The games are ARC Prize's and reach us as competition data, so only their
``metadata.json`` is kept here: listing games and filtering by tag work without
them. Playing one needs its code, which this module downloads from the
competition's data, for a Kaggle account that has accepted the competition's
rules, caches, and installs next to the metadata.

    python -m re_arc.official_games             # download what is missing, install it
    python -m re_arc.official_games --optional  # the same, but failing is only a warning

``RE_ARC_OFFICIAL_GAMES_DIR`` moves the cache from ``~/.cache/re_arc/official_games``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

COMPETITION = "arc-prize-2026-arc-agi-3"
ENVIRONMENTS_DIR = Path(__file__).resolve().parent / "environment_files"


def official_sources() -> list[Path]:
    """Where each official game's code belongs, whether or not it is there yet."""
    sources: list[Path] = []
    for metadata_file in sorted(ENVIRONMENTS_DIR.glob("*/*/metadata.json")):
        if "official" in json.loads(metadata_file.read_text()).get("tags", []):
            version_dir = metadata_file.parent
            sources.append(version_dir / f"{version_dir.parent.name}.py")
    return sources


def missing_sources() -> list[Path]:
    """The official games whose code is not installed."""
    return [source for source in official_sources() if not source.is_file()]


def cache_dir() -> Path:
    """Where downloads are kept, so later installs and other jobs need no network."""
    override = os.environ.get("RE_ARC_OFFICIAL_GAMES_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "re_arc" / "official_games"


def _download(relative: Path, destination: Path) -> None:
    """Download one game file from the competition's data to ``destination``."""
    kaggle = shutil.which("kaggle", path=f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}")
    if kaggle is None:
        raise RuntimeError("the Kaggle CLI is not installed (pip install kaggle)")
    with tempfile.TemporaryDirectory() as scratch:
        result = subprocess.run(
            [
                kaggle,
                "competitions",
                "download",
                COMPETITION,
                "-f",
                f"environment_files/{relative.as_posix()}",
                "-p",
                scratch,
                "-q",
            ],
            capture_output=True,
            text=True,
        )
        fetched = Path(scratch) / relative.name
        if result.returncode != 0 or not fetched.is_file():
            raise RuntimeError(
                f"could not download {relative} (has this Kaggle account accepted the rules of {COMPETITION}?): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _place(fetched, destination)


def _place(source: Path, destination: Path) -> None:
    """Copy a file so a concurrent reader sees all of it or none of it."""
    partial = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    shutil.copyfile(source, partial)
    partial.replace(destination)


def install() -> int:
    """Install every missing game's code, downloading what the cache lacks; returns how many."""
    missing = missing_sources()
    cache = cache_dir()
    for source in missing:
        relative = source.relative_to(ENVIRONMENTS_DIR)
        cached = cache / relative
        if not cached.is_file():
            _download(relative, cached)
        _place(cached, source)
    return len(missing)


def main(argv: list[str] | None = None) -> int:
    """Command line entry point."""
    parser = argparse.ArgumentParser(description="Install the code of the official ARC-AGI-3 games.")
    parser.add_argument("--optional", action="store_true", help="warn rather than fail when a game cannot be fetched")
    args = parser.parse_args(argv)
    try:
        installed = install()
    except (OSError, RuntimeError) as error:
        if not args.optional:
            raise SystemExit(f"re_arc.official_games: {error}") from error
        print(f"re_arc.official_games: official games not installed: {error}", file=sys.stderr)
        return 0
    print(f"re_arc.official_games: {installed} installed, {len(official_sources())} official games present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
