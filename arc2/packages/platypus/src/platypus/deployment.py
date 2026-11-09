"""Shared local resolution used when constructing TAAF deployment targets."""

from __future__ import annotations

from pathlib import Path

import taaf.deploy
import taaf.deploy_kaggle


def detect_main_project(start: Path | None = None) -> str:
    """Return the checkout directory name bundled for a Slurm worker.

    Thin alias for TAAF's own derivation, which is also what
    ``TufaSlurmTarget.main_project`` defaults to — so a caller that passes
    this explicitly and one that passes nothing cannot disagree."""
    return taaf.deploy.detect_bundled_repo_name(start)


def kaggle_owner(start: Path | None = None) -> str:
    """Resolve the owner through TAAF's exact dotenv and credential rules."""
    taaf.deploy_kaggle._load_dotenv_upwards(start)  # pyright: ignore[reportPrivateUsage]
    owner, _ = taaf.deploy_kaggle._resolve_kaggle_credentials(None)  # pyright: ignore[reportPrivateUsage]
    return owner
