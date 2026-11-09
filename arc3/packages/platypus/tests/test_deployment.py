"""Shared deployment-target resolution used by tools and notebooks (I2)."""

from __future__ import annotations

from pathlib import Path

import pytest

import platypus.deployment


def test_main_project_uses_taaf_workspace_root_resolution(tmp_path: Path) -> None:
    """A nested caller resolves to the checkout directory TAAF will bundle."""
    root = tmp_path / "checkout"
    nested = root / "notebooks" / "scratch"
    nested.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0'\n", encoding="utf-8")

    assert platypus.deployment.detect_main_project(nested) == "checkout"


def test_kaggle_owner_uses_taaf_dotenv_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A credential stored only in a parent .env resolves exactly as deploy does."""
    nested = tmp_path / "repo" / "tools"
    nested.mkdir(parents=True)
    (tmp_path / "repo" / ".env").write_text(
        "KAGGLE_USERNAME=dotenv-owner\nKAGGLE_KEY=placeholder-key\n", encoding="utf-8"
    )
    for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)

    assert platypus.deployment.kaggle_owner(nested) == "dotenv-owner"
