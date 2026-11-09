"""Shared pytest configuration."""

from __future__ import annotations

import pytest

from re_arc import official_games

_PLAY_EVERY_GAME = ("test_envsampler_make_augmentation_is_deterministic_under_concurrency",)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip what plays an official game when its code, fetched and not shipped, is absent."""
    missing = {source.parent.parent.name for source in official_games.missing_sources()}
    if not missing:
        return
    skip = pytest.mark.skip(reason="official game code not installed: run python -m re_arc.official_games")
    for item in items:
        if item.name in _PLAY_EVERY_GAME or any(f"[{game}-" in item.nodeid for game in missing):
            item.add_marker(skip)
