from __future__ import annotations

import json
from pathlib import Path

import pytest

from re_arc import EnvSampler, list_game_ids


def test_list_game_ids_can_filter_by_tag() -> None:
    game_ids = list_game_ids(include_tags="official")

    assert len(game_ids) >= 25
    assert "ft09-0d8bbf25" in game_ids
    assert "lp85-305b61c3" in game_ids
    assert "ls20-9607627b" in game_ids
    assert "vc33-5430563c" in game_ids


def test_env_sampler_can_filter_by_include_and_exclude_tags() -> None:
    sampler = EnvSampler(include_tags="click-only", exclude_tags="official", seed=0)

    assert len(sampler.game_ids) > 0
    assert "easy_taps-0001" in sampler.game_ids
    assert "junction_click_maze-0001" in sampler.game_ids
    assert "airtime_glider-0001" not in sampler.game_ids
    assert "ft09-0d8bbf25" not in sampler.game_ids
    assert "ls20-9607627b" not in sampler.game_ids


def test_list_game_ids_reports_tag_filter_context_on_no_match() -> None:
    with pytest.raises(ValueError, match="include_tags=\\['offical'\\]") as excinfo:
        list_game_ids(include_tags="offical")

    message = str(excinfo.value)
    assert "include_tags=['offical']" in message
    assert "include_tag_suggestions=" in message
    assert "official" in message


def test_all_games_have_exactly_one_input_modality_tag() -> None:
    metadata_paths = sorted(
        (Path(__file__).resolve().parents[1] / "re_arc" / "environment_files").glob("*/**/metadata.json")
    )
    modality_tags = {"click-only", "keys-only", "click-keys"}

    for path in metadata_paths:
        rel = path.relative_to(Path(__file__).resolve().parents[1] / "re_arc" / "environment_files")
        if len(rel.parts) != 3:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        tags = {str(tag).strip() for tag in payload.get("tags", []) if str(tag).strip()}
        present = tags & modality_tags
        assert len(present) == 1, (
            f"{path}: expected exactly one modality tag, found {sorted(present)} in {sorted(tags)}"
        )
