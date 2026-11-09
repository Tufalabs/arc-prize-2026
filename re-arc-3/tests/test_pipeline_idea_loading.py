from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.idea_loading import load_idea_specs, load_official_description_pair_specs


def test_load_idea_specs_accepts_title_description_shape(tmp_path: Path):
    path = tmp_path / "ideas.json"
    path.write_text(
        json.dumps(
            [
                {"title": "Maze Runner", "description": "Reach the exit.", "num_levels": 4},
                {"title": "Maze Runner", "description": "Variant layout.", "num_levels": 6},
            ]
        ),
        encoding="utf-8",
    )

    ideas = load_idea_specs(path)

    assert len(ideas) == 2
    assert ideas[0].idea_id == "maze-runner"
    assert ideas[1].idea_id == "maze-runner-02"
    assert ideas[0].description == "Reach the exit."
    assert ideas[0].num_levels == 4
    assert ideas[1].num_levels == 6


def test_load_idea_specs_requires_description_and_num_levels(tmp_path: Path):
    path = tmp_path / "ideas.json"
    path.write_text(json.dumps([{"title": "No Description"}]), encoding="utf-8")

    with pytest.raises(ValueError, match="description"):
        load_idea_specs(path)

    path.write_text(json.dumps([{"title": "No Levels", "description": "Reach the exit."}]), encoding="utf-8")

    with pytest.raises(ValueError, match="num_levels"):
        load_idea_specs(path)


def test_load_official_description_pair_specs_expands_all_combinations(tmp_path: Path):
    paths = [tmp_path / "alpha_game.md", tmp_path / "beta_game.md", tmp_path / "gamma_game.md"]
    for path in paths:
        path.write_text(f"# {path.stem}\n\nOfficial description text.\n", encoding="utf-8")

    ideas = load_official_description_pair_specs(paths, root=tmp_path, num_levels=4)

    assert [idea.idea_id for idea in ideas] == ["alpha-game-beta-game", "alpha-game-gamma-game", "beta-game-gamma-game"]
    assert [idea.num_levels for idea in ideas] == [4, 4, 4]
    assert all(len(idea.source_official_descriptions) == 2 for idea in ideas)
    assert ideas[0].source_official_descriptions[0]["path"] == "alpha_game.md"
