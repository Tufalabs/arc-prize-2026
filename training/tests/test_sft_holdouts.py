"""Prevent CD82 reimplementation leakage through native rows and TAAF source selection."""

import hashlib
import json
from pathlib import Path

import pytest

from sft.selection import DatasetSpecLike, resolve_sources
from training.flash_sft import RowIndex


def _rows(path: Path, game_ids: list[str]) -> tuple[Path, str]:
    """Write native rows with real game IDs and a matching input checksum."""
    path.write_text("".join(json.dumps({"game_id": game, "input_token_ids": [1],
                                    "output_token_ids": [2]}) + "\n" for game in game_ids))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("game_id", ["crane_dipper-0001", " CRANE_DIPPER ", "crane_dipper-0002", "cd82-deadbeef"])
def test_native_training_refuses_alias_and_versions_but_validation_accepts(tmp_path: Path, game_id: str) -> None:
    path, digest = _rows(tmp_path / "rows.jsonl", ["s5i5-18d95033", game_id])
    with pytest.raises(ValueError, match="holdout cd82-fb555c5d"):
        RowIndex(path, digest, max_length=10, for_training=True)
    assert len(RowIndex(path, digest, max_length=10)) == 2


def test_non_holdout_reimplementations_and_unrelated_names_stay_eligible(tmp_path: Path) -> None:
    games = ["sliders-0001", "s5i5-18d95033", "easy_taps-0001", "safe_tile_taps-0001", "crane_dipper_extra-0001", "0934a4d8"]
    path, digest = _rows(tmp_path / "rows.jsonl", games)
    assert len(RowIndex(path, digest, max_length=10, for_training=True)) == len(games)


def test_source_selection_catches_derivative_even_when_original_is_excluded(tmp_path: Path) -> None:
    records = [{"run_id": str(i), "game_id": game} for i, game in enumerate(
        ["cd82-fb555c5d", "crane_dipper-0001", "sliders-0001"])]
    (tmp_path / "benchmark.json").write_text(json.dumps({"game_runs": records}))
    for record in records:
        trace = tmp_path / "traces" / record["run_id"]
        trace.mkdir(parents=True)
        (trace / "steps.jsonl").write_text('{"step_index":0}\n')
        (trace / "_SUCCESS").touch()
    spec = DatasetSpecLike(path=tmp_path, exclude_games=["cd82-fb555c5d"])
    with pytest.raises(ValueError, match="crane_dipper-0001.*holdout cd82-fb555c5d"):
        resolve_sources([spec], for_training=True)
    validation, _ = resolve_sources([spec])
    assert {trace.game_id for trace in validation} == {"crane_dipper-0001", "sliders-0001"}
    spec.exclude_games.append("crane_dipper-0001")
    training, _ = resolve_sources([spec], for_training=True)
    assert [trace.game_id for trace in training] == ["sliders-0001"]
