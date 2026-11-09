from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from re_arc import EnvSampler, list_game_ids
from re_arc.cli import _build_env_sampler, _list_game_ids, _load_runtime_config


def _write_dataset(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_list_game_ids_can_filter_by_named_dataset(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "train.json", {"game_ids": ["maze-0001", {"game_id": "push"}]})

    game_ids = list_game_ids(datasets="train", datasets_dir=datasets_dir)

    assert game_ids == ("maze-0001", "push-0001")


def test_list_game_ids_can_filter_by_dataset_tag(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "route_games.json", {"tags": ["train"], "game_ids": ["maze-0001"]})
    _write_dataset(
        datasets_dir / "tap_games.json", {"name": "tap-games", "tags": ["train"], "game_ids": ["easy_taps-0001"]}
    )

    game_ids = list_game_ids(datasets="train", datasets_dir=datasets_dir)

    assert set(game_ids) == {"easy_taps-0001", "maze-0001"}


def test_env_sampler_can_exclude_dataset_from_dataset_tag_group(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "route_games.json", {"tags": ["train"], "game_ids": ["maze-0001", "push-0001"]})
    _write_dataset(datasets_dir / "heldout.json", {"tags": ["train"], "game_ids": ["push-0001"]})

    sampler = EnvSampler(datasets="train", exclude_datasets="heldout", datasets_dir=datasets_dir, seed=0)

    assert sampler.game_ids == ("maze-0001",)


def test_dataset_filters_combine_with_game_metadata_tags(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "train.json", {"game_ids": ["easy_taps-0001", "maze-0001"]})

    game_ids = list_game_ids(datasets="train", datasets_dir=datasets_dir, include_tags="click-only")

    assert game_ids == ("easy_taps-0001",)


def test_cli_config_can_select_dataset_for_random_sampler(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "train.json", {"game_ids": ["maze-0001", "push-0001"]})
    config = {"DATASET": "train", "DATASETS_DIR": str(datasets_dir)}

    assert _list_game_ids(config) == ["maze-0001", "push-0001"]
    assert _build_env_sampler(config, seed=0).game_ids == ("maze-0001", "push-0001")


def test_cli_dataset_flag_overrides_dataset_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATASET", "train")
    args = SimpleNamespace(config=str(config_path), dataset=["eval"], exclude_dataset=None, datasets_dir=None)

    config = _load_runtime_config(args)

    assert _list_game_ids(config) == list(list_game_ids(datasets="eval"))


def test_cli_datasets_dir_flag_overrides_datasets_dir_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_dir = tmp_path / "env_datasets"
    cli_dir = tmp_path / "cli_datasets"
    _write_dataset(env_dir / "custom.json", {"game_ids": ["maze-0001"]})
    _write_dataset(cli_dir / "custom.json", {"game_ids": ["push-0001"]})
    config_path = tmp_path / "config.env"
    config_path.write_text("", encoding="utf-8")
    monkeypatch.setenv("DATASETS_DIR", str(env_dir))
    args = SimpleNamespace(config=str(config_path), dataset=["custom"], exclude_dataset=None, datasets_dir=str(cli_dir))

    config = _load_runtime_config(args)

    assert _list_game_ids(config) == ["push-0001"]


def test_unknown_dataset_reports_available_dataset_context(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "training.json", {"tags": ["optimisation"], "game_ids": ["maze-0001"]})

    with pytest.raises(ValueError, match="Unknown datasets") as excinfo:
        list_game_ids(datasets="trning", datasets_dir=datasets_dir)

    message = str(excinfo.value)
    assert "available_datasets=" in message
    assert "training" in message
    assert "dataset_suggestions=" in message


def test_dataset_rejects_unknown_game_ids(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_dataset(datasets_dir / "train.json", {"game_ids": ["missing-game-0001"]})

    with pytest.raises(ValueError, match="datasets reference unknown game ids") as excinfo:
        list_game_ids(datasets="train", datasets_dir=datasets_dir)

    assert "missing-game-0001" in str(excinfo.value)
