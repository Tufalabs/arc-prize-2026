# type: ignore
"""Tests for the SolverRandom Kaggle launcher."""

import asyncio
import json
import os
import sys
from pathlib import Path

import re_arc

import taaf
import taaf.kaggle_random as kaggle_random


def _fake_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake_repo"
    (repo / "src" / "fake_pkg").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        '[project]\nname = "fake-repo"\nversion = "0.1.0"\n'
        '[tool.setuptools.packages.find]\nwhere = ["src"]\ninclude = ["fake_pkg*"]\n',
        encoding="utf-8",
    )
    (repo / "README.md").write_text("# fake\n", encoding="utf-8")
    (repo / "src" / "fake_pkg" / "__init__.py").write_text("", encoding="utf-8")
    return repo


def test_random_kaggle_benchmark_defaults_to_25_official_games_without_action_cap(tmp_path: Path, monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    benchmark = kaggle_random.make_random_kaggle_benchmark(
        job_dir=tmp_path / "job",
    )

    assert benchmark.job_dir == tmp_path / "job"
    assert len(benchmark.games) == 25
    assert [game.env_name for game in benchmark.games] == official_ids
    assert isinstance(benchmark.solver, taaf.solver_examples.SolverRandom)
    assert benchmark.solver.max_actions_per_game is None
    assert benchmark.solver.click_on_color is True


def test_random_kaggle_benchmark_accepts_action_cap_and_game_limit(tmp_path: Path, monkeypatch):
    official_ids = [f"official_{i:02d}" for i in range(25)]
    monkeypatch.setattr(re_arc, "list_game_ids", lambda **_kwargs: official_ids)

    benchmark = kaggle_random.make_random_kaggle_benchmark(
        job_dir=tmp_path / "job",
        label="random_test",
        game_limit=2,
        seed=11,
        max_actions_per_game=7,
    )

    assert benchmark.label == "random_test"
    assert len(benchmark.games) == 2
    assert isinstance(benchmark.solver, taaf.solver_examples.SolverRandom)
    assert benchmark.solver.max_actions_per_game == 7
    assert benchmark.solver.seed == 11


def test_random_kaggle_benchmark_can_use_explicit_games(tmp_path: Path):
    benchmark = kaggle_random.make_random_kaggle_benchmark(
        job_dir=tmp_path / "job",
        game_ids=["ls20-9607627b", "ls20-9607627b", "easy_taps-0001"],
    )

    assert [game.env_name for game in benchmark.games] == ["ls20-9607627b", "easy_taps-0001"]
    assert benchmark.n_passes == 1


def test_random_kaggle_target_defaults_to_cpu_only():
    target = kaggle_random.make_random_kaggle_target(
        run_name="Random Smoke",
        dataset_ref="owner/random-source",
        dry_run=True,
    )

    assert target.kernel_slug == "random-smoke"
    assert target.kernel_title == "random-smoke"
    assert target.dataset_ref == "owner/random-source"
    assert target.cpu_only is True
    assert target.dry_run is True
    assert target.max_runtime_s == kaggle_random.DEFAULT_MAX_RUNTIME_MINUTES * 60


def test_random_kaggle_parse_game_ids_accepts_csv_and_json():
    assert kaggle_random.parse_game_ids("ls20, taps ,ls20") == ["ls20", "taps"]
    assert kaggle_random.parse_game_ids('["ls20", "taps"]') == ["ls20", "taps"]


def test_random_launcher_prepends_current_python_bin_to_path(monkeypatch):
    script_dir = "Scripts" if os.name == "nt" else "bin"
    python_bin = str(Path(sys.prefix) / script_dir)
    monkeypatch.setenv("PATH", "/usr/bin")

    kaggle_random._prepend_current_python_bin_to_path()
    kaggle_random._prepend_current_python_bin_to_path()

    path_entries = os.environ["PATH"].split(os.pathsep)
    assert path_entries[0] == python_bin
    assert path_entries.count(python_bin) == 1


def test_random_kaggle_dry_run_writes_cpu_kernel_bundle(tmp_path: Path, monkeypatch):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    args = kaggle_random.parse_args(
        [
            "--run-name",
            "random-dry",
            "--job-dir",
            str(tmp_path / "job"),
            "--game",
            "ls20-9607627b",
            "--max-actions-per-game",
            "1",
            "--kernel-slug",
            "taaf-random-dry",
            "--dataset-ref",
            "dry-user/taaf-release-source-random-dry",
            "--dry-run",
        ]
    )

    handle = asyncio.run(kaggle_random.deploy_random_kaggle(args))

    assert handle.kernel_id == "dry-user/taaf-random-dry"
    assert handle.dataset_ref == "dry-user/taaf-release-source-random-dry"
    assert handle.uploaded is False
    metadata = json.loads((tmp_path / "job" / "kaggle" / "kernel" / "kernel-metadata.json").read_text())
    assert metadata["enable_gpu"] is False
    assert "machine_shape" not in metadata
    assert metadata["dataset_sources"] == ["dry-user/taaf-release-source-random-dry"]


def test_random_kaggle_main_reports_deployed_job_dir(tmp_path: Path, monkeypatch, capsys):
    fake = _fake_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setenv("KAGGLE_USERNAME", "dry-user")

    actual_job = tmp_path / "actual-job"
    wrong_job = tmp_path / "wrong-job"
    resolved_dirs = iter([actual_job, wrong_job])
    monkeypatch.setattr(kaggle_random, "_resolve_job_dir", lambda _args: next(resolved_dirs))

    kaggle_random.main(
        [
            "--run-name",
            "random-main",
            "--game",
            "ls20-9607627b",
            "--max-actions-per-game",
            "1",
            "--kernel-slug",
            "taaf-random-main",
            "--dataset-ref",
            "dry-user/taaf-release-source-random-main",
            "--dry-run",
        ]
    )

    output = capsys.readouterr().out
    assert f"Run directory: {actual_job.absolute()}" in output
    assert "Games: 1" in output
    assert f"Kaggle dry-run bundles: {actual_job / 'kaggle'}" in output
    assert str(wrong_job) not in output
