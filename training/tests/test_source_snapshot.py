import json
import subprocess
import sys
from pathlib import Path

import pytest

from training.launch_frozen import _exec_training_from_snapshot
from training.source_snapshot import (
    FrozenRunPaths,
    _snapshot_git_state,
    create_source_snapshot,
    default_source_workspace,
    materialize_source_snapshot,
    prepare_frozen_config,
    prepare_frozen_run,
    python_environment_fingerprint,
    verify_runtime_fingerprint,
)


def test_snapshot_git_state_queries_only_the_root_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git_output(_repo_root: Path, args: list[str]) -> str:
        assert "submodule" not in args
        return "value"

    monkeypatch.setattr("training.source_snapshot._git_output", fake_git_output)

    assert _snapshot_git_state(tmp_path) == {
        "root_commit": "value",
        "root_branch": "value",
        "root_status": "value",
        "harness_gitlinks": "value",
    }


def _unpack(archive: Path, workspace: Path) -> Path:
    return materialize_source_snapshot(archive_path=archive, workspace_root=workspace)


def test_create_source_snapshot_archives_current_source_and_excludes_runtime_dirs(
    tmp_path: Path,
) -> None:
    live_repo = tmp_path / "repo"
    live_repo.mkdir()
    (live_repo / "training").mkdir()
    (live_repo / "training" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (live_repo / ".venv").mkdir()
    (live_repo / ".venv" / "pyvenv.cfg").write_text("ignored\n", encoding="utf-8")
    (live_repo / ".venvs" / "root").mkdir(parents=True)
    (live_repo / ".venvs" / "root" / "pyvenv.cfg").write_text(
        "ignored\n", encoding="utf-8"
    )
    (live_repo / ".temp_venv").mkdir()
    (live_repo / ".temp_venv" / "pyvenv.cfg").write_text("ignored\n", encoding="utf-8")
    (live_repo / ".git").mkdir()
    (live_repo / ".git" / "objects").mkdir()
    (live_repo / ".git" / "objects" / "large-pack").write_text(
        "ignored\n",
        encoding="utf-8",
    )
    (live_repo / "runs").mkdir()
    (live_repo / "runs" / "old_run").mkdir()
    (live_repo / "runs_old").mkdir()
    (live_repo / "runs_old" / "old_run").mkdir()
    (live_repo / "platypus" / "packages").mkdir(parents=True)
    (live_repo / "platypus" / "packages" / "big.py").write_text(
        "ignored\n", encoding="utf-8"
    )
    (live_repo / "arc3_training.egg-info").mkdir()
    (live_repo / "arc3_training.egg-info" / "PKG-INFO").write_text(
        "ignored\n",
        encoding="utf-8",
    )
    (live_repo / "training" / "__pycache__").mkdir()
    (live_repo / "training" / "__pycache__" / "example.pyc").write_bytes(b"ignored")

    run_root = tmp_path / "run"

    archive = create_source_snapshot(live_repo_root=live_repo, run_root=run_root)

    assert archive == run_root / "source.tar"
    # One archive plus its metadata: nothing else lands in the run directory.
    assert sorted(path.name for path in run_root.iterdir()) == [
        "source.tar",
        "source_snapshot.json",
    ]
    source_root = _unpack(archive, tmp_path / "workspace")
    assert (source_root / "training" / "example.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
    assert not (source_root / ".venv").exists()
    assert not (source_root / ".venvs").exists()
    assert (source_root / ".temp_venv" / "pyvenv.cfg").read_text(
        encoding="utf-8"
    ) == "ignored\n"
    assert not (source_root / ".git").exists()
    assert not (source_root / "runs").exists()
    assert not (source_root / "runs_old").exists()
    assert not (source_root / "platypus").exists()
    assert not (source_root / "arc3_training.egg-info").exists()
    assert not (source_root / "training" / "__pycache__").exists()
    metadata = json.loads(
        (run_root / "source_snapshot.json").read_text(encoding="utf-8")
    )
    assert metadata["source_archive"] == str(archive)
    assert metadata["file_count"] == 2
    assert (source_root / ".arc3_source_ready").read_text(
        encoding="utf-8"
    ).strip() == metadata["archive_sha256"]


def test_create_source_snapshot_excludes_active_run_dir_inside_repo(
    tmp_path: Path,
) -> None:
    live_repo = tmp_path / "repo"
    live_repo.mkdir()
    (live_repo / "training").mkdir()
    (live_repo / "training" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_root = live_repo / "experiments" / "run1"
    run_root.mkdir(parents=True)
    (run_root / "existing_artifact.txt").write_text("do not copy\n", encoding="utf-8")

    archive = create_source_snapshot(live_repo_root=live_repo, run_root=run_root)

    source_root = _unpack(archive, tmp_path / "workspace")
    assert (source_root / "training" / "example.py").exists()
    assert not (source_root / "experiments").exists()


def test_create_source_snapshot_in_git_worktree_follows_git_visibility(
    tmp_path: Path,
) -> None:
    live_repo = tmp_path / "repo"
    live_repo.mkdir()
    subprocess.run(
        ["git", "init"], cwd=live_repo, check=True, stdout=subprocess.DEVNULL
    )
    (live_repo / ".gitignore").write_text("ignored/\n*.out\n", encoding="utf-8")
    (live_repo / "tracked.py").write_text("old\n", encoding="utf-8")
    (live_repo / "deleted.py").write_text("delete me\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", ".gitignore", "tracked.py", "deleted.py"],
        cwd=live_repo,
        check=True,
    )
    (live_repo / "tracked.py").write_text("new\n", encoding="utf-8")
    (live_repo / "deleted.py").unlink()
    (live_repo / "untracked.py").write_text("include me\n", encoding="utf-8")
    (live_repo / "slurm-1.out").write_text("ignored\n", encoding="utf-8")
    (live_repo / "ignored").mkdir()
    (live_repo / "ignored" / "ignored.py").write_text("ignored\n", encoding="utf-8")
    (live_repo / ".venv").mkdir()
    (live_repo / ".venv" / "pyvenv.cfg").write_text("ignored\n", encoding="utf-8")
    (live_repo / ".venvs").mkdir()
    (live_repo / ".venvs" / "tracked.txt").write_text("ignored\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "-f", ".venvs/tracked.txt"],
        cwd=live_repo,
        check=True,
    )

    run_root = tmp_path / "run"

    archive = create_source_snapshot(live_repo_root=live_repo, run_root=run_root)

    source_root = _unpack(archive, tmp_path / "workspace")
    assert (source_root / "tracked.py").read_text(encoding="utf-8") == "new\n"
    assert (source_root / "untracked.py").read_text(encoding="utf-8") == "include me\n"
    assert not (source_root / "deleted.py").exists()
    assert not (source_root / "slurm-1.out").exists()
    assert not (source_root / "ignored").exists()
    assert not (source_root / ".venv").exists()
    assert not (source_root / ".venvs").exists()
    assert not (source_root / ".git").exists()


def test_create_source_snapshot_reuses_existing_archive_unless_forced(
    tmp_path: Path,
) -> None:
    live_repo = tmp_path / "repo"
    (live_repo / "training").mkdir(parents=True)
    (live_repo / "training" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_root = tmp_path / "run"

    first = create_source_snapshot(live_repo_root=live_repo, run_root=run_root)
    first_sha = json.loads((run_root / "source_snapshot.json").read_text())[
        "archive_sha256"
    ]
    (live_repo / "training" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")

    assert create_source_snapshot(live_repo_root=live_repo, run_root=run_root) == first
    assert (
        json.loads((run_root / "source_snapshot.json").read_text())["archive_sha256"]
        == first_sha
    )
    create_source_snapshot(live_repo_root=live_repo, run_root=run_root, force=True)
    forced = _unpack(first, tmp_path / "workspace")
    assert (forced / "training" / "example.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"


def test_materialize_source_snapshot_unpacks_once_per_content(tmp_path: Path) -> None:
    live_repo = tmp_path / "repo"
    (live_repo / "training").mkdir(parents=True)
    (live_repo / "training" / "example.py").write_text("VALUE = 1\n", encoding="utf-8")
    run_root = tmp_path / "my_run"
    archive = create_source_snapshot(live_repo_root=live_repo, run_root=run_root)
    workspace = tmp_path / "workspace"

    first = materialize_source_snapshot(archive_path=archive, workspace_root=workspace)
    assert first.parent == workspace
    assert first.name.startswith("my_run-")
    (first / "scratch.txt").write_text("left by a previous launch\n", encoding="utf-8")

    again = materialize_source_snapshot(archive_path=archive, workspace_root=workspace)
    assert again == first
    assert (again / "scratch.txt").exists()

    create_source_snapshot(live_repo_root=live_repo, run_root=run_root, force=True)
    (live_repo / "training" / "example.py").write_text("VALUE = 2\n", encoding="utf-8")
    create_source_snapshot(live_repo_root=live_repo, run_root=run_root, force=True)
    changed = materialize_source_snapshot(
        archive_path=archive, workspace_root=workspace
    )
    assert changed != first
    assert (changed / "training" / "example.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert not any(path.name.startswith(".my_run") for path in workspace.iterdir())


def test_default_source_workspace_prefers_the_node_cache_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ARC3_SOURCE_WORKSPACE", raising=False)
    monkeypatch.setenv("ARC3_RL_CACHE_ROOT", str(tmp_path / "cache"))
    assert default_source_workspace() == tmp_path / "cache" / "source"
    monkeypatch.setenv("ARC3_SOURCE_WORKSPACE", str(tmp_path / "explicit"))
    assert default_source_workspace() == tmp_path / "explicit"


def test_prepare_frozen_run_executes_from_a_node_local_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live_repo = tmp_path / "repo"
    (live_repo / "configs").mkdir(parents=True)
    (live_repo / "configs" / "debug.json").write_text(
        json.dumps({"training": {"seed": 1}, "inference": {}}), encoding="utf-8"
    )
    run_root = tmp_path / "run"
    monkeypatch.setattr(
        "training.source_snapshot.write_runtime_fingerprint",
        lambda **kwargs: kwargs["destination"].write_text("{}", encoding="utf-8"),
    )

    paths = prepare_frozen_run(
        config_path="configs/debug.json",
        live_repo_root=live_repo,
        run_root=run_root,
        training_python=Path(sys.executable),
        source_workspace=tmp_path / "workspace",
    )

    assert paths.source_archive_path == run_root / "source.tar"
    assert paths.source_root.parent == tmp_path / "workspace"
    assert (paths.source_root / "configs" / "debug.json").exists()
    assert json.loads(paths.frozen_config_path.read_text())["training"]["seed"] == 1


def test_prepare_frozen_config_keeps_code_relative_but_externalizes_live_paths(
    tmp_path: Path,
) -> None:
    live_repo = tmp_path / "repo"
    source_root = tmp_path / "run" / "source"
    run_root = tmp_path / "run"
    (live_repo / "configs").mkdir(parents=True)
    (source_root / "configs").mkdir(parents=True)
    payload = {
        "training": {
            "output_dir": "runs",
        },
        "inference": {
            "platypus_venv": "platypus/.venv",
        },
    }
    (source_root / "configs" / "ppo_8gpu_official.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    frozen_config = prepare_frozen_config(
        config_path="configs/ppo_8gpu_official.json",
        live_repo_root=live_repo,
        source_root=source_root,
        run_root=run_root,
    )

    rewritten = json.loads(frozen_config.read_text(encoding="utf-8"))
    assert rewritten["inference"]["platypus_venv"] == str(live_repo / "platypus/.venv")


def test_prepare_frozen_config_applies_training_seed_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live_repo = tmp_path / "repo"
    source_root = tmp_path / "run" / "source"
    run_root = tmp_path / "run"
    (source_root / "configs").mkdir(parents=True)
    (source_root / "configs" / "ppo_8gpu_official.json").write_text(
        json.dumps({"training": {"seed": 42}, "inference": {}}),
        encoding="utf-8",
    )

    monkeypatch.setenv("ARC3_TRAINING_SEED", "303")

    frozen_config = prepare_frozen_config(
        config_path="configs/ppo_8gpu_official.json",
        live_repo_root=live_repo,
        source_root=source_root,
        run_root=run_root,
    )

    rewritten = json.loads(frozen_config.read_text(encoding="utf-8"))
    assert rewritten["training"]["seed"] == 303


def test_prepare_frozen_config_reuses_existing_config_on_resume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live_repo = tmp_path / "repo"
    source_root = tmp_path / "run" / "source"
    run_root = tmp_path / "run"
    (source_root / "configs").mkdir(parents=True)
    (source_root / "configs" / "ppo_8gpu_official.json").write_text(
        json.dumps({"training": {"seed": 42}, "inference": {}}),
        encoding="utf-8",
    )
    existing_config = run_root / "frozen_config.json"
    existing_config.write_text(
        json.dumps({"training": {"seed": 101}, "inference": {}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ARC3_TRAINING_SEED", "303")

    frozen_config = prepare_frozen_config(
        config_path="configs/ppo_8gpu_official.json",
        live_repo_root=live_repo,
        source_root=source_root,
        run_root=run_root,
    )

    assert frozen_config == existing_config
    rewritten = json.loads(frozen_config.read_text(encoding="utf-8"))
    assert rewritten["training"]["seed"] == 101


def test_python_environment_fingerprint_ignores_editable_metadata_churn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    editable = site_packages / "arc3_training-0.1.0.dist-info"
    editable.mkdir()
    (editable / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: arc3-training\nVersion: 0.1.0\n\nold\n",
        encoding="utf-8",
    )
    (editable / "direct_url.json").write_text(
        json.dumps({"url": "file:///repo", "dir_info": {"editable": True}}),
        encoding="utf-8",
    )
    (editable / "RECORD").write_text("old editable record\n", encoding="utf-8")
    (site_packages / "__editable__.arc3_training-0.1.0.pth").write_text(
        "import __editable___arc3_training_0_1_0_finder\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "training.source_snapshot._site_paths_from_python",
        lambda _python_bin: [str(site_packages)],
    )
    before = python_environment_fingerprint(tmp_path / "python")

    (editable / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: arc3-training\nVersion: 0.1.0\n\nnew\n",
        encoding="utf-8",
    )
    (editable / "RECORD").write_text("new editable record\n", encoding="utf-8")
    after = python_environment_fingerprint(tmp_path / "python")

    assert after["metadata_sha256"] == before["metadata_sha256"]


def test_python_environment_fingerprint_detects_dependency_version_change(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    dependency = site_packages / "example_dep-1.0.0.dist-info"
    dependency.mkdir()
    (dependency / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: example-dep\nVersion: 1.0.0\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "training.source_snapshot._site_paths_from_python",
        lambda _python_bin: [str(site_packages)],
    )
    before = python_environment_fingerprint(tmp_path / "python")

    (dependency / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: example-dep\nVersion: 1.0.1\n",
        encoding="utf-8",
    )
    after = python_environment_fingerprint(tmp_path / "python")

    assert after["metadata_sha256"] != before["metadata_sha256"]


def test_verify_runtime_fingerprint_rejects_unknown_format(tmp_path: Path) -> None:
    fingerprint_path = tmp_path / "runtime_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps({"training": {"python": str(tmp_path / "python")}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="Unsupported runtime Python"):
        verify_runtime_fingerprint(fingerprint_path)


def test_frozen_launcher_execs_from_snapshot_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source_root = tmp_path / "run" / "source"
    source_root.mkdir(parents=True)
    paths = FrozenRunPaths(
        live_repo_root=tmp_path / "repo",
        run_root=tmp_path / "run",
        source_archive_path=tmp_path / "run" / "source.tar",
        source_root=source_root,
        frozen_config_path=tmp_path / "run" / "frozen_config.json",
        snapshot_metadata_path=tmp_path / "run" / "source_snapshot.json",
        runtime_fingerprint_path=tmp_path / "run" / "runtime_fingerprint.json",
    )
    paths.frozen_config_path.write_text(
        (
            Path(__file__).resolve().parents[1]
            / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json"
        ).read_text()
    )
    calls: dict[str, object] = {}

    def fake_execvpe(executable: str, command: list[str], env: dict[str, str]) -> None:
        calls["executable"] = executable
        calls["command"] = command
        calls["env"] = env
        raise RuntimeError("exec intercepted")

    monkeypatch.setattr(
        "training.launch_frozen.os.chdir",
        lambda path: calls.setdefault("cwd", Path(path)),
    )
    monkeypatch.setattr("training.launch_frozen.os.execvpe", fake_execvpe)
    monkeypatch.setenv("ARC3_ORIGINAL_REPO_ROOT", str(tmp_path / "live_repo"))

    with pytest.raises(RuntimeError, match="exec intercepted"):
        _exec_training_from_snapshot(
            paths=paths,
            run_dir=str(paths.run_root),
            argv0_python="/usr/bin/python",
        )

    env = calls["env"]
    assert isinstance(env, dict)
    assert calls["cwd"] == source_root
    assert env["PWD"] == str(source_root)
    assert env["ARC3_ORIGINAL_REPO_ROOT"] == str(tmp_path / "live_repo")
    assert env["ARC3_FROZEN_REPO_ROOT"] == str(source_root)
    assert calls["command"] == [
        "/usr/bin/python",
        "-m",
        "training.cli",
        "--config",
        str(paths.frozen_config_path),
        "--run-dir",
        str(paths.run_root),
    ]


def test_prepare_frozen_run_forces_config_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    source_root = run_root / "source"
    source_root.mkdir(parents=True)
    frozen_config = run_root / "frozen_config.json"
    frozen_config.write_text('{"inference": {}}\n', encoding="utf-8")
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "training.source_snapshot.create_source_snapshot",
        lambda **_kwargs: run_root / "source.tar",
    )
    (run_root / "source_snapshot.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "training.source_snapshot.materialize_source_snapshot",
        lambda **_kwargs: source_root,
    )

    def fake_prepare_frozen_config(**kwargs) -> Path:  # noqa: ANN003
        observed.update(kwargs)
        return frozen_config

    monkeypatch.setattr(
        "training.source_snapshot.prepare_frozen_config",
        fake_prepare_frozen_config,
    )
    monkeypatch.setattr(
        "training.source_snapshot.write_runtime_fingerprint",
        lambda **_kwargs: None,
    )

    prepare_frozen_run(
        config_path="configs/ppo.json",
        live_repo_root=tmp_path / "repo",
        run_root=run_root,
        training_python=tmp_path / "python",
        force_snapshot=True,
    )

    assert observed["force"] is True
