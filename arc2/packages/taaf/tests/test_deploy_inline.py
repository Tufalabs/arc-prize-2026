# type: ignore
"""Tests for taaf.deploy / taaf.deploy_inline — the in-process target, and
the parts of the deploy contract every target shares."""

import json
from pathlib import Path

import pytest

import taaf

# The snapshot names each repo by its checkout directory, and so does
# ``main_project``'s default — a worktree checks the same repo out under a
# branch-specific name, so neither may be written down as a literal.
REPO_DIR = Path(__file__).resolve().parents[3].name


def _winning_benchmark(job_dir: Path) -> taaf.benchmark.Benchmark:
    return taaf.benchmark.Benchmark(
        label="test_deploy_inline",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverFlipLR(),
        n_passes=1,
        job_dir=job_dir,
    )


# --- Benchmark.deploy(InlineTarget) end-to-end --------------------------


async def test_inline_deploy_runs_to_completion(tmp_path: Path):
    bm = _winning_benchmark(tmp_path)
    handle = await bm.deploy(taaf.deploy_inline.InlineTarget())
    assert handle.is_done
    assert handle.job_dir == tmp_path
    assert len(bm.game_runs) == 1
    assert bm.game_runs[0].state == "won"


async def test_inline_deploy_requires_job_dir():
    bm = taaf.benchmark.Benchmark(
        label="no_job_dir",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverFlipLR(),
        n_passes=1,
        job_dir=None,
    )
    with pytest.raises(ValueError, match="job_dir"):
        await bm.deploy(taaf.deploy_inline.InlineTarget())


async def test_inline_deploy_refuses_existing_deploy_meta(tmp_path: Path):
    """deploy() must refuse a job_dir that already contains
    ``deploy_meta.json``. Silently overwriting would either trample a
    prior run's artifacts or race a still-running deploy on every
    output file (.venv, stdout.log, benchmark.pkl, ...). Fail fast on
    the launcher with a message telling the user to rm/mv the dir."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "deploy_meta.json").write_text("{}")
    bm = _winning_benchmark(tmp_path)
    with pytest.raises(FileExistsError, match="deploy_meta.json"):
        await bm.deploy(taaf.deploy_inline.InlineTarget())


# --- source snapshot in job_dir/src/ ------------------------------------


async def test_inline_deploy_snapshots_tufalabs_sources(tmp_path: Path):
    bm = _winning_benchmark(tmp_path)
    await bm.deploy(taaf.deploy_inline.InlineTarget())
    src = tmp_path / "src"
    assert (src / REPO_DIR / "pyproject.toml").is_file()
    assert (src / REPO_DIR / "README.md").is_file()
    # The member tree is mirrored at its workspace-relative path, so the
    # bundled root can `uv sync --all-packages` on the worker.
    assert (src / REPO_DIR / "packages" / "taaf" / "src" / "taaf" / "deploy.py").is_file()
    # And the packed datasets travel with it: a worker has no checkout and
    # no network, so a dataset left behind is a run that cannot start.
    assert (src / REPO_DIR / "packages" / "taaf" / "src" / "taaf" / "data" / "arc2-train.npz").is_file()


def test_discover_editable_repos_picks_only_editable_distributions(tmp_path: Path, monkeypatch):
    """``_discover_editable_repos`` walks ``importlib.metadata``
    and surfaces every distribution with ``direct_url.json`` recording
    an editable install. Non-editable, missing-metadata, malformed-JSON,
    and non-``file://`` URLs are all skipped silently — none of them are
    "an editable repo this venv depends on", so they don't need shipping.
    Two distributions in the same repo collapse to one entry (monorepo
    case).
    """
    import importlib.metadata
    import json as _json

    repo_a = tmp_path / "repo_a"
    repo_a.mkdir()
    (repo_a / "pyproject.toml").write_text("")
    repo_b = tmp_path / "repo_b"
    repo_b.mkdir()
    (repo_b / "pyproject.toml").write_text("")
    no_pyproject = tmp_path / "no_pyproject_here"
    no_pyproject.mkdir()

    class _Dist:
        def __init__(self, direct_url: str | None) -> None:
            self._du = direct_url

        def read_text(self, filename: str) -> str | None:
            if filename == "direct_url.json":
                return self._du
            return None

    editable_a = _json.dumps({"url": f"file://{repo_a}", "dir_info": {"editable": True}})
    editable_a_dup = _json.dumps(
        # Second distribution rooted in the same repo (monorepo) — must dedupe.
        {"url": f"file://{repo_a}", "dir_info": {"editable": True}}
    )
    editable_b = _json.dumps({"url": f"file://{repo_b}", "dir_info": {"editable": True}})
    not_editable = _json.dumps({"url": f"file://{repo_a}", "dir_info": {"editable": False}})
    no_dir_info = _json.dumps({"url": f"file://{repo_a}"})  # no dir_info key at all
    non_file = _json.dumps({"url": "https://example.com/x", "dir_info": {"editable": True}})
    no_pyproject_url = _json.dumps({"url": f"file://{no_pyproject}", "dir_info": {"editable": True}})
    fakes = [
        _Dist(editable_a),
        _Dist(editable_a_dup),
        _Dist(editable_b),
        _Dist(not_editable),
        _Dist(no_dir_info),
        _Dist(non_file),
        _Dist(no_pyproject_url),
        _Dist(None),  # legacy install with no direct_url.json
        _Dist("not json"),
    ]
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: iter(fakes))

    result = taaf.deploy._discover_editable_repos()
    assert set(result) == {repo_a.resolve(), repo_b.resolve()}


def test_snapshot_editable_sources_raises_on_per_repo_failure(tmp_path: Path, monkeypatch):
    """If any single repo fails to snapshot, the whole call
    raises ``RuntimeError`` rather than silently skipping. Silent skip
    let the deploy queue and only fail cryptically inside ``uv pip
    install -e <missing-repo>`` on the worker; the user wants the
    failure surfaced on the launcher, where they have full context."""
    broken = tmp_path / "broken_repo"
    broken.mkdir()
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [broken])

    def fail_for_one(repo_root: Path, dest: Path):
        if repo_root.name == "broken_repo":
            raise OSError("simulated disk full")
        raise AssertionError(f"unexpected repo: {repo_root}")

    monkeypatch.setattr(taaf.deploy, "_snapshot_one_repo", fail_for_one)
    with pytest.raises(RuntimeError, match="broken_repo"):
        taaf.deploy.snapshot_editable_sources(tmp_path / "src")


def test_snapshot_editable_sources_includes_extra_repos(tmp_path: Path, monkeypatch):
    """Callers can force local repos into the deploy snapshot even when they
    are not installed editable in the launcher venv."""
    extra = tmp_path / "extra_repo"
    (extra / "extra_pkg").mkdir(parents=True)
    (extra / "pyproject.toml").write_text(
        '[project]\nname = "extra-repo"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["extra_pkg*"]\n'
    )
    (extra / "extra_pkg" / "__init__.py").write_text("")
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", list)
    monkeypatch.setattr(taaf.deploy, "_discover_noneditable_tufa_packages", list)

    written = taaf.deploy.snapshot_editable_sources(tmp_path / "src", extra_repos=[extra])

    assert written == [(tmp_path / "src" / "extra_repo")]
    assert (tmp_path / "src" / "extra_repo" / "pyproject.toml").is_file()
    assert (tmp_path / "src" / "extra_repo" / "extra_pkg" / "__init__.py").is_file()


# --- stdout teed to a log file ------------------------------------------


async def test_inline_deploy_tees_stdout_to_log(tmp_path: Path):
    bm = _winning_benchmark(tmp_path)
    await bm.deploy(taaf.deploy_inline.InlineTarget())
    log = (tmp_path / "stdout.log").read_text()
    # Preamble lines are guaranteed; the summary may or may not be present
    # depending on whether the benchmark wrote it before the tee teardown.
    assert "benchmark.label : test_deploy_inline" in log
    assert "git status:" in log


# --- deploy_meta.json and attach() round-trip ---------------------------


async def test_inline_deploy_writes_deploy_meta(tmp_path: Path):
    bm = _winning_benchmark(tmp_path)
    await bm.deploy(taaf.deploy_inline.InlineTarget())
    meta_path = tmp_path / "deploy_meta.json"
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text())
    assert meta["schema_version"] == taaf.deploy.DEPLOY_META_SCHEMA_VERSION
    assert meta["target_class"] == "taaf.deploy_inline.InlineTarget"
    assert meta["handle_class"] == "taaf.deploy_inline.InlineHandle"
    assert meta["job_id"] is None  # inline has no external scheduler
    assert meta["benchmark_label"] == "test_deploy_inline"
    assert "started_at" in meta
    assert "host" in meta
    assert "pid" in meta


async def test_attach_reconstructs_handle_for_finished_inline_run(tmp_path: Path):
    """Roundtrip: deploy → forget the in-memory handle → attach() from disk."""
    bm = _winning_benchmark(tmp_path)
    handle = await bm.deploy(taaf.deploy_inline.InlineTarget())
    original_job_dir = handle.job_dir
    del handle, bm  # simulate Python process death

    reattached = taaf.deploy.DeploymentHandle.attach(original_job_dir)
    assert isinstance(reattached, taaf.deploy_inline.InlineHandle)
    assert reattached.is_done
    assert reattached.job_dir == original_job_dir
    # The benchmark loaded from disk reproduces the won state.
    reloaded = reattached.wait()
    assert len(reloaded.game_runs) == 1
    assert reloaded.game_runs[0].state == "won"
    assert reloaded.game_runs[0].final_score == pytest.approx(100.0)


def test_attach_errors_when_no_meta(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="deploy_meta.json"):
        taaf.deploy.DeploymentHandle.attach(tmp_path)


async def test_inline_deploy_clamps_soft_deadline_for_short_max_runtime(tmp_path: Path):
    """With a short max_runtime_s, a blind 10-min buffer would set
    soft_end_time in the past and cancel the run before it starts. The
    clamp picks half the budget instead, so a 60s budget still gives
    the solver 30s to do work."""
    # Patch Benchmark.run so we can observe the soft_end_time without
    # actually running the benchmark to completion.
    captured: dict = {}

    real_run = taaf.benchmark.Benchmark.run

    async def fake_run(self, soft_end_time=None, runtime_environment=None):
        captured["soft_end_time"] = soft_end_time
        captured["called_at"] = __import__("datetime").datetime.now()
        # Call through with no time bound so teardown happens cleanly.
        await real_run(self, soft_end_time=None, runtime_environment=runtime_environment)

    taaf.benchmark.Benchmark.run = fake_run
    try:
        bm = _winning_benchmark(tmp_path)
        # 60s budget; with naive math this lands in the past.
        await bm.deploy(taaf.deploy_inline.InlineTarget(max_runtime_s=60.0))
    finally:
        taaf.benchmark.Benchmark.run = real_run

    assert captured["soft_end_time"] is not None
    delta = (captured["soft_end_time"] - captured["called_at"]).total_seconds()
    # Half-of-budget clamp: 60s budget -> 30s headroom (give or take a
    # few hundred ms for the deploy() preamble).
    assert 25 < delta < 35


async def test_inline_deploy_external_cancellation_runs_teardown(tmp_path: Path):
    """Cancelling the task awaiting InlineTarget.deploy()
    (notebook cell interrupt, ``asyncio.run`` Ctrl-C, explicit
    ``task.cancel()``) must route through ``Benchmark.run``'s teardown
    path: pickle + JSON + diagnostics land in ``job_dir`` before
    deploy() returns. Locks in the cancellation contract the async
    refactor provides.
    """
    import asyncio

    bm = taaf.benchmark.Benchmark(
        label="test_inline_cancel",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverFlipLR(delay_s=0.05),
        n_passes=1,
        job_dir=tmp_path,
    )
    deploy_task = asyncio.create_task(bm.deploy(taaf.deploy_inline.InlineTarget()))
    # Wait until Benchmark.run is actually inside its solver-await — request
    # before that point is a no-op and the test would race.
    for _ in range(500):
        if bm._solver_task is not None:
            break
        await asyncio.sleep(0.01)
    assert bm._solver_task is not None, "deploy() never reached solver-await"
    deploy_task.cancel()
    handle = await deploy_task  # InlineTarget swallows CancelledError → returns clean
    assert handle.is_done
    assert (tmp_path / "benchmark.json").is_file()
    assert (tmp_path / "game_specs.pkl").is_file()
    assert (tmp_path / "diagnostics.html").is_file()
    assert any(r.state == "cancelled" for r in bm.game_runs)


def test_attach_errors_when_no_benchmark_snapshot(tmp_path: Path):
    """Simulate deploy() that wrote metadata but Benchmark.run() never finished."""
    # Hand-craft an inline-flavoured deploy_meta.json with no benchmark.json
    # alongside (mimics a crash between metadata write and snapshot write).
    meta = {
        "schema_version": taaf.deploy.DEPLOY_META_SCHEMA_VERSION,
        "target_class": "taaf.deploy_inline.InlineTarget",
        "target_config": {},
        "handle_class": "taaf.deploy_inline.InlineHandle",
        "job_id": None,
        "benchmark_label": "partial",
        "started_at": "2026-05-11T13:42:04",
        "host": "test",
        "pid": 1234,
    }
    (tmp_path / "deploy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="benchmark.json"):
        taaf.deploy.DeploymentHandle.attach(tmp_path)


# --- uv-workspace launchers bundle the whole workspace ------------------


def _make_workspace_repo(tmp_path: Path) -> Path:
    """A uv workspace: virtual root, two members (one depending on the
    other via a `{ workspace = true }` source), one excluded dir."""
    root = tmp_path / "wsrepo"
    (root / "pyproject.toml").parent.mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        '[tool.uv.workspace]\nmembers = ["packages/*"]\nexclude = ["packages/skipme"]\n'
    )
    (root / "uv.lock").write_text("# placeholder lock\n")
    for name, sources in (("alpha", ""), ("beta", "[tool.uv.sources]\nalpha = { workspace = true }\n")):
        member = root / "packages" / name
        (member / "src" / f"{name}_pkg").mkdir(parents=True)
        deps = '"alpha"' if name == "beta" else ""
        (member / "pyproject.toml").write_text(
            f'[project]\nname = "{name}"\nversion = "0.1.0"\ndependencies = [{deps}]\n'
            f'[tool.setuptools.packages.find]\nwhere = ["src"]\n{sources}'
        )
        (member / "src" / f"{name}_pkg" / "__init__.py").write_text("")
    skipped = root / "packages" / "skipme"
    skipped.mkdir()
    (skipped / "pyproject.toml").write_text('[project]\nname = "skipme"\nversion = "0.1.0"\n')
    return root


def test_find_repo_root_resolves_workspace_member_to_workspace_root(tmp_path: Path):
    root = _make_workspace_repo(tmp_path)
    member_src = root / "packages" / "beta" / "src" / "beta_pkg"
    assert taaf.deploy._find_repo_root(member_src) == root
    # An excluded dir is standalone: its own pyproject is the repo root.
    assert taaf.deploy._find_repo_root(root / "packages" / "skipme") == root / "packages" / "skipme"


def test_snapshot_bundles_whole_uv_workspace(tmp_path: Path, monkeypatch):
    """An editable uv-workspace member is bundled as its whole
    workspace (root pyproject + uv.lock + every member at its
    workspace-relative path): the member pyproject alone cannot resolve
    its `{ workspace = true }` sources on the worker."""
    root = _make_workspace_repo(tmp_path)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [root])
    monkeypatch.setattr(taaf.deploy, "_discover_noneditable_tufa_packages", list)

    written = taaf.deploy.snapshot_editable_sources(tmp_path / "src")

    dest = tmp_path / "src" / "wsrepo"
    assert written == [dest]
    assert (dest / "pyproject.toml").is_file()
    assert (dest / "uv.lock").is_file()
    for name in ("alpha", "beta"):
        assert (dest / "packages" / name / "pyproject.toml").is_file()
        assert (dest / "packages" / name / "src" / f"{name}_pkg" / "__init__.py").is_file()
    # Excluded members stay out of the bundle.
    assert not (dest / "packages" / "skipme").exists()
