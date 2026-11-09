# type: ignore
"""Tests for taaf.deploy_slurm — the cluster target.

The Tufa cluster isn't reachable from CI, so every test stubs the three
subprocess callers (sbatch / squeue / scancel) using monkeypatch. The
tests focus on:

- That ``deploy()`` invokes ``sbatch`` with the right input and writes
  the right files.
- That the rendered sbatch script carries the directives the cluster
  needs (gres, --time, --output/--error, partition / nodelist / image /
  mounts pass-through).
- That ``TufaSlurmHandle.is_done`` / ``wait()`` / ``stop()`` use squeue
  / scancel as designed.
- That ``DeploymentHandle.attach(job_dir)`` reconstructs a
  ``TufaSlurmHandle`` from ``deploy_meta.json``.
"""

import json
import os
import pickle
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

import taaf

# The snapshot names each repo by its checkout directory, and so does
# ``main_project``'s default — a worktree checks the same repo out under a
# branch-specific name, so neither may be written down as a literal.
REPO_DIR = Path(__file__).resolve().parents[3].name


def _extracted_src(root: Path) -> Path:
    """deploy() ships the snapshot as a single ``src.tar`` (one NFS write
    instead of ~1k small files); extract it so tests can assert on the same
    ``src/<repo>/`` layout that ``snapshot_editable_sources`` produces."""
    import tarfile

    src_dir = root / "src"
    with tarfile.open(root / "src.tar") as tf:
        tf.extractall(src_dir, filter="data")
    return src_dir


def _benchmark(job_dir: Path, **kwargs) -> taaf.benchmark.Benchmark:
    defaults = dict(
        label="test_slurm",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverFlipLR(),
        n_passes=1,
        job_dir=job_dir,
    )
    defaults.update(kwargs)
    return taaf.benchmark.Benchmark(**defaults)


class _SubprocessRecorder:
    """Capture every ``subprocess.run`` call. Each entry maps a leading
    argv token to a response (stdout, returncode). Missing tokens fall
    through to a sensible default for tests that don't care."""

    def __init__(self, responses: dict[str, tuple[str, int]] | None = None) -> None:
        self.responses = responses or {}
        self.calls: list[dict] = []

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), "kwargs": kwargs})
        cmd = argv[0]
        stdout, rc = self.responses.get(cmd, ("", 0))
        if kwargs.get("check") and rc != 0:
            raise subprocess.CalledProcessError(rc, argv, output=stdout, stderr="")
        return subprocess.CompletedProcess(args=argv, returncode=rc, stdout=stdout, stderr="")


@pytest.fixture
def stub_subprocess(monkeypatch):
    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0), "squeue": ("", 0), "scancel": ("", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    return rec


# --- deploy() end-to-end (mocked sbatch) ---------------------------------


async def test_deploy_invokes_sbatch_and_returns_handle(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    handle = await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="04:00:00"))

    assert isinstance(handle, taaf.deploy_slurm.TufaSlurmHandle)
    assert handle.job_id == "1234567"
    assert handle.job_dir == tmp_path.resolve()

    sbatch_calls = [c for c in stub_subprocess.calls if c["argv"][0] == "sbatch"]
    assert len(sbatch_calls) == 1
    assert sbatch_calls[0]["argv"] == ["sbatch", "--parsable"]
    assert "input" in sbatch_calls[0]["kwargs"]
    assert "#SBATCH --time=04:00:00" in sbatch_calls[0]["kwargs"]["input"]


async def test_deploy_requires_job_dir(stub_subprocess):
    bm = taaf.benchmark.Benchmark(
        label="no_job_dir",
        games=[taaf.game_examples.flip_game()],
        solver=taaf.solver_examples.SolverFlipLR(),
        n_passes=1,
        job_dir=None,
    )
    with pytest.raises(ValueError, match="job_dir"):
        await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))


async def test_deploy_refuses_existing_deploy_meta(tmp_path: Path, stub_subprocess):
    """deploy() must refuse a job_dir that already contains
    ``deploy_meta.json``. The slurm-specific failure mode this prevents:
    a re-deploy submits a second sbatch against the same job_dir, both
    jobs eventually run, and they race on .venv/, stdout.log,
    benchmark.pkl, etc. Fail fast on the launcher (before sbatch) and
    leave the prior run's artifacts intact for the user to inspect."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "deploy_meta.json").write_text("{}")
    bm = _benchmark(tmp_path)
    with pytest.raises(FileExistsError, match="deploy_meta.json"):
        await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    # sbatch must NOT be invoked when the gate fires.
    assert not any(c["argv"][0] == "sbatch" for c in stub_subprocess.calls)


async def test_deploy_parses_parsable_cluster_suffix(tmp_path: Path, monkeypatch):
    """``sbatch --parsable`` may emit ``<id>;<cluster_name>``; the suffix
    must not contaminate the job_id."""
    rec = _SubprocessRecorder({"sbatch": ("9999999;tufa-cluster\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    bm = _benchmark(tmp_path)
    handle = await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    assert handle.job_id == "9999999"


async def test_deploy_writes_deploy_meta_with_job_id(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B300", time="02:00:00", partition="gpu"))
    meta = json.loads((tmp_path / "deploy_meta.json").read_text())
    assert meta["target_class"] == "taaf.deploy_slurm.TufaSlurmTarget"
    assert meta["handle_class"] == "taaf.deploy_slurm.TufaSlurmHandle"
    assert meta["job_id"] == "1234567"
    assert meta["target_config"]["gpu"] == "B300"
    assert meta["target_config"]["time"] == "02:00:00"
    assert meta["target_config"]["partition"] == "gpu"


# --- source snapshot also runs on cluster deploys ------------------------


async def test_deploy_snapshots_tufalabs_sources(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    _extracted_src(tmp_path)
    assert (tmp_path / "src" / REPO_DIR / "packages" / "taaf" / "src" / "taaf" / "deploy_slurm.py").is_file()


# --- Initial pickle + worker script are written --------------------------


async def test_deploy_writes_initial_pickle(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    pkl_path = tmp_path / "benchmark_initial.pkl"
    assert pkl_path.is_file()
    with open(pkl_path, "rb") as f:
        reloaded = pickle.load(f)
    assert reloaded.label == "test_slurm"
    assert len(reloaded.games) == 1


async def test_deploy_writes_worker_script(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    worker = (tmp_path / "run_in_worker.py").read_text()
    # Anchors the soft deadline to Slurm's scheduled end time.
    assert "SLURM_JOB_END_TIME" in worker
    # ``_amain`` wraps ``bm.run(soft_end_time=..., runtime_environment=...)``
    # so the SIGUSR1 handler (graceful stop) can be registered inside
    # the asyncio loop and the solver sees the target it's running under
    # .
    assert "asyncio.run(_amain(bm, SOFT_END, target))" in worker
    assert "bm.run(soft_end_time=soft_end, runtime_environment=target)" in worker
    # The worker loads the pickled target so the solver gets the live
    # TufaSlurmTarget reference, not a string mirror.
    assert 'open(JOB_DIR / "deploy_target.pkl", "rb")' in worker
    # The job_dir is baked in absolute so the worker can find its pickle
    # regardless of where it's launched from.
    assert repr(str(tmp_path.resolve())) in worker


async def test_deploy_writes_target_pickle(tmp_path: Path, stub_subprocess):
    """The target is pickled alongside the benchmark so the worker
    can pass it as ``runtime_environment=`` into ``bm.run``. Reloading must
    yield an equal ``TufaSlurmTarget`` so the solver sees the real config
    (gpu, time, partition, etc.) of the environment it's running on."""
    target = taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00", partition="llm")
    bm = _benchmark(tmp_path)
    await bm.deploy(target)
    pkl_path = tmp_path / "deploy_target.pkl"
    assert pkl_path.is_file()
    with open(pkl_path, "rb") as f:
        reloaded = pickle.load(f)
    assert isinstance(reloaded, taaf.deploy_slurm.TufaSlurmTarget)
    assert reloaded.gpu == "B200"
    assert reloaded.time == "01:00:00"
    assert reloaded.partition == "llm"


async def test_worker_script_is_syntactically_valid_python(tmp_path: Path, stub_subprocess):
    """Belt-and-suspenders: the rendered worker must always parse,
    even when the embedded preamble contains shell metacharacters,
    quotes, newlines, etc. ``compile()`` catches template breakage
    before the worker is shipped to the cluster."""
    bm = _benchmark(tmp_path, label='funky: "quotes" and \nnewlines and \\backslashes')
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    worker_src = (tmp_path / "run_in_worker.py").read_text()
    compile(worker_src, str(tmp_path / "run_in_worker.py"), "exec")


# --- preamble emitted by the worker --------------------------------------


async def test_worker_script_clamps_soft_deadline_to_half_remaining(tmp_path: Path, stub_subprocess):
    """The rendered worker must route deadline calculation through
    the tested helper that clamps short walltimes and applies any target
    experiment budget."""
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="00:05:00"))
    worker = (tmp_path / "run_in_worker.py").read_text()
    assert "SOFT_END = taaf.deploy_slurm.worker_soft_end_time(" in worker
    assert 'slurm_end_epoch=os.environ.get("SLURM_JOB_END_TIME")' in worker
    assert 'max_runtime_s=getattr(target, "max_runtime_s", None)' in worker


async def test_worker_script_embeds_launcher_side_preamble(tmp_path: Path, stub_subprocess):
    """Label / solver repr / git block, captured on the
    launcher (where the real git checkout lives) and baked into the
    worker script as a literal so the bundled-snapshot side doesn't
    need git to print the preamble."""
    bm = _benchmark(tmp_path)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    worker = (tmp_path / "run_in_worker.py").read_text()
    # Preamble fields from format_preamble are present in the embedded
    # literal. We don't pin the exact string because git status output
    # depends on the working tree, but the structural lines are stable.
    assert "benchmark.label : test_slurm" in worker
    assert "benchmark.solver:" in worker
    assert "git status:" in worker
    # And the worker actually prints it (and the slurm-id line) before
    # the run.
    assert 'print(f"deploy.slurm: SLURM_JOB_ID' in worker


# --- sbatch script rendering ---------------------------------------------


def _sbatch_script_for(target: taaf.deploy_slurm.TufaSlurmTarget, job_dir: Path, **bm_kw) -> str:
    bm = _benchmark(job_dir, **bm_kw)
    job_dir.mkdir(parents=True, exist_ok=True)
    # Render directly without invoking deploy() so we don't need to stub
    # subprocess for these pure-render tests.
    return taaf.deploy_slurm._render_sbatch_script(target, bm, job_dir.resolve())


def test_sbatch_script_sets_ntasks(tmp_path: Path):
    """Tufa cluster's submit plugin refuses GPU jobs without --ntasks set;
    we emit ``--ntasks=1`` unconditionally for our single-process worker."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"),
        tmp_path,
    )
    assert "#SBATCH --ntasks=1" in script


async def test_deploy_surfaces_sbatch_stderr_on_failure(tmp_path: Path, monkeypatch):
    """When sbatch rejects the script, the user must see the actual
    error (not just a CalledProcessError with a returncode)."""
    rec = _SubprocessRecorder({"sbatch": ("", 1)})

    # Patch the recorder to also emit stderr — simulates the cluster's
    # ``--ntasks`` error we hit during the first real deploy.
    def fake_run_with_stderr(argv, **kwargs):
        rec.calls.append({"argv": list(argv), "kwargs": kwargs})
        if argv[0] == "sbatch":
            return subprocess.CompletedProcess(argv, 1, "", "ERROR: GPU jobs must set --ntasks")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run_with_stderr)
    bm = _benchmark(tmp_path)
    with pytest.raises(RuntimeError, match="ntasks"):
        await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))


def test_sbatch_script_includes_gres_for_gpu_type(tmp_path: Path):
    """Each declared GPU type renders to its matching
    ``--gres=gpu:<type>:1`` line. B200 is live on the cluster
    today (gres ``b200``); B300 follows the naming convention so
    it's ready when the cluster grows that type."""
    for gpu, expected in [("B200", "b200"), ("B300", "b300")]:
        script = _sbatch_script_for(
            taaf.deploy_slurm.TufaSlurmTarget(gpu=gpu, time="01:00:00"),
            tmp_path,
        )
        assert f"--gres=gpu:{expected}:1" in script


def test_sbatch_script_includes_configured_gpu_count(tmp_path: Path):
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            gpu_count=2,
            time="01:00:00",
        ),
        tmp_path,
    )

    assert "--gres=gpu:b200:2" in script


def test_slurm_target_preserves_positional_gpu_time_constructor(tmp_path: Path):
    """Existing callers may pass ``gpu`` and ``time`` positionally."""
    target = taaf.deploy_slurm.TufaSlurmTarget("B200", "01:00:00")

    assert target.gpu == "B200"
    assert target.time == "01:00:00"
    assert target.gpu_count == 1
    script = _sbatch_script_for(target, tmp_path)
    assert "#SBATCH --time=01:00:00" in script
    assert "--gres=gpu:b200:1" in script


def test_sbatch_script_default_resources_match_pre_rename_behaviour(tmp_path: Path):
    """Constructing the target with no explicit resource fields uses the
    per-gpu default (8 CPUs + 32 GiB per gpu) — the historical default
    before the cpu_count/memory_limit_bytes rename. Guards the
    regression flagged in PR #19: a multi-GPU caller relying on the
    default used to get 8 CPUs / 32 GiB *per GPU*; the totals-only
    rewrite silently dropped this to per-job, which is what the rename
    + per-gpu defaults restore."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            "B200",
            "01:00:00",
            gpu_count=4,
        ),
        tmp_path,
    )
    assert "#SBATCH --cpus-per-gpu=8" in script
    assert f"#SBATCH --mem-per-gpu={32 * 1024}M" in script


def test_sbatch_script_per_gpu_explicit(tmp_path: Path):
    """Per-gpu values are emitted directly to ``--cpus-per-gpu`` /
    ``--mem-per-gpu`` without division by gpu_count."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            "B200",
            "01:00:00",
            gpu_count=2,
            cpus_per_gpu=12,
            mem_per_gpu_bytes=48 * 1024**3,
        ),
        tmp_path,
    )
    assert "#SBATCH --cpus-per-gpu=12" in script
    assert f"#SBATCH --mem-per-gpu={48 * 1024}M" in script


def test_sbatch_script_totals_divided_across_gpus(tmp_path: Path):
    """Totals are divided by gpu_count (ceiling) before rendering, so
    the per-gpu sbatch lines sum to at least the user's ask."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            "B200",
            "01:00:00",
            gpu_count=4,
            total_cpus=20,
            total_mem_bytes=63 * 1024**3,
        ),
        tmp_path,
    )
    # 20 / 4 = 5; 63 GiB / 4 = 16128 MiB exactly (no ceiling bump here).
    assert "#SBATCH --cpus-per-gpu=5" in script
    assert "#SBATCH --mem-per-gpu=16128M" in script


def test_sbatch_script_mixed_coordinate_systems_errors(tmp_path: Path):
    """Setting both per-gpu and totals is rejected — the rename's point
    is to force callers to pick one."""
    target = taaf.deploy_slurm.TufaSlurmTarget(
        "B200",
        "01:00:00",
        gpu_count=2,
        cpus_per_gpu=8,
        total_cpus=16,
    )
    with pytest.raises(ValueError, match="cannot mix per-gpu"):
        _sbatch_script_for(target, tmp_path)


def test_sbatch_script_partial_per_gpu_pair_errors(tmp_path: Path):
    target = taaf.deploy_slurm.TufaSlurmTarget(
        "B200",
        "01:00:00",
        gpu_count=2,
        cpus_per_gpu=8,
    )
    with pytest.raises(ValueError, match="cpus_per_gpu and mem_per_gpu_bytes must be set together"):
        _sbatch_script_for(target, tmp_path)


def test_sbatch_script_partial_totals_pair_errors(tmp_path: Path):
    target = taaf.deploy_slurm.TufaSlurmTarget(
        "B200",
        "01:00:00",
        gpu_count=2,
        total_cpus=16,
    )
    with pytest.raises(ValueError, match="total_cpus and total_mem_bytes must be set together"):
        _sbatch_script_for(target, tmp_path)


def test_sbatch_script_cpu_accepts_per_gpu_as_convenience(tmp_path: Path):
    """CPU jobs accept the per-gpu coordinate system (treated as totals
    with gpu_count=1) so callers don't have to switch coordinate
    systems just to smoke-test on CPU."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="CPU",
            gpu_count=0,
            time="01:00:00",
            cpus_per_gpu=4,
            mem_per_gpu_bytes=8 * 1024**3,
        ),
        tmp_path,
    )
    assert "#SBATCH --cpus-per-task=4" in script
    assert f"#SBATCH --mem={8 * 1024}M" in script


def test_sbatch_script_cpu_omits_gres_and_uses_per_task_flags(tmp_path: Path):
    """``gpu="CPU"`` selects a CPU-only job: no ``--gres=gpu:…``, and the
    per-gpu CPU/memory forms (``--cpus-per-gpu`` / ``--mem-per-gpu``) — which
    are nonsensical without a GPU — switch to the standard per-task / total
    forms. Used for cheap smoke tests where the solver doesn't need a GPU."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="CPU",
            gpu_count=0,
            time="01:00:00",
        ),
        tmp_path,
    )
    assert "--gres=gpu:" not in script
    assert "--cpus-per-gpu" not in script
    assert "--mem-per-gpu" not in script
    assert "#SBATCH --cpus-per-task=" in script
    assert "#SBATCH --mem=" in script


def test_sbatch_script_cpu_rejects_nonzero_gpu_count(tmp_path: Path):
    target = taaf.deploy_slurm.TufaSlurmTarget(
        gpu="CPU",
        gpu_count=1,
        time="01:00:00",
    )

    with pytest.raises(ValueError, match="gpu_count must be 0 for CPU Slurm jobs"):
        _sbatch_script_for(target, tmp_path)


def test_sbatch_script_routes_output_to_job_dir(tmp_path: Path):
    """stdout/stderr piped into the job dir via sbatch directives."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"),
        tmp_path,
    )
    job_dir = tmp_path.resolve()
    assert f"#SBATCH --output={job_dir}/stdout.log" in script
    assert f"#SBATCH --error={job_dir}/stderr.log" in script


def test_sbatch_script_creates_per_run_venv(tmp_path: Path):
    """Fresh venv per run, materialized via ``uv sync`` from the
    main project's bundled pyproject. ``uv sync`` is the only uv command
    that honors ``[tool.uv.sources]`` path rewrites, so it's the single
    install command for both downstream-with-uv.sources setups and
    self-contained snapshots (where uv sync degenerates to a plain
    locked install)."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00", node_local_storage=False),
        tmp_path,
    )
    job_dir = tmp_path.resolve()
    assert f'export UV_PROJECT_ENVIRONMENT="{job_dir}/.venv"' in script
    assert f"cd {job_dir}/src/{REPO_DIR}" in script
    assert "uv sync" in script
    assert 'uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"' in script
    assert "TAAF worker dependency install failed" in script
    # The dropped editable-loop fallback must not leak back in.
    assert "uv pip install $EDITABLE_ARGS" not in script
    assert 'uv pip install --python "$(pwd)/.venv" $EDITABLE_ARGS' not in script
    assert 'EDITABLE_ARGS=""' not in script
    # the flock contract: uv sync runs inside a
    # flock'd subshell on a lockfile next to UV_CACHE_DIR. Image mode
    # makes this a no-op; no-image mode is where it does the work.
    assert "flock 9" in script
    assert '9>"${UV_CACHE_DIR}.lock"' in script
    # ``exec`` is what makes SIGUSR1 actually reach python. Without it
    # the signal lands on bash, bash terminates by default, python orphans
    # and is killed ungracefully. See _render_sbatch_script for the chain.
    assert "exec python -u run_in_worker.py" in script
    assert "\npython -u run_in_worker.py" not in script  # no bare invocation slips through


def test_sbatch_script_uses_configured_uv_override_skip_packages(tmp_path: Path):
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            uv_override_skip_packages=["some-local-pkg", "arc-agi-3-local"],
        ),
        tmp_path,
    )

    assert "--no-install-package some-local-pkg" in script
    assert "--no-install-package arc-agi-3-local" in script
    assert 'if [ -n "${UV_OVERRIDE:-}" ]; then' in script
    assert 'uv pip install --python "$UV_PROJECT_ENVIRONMENT" --no-deps -r "$UV_OVERRIDE"' in script


def test_sbatch_script_node_local_storage_localizes_uv_and_sources(tmp_path: Path):
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="CPU",
            gpu_count=0,
            time="00:10:00",
            main_project="ARC3-Inference",
            node_local_storage=True,
            uv_override_skip_packages=["some-local-pkg", "arc-agi-3-local"],
        ),
        tmp_path,
    )

    assert ". /etc/kander/job-env.sh" in script
    assert "/scratch/jobs/${SLURM_JOB_ID}" in script
    assert "/cache/users/${taaf_uid}" in script
    assert "TAAF_LOCAL_JOB_ROOT" in script
    assert "TAAF_LOCAL_SRC_ROOT" in script
    assert "TAAF_LOCAL_VENV_ROOT" in script
    assert 'export UV_CACHE_DIR="${UV_CACHE_DIR:-${TAAF_NODE_CACHE_ROOT}/uv}"' in script
    # The base CPython (stdlib read on every import) and the $HOME-defaulting
    # startup caches must be node-local too, not just the venv/uv-cache.
    assert 'export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-${TAAF_NODE_CACHE_ROOT}/python}"' in script
    assert "export UV_PYTHON_PREFERENCE=only-managed" in script
    assert 'export MPLCONFIGDIR="${MPLCONFIGDIR:-${TAAF_NODE_CACHE_ROOT}/matplotlib}"' in script
    assert 'export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${TAAF_NODE_CACHE_ROOT}/xdg}"' in script
    assert 'export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${TAAF_NODE_CACHE_ROOT}/triton}"' in script
    assert '"${UV_PYTHON_INSTALL_DIR}" "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}"' in script
    assert 'export UV_PROJECT_ENVIRONMENT="${TAAF_LOCAL_VENV_ROOT}"/ARC3-Inference.venv' in script
    assert "deployment-overrides.txt" in script
    assert "UV_LINK_MODE" in script
    # uv is curl-installed on the node; no pre-staged seed cache.
    assert "astral.sh/uv/install.sh" in script
    assert "cache-seed.tar" not in script
    assert "TAAF_UV_CACHE_SEED" not in script
    assert "export UV_OFFLINE" not in script
    assert 'exec {_taaf_sync_lock_fd}>"${UV_CACHE_DIR}.lock"' in script
    assert 'flock -w 900 "${_taaf_sync_lock_fd}"' in script
    assert "TAAF worker dependency install failed" in script
    assert str((tmp_path.resolve() / "src").as_uri()) in script


def _run_path_resolution_prefix(block: str, probe: str, env: dict[str, str]) -> str:
    """Bash-execute only the path-resolution prefix of the install block,
    stopping before the uv-sync / src-extract machinery so the probe needs
    no real sources or network."""
    marker = 'export TEMPDIR="${TEMPDIR:-${TMPDIR}}"'
    prefix = block[: block.index(marker) + len(marker)]
    return subprocess.run(["bash", "-c", prefix + probe], env=env, capture_output=True, text=True, check=True).stdout


def test_node_local_storage_ignores_leaked_submitter_paths(tmp_path: Path):
    """``sbatch --export=ALL`` ships the launcher's shell env to the worker.
    A login-node ``UV_CACHE_DIR`` / ``TMPDIR`` under shared storage would
    otherwise win the ``${VAR:-...}`` fallbacks and get mkdir'd on
    node-local storage where it isn't writable, killing the job. Execute
    the resolution block with poisoned values and assert they're discarded
    in favour of node-local paths."""
    block = "\n".join(
        taaf.deploy_slurm._render_node_local_install_block(
            taaf.deploy_slurm.TufaSlurmTarget(gpu="CPU", gpu_count=0, time="00:10:00", main_project="ARC3-Inference"),
            tmp_path,
        )
    )
    probe = (
        '\necho "UV=${UV_CACHE_DIR}"\necho "TMP=${TMPDIR}"'
        '\necho "PYI=${UV_PYTHON_INSTALL_DIR}"\necho "MPL=${MPLCONFIGDIR}"\necho "XDG=${XDG_CACHE_HOME}"'
        '\necho "TRITON=${TRITON_CACHE_DIR}"\n'
    )

    poison = "/poison/shared/scratch"
    env = {
        "PATH": os.environ["PATH"],
        "USER": "tester",
        "UV_CACHE_DIR": f"{poison}/uv",  # the real-world footgun
        "TMPDIR": f"{poison}/tmp",
        "UV_PYTHON_INSTALL_DIR": f"{poison}/python",  # leaked $HOME interpreter dir
        "MPLCONFIGDIR": f"{poison}/matplotlib",
        "XDG_CACHE_HOME": f"{poison}/xdg",
        "TRITON_CACHE_DIR": f"{poison}/triton",
        # SLURM_JOB_ID unset -> resolution falls through to the /tmp branch.
    }
    out = _run_path_resolution_prefix(block, probe, env)
    resolved = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    assert poison not in resolved["UV"]
    assert poison not in resolved["TMP"]
    assert resolved["UV"].endswith("/uv")
    # The interpreter + startup caches must shed leaked shared-storage paths too.
    assert poison not in resolved["PYI"] and resolved["PYI"].endswith("/python")
    assert poison not in resolved["MPL"] and resolved["MPL"].endswith("/matplotlib")
    assert poison not in resolved["XDG"] and resolved["XDG"].endswith("/xdg")
    assert poison not in resolved["TRITON"] and resolved["TRITON"].endswith("/triton")


def test_node_local_cache_root_is_per_user_not_per_project(tmp_path: Path):
    """The cache root must depend on the uid alone: its contents are shared
    across projects, so a per-project root costs every new project a cold
    download of wheels a sibling already holds. ``TAAF_NODE_CACHE_ROOT`` is the
    escape hatch for pointing it elsewhere."""
    scripts = [
        _sbatch_script_for(
            taaf.deploy_slurm.TufaSlurmTarget(
                gpu="CPU", gpu_count=0, time="00:10:00", main_project=project, node_local_storage=True
            ),
            tmp_path / project,
        )
        for project in ("ARC3-Inference", "sandbox")
    ]
    # The rendered root is unexpanded template text, so the guard lives at
    # template level: the probe targets the uid-only path and the
    # project-suffixed root appears nowhere. Executing the /cache/users
    # branch is out of reach for a test — the path is fixed and root-owned.
    for script in scripts:
        assert 'elif mkdir -p "/cache/users/${taaf_uid}" 2>/dev/null' in script
        assert "/cache/users/${taaf_uid}/${taaf_storage_name}" not in script

    # The override wins over the probe, so a launcher can redirect the root.
    block = "\n".join(
        taaf.deploy_slurm._render_node_local_install_block(
            taaf.deploy_slurm.TufaSlurmTarget(gpu="CPU", gpu_count=0, time="00:10:00", main_project="ARC3-Inference"),
            tmp_path,
        )
    )
    out = _run_path_resolution_prefix(
        block,
        '\necho "UV=${UV_CACHE_DIR}"\n',
        {"PATH": os.environ["PATH"], "USER": "tester", "TAAF_NODE_CACHE_ROOT": str(tmp_path / "shared-cache")},
    )
    assert f"UV={tmp_path / 'shared-cache'}/uv" in out


def test_worker_removes_a_managed_interpreter_that_no_longer_runs(tmp_path: Path):
    """A node's /cache keeps uv's managed interpreter across jobs, and a
    cleanup took ``bin/``, ``lib/`` and ``include/`` from one and left
    ``share/``: uv took it as installed and every job on that node died in a
    second writing into the missing ``lib/``. Execute the heal block over a
    gutted interpreter, a working one, and a gutted one young enough to be
    another job's install still in progress."""
    py = tmp_path / "python"
    gutted = py / "cpython-3.12.12-linux-x86_64-gnu"
    (gutted / "share").mkdir(parents=True)
    alias = py / "cpython-3.12-linux-x86_64-gnu"
    alias.symlink_to(gutted)
    working = py / "cpython-3.13.1-linux-x86_64-gnu"
    (working / "bin").mkdir(parents=True)
    interpreter = working / "bin" / "python3"
    interpreter.write_text("#!/bin/sh\nexit 0\n")
    interpreter.chmod(0o755)
    in_progress = py / "cpython-3.11.9-linux-x86_64-gnu"
    (in_progress / "share").mkdir(parents=True)
    an_hour_ago = time.time() - 3600
    for directory in (gutted, working):
        os.utime(directory, (an_hour_ago, an_hour_ago))

    block = "\n".join(taaf.deploy_slurm._render_managed_python_heal_block(indent=""))
    subprocess.run(
        ["bash", "-c", block],
        env={"PATH": os.environ["PATH"], "UV_PYTHON_INSTALL_DIR": str(py)},
        capture_output=True,
        text=True,
        check=True,
    )

    assert not gutted.exists(), "the gutted interpreter is removed, so the sync reinstalls it"
    assert not alias.is_symlink(), "and the version alias left pointing at nothing"
    assert interpreter.is_file(), "an interpreter that runs is kept"
    assert in_progress.is_dir(), "a young directory may be an install in progress"


def test_worker_drops_uv_binary_that_does_not_resolve(tmp_path: Path):
    """``/etc/kander/job-env.sh`` exports ``UV`` as an absolute host path.
    Under a container image that path is outside it, so a project Makefile doing
    ``UV ?= uv`` inherits a dead binary and the run dies at install time.
    Execute the block with a dangling and with a live ``UV``."""
    block = "\n".join(taaf.deploy_slurm._render_uv_binary_sanitize_block())
    probe = block + '\necho "UV=${UV:-<unset>}"\n'
    base_env = {"PATH": os.environ["PATH"]}

    dangling = subprocess.run(
        ["bash", "-c", probe],
        env={**base_env, "UV": str(tmp_path / "opt-tufa" / "bin" / "uv")},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert dangling.strip() == "UV=<unset>"

    live = tmp_path / "uv"
    live.write_text("#!/bin/sh\n")
    live.chmod(0o755)
    kept = subprocess.run(
        ["bash", "-c", probe],
        env={**base_env, "UV": str(live)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert kept.strip() == f"UV={live}"


def test_sbatch_script_sanitizes_uv_after_sourcing_cluster_job_env(tmp_path: Path):
    """The dangling ``UV`` arrives from ``/etc/kander/job-env.sh``, which the
    node-local install block sources — so the guard has to run after it, not
    with the rest of the uv bootstrap."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="CPU",
            gpu_count=0,
            time="00:10:00",
            main_project="ARC3-Inference",
            node_local_storage=True,
        ),
        tmp_path,
    )

    assert script.index(". /etc/kander/job-env.sh") < script.index("    unset UV\n")


def test_main_project_defaults_to_what_the_snapshot_bundles():
    """The default must be the checkout's own directory name, because that
    is the name :func:`snapshot_editable_sources` bundles it under and the
    worker ``uv sync``s from ``src/<main_project>``.

    Requiring the caller to pass it was the older contract, and it made the
    failure worse rather than better: a literal is right until the checkout
    is renamed or a worktree checks the same repo out under a branch name,
    and then the worker syncs a path that does not exist. Deriving both
    sides from one source is what makes them unable to disagree."""
    target = taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00")
    assert target.main_project == REPO_DIR
    # Still overridable, for a caller bundling something else as the root.
    assert taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00", main_project="other").main_project == "other"


async def test_deploy_bundles_every_editable_repo_via_auto_discovery(tmp_path: Path, monkeypatch):
    """Every editable distribution in the launcher's venv lands
    in ``job_dir/src/``. The mechanism (``_discover_editable_repos``)
    walks ``importlib.metadata`` and parses PEP 610 ``direct_url.json``,
    so monkeypatching the discovery is the right seam for unit tests —
    we don't have to install fixture packages into the running venv."""
    # Build a minimal "downstream" repo on disk so the snapshot helper
    # has something concrete to copy. In production, this would be a
    # ``uv pip install -e ../downstream-sandbox`` in the launcher's venv.
    sibling = tmp_path / "fake_sibling"
    (sibling / "fake_sibling").mkdir(parents=True)
    (sibling / "pyproject.toml").write_text(
        '[project]\nname = "fake-sibling"\nversion = "0.1.0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["fake_sibling*"]\n'
    )
    (sibling / "fake_sibling" / "__init__.py").write_text("")

    # Pretend ``importlib.metadata`` sees one editable install at the fixture
    # path. The real discovery would also surface TAAF + re-arc-3 from the
    # pytest venv, but this test cares about the discovery seam, not the
    # full launcher venv.
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [sibling])

    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)

    job_dir = tmp_path / "job"
    bm = _benchmark(job_dir)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))

    # The auto-discovered editable lands in the snapshot.
    _extracted_src(job_dir)
    assert (job_dir / "src" / "fake_sibling" / "pyproject.toml").is_file()
    assert (job_dir / "src" / "fake_sibling" / "fake_sibling" / "__init__.py").is_file()


async def test_deploy_bundles_extra_source_repos(tmp_path: Path, monkeypatch):
    """``TufaSlurmTarget.extra_source_repos`` lets downstream launchers
    force additional local repos into the worker snapshot even if those repos
    are not installed editable in the launcher's venv."""
    sibling = tmp_path / "forced_sibling"
    (sibling / "forced_sibling").mkdir(parents=True)
    (sibling / "pyproject.toml").write_text(
        '[project]\nname = "forced-sibling"\nversion = "0.1.0"\n'
        '[tool.setuptools.packages.find]\ninclude = ["forced_sibling*"]\n'
    )
    (sibling / "forced_sibling" / "__init__.py").write_text("")
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", list)

    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)

    job_dir = tmp_path / "job"
    bm = _benchmark(job_dir)
    await bm.deploy(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            extra_source_repos=[sibling],
        )
    )

    _extracted_src(job_dir)
    assert (job_dir / "src" / "forced_sibling" / "pyproject.toml").is_file()
    assert (job_dir / "src" / "forced_sibling" / "forced_sibling" / "__init__.py").is_file()


def test_sbatch_script_uses_uv_sync_when_main_project_set(tmp_path: Path):
    """When the launcher comes from a downstream project whose pyproject
    uses ``[tool.uv.sources]`` to rewrite abstract dep names to bundled
    siblings (e.g. ``arc-agi-3-local = { path = "../re-arc-3" }``), the
    worker must use ``uv sync`` to honor those rewrites — ``uv pip install
    -e`` ignores ``[tool.uv.sources]`` and falls through to PyPI / git."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            main_project="sandbox",
            node_local_storage=False,
        ),
        tmp_path,
    )
    job_dir = tmp_path.resolve()
    assert f'export UV_PROJECT_ENVIRONMENT="{job_dir}/.venv"' in script
    assert f"cd {job_dir}/src/sandbox" in script
    assert "uv sync" in script
    # And the fallback editable-loop path is NOT in this branch.
    assert 'EDITABLE_ARGS=""' not in script
    assert 'uv pip install --python "$(pwd)/.venv" $EDITABLE_ARGS' not in script


@pytest.mark.local
@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="install.sh uses flock, which ships only on Linux (the SLURM target); not on macOS",
)
def test_main_project_install_block_actually_installs_bundled_siblings(tmp_path: Path):
    """End-to-end regression test for the ``main_project`` install path.
    Template-rendering tests above assert the bash string
    contains ``uv sync`` etc., but would still pass if ``uv sync``
    itself broke against the bundled layout — they don't *execute* the
    install. This test does.

    Setup mimics a downstream project (``sandbox``-shaped)
    pulling a sibling via ``[tool.uv.sources]``:

    - ``downstream-pkg`` declares ``sibling-dep`` as a dependency.
    - ``[tool.uv.sources] sibling-dep = { path = "../sibling-pkg" }``
      redirects that abstract name to a bundled sibling.
    - ``sibling-pkg`` ships ``sibling_pkg`` with a ``SIGIL`` constant.
    - ``downstream-pkg`` ships ``downstream_pkg`` that imports the
      sibling and surfaces ``HELLO``.

    The test: snapshot both fixtures, render the install block via
    ``_render_install_block(main_project="downstream-pkg")``, execute
    it as bash, then import the downstream package from the resulting
    ``.venv`` and assert the import chain works.

    Slow (~10s — network for setuptools fetch). Marked ``slow`` so
    ``make prepare`` skips it; ``make test-all`` runs it. Network and
    a working ``uv`` on ``$PATH`` required."""
    fixtures = tmp_path / "fixtures"
    fixtures.mkdir()

    # Sibling fixture: distribution name "sibling-dep", module name "sibling_pkg".
    # The dist/module mismatch mirrors how arc-agi-3-local distributes re_arc.
    sibling = fixtures / "sibling-pkg"
    (sibling / "sibling_pkg").mkdir(parents=True)
    (sibling / "pyproject.toml").write_text(
        "[project]\n"
        'name = "sibling-dep"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.12"\n'
        "\n"
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[tool.setuptools.packages.find]\n"
        'include = ["sibling_pkg*"]\n'
    )
    (sibling / "sibling_pkg" / "__init__.py").write_text('SIGIL = "sibling-loaded"\n')

    # Downstream fixture: depends on sibling-dep, redirected via [tool.uv.sources].
    downstream = fixtures / "downstream-pkg"
    (downstream / "src" / "downstream_pkg").mkdir(parents=True)
    (downstream / "pyproject.toml").write_text(
        "[project]\n"
        'name = "downstream-pkg"\n'
        'version = "0.1.0"\n'
        'requires-python = ">=3.12"\n'
        'dependencies = ["sibling-dep"]\n'
        "\n"
        "[build-system]\n"
        'requires = ["setuptools>=61"]\n'
        'build-backend = "setuptools.build_meta"\n'
        "\n"
        "[tool.uv.sources]\n"
        'sibling-dep = { path = "../sibling-pkg" }\n'
        "\n"
        "[tool.setuptools.packages.find]\n"
        'where = ["src"]\n'
        'include = ["downstream_pkg*"]\n'
    )
    (downstream / "src" / "downstream_pkg" / "__init__.py").write_text(
        'from sibling_pkg import SIGIL\nHELLO = f"downstream + {SIGIL}"\n'
    )

    # Snapshot both fixtures into the job dir under src/, mimicking what
    # TufaSlurmTarget.deploy does on the launcher. Auto-discovery walks
    # ``importlib.metadata`` of the running venv; for this test we
    # monkeypatch it to return our two fixture paths (we don't want to
    # install fixture packages into the pytest venv).
    job_dir = (tmp_path / "job").resolve()
    job_dir.mkdir()
    import unittest.mock

    with (
        unittest.mock.patch.object(taaf.deploy, "_discover_editable_repos", return_value=[downstream, sibling]),
        unittest.mock.patch.object(taaf.deploy, "_discover_noneditable_tufa_packages", return_value=[]),
    ):
        taaf.deploy.snapshot_editable_sources(job_dir / "src")
    assert (job_dir / "src" / "downstream-pkg" / "pyproject.toml").is_file()
    assert (job_dir / "src" / "sibling-pkg" / "pyproject.toml").is_file()

    # Render the install block via the production helper, so any drift
    # in the production renderer breaks this test (rather than the test
    # passing while production silently regresses).
    target = taaf.deploy_slurm.TufaSlurmTarget(
        gpu="B200", time="01:00:00", main_project="downstream-pkg", node_local_storage=False
    )
    install_block = taaf.deploy_slurm._render_install_block(target, job_dir)
    script = "\n".join(["set -euo pipefail", *install_block]) + "\n"
    script_path = job_dir / "install.sh"
    script_path.write_text(script)

    result = subprocess.run(
        ["bash", str(script_path)],
        capture_output=True,
        text=True,
        cwd=job_dir,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"install_block failed (rc={result.returncode})\n"
        f"--- script ---\n{script}\n"
        f"--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )

    # Now exercise the resulting venv: import both packages and verify
    # the cross-package linkage. If uv sync had picked sibling-dep from
    # PyPI instead of honoring [tool.uv.sources], the import would either
    # fail (no such package on PyPI) or succeed but yield a different
    # SIGIL. Either case fails this assertion.
    verify = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{job_dir}/.venv/bin/activate" && python -c "import downstream_pkg; print(downstream_pkg.HELLO)"',
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert verify.returncode == 0, f"verify failed:\nstdout={verify.stdout}\nstderr={verify.stderr}"
    assert "downstream + sibling-loaded" in verify.stdout


async def test_snapshot_includes_uv_lock_when_present(tmp_path: Path, monkeypatch):
    """uv.lock is not importable source, but the worker's ``uv sync``
    benefits from a frozen resolve — include it when present."""
    fake = tmp_path / "fake_repo"
    (fake / "pkg").mkdir(parents=True)
    (fake / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["pkg*"]\n'
    )
    (fake / "pkg" / "__init__.py").write_text("")
    (fake / "uv.lock").write_text("# fake lock\n")

    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])

    job_dir = tmp_path / "job"
    bm = _benchmark(job_dir)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    _extracted_src(job_dir)
    assert (job_dir / "src" / "fake_repo" / "uv.lock").is_file()


async def test_slurm_src_tar_omits_installed_tufa_packages(tmp_path: Path, monkeypatch):
    """deploy() keeps installed-package bundles out of src.tar.

    The worker ``uv sync``s from the bundled main_project's lock, so it
    resolves these itself; bundled, the vLLM fork alone would put hundreds
    of MB on every job's NFS write.
    """
    fake = tmp_path / "fake_repo"
    (fake / "pkg").mkdir(parents=True)
    (fake / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["pkg*"]\n'
    )
    (fake / "pkg" / "__init__.py").write_text("")
    installed = tmp_path / "site-packages" / "vllm"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("# installed\n")

    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])
    monkeypatch.setattr(
        taaf.deploy, "_discover_noneditable_tufa_packages", lambda: [("arc-prize-2026-vllm", [installed])]
    )

    job_dir = tmp_path / "job"
    bm = _benchmark(job_dir)
    await bm.deploy(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00", main_project="tufa-arc-agi-framework")
    )
    _extracted_src(job_dir)
    assert (job_dir / "src" / "fake_repo" / "pyproject.toml").is_file()
    assert not (job_dir / "src" / "arc-prize-2026-vllm").exists()


async def test_snapshot_includes_project_support_files(tmp_path: Path, monkeypatch):
    """Slurm workers may need repo-local commands/configs after snapshot.
    ARC3-Inference starts its local server via ``make server`` from the
    bundled repo, so the snapshot must include the Makefile plus the
    config/script directories that target uses."""
    fake = tmp_path / "fake_repo"
    (fake / "pkg").mkdir(parents=True)
    (fake / "configs").mkdir()
    (fake / "scripts").mkdir()
    (fake / "pyproject.toml").write_text(
        '[project]\nname = "pkg"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["pkg*"]\n'
    )
    (fake / "pkg" / "__init__.py").write_text("")
    (fake / "Makefile").write_text("server:\n\t@true\n")
    (fake / "configs" / "inference.json").write_text("{}\n")
    (fake / "scripts" / "server_curl.sh").write_text("#!/bin/sh\n")

    rec = _SubprocessRecorder({"sbatch": ("1234567\n", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [fake])

    job_dir = tmp_path / "job"
    bm = _benchmark(job_dir)
    await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    _extracted_src(job_dir)
    snapshot = job_dir / "src" / "fake_repo"
    assert (snapshot / "Makefile").is_file()
    assert (snapshot / "configs" / "inference.json").is_file()
    assert (snapshot / "scripts" / "server_curl.sh").is_file()


def test_is_tufa_url_matches_tufalabs_only():
    """Non-editable bundling keys on a github.com/Tufalabs VCS URL (any
    scheme/case). Non-Tufa URLs must not match, else we'd try to copy
    PyPI deps like numpy into the bundle."""
    assert taaf.deploy._is_tufa_url("https://github.com/Tufalabs/arc-prize-2026.git")
    assert taaf.deploy._is_tufa_url("git+ssh://git@github.com/tufalabs/arc-prize-2026.git")
    assert not taaf.deploy._is_tufa_url("https://github.com/other-org/pkg.git")
    assert not taaf.deploy._is_tufa_url("https://files.pythonhosted.org/numpy.whl")
    assert taaf.deploy._repo_name_from_url("https://github.com/Tufalabs/arc-prize-2026.git") == "arc-prize-2026"


def test_snapshot_bundles_noneditable_tufa_package(tmp_path: Path, monkeypatch):
    """A non-editable Tufa package has no checkout, so the snapshot copies
    its installed top-level package tree into ``<repo_name>/<pkg>/`` — and
    only the package code (no pyproject/Makefile)."""
    pkg = tmp_path / "site-packages" / "re_arc"
    (pkg / "environment_files").mkdir(parents=True)
    (pkg / "__init__.py").write_text("VERSION = 'pinned'\n")
    (pkg / "environment_files" / "ls20.py").write_text("# env\n")

    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", list)
    monkeypatch.setattr(taaf.deploy, "_discover_noneditable_tufa_packages", lambda: [("re-arc-3", [pkg])])

    dest = tmp_path / "src"
    written = taaf.deploy.snapshot_editable_sources(dest)

    assert dest / "re-arc-3" in [Path(p) for p in written]
    assert (dest / "re-arc-3" / "re_arc" / "__init__.py").is_file()
    assert (dest / "re-arc-3" / "re_arc" / "environment_files" / "ls20.py").is_file()
    assert not (dest / "re-arc-3" / "pyproject.toml").exists()


def test_snapshot_can_drop_noneditable_tufa_packages(tmp_path: Path, monkeypatch):
    """``include_noneditable_tufa=False`` keeps installed-package
    bundles out while the editable repos stay. The Slurm worker resolves
    its own dependencies, so an installed tree is never read there — and
    the vLLM fork alone is hundreds of MB on every job's NFS write."""
    editable = tmp_path / "fake_repo"
    (editable / "fake_pkg").mkdir(parents=True)
    (editable / "pyproject.toml").write_text(
        '[project]\nname = "fake-repo"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["fake_pkg*"]\n'
    )
    (editable / "fake_pkg" / "__init__.py").write_text("# editable\n")
    noned = tmp_path / "site-packages" / "vllm"
    noned.mkdir(parents=True)
    (noned / "__init__.py").write_text("# installed\n")

    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [editable])
    monkeypatch.setattr(taaf.deploy, "_discover_noneditable_tufa_packages", lambda: [("arc-prize-2026-vllm", [noned])])

    dest = tmp_path / "src"
    written = taaf.deploy.snapshot_editable_sources(dest, include_noneditable_tufa=False)

    assert (dest / "fake_repo" / "pyproject.toml").is_file()
    assert not (dest / "arc-prize-2026-vllm").exists()
    assert dest / "arc-prize-2026-vllm" not in [Path(p) for p in written]


def test_snapshot_editable_wins_over_noneditable_same_name(tmp_path: Path, monkeypatch):
    """If a repo is bundled editable, a non-editable dist of the same name
    is skipped — the full editable repo (with pyproject) is strictly
    better than a package-only copy."""
    editable = tmp_path / "re-arc-3"
    (editable / "re_arc").mkdir(parents=True)
    (editable / "pyproject.toml").write_text(
        '[project]\nname = "re-arc-3"\nversion = "0.1.0"\n[tool.setuptools.packages.find]\ninclude = ["re_arc*"]\n'
    )
    (editable / "re_arc" / "__init__.py").write_text("# editable\n")

    noned = tmp_path / "site-packages" / "re_arc"
    noned.mkdir(parents=True)
    (noned / "__init__.py").write_text("# non-editable\n")

    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", lambda: [editable])
    monkeypatch.setattr(taaf.deploy, "_discover_noneditable_tufa_packages", lambda: [("re-arc-3", [noned])])

    dest = tmp_path / "src"
    taaf.deploy.snapshot_editable_sources(dest)

    assert (dest / "re-arc-3" / "pyproject.toml").is_file()
    assert (dest / "re-arc-3" / "re_arc" / "__init__.py").read_text() == "# editable\n"


def test_format_git_status_lists_noneditable_tufa_repos(monkeypatch):
    """Non-editable Tufa repos appear in the git-status overview with their
    pinned commit (no checkout to read clean/DIRTY from)."""
    monkeypatch.setattr(taaf.deploy, "_discover_editable_repos", list)
    monkeypatch.setattr(taaf.deploy, "_noneditable_tufa_git_lines", lambda: [("re-arc-3", "793679eef5", "master")])
    status = taaf.deploy.format_git_status()
    assert "re-arc-3" in status
    assert "793679eef5" in status
    assert "pinned" in status


def test_sbatch_script_prepends_user_local_bin(tmp_path: Path):
    """The worker shell is non-login; Slurm propagates the submitter's PATH.
    If the launcher itself wasn't a login shell, ~/.local/bin (where ``uv``
    typically lives) isn't on PATH and the venv setup fails with
    ``uv: command not found``. The rendered script prepends ~/.local/bin
    unconditionally so the deploy is robust to how the launcher was run."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"),
        tmp_path,
    )
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script


def test_sbatch_script_pins_mplbackend_to_agg(tmp_path: Path):
    """Slurm propagates the submitter's env to the worker. A launcher
    running inside a Jupyter / VS Code Python kernel typically has
    ``MPLBACKEND=module://matplotlib_inline.backend_inline`` set —
    that backend isn't installed in the worker venv and matplotlib
    refuses to import. The worker runs headless so ``Agg`` is the
    right override; the rendered script must set it unconditionally
    so the worker doesn't depend on a clean launcher env."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"),
        tmp_path,
    )
    assert "export MPLBACKEND=Agg" in script


def test_sbatch_script_bootstraps_uv_when_missing(tmp_path: Path):
    """Inside a pyxis container the host's ~/.local/bin isn't mounted and
    the base image (e.g. nvcr.io/nvidia/pytorch) doesn't ship ``uv``. The
    rendered script must guard against this with a ``command -v uv`` check
    plus a fallback installer; the host path remains a no-op because uv is
    already on PATH there."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            image="nvcr.io/nvidia/pytorch:25.01-py3",
        ),
        tmp_path,
    )
    assert "command -v uv" in script
    assert "astral.sh/uv/install.sh" in script


def test_sbatch_script_runs_bare_without_image(tmp_path: Path):
    """No image means no srun --container-image wrapping."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00", image=None),
        tmp_path,
    )
    assert "--container-image" not in script
    assert "exec python -u run_in_worker.py" in script


def test_sbatch_script_wraps_in_srun_when_image_set(tmp_path: Path):
    """Image set → srun --container-image=... bash -c '<work>'."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            image="nvcr.io/nvidia/pytorch:25.01-py3",
            extra_mounts=[("/data", "/data")],
        ),
        tmp_path,
    )
    job_dir = tmp_path.resolve()
    assert "srun --container-image=" in script
    assert "nvcr.io/nvidia/pytorch:25.01-py3" in script
    assert f"{job_dir}:{job_dir}" in script  # default job_dir mount
    assert "/data:/data" in script  # extra mount


def test_a_container_step_overlaps_the_allocation_too(tmp_path: Path):
    """Under ``jobid`` the outer step already holds the allocation's
    resources, and ``--overlap`` does not carry from a step to the steps it
    starts. Without its own, the container step waits for resources the run
    it is supposed to be is holding, while deploy() reports success."""
    for_allocation = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            image="nvcr.io/nvidia/pytorch:25.01-py3",
            main_project="tufa-arc-agi-framework",
            jobid="2554274",
        ),
        tmp_path,
    )
    for_batch = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            image="nvcr.io/nvidia/pytorch:25.01-py3",
            main_project="tufa-arc-agi-framework",
        ),
        tmp_path,
    )
    assert "srun --overlap --container-image=" in for_allocation
    assert "srun --container-image=" in for_batch


def test_sbatch_script_passes_through_partition_and_nodelist(tmp_path: Path):
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            partition="gpu-high",
            nodelist="h200-01",
        ),
        tmp_path,
    )
    assert "#SBATCH --partition=gpu-high" in script
    assert "#SBATCH --nodelist=h200-01" in script


def test_sbatch_script_divides_totals_with_ceiling_across_gpus(tmp_path: Path):
    """Totals are divided by ``gpu_count`` with ceiling rounding so the
    resulting per-gpu allocation ≥ the user's ask. Uses an odd CPU
    total (17 / 4 → 5 per-gpu = 20 total) and a memory total whose
    MiB-per-gpu quotient is not exact to exercise both ceilings."""
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            gpu_count=4,
            total_cpus=17,
            total_mem_bytes=64 * 1024**3 + 1,
        ),
        tmp_path,
    )
    assert "#SBATCH --cpus-per-gpu=5" in script
    assert "#SBATCH --mem-per-gpu=16385M" in script


def test_sbatch_script_includes_extra_sbatch_flags(tmp_path: Path):
    script = _sbatch_script_for(
        taaf.deploy_slurm.TufaSlurmTarget(
            gpu="B200",
            time="01:00:00",
            extra_sbatch_flags=["--qos=high", "--account=tufa"],
        ),
        tmp_path,
    )
    assert "#SBATCH --qos=high" in script
    assert "#SBATCH --account=tufa" in script


# --- TufaSlurmHandle: is_done / wait / stop ------------------------------


def test_handle_is_done_true_when_squeue_empty(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder({"squeue": ("", 0)}))
    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567")
    assert handle.is_done is True


def test_handle_is_done_false_when_squeue_lists_job(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder({"squeue": ("RUNNING\n", 0)}))
    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567")
    assert handle.is_done is False


def test_handle_stop_runs_scancel_with_sigusr1(tmp_path: Path, monkeypatch):
    """Graceful stop — ``stop()`` sends SIGUSR1 (not bare SIGTERM) so the
    worker's signal handler can route through ``Benchmark.run``'s teardown
    path. Bare scancel is left as the explicit hard-kill escape hatch.

    The scope of the signal is the load-bearing part, and both other
    choices are actively wrong. ``--full`` signals the batch step's
    CHILDREN too, and SIGUSR1's default disposition is Terminate — a
    solver's per-game workers and inference server install no handler, so
    they are killed outright and teardown runs against a gutted job.
    Bare ``--signal`` skips the batch step, which for the non-image
    rendering is where the work actually runs. ``--batch`` hits the batch
    step alone, which ``exec python -u`` has made the worker itself.
    """
    rec = _SubprocessRecorder({"scancel": ("", 0)})
    monkeypatch.setattr(subprocess, "run", rec)
    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567")
    handle.stop()
    scancel_calls = [c for c in rec.calls if c["argv"][0] == "scancel"]
    assert len(scancel_calls) == 1
    assert scancel_calls[0]["argv"] == ["scancel", "--signal=USR1", "--batch", "1234567"]
    # --full would take the solver's children with it; --batch must not appear
    # alongside it either, since scancel ignores --batch when --full is given.
    assert "--full" not in scancel_calls[0]["argv"]


def test_worker_script_installs_sigusr1_handler(tmp_path: Path):
    """The rendered ``run_in_worker.py`` must register an asyncio signal
    handler that calls ``bm.request_stop`` on SIGUSR1, so ``handle.stop()``'s
    ``scancel --signal=USR1`` actually triggers graceful teardown. It must
    also swallow the re-raised ``CancelledError`` after teardown so the
    worker exits 0 and Slurm reports ``COMPLETED`` rather than ``FAILED``
    for what is by design a successful graceful stop."""
    script = taaf.deploy_slurm._render_worker_script(tmp_path, preamble="")
    assert "import signal" in script
    assert "add_signal_handler(signal.SIGUSR1, bm.request_stop)" in script
    assert "except asyncio.CancelledError:" in script


def test_worker_soft_end_uses_experiment_runtime_before_slurm_walltime():
    now = datetime(2026, 5, 13, 9, 30, 0)
    slurm_end_epoch = str(int((now + timedelta(hours=4)).timestamp()))

    soft_end = taaf.deploy_slurm.worker_soft_end_time(
        slurm_end_epoch=slurm_end_epoch,
        max_runtime_s=3 * 60 * 60,
        now=now,
    )

    assert soft_end == now + timedelta(hours=3)


def test_worker_soft_end_still_respects_slurm_walltime_buffer():
    now = datetime(2026, 5, 13, 9, 30, 0)
    slurm_end_epoch = str(int((now + timedelta(hours=2)).timestamp()))

    soft_end = taaf.deploy_slurm.worker_soft_end_time(
        slurm_end_epoch=slurm_end_epoch,
        max_runtime_s=3 * 60 * 60,
        now=now,
    )

    assert soft_end == now + timedelta(hours=1, minutes=50)


@pytest.mark.local
def test_bash_exec_routes_sigusr1_to_python_directly(tmp_path: Path):
    """Behavioural test for the ``exec python -u`` line in the
    rendered work block. Spawn a bash subprocess that ``exec``s a python
    program with a SIGUSR1 handler; send SIGUSR1 to the bash PID; assert
    python received it (wrote a sentinel file).

    Without ``exec`` the equivalent setup would have bash terminate
    (default SIGUSR1 action = Terminate) and orphan python, which never
    sees the signal. The audit on tufaslurm showed this is exactly how
    ``handle.stop()`` was silently failing before this fix. This test
    locks in the ``exec`` mechanism end-to-end on the local machine —
    same shell semantics as the cluster, no Slurm needed.

    Slow because spawning subprocesses + a small wait for python startup
    pushes us past the cheap-unit-test boundary. Still under 1s on a
    laptop.
    """
    import os
    import signal
    import time

    sentinel = tmp_path / "saw_sigusr1"
    python_src = tmp_path / "child.py"
    python_src.write_text(
        "import asyncio, signal, sys\n"
        f"SENTINEL = {str(sentinel)!r}\n"
        "async def main():\n"
        "    fired = asyncio.Event()\n"
        "    loop = asyncio.get_running_loop()\n"
        "    loop.add_signal_handler(signal.SIGUSR1, fired.set)\n"
        "    print('ready', flush=True)\n"
        "    try:\n"
        "        await asyncio.wait_for(fired.wait(), timeout=10)\n"
        "    except asyncio.TimeoutError:\n"
        "        sys.exit(2)\n"
        "    with open(SENTINEL, 'w') as f:\n"
        "        f.write('signal received')\n"
        "asyncio.run(main())\n"
    )
    bash_src = tmp_path / "wrapper.sh"
    # Use the running interpreter explicitly — the spawned bash inherits its
    # parent's PATH but in some environments ``python`` is only reachable
    # as ``python3`` or via the venv's full path.
    bash_src.write_text(f"#!/bin/bash\nset -euo pipefail\nexec {sys.executable} -u {python_src}\n")
    bash_src.chmod(0o755)

    proc = subprocess.Popen([str(bash_src)], stdout=subprocess.PIPE, text=True)
    try:
        # Wait for python to print 'ready' so its signal handler is installed.
        assert proc.stdout is not None
        line = proc.stdout.readline().strip()
        assert line == "ready", f"unexpected child output: {line!r}"
        # The pid we have is bash's — exec has now replaced bash with python
        # at that same pid. Sending SIGUSR1 here lands directly on python.
        os.kill(proc.pid, signal.SIGUSR1)
        rc = proc.wait(timeout=15)
        assert rc == 0, f"child exited non-zero ({rc}); SIGUSR1 was not delivered"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=2)
        # Popen does not close the pipe it opened; leaving it to the garbage
        # collector raises an unraisable ResourceWarning at an unrelated
        # point in the suite.
        if proc.stdout is not None:
            proc.stdout.close()
    # Give the OS a beat to flush the file (the python wrote it before exit,
    # but on rare schedulers we've seen a race between exit and fs flush).
    for _ in range(20):
        if sentinel.is_file():
            break
        time.sleep(0.05)
    assert sentinel.is_file(), "python's SIGUSR1 handler did not run"
    assert sentinel.read_text() == "signal received"


def test_handle_stop_swallows_nonzero_scancel(tmp_path: Path, monkeypatch):
    """Calling stop() on an already-terminated job should not raise."""
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder({"scancel": ("", 1)}))
    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567")
    handle.stop()  # would raise if check=True


def test_handle_wait_polls_then_loads_snapshot(tmp_path: Path, monkeypatch):
    """wait() should poll squeue until empty, then load benchmark.json."""
    states = iter(["RUNNING\n", "RUNNING\n", ""])

    def fake_run(argv, **kwargs):
        cmd = argv[0]
        if cmd == "squeue":
            return subprocess.CompletedProcess(argv, 0, next(states), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Create a fake benchmark.json that wait() will load on completion.
    bm = _benchmark(tmp_path)
    bm.job_dir = tmp_path
    bm._save_json()

    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567", poll_interval_s=0.0)
    loaded = handle.wait()
    assert loaded.label == "test_slurm"


def test_handle_wait_errors_if_snapshot_missing_after_done(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _SubprocessRecorder({"squeue": ("", 0)}))
    handle = taaf.deploy_slurm.TufaSlurmHandle(job_dir=tmp_path, job_id="1234567", poll_interval_s=0.0)
    with pytest.raises(RuntimeError, match="no benchmark.json"):
        handle.wait()


# --- DeploymentHandle.attach() for Slurm ---------------------------------


async def test_attach_reconstructs_slurm_handle(tmp_path: Path, stub_subprocess):
    bm = _benchmark(tmp_path)
    handle = await bm.deploy(taaf.deploy_slurm.TufaSlurmTarget(gpu="B200", time="01:00:00"))
    job_id = handle.job_id
    job_dir = handle.job_dir
    del handle, bm  # simulate launcher process exit

    reattached = taaf.deploy.DeploymentHandle.attach(job_dir)
    assert isinstance(reattached, taaf.deploy_slurm.TufaSlurmHandle)
    assert reattached.job_id == job_id
    assert reattached.job_dir == job_dir


def test_attach_errors_when_meta_missing_job_id(tmp_path: Path):
    # Hand-craft a meta with handle_class=TufaSlurmHandle but no job_id.
    meta = {
        "schema_version": taaf.deploy.DEPLOY_META_SCHEMA_VERSION,
        "target_class": "taaf.deploy_slurm.TufaSlurmTarget",
        "target_config": {},
        "handle_class": "taaf.deploy_slurm.TufaSlurmHandle",
        "job_id": None,
        "benchmark_label": "broken",
        "started_at": "2026-05-11T13:42:04",
        "host": "test",
        "pid": 0,
    }
    (tmp_path / "deploy_meta.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="job_id"):
        taaf.deploy.DeploymentHandle.attach(tmp_path)


def test_uv_sync_gets_all_packages_flag_for_workspace_main_project():
    """The worker detects a uv-workspace main_project from
    its bundled pyproject and syncs with --all-packages (a plain sync of
    a virtual workspace root would install nothing)."""
    target = taaf.deploy_slurm.TufaSlurmTarget(gpu="CPU", gpu_count=0, main_project="wsrepo")
    rendered = "\n".join(taaf.deploy_slurm._render_uv_sync_args_block(target))
    assert "[tool.uv.workspace]" in rendered
    assert "--all-packages" in rendered


# --- Deploying into an existing allocation ---------------------------------


class _FakeLauncher:
    """Stands in for the detached ``srun``; records what was done to it."""

    def __init__(self) -> None:
        self.signals: list[str] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.signals.append("terminate")
        self.returncode = -15

    def kill(self) -> None:
        self.signals.append("kill")
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.signals.append("wait")
        return self.returncode or 0


class _PopenRecorder:
    """Capture ``subprocess.Popen`` calls; the step is never really started.

    With ``squeue_stub`` set, a launch registers the step it asked for in
    that stub's queue listing, the way srun would; without one the step
    never appears and the caller runs into its registration timeout.
    """

    def __init__(self, squeue_stub=None, step_id: str = "2554274.1") -> None:
        self.calls: list[dict] = []
        self.launchers: list[_FakeLauncher] = []
        self._squeue_stub = squeue_stub
        self._step_id = step_id

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": list(argv), "kwargs": kwargs})
        for stream in ("stdout", "stderr"):
            f = kwargs.get(stream)
            if hasattr(f, "close"):
                f.close()
        if self._squeue_stub is not None:
            name = next(a.split("=", 1)[1] for a in argv if a.startswith("--job-name="))
            listed, _ = self._squeue_stub.responses["squeue"]
            self._squeue_stub.responses["squeue"] = (f"{listed}{self._step_id} {name}\n", 0)
        self.launchers.append(_FakeLauncher())
        return self.launchers[-1]


async def test_deploy_into_an_allocation_starts_an_overlapping_step(tmp_path: Path, stub_subprocess, monkeypatch):
    """With ``jobid`` set the rendered script runs as a step of that
    allocation: no sbatch, ``--overlap`` because the allocation's own shell
    holds its resources, stdio to the files the directives would have
    named, and a handle that knows the allocation and the step's name."""
    stub_subprocess.responses["squeue"] = ("2554274.0 bash\n", 0)
    popen = _PopenRecorder(stub_subprocess)
    monkeypatch.setattr(subprocess, "Popen", popen)
    job_dir = tmp_path / "run_20260910"
    target = taaf.deploy_slurm.TufaSlurmTarget(main_project="taaf", jobid="2554274")
    handle = await target.deploy(_benchmark(job_dir))

    assert not [c for c in stub_subprocess.calls if c["argv"][0] == "sbatch"]
    (call,) = popen.calls
    argv = call["argv"]
    assert argv[:3] == ["srun", "--jobid=2554274", "--overlap"]
    assert f"--job-name={handle.step_name}" in argv and handle.step_name.startswith(f"{job_dir.name}-")
    assert f"--output={job_dir / 'stdout.log'}" in argv
    assert argv[-2:] == ["bash", str(job_dir / "sbatch_script.sh")]
    assert (handle.job_id, handle.step_id) == ("2554274", "2554274.1")
    meta = json.loads((job_dir / "deploy_meta.json").read_text())
    assert (meta["job_id"], meta["step_name"], meta["step_id"]) == ("2554274", handle.step_name, "2554274.1")
    attached = taaf.deploy.DeploymentHandle.attach(job_dir)
    assert isinstance(attached, taaf.deploy_slurm.TufaSlurmHandle) and attached.step_id == "2554274.1"


def test_a_step_keys_its_node_local_root_by_step(tmp_path: Path):
    """Two deploys into one allocation must not share sources or a venv:
    the script's install block would find the first's in place and skip."""
    job_dir = tmp_path / "run_x"
    step = taaf.deploy_slurm._render_sbatch_script(
        taaf.deploy_slurm.TufaSlurmTarget(main_project="taaf", jobid="2554274"), _benchmark(job_dir), job_dir
    )
    batch = taaf.deploy_slurm._render_sbatch_script(
        taaf.deploy_slurm.TufaSlurmTarget(main_project="taaf"), _benchmark(job_dir), job_dir
    )
    assert 'taaf_job_id="${SLURM_JOB_ID:-$$}.${SLURM_STEP_ID:-0}"' in step
    assert 'taaf_job_id="${SLURM_JOB_ID:-$$}"' in batch


def test_a_step_handle_stops_its_step_and_leaves_the_allocation(tmp_path: Path, stub_subprocess):
    handle = taaf.deploy_slurm.TufaSlurmHandle(
        job_dir=tmp_path, job_id="2554274", step_name="run_x", step_id="2554274.1"
    )
    stub_subprocess.responses["squeue"] = ("2554274.0 bash\n2554274.1 run_x\n", 0)
    assert handle.is_done is False
    handle.stop()
    (cancel,) = [c for c in stub_subprocess.calls if c["argv"][0] == "scancel"]
    assert cancel["argv"] == ["scancel", "--signal=USR1", "2554274.1"]
    stub_subprocess.responses["squeue"] = ("2554274.0 bash\n", 0)
    assert handle.is_done is True


async def test_two_deployments_sharing_a_job_dir_name_keep_separate_steps(tmp_path: Path, stub_subprocess, monkeypatch):
    """``job_dir.name`` is not unique within an allocation: two runs under
    different parents share a basename. Each deployment must end up on its
    own step, or the second's stop() signals the first's run."""
    stub_subprocess.responses["squeue"] = ("2554274.0 bash\n", 0)
    target = taaf.deploy_slurm.TufaSlurmTarget(main_project="taaf", jobid="2554274")
    handles = []
    for parent, step_id in (("experiment_a", "2554274.1"), ("experiment_b", "2554274.2")):
        monkeypatch.setattr(subprocess, "Popen", _PopenRecorder(stub_subprocess, step_id=step_id))
        handles.append(await target.deploy(_benchmark(tmp_path / parent / "run_x")))

    first, second = handles
    assert first.step_name != second.step_name
    assert (first.step_id, second.step_id) == ("2554274.1", "2554274.2")

    second.stop()
    (cancel,) = [c for c in stub_subprocess.calls if c["argv"][0] == "scancel"]
    assert cancel["argv"] == ["scancel", "--signal=USR1", "2554274.2"]

    # The first step outliving the second must not draw the second's
    # handle onto it.
    stub_subprocess.responses["squeue"] = ("2554274.0 bash\n2554274.1 " + first.step_name + "\n", 0)
    assert second.is_done is True and first.is_done is False


async def test_a_step_that_never_registers_cancels_its_launcher(tmp_path: Path, stub_subprocess, monkeypatch):
    """The launch is still pending when the timeout fires, and nothing
    records it: left running it would start the benchmark after the caller
    was told the deploy failed."""
    popen = _PopenRecorder()  # no squeue stub -- the step never appears
    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(taaf.deploy_slurm, "_STEP_REGISTER_TIMEOUT_S", 0.05)
    monkeypatch.setattr(taaf.deploy_slurm, "_STEP_REGISTER_POLL_S", 0.01)
    job_dir = tmp_path / "run_x"
    target = taaf.deploy_slurm.TufaSlurmTarget(main_project="taaf", jobid="2554274")

    with pytest.raises(RuntimeError, match="did not appear in allocation"):
        await target.deploy(_benchmark(job_dir))

    (launcher,) = popen.launchers
    assert launcher.signals[:2] == ["terminate", "wait"]
    assert not (job_dir / "deploy_meta.json").exists()


def test_only_the_installed_members_of_another_workspace_are_bundled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Another project's workspace brings only what this venv installed from it."""
    root = tmp_path / "other"
    for member in ("ullm", "platypus"):
        (root / "packages" / member).mkdir(parents=True)
        (root / "packages" / member / "pyproject.toml").write_text(f'[project]\nname = "{member}"\nversion = "0"\n')
    (root / "pyproject.toml").write_text('[tool.uv.workspace]\nmembers = ["packages/*"]\n')
    ullm = (root / "packages" / "ullm").resolve()
    monkeypatch.setattr(taaf.deploy, "_editable_source_paths", lambda: {ullm})

    assert taaf.deploy._members_to_snapshot(root) == [root / "packages" / "ullm"]

    monkeypatch.setattr(taaf.deploy, "_editable_source_paths", set)
    assert len(taaf.deploy._members_to_snapshot(root)) == 2
