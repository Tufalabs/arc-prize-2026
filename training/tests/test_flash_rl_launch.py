"""Exercise Flash job preparation and shutdown without CUDA or Slurm."""

import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from flash_configs import RECIPE, write_arc3_flash_config
from training.config import ExperimentConfig, max_served_sequence_tokens
from training.job_shape import job_shape, write_launch_config

ROOT = Path(__file__).resolve().parents[1]
JOB = (ROOT / "scripts/flash_rl.sbatch").read_text()
DATA = "arc2-platypus/packages/taaf/src/taaf/data"
# Mock only staging and the timeout wrapper; everything else runs.
RUN_STUB = 'run() { shift; if [[ "$1" == -k ]]; then shift 2; fi; shift; "$@"; }\n'
STAGE_STUB = 'stage() { [[ "$1" == *.tar.zst ]] && echo "$WORKER_STUB_PYTHON"; }\n'
# The CPU launch preparation, from the config to the printed commands.
LAUNCH_TAIL = JOB[JOB.index("# CPU launch preparation:"):]
# The rollout worker's environment: the harness for ARC-3, a staged worker for ARC-2.
WORKER_BLOCK = JOB[JOB.index("# The rollout worker's environment"):JOB.index("port_free() {")]
# The job's helpers through cleanup, with its real run() (setsid and timeout).
HELPERS = JOB[JOB.index("run() {"):JOB.index("trap cleanup EXIT")]
cluster_tools = pytest.mark.skipif(
    sys.platform != "linux" or not all(map(shutil.which, ("setsid", "timeout", "zstd", "sha256sum"))),
    reason="the job's helpers need cluster tools (util-linux setsid, coreutils, zstd)")


def clean_env():
    """The test process's environment without launcher, server or harness settings
    (nor a GPU allocation, which a dry run would check)."""
    return {key: value for key, value in os.environ.items()
            if not key.startswith(("ARC3_", "SERVER_", "PLATYPUS_")) and key != "SLURM_GPUS_ON_NODE"}


def stub_venv(path, versions, taaf=False):
    """A real (pip-less) venv with the given distributions and, for the worker, taaf's datasets."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True)
    site = next(path.glob("lib/python3.*/site-packages"))
    for name, version in versions.items():
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    if taaf:
        (site / "taaf/data").mkdir(parents=True)
        (site / "taaf/__init__.py").write_text("raise AssertionError('the launcher must not import taaf')\n")
        for name in ("arc2-train", "arc2-eval"):
            (site / f"taaf/data/{name}.npz").write_bytes(f"pinned {name}".encode())
    return path


STACK = {"transformers": "5.16.1", "torch": "2.13.0", "vllm": "0.29.1.dev63+g077920f36"}


def arc2_env(tmp_path, worker_stack=STACK):
    work, run = tmp_path / "work", tmp_path / "run"
    run.mkdir(parents=True)
    # The job's archived revision: no submodule checkouts, no .git.
    shutil.copytree(ROOT, work / "source", ignore=shutil.ignore_patterns(
        ".git", "arc2-platypus", "platypus", ".venv", "__pycache__", "runs", "flash_bench", "*.egg-info",
        ".pytest_cache", ".ruff_cache"))
    harness = stub_venv(tmp_path / "harness", STACK)
    worker = stub_venv(tmp_path / "worker", worker_stack, taaf=True)
    env = clean_env()
    env.update(WORK=str(work), RUN_DIR=str(run), REPO_ROOT=str(work / "source"), SOURCE_REPO=str(ROOT),
               ARC3_CONFIG=str(RECIPE), ARC3_FLASH_PYTHON=sys.executable, PLATYPUS_VENV=str(harness),
               ARC3_ARC2_WORKER_VENV_TARBALL="/data/envs/arc2-worker-vllm-fork-0123456789abcdef.tar.zst",
               WORKER_STUB_PYTHON=str(worker / "bin/python"),
               ARC3_RUN_DIR=str(run), ARC3_SOURCE_WORKSPACE=str(work / "frozen-source"),
               ARC3_RL_CACHE_ROOT=str(tmp_path / "cache"), ARC3_DRY_RUN="1", SLURM_JOB_ID="123",
               OMP_NUM_THREADS="1", PYTHONPATH=str(work / "source"), PYTHONDONTWRITEBYTECODE="1")
    return env, work, run, harness, worker


def test_flash_shape_uses_checkpoint_and_all_eight_devices(tmp_path):
    cfg = ExperimentConfig.load(write_arc3_flash_config(tmp_path))
    shape = job_shape(cfg, {})
    assert shape["SERVER_MODEL"] == cfg.training.flash.checkpoint_root
    assert shape["SERVER_SERVED_NAME"] == "Qwen/Qwen3.8-Flash-Next"
    assert shape["SERVER_GPU_LIST"] == "0 1 2 3"
    assert shape["TRAINER_GPUS"] == "4,5,6,7"
    assert shape["DEFAULT_VISIBLE_GPUS"] == "0,1,2,3,4,5,6,7"
    assert max_served_sequence_tokens(cfg.inference) == 32768
    assert shape["SERVER_CONTEXT_LIST"] == "32768 32768 32768 32768"
    assert shape["SERVER_ENVIRONMENT"] == "arc3"
    assert shape["SERVER_MULTIMODAL_CONTEXT"] == "current_grid"


@pytest.mark.parametrize(("env", "error"), [
    ({"ARC3_SERVER_GPUS": "0,1,2,4"}, "claimed twice"),
    ({"ARC3_SERVER_GPUS": "0,1,2,2"}, "claimed twice"),
    ({"ARC3_SERVER_GPUS": "0,1,2,-1"}, "nonnegative GPU"),
    ({"ARC3_SERVER_GPUS": "0,1"}, "preserve the Flash server count"),
    ({"ARC3_SERVER_TENSOR_PARALLEL": "4"}, "tensor parallelism 1"),
])
def test_flash_effective_allocation_rejects_overrides(env, error):
    with pytest.raises(ValueError, match=error):
        job_shape(ExperimentConfig.load(RECIPE), env)


@pytest.mark.parametrize("mode", ["lora", "full_weight"])
@pytest.mark.parametrize("evaluation", [False, True])
def test_27b_shape_matches_existing_heredoc(tmp_path, mode, evaluation):
    payload = json.loads((ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json").read_text())
    payload["inference"]["server_gpus"] = ["0"]
    payload["training"].update(device="cuda:2", device_map_gpus=[2], worker_device_map_gpus=[])
    if mode == "full_weight":
        payload["training"].update(weight_mode=mode, device="cuda", device_map_gpus=[],
                                    full_weight={"device_ids": [2, 4, 6]})
    if evaluation:
        payload["official_eval"] = {"enabled": True, "games": ["ar25-0c556536"],
                                    "server_gpus": ["7"], "context_window_tokens": 65536}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    # Execute the original, unchanged SERVER_SHAPE Python, not a second copy
    # of its allocation algorithm. Its multi-server override is ignored.
    original = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    snippet = original.split('SERVER_SHAPE="$(')[1].split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    env = {**os.environ, "ARC3_SERVER_GPUS": "1", "ARC3_SERVER_TENSOR_PARALLEL": "1"}
    result = subprocess.run([sys.executable, "-", str(config)], input=snippet,
                            env=env, text=True, capture_output=True, check=True)
    shape = job_shape(ExperimentConfig.load(config), env)
    assert result.stdout.splitlines() == list(shape.values())[:10]


@pytest.mark.parametrize(("rollouts", "cap", "explicit", "seqs"), [
    (128, None, None, "40"),  # the recipe: ceil(128 / 4) + 8
    (130, None, None, "41"),  # an uneven share rounds up
    (128, 16, None, "24"),  # inference.max_concurrent_episodes caps the share
    (128, None, "32", "32"),  # an explicit value holding the share wins
    (128, None, "31", "at least 32"),
    (128, 16, "15", "at least 16"),
    (128, None, "lots", "at least 32"),
])
def test_flash_engines_hold_each_servers_share(tmp_path, rollouts, cap, explicit, seqs):
    payload = json.loads(RECIPE.read_text())
    payload.update(parallel_rollouts=rollouts)
    payload["inference"]["max_concurrent_episodes"] = cap
    if rollouts % 16:
        payload["sampling"]["game_sampling_games_per_window"] = rollouts // 2
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload))
    env = {} if explicit is None else {"SERVER_MAX_NUM_SEQS": explicit}
    cfg = ExperimentConfig.load(config)
    if seqs.isdigit():
        assert job_shape(cfg, env)["SERVER_MAX_NUM_SEQS"] == seqs
        return
    with pytest.raises(ValueError, match=f"SERVER_MAX_NUM_SEQS={explicit} must be an integer of {seqs}"):
        job_shape(cfg, env)
    # The job stops at its first shape, before staging a server.
    result = subprocess.run([sys.executable, "-m", "training.job_shape", str(config), "--flash-job"],
                            cwd=ROOT, env={**clean_env(), **env, "PYTHONPATH": str(ROOT)},
                            capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert f"must be an integer of {seqs}" in result.stderr


def test_config_rewrite_records_effective_gpus_and_actual_logs(tmp_path):
    shape = job_shape(ExperimentConfig.load(RECIPE), {"ARC3_SERVER_GPUS": "3,2,1,0"})
    target = tmp_path / "frozen_config.json"
    write_launch_config(RECIPE, target, shape, 50123, "/cache/harness")
    cfg = ExperimentConfig.load(target)
    assert cfg.inference.base_url == "http://127.0.0.1:50123"
    assert cfg.inference.platypus_venv == "/cache/harness"
    assert cfg.inference.server_gpus == ["3", "2", "1", "0"]
    assert cfg.inference.server_log_paths == [str(tmp_path / f"server_{i}.log") for i in range(4)]
    write_launch_config(target, target, shape, 50999, "/cache/new-harness")
    assert ExperimentConfig.load(target).inference.base_url.endswith(":50999")


@pytest.mark.parametrize(("resume", "kernel_mode"), [(False, None), (True, None), (False, "reference")])
def test_bash_dry_run_prepares_real_snapshot_without_launching(tmp_path, resume, kernel_mode):
    work = tmp_path / "work"
    run = tmp_path / "run"
    work.mkdir()
    run.mkdir()
    venv = tmp_path / "harness"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin/python").symlink_to(sys.executable)
    env = clean_env()
    env.update(WORK=str(work), RUN_DIR=str(run), REPO_ROOT=str(ROOT), SOURCE_REPO=str(ROOT),
               ARC3_CONFIG=str(write_arc3_flash_config(
                   tmp_path, **({} if kernel_mode is None else {"kernel_mode": kernel_mode}))),
               ARC3_FLASH_PYTHON=sys.executable, PLATYPUS_VENV=str(venv),
               ARC3_RUN_DIR=str(run), ARC3_SOURCE_WORKSPACE=str(work / "source"),
               ARC3_RL_CACHE_ROOT=str(tmp_path / "cache"), ARC3_DRY_RUN="1", SLURM_JOB_ID="123",
               OMP_NUM_THREADS="1", PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE="1",
               # An explicit override: this config caps each server at 4 episodes (the job derives 12).
               SERVER_MAX_NUM_SEQS="40")
    # A previous segment's kernel cache: a dry run neither restores nor saves it.
    cache = tmp_path / "kernels"
    (cache / "triton").mkdir(parents=True)
    (cache / "inductor").mkdir()
    (cache / "triton/marker").write_text("unchanged")
    (run / "kernel_cache.tar.zst").write_bytes(b"previous segment")
    env.update(TRAINER_CACHE=str(cache), TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), STARTED_AT="0")
    # Skip only cluster provisioning/staging. Run actual config validation,
    # socket probes, source snapshot, fingerprint and command construction.
    script = (HELPERS + RUN_STUB + 'SERVER_PIDS=() TRAINER_PID="" CHILD="" LOGGER_PID=""\n'
              + "trap cleanup EXIT\n" + LAUNCH_TAIL)
    first_archive = None
    for iteration in range(2 if resume else 1):
        env["SLURM_JOB_ID"] = str(123 + iteration)
        result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                                cwd=ROOT, env=env, capture_output=True, text=True, timeout=60, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "kernel cache" not in result.stdout + result.stderr
        assert (run / "kernel_cache.tar.zst").read_bytes() == b"previous segment"
        assert (cache / "triton/marker").read_text() == "unchanged"
        assert not (run / "kernel_cache.tar.zst.tmp").exists()
        frozen = json.loads((run / "frozen_config.json").read_text())
        # The launcher starts at 50000 + job id % 4000 and moves up by 101 until the
        # four server ports and their +5000 twins are all free. They lie in the
        # ephemeral range, so on a shared host one may be in transient use.
        base = int(frozen["inference"]["base_url"].rsplit(":", 1)[1])
        assert frozen["inference"]["base_url"] == f"http://127.0.0.1:{base}"
        assert base >= 50123 + iteration and (base - 50123 - iteration) % 101 == 0
        # The learner the frozen config builds: the switch survives the launcher.
        learner = ExperimentConfig.load(run / "frozen_config.json").training.flash
        assert learner.kernel_mode == (kernel_mode or "fast")
        assert learner.attention_backend == ("sdpa" if kernel_mode == "reference" else "flex_qsa")
        assert frozen["inference"]["platypus_venv"] == str(venv)
        commands = [shlex.split(line) for line in result.stdout.splitlines()
                    if line.startswith(("env ", "setsid "))]
        assert len(commands) == 5
        for index, command in enumerate(commands[:4]):
            assert f"SERVER_GPU={index}" in command
            assert f"SERVER_PORT={base + index * 100}" in command
            assert "SERVER_WEIGHT_MODE=flash" in command
            assert "SERVER_MAX_MODEL_LEN=32768" in command
            assert "SERVER_MAX_NUM_SEQS=40" in command  # the explicit override
            assert f"SERVER_LOG={run}/server_{index}.log" in command
            assert str(work / "source") in command[-1]
        assert commands[-1][-6:] == ["-m", "training.launch_frozen", "--config",
                                     str(run / "launch_config.json"), "--run-dir", str(run)]
        assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7" in commands[-1]
        assert not (run / "trainer.log").exists()
        assert not list(run.glob("server_*.log"))
        archive = (run / "source.tar").read_bytes()
        if iteration:
            assert archive == first_archive
        first_archive = archive
    fingerprint = json.loads((run / "runtime_fingerprint.json").read_text())
    assert {"training", "inference", "server"} <= fingerprint.keys()


def test_bash_dry_run_arc2_runs_the_worker_from_its_own_environment(tmp_path):
    env, work, run, harness, worker = arc2_env(tmp_path)
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", RUN_STUB + STAGE_STUB + LAUNCH_TAIL],
                            cwd=work / "source", env=env, capture_output=True, text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    frozen = json.loads((run / "frozen_config.json").read_text())
    assert frozen["inference"]["environment"] == "arc2"
    assert frozen["inference"]["platypus_venv"] == str(worker)
    assert f"rollout worker: {worker}" in result.stdout
    assert "Each engine's --max-num-seqs: 40" in result.stdout
    commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith(("env ", "setsid "))]
    assert len(commands) == 5
    for command in commands[:4]:
        assert f"PLATYPUS_VENV={harness}" in command  # the servers keep the harness
        assert "SERVER_MAX_NUM_SEQS=40" in command  # unset: the job derives ceil(128 / 4) + 8
        assert "SERVER_ENVIRONMENT=arc2" in command
        assert "SERVER_MAX_MODEL_LEN=262144" in command  # inference.agent_max_served_sequence_tokens
    # The datasets the trainer resolves come from the worker's taaf into the run's snapshot.
    with tarfile.open(run / "source.tar") as archive:
        for name in ("arc2-train", "arc2-eval"):
            assert archive.extractfile(f"{DATA}/{name}.npz").read() == f"pinned {name}".encode()
        assert not [member for member in archive.getnames() if member.endswith("taaf/__init__.py")]
    fingerprint = json.loads((run / "runtime_fingerprint.json").read_text())
    assert fingerprint["inference"]["python"] == str(worker / "bin/python")
    assert fingerprint["server"]["python"] == str(harness / "bin/python")


def worker_block(env, cwd):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", RUN_STUB + STAGE_STUB + WORKER_BLOCK],
                          cwd=cwd, env=env, capture_output=True, text=True, timeout=60, check=False)


def test_arc2_worker_requires_its_tarball_and_the_harness_stack(tmp_path):
    env, work, *_ = arc2_env(tmp_path, worker_stack={**STACK, "transformers": "5.14.1"})
    env["SRC_CONFIG"] = str(RECIPE)
    result = worker_block(env, work / "source")
    assert result.returncode == 1
    assert "differ in their serving stack" in result.stderr
    assert "transformers 5.16.1 transformers 5.14.1" in result.stderr
    del env["ARC3_ARC2_WORKER_VENV_TARBALL"]
    result = worker_block(env, work / "source")
    assert result.returncode != 0
    assert "ARC-2 configs need ARC3_ARC2_WORKER_VENV_TARBALL" in result.stderr
    # ARC-3 ignores the variable and stages nothing.
    env.update(SRC_CONFIG=str(write_arc3_flash_config(tmp_path)), ARC3_ARC2_WORKER_VENV_TARBALL="/data/not-staged.tar.zst")
    script = RUN_STUB + 'stage() { echo staged >&2; exit 9; }\n'
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script + WORKER_BLOCK + 'echo "$WORKER_VENV"\n'],
                            cwd=work / "source", env=env, capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == env["PLATYPUS_VENV"]


def test_arc2_datasets_are_never_written_into_a_checkout(tmp_path):
    env, work, *_ = arc2_env(tmp_path)
    checkout = tmp_path / "checkout"
    (work / "source").rename(checkout)
    env.update(SRC_CONFIG=str(RECIPE), REPO_ROOT=str(checkout))
    result = worker_block(env, checkout)
    assert result.returncode == 1
    assert f"{checkout} has no {DATA}; initialize the arc2-platypus submodule" in result.stderr
    assert not (checkout / DATA).exists()


def test_shutdown_waits_for_trainer_before_stopping_every_server_group():
    start = JOB.index("stop_group()")
    end = JOB.index("# Defined early", start)
    script = '''
TRAINER_PID=100001
SERVER_PIDS=(100002 100003 100004 100005)
CHILD="" LOGGER_PID="" WORK=/not-scratch RUN_DIR=/unused
kill() { if [[ "$1" != -0 ]]; then printf '%s\\n' "$*"; fi; }
sleep() { SECONDS=$((SECONDS + 30)); }
wait() { :; }
pgrep() { return 1; }
maybe_relaunch() { :; }
'''
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script + JOB[start:end] + "\ncleanup\n"],
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [
        "-TERM 100001", "-KILL -- -100001",
        "-TERM 100002", "-TERM 100003", "-TERM 100004", "-TERM 100005",
        "-KILL -- -100002", "-KILL -- -100003", "-KILL -- -100004", "-KILL -- -100005",
    ]


def kernel_cache_env(tmp_path):
    """A learner cache (triton/, inductor/) and a run directory, as the job prepares them."""
    cache, run = tmp_path / "kernels", tmp_path / "run"
    (cache / "triton").mkdir(parents=True)
    (cache / "inductor").mkdir()
    run.mkdir()
    env = clean_env()
    env.update(RUN_DIR=str(run), TRAINER_CACHE=str(cache), TRITON_CACHE_DIR=str(cache / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor"), STARTED_AT="0", LIMIT_SECONDS="")
    return env, cache, run


def job_helpers(env, commands):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", HELPERS + commands],
                          env=env, capture_output=True, text=True, timeout=120, check=False)


def write_kernel_archive(source, archive, files):
    for name in ("triton", "inductor"):
        (source / name).mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        (source / name).write_bytes(data)
    subprocess.run(["tar", "-I", "zstd", "-cf", str(archive), "-C", str(source), "triton", "inductor"],
                   check=True)


def cache_files(cache):
    return sorted(str(path.relative_to(cache)) for path in cache.rglob("*") if path.is_file())


@cluster_tools
def test_kernel_cache_restore_unpacks_the_run_archive_if_present(tmp_path):
    env, cache, run = kernel_cache_env(tmp_path)
    result = job_helpers(env, "restore_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "kernel cache" not in result.stdout + result.stderr
    assert cache_files(cache) == []
    files = {"triton/0abc/kernel.cubin": b"cubin", "triton/0abc/kernel.autotune.json": b"{}",
             "inductor/fxgraph/entry": b"graph"}
    write_kernel_archive(tmp_path / "previous", run / "kernel_cache.tar.zst", files)
    result = job_helpers(env, "restore_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "Restored the learner's kernel cache" in result.stdout
    assert cache_files(cache) == sorted(files)
    assert (cache / "triton/0abc/kernel.cubin").read_bytes() == b"cubin"


@cluster_tools
def test_kernel_cache_restore_failure_continues_with_an_empty_cache(tmp_path):
    env, cache, run = kernel_cache_env(tmp_path)
    archive = run / "kernel_cache.tar.zst"
    # Incompressible files, so half the archive still unpacks some of them.
    write_kernel_archive(tmp_path / "previous", archive,
                         {f"triton/{index:03d}/kernel.cubin": os.urandom(8192) for index in range(64)})
    archive.write_bytes(archive.read_bytes()[: archive.stat().st_size // 2])
    result = job_helpers(env, "restore_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "the learner starts with an empty kernel cache" in result.stderr
    assert sorted(path.name for path in cache.iterdir()) == ["inductor", "triton"]
    assert cache_files(cache) == []


@cluster_tools
def test_kernel_cache_save_replaces_the_archive_and_keeps_it_when_packing_fails(tmp_path):
    env, cache, run = kernel_cache_env(tmp_path)
    archive = run / "kernel_cache.tar.zst"
    archive.write_bytes(b"previous segment")
    (cache / "triton/0abc").mkdir()
    (cache / "triton/0abc/kernel.cubin").write_bytes(b"cubin")
    (cache / "inductor/entry").write_bytes(b"graph")
    # A start time far ahead leaves more than the 10-minute cap before the limit.
    env.update(STARTED_AT=str(10**10), LIMIT_SECONDS="21600")
    result = job_helpers(env, "save_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "Saved the learner's kernel cache" in result.stdout
    listing = subprocess.run(["tar", "-I", "zstd", "-tf", str(archive)], capture_output=True, text=True,
                             check=True).stdout.split()
    assert {"triton/0abc/kernel.cubin", "inductor/entry"} <= set(listing)
    assert not (run / "kernel_cache.tar.zst.tmp").exists()
    saved = archive.read_bytes()
    # A pack that dies midway leaves a partial temporary file and the old archive.
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "tar").write_text("#!/bin/sh\n"
                              'for arg; do case "$arg" in *.tmp) printf partial > "$arg" ;; esac; done\n'
                              "exit 2\n")
    (fake / "tar").chmod(0o755)
    env["PATH"] = f"{fake}:{env['PATH']}"
    result = job_helpers(env, "save_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "Could not save the kernel cache" in result.stderr
    assert archive.read_bytes() == saved
    assert not (run / "kernel_cache.tar.zst.tmp").exists()
    # Too close to the time limit, the job does not start a pack.
    env.update(STARTED_AT="0", LIMIT_SECONDS="60")
    result = job_helpers(env, "save_kernel_cache\n")
    assert result.returncode == 0, result.stderr
    assert "No time left to save the kernel cache" in result.stderr
    assert archive.read_bytes() == saved


@cluster_tools
@pytest.mark.parametrize("launched", [False, True])
def test_cleanup_saves_the_kernel_cache_after_the_relaunch_when_the_learner_ran(tmp_path, launched):
    env, cache, run = kernel_cache_env(tmp_path)
    (cache / "triton/kernel.cubin").write_bytes(b"cubin")
    env.update(TRAINER_LAUNCHED=str(int(launched)), WORK=str(tmp_path / "work"))
    # The relaunch is submitted before the pack starts.
    script = (HELPERS + 'SERVER_PIDS=() TRAINER_PID="" CHILD="" LOGGER_PID=""\n'
              'maybe_relaunch() { [[ -f "$RUN_DIR/kernel_cache.tar.zst" ]] && echo late || echo first; }\n'
              "cleanup\n")
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True,
                            text=True, timeout=120, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "first"
    assert (run / "kernel_cache.tar.zst").exists() == launched


@cluster_tools
def test_learner_cache_directory_is_per_run_not_per_job(tmp_path):
    block = JOB[JOB.index("# FLA persists autotune results"):JOB.index("export XDG_CACHE_HOME")]

    def triton_cache(run_dir, job):
        env = {**clean_env(), "RUN_DIR": run_dir, "SLURM_JOB_ID": job, "ARC3_COMPILE_CACHE_ROOT": str(tmp_path)}
        return subprocess.run(["bash", "-euo", "pipefail", "-c", block + 'echo "$TRITON_CACHE_DIR"\n'],
                              env=env, capture_output=True, text=True, check=True).stdout.strip()

    # Triton's cache index holds absolute paths: every segment must use the same one.
    assert triton_cache("/data/run-a", "1") == triton_cache("/data/run-a", "2")
    assert triton_cache("/data/run-a", "1") != triton_cache("/data/run-b", "1")
    assert triton_cache("/data/run-a", "1").startswith(f"{tmp_path}/trainer-run-")


def test_term_reaches_learner_ranks_without_stopping_torchrun():
    snippet = JOB[JOB.index("signal_trainer()"):JOB.index("on_signal()")]
    script = '''
TRAINER_PID=100001
pgrep() {
  case "$2" in
    100001) echo 100002 ;;
    100002) echo '100003 100004 100005 100006' ;;
  esac
}
kill() { printf '%s\\n' "$*"; }
'''
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script + snippet + "\nsignal_trainer\n"],
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == [f"-TERM {pid}" for pid in range(100003, 100007)]


# The TERM handler and the relaunch decision, as cleanup runs them after the job ends.
RELAUNCH = (JOB[JOB.index("on_signal() {"):JOB.index("# Segments of a run share")]
            + JOB[JOB.index("# Slurm's end-of-job TERM"):JOB.index("# Stage each immutable")])
# Mock only Slurm, the clock, and the env/timeout wrappers. No job is submitted.
SLURM_STUBS = """
timeout() { shift; "$@"; }
scontrol() { echo "JobState=$STATE"; }
date() { echo "$NOW"; }
env() { while [[ "$1" == -u ]]; do shift 2; done; "$@"; }
sbatch() { printf '%s\\n' "$*"; }
"""


def relaunch(run_dir, *, rc=0, term_at=None, state="RUNNING", **variables):
    """The job's ending in a 1200 s segment: the learner exits RC, or a TERM
    arrives TERM_AT seconds in and on_signal exits; cleanup then stops the
    learner and servers (5 s here) and calls maybe_relaunch. VARIABLES replace
    the job's values; ARC3_AUTO_RELAUNCH is left unset unless given, so the
    chaining cases exercise the script's default. Returns what maybe_relaunch printed."""
    settings = {
        "ARC3_DRY_RUN": "0", "TRAINER_RC": str(rc), "TRAINER_PID": "",
        "TRAINER_LAUNCHED": "1", "TERM_ELAPSED": "", "RUN_DIR": str(run_dir), "SLURM_JOB_ID": "123",
        "STARTED_AT": "0", "LIMIT_SECONDS": "1200", "TIME_LIMIT": "00:20:00", "STATE": state,
        "NOW": str(term_at or 0), "REPO_ROOT": "/source", "SOURCE_REPO": "/fetch-only",
        "REVISION": "abcdef", **variables,
    }
    script = ("".join(f"{name}={shlex.quote(value)}\n" for name, value in settings.items())
              + SLURM_STUBS + RELAUNCH
              + "trap 'NOW=$((NOW + 5)); maybe_relaunch' EXIT\n"
              + ("on_signal\n" if term_at is not None else f"exit {rc}\n"))
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script],
                            capture_output=True, text=True, check=False)
    assert result.returncode == (143 if term_at is not None else rc), result.stderr
    return result.stdout


@pytest.mark.parametrize(("rc", "term_at", "state", "variables", "message"), [
    (75, None, "RUNNING", {}, "Relaunching: the learner yielded at a checkpoint (exit 75)."),
    (1, None, "RUNNING", {},
     "Not relaunching: the learner exited 1 without yielding at a checkpoint (75)."),
    # The end-of-job TERM is due 600 s in and may come up to a minute early; a
    # learner waiting on a slow collection stops at once, so only the time the
    # TERM arrived tells it from a stop.
    (0, 555, "RUNNING", {}, "Relaunching: TERM 555s into a 1200s limit, the end-of-job signal."),
    (0, 480, "RUNNING", {}, "Relaunching: TERM 480s into a 1200s limit, the end-of-job signal."),
    (0, 479, "RUNNING", {},
     "Not relaunching: TERM 479s into a 1200s limit, before its last 720s: a stop, "
     "not the end-of-job signal."),
    (0, 10, "PREEMPTED", {}, "Relaunching: TERM with JobState=PREEMPTED."),
    (0, 555, "CANCELLED", {}, "Not relaunching: the job was cancelled (JobState=CANCELLED)."),
    (0, 555, "RUNNING", {"LIMIT_SECONDS": ""},
     "Not relaunching: TERM 555s into the segment, whose time limit is unknown."),
    (75, None, "RUNNING", {"ARC3_DRY_RUN": "1"}, "Not relaunching: a dry run."),
    (75, None, "RUNNING", {"ARC3_AUTO_RELAUNCH": "0"},
     "Not relaunching: ARC3_AUTO_RELAUNCH is not 1."),
    (0, 555, "RUNNING", {"STOP": "1"}, "Not relaunching: {run}/STOP_AUTO_RELAUNCH exists."),
    (0, None, "RUNNING", {"TRAINER_LAUNCHED": "0"},
     "Not relaunching: the job ended before the learner started."),
])
def test_auto_relaunch_only_chains_scheduled_or_checkpointed_segments(
    tmp_path, rc, term_at, state, variables, message,
):
    if variables.pop("STOP", None):
        (tmp_path / "STOP_AUTO_RELAUNCH").touch()
    output = relaunch(tmp_path, rc=rc, term_at=term_at, state=state, **variables)
    lines = output.splitlines()
    # One line says why, every time; a relaunch is followed by its submission.
    assert lines[0] == message.format(run=tmp_path)
    chains = message.startswith("Relaunching")
    assert len(lines) == (2 if chains else 1)
    if chains:
        assert "--dependency=afterany:123" in lines[1]
        assert lines[1].endswith("/source/scripts/flash_rl.sbatch /fetch-only abcdef")
        assert f"--output={tmp_path}/slurm-%j.out" in lines[1]
        assert "--exclude" not in lines[1]


@pytest.mark.parametrize("exclude", ["", "dgx02,dgx06"])
@pytest.mark.parametrize(("rc", "term_at"), [(75, None), (0, 555)])
def test_auto_relaunch_keeps_the_node_exclusions(tmp_path, exclude, rc, term_at):
    # The submit helper exports ARC3_EXCLUDE with the first segment; a single-node
    # continuation must carry it like the heterogeneous relaunch does, after a
    # checkpointed yield and after the end-of-job TERM alike.
    reason, submission = relaunch(tmp_path, rc=rc, term_at=term_at, ARC3_EXCLUDE=exclude).splitlines()
    assert reason.startswith("Relaunching: ")
    args = shlex.split(submission)
    assert "--dependency=afterany:123" in args
    assert args[-3:] == ["/source/scripts/flash_rl.sbatch", "/fetch-only", "abcdef"]
    assert [a for a in args if a.startswith("--exclude")] == ([f"--exclude={exclude}"] if exclude else [])


def test_batch_script_and_staging_helper():
    subprocess.run(["bash", "-n", str(ROOT / "scripts/flash_rl.sbatch")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "scripts/stage_venv.sh")], check=True)
