"""slurm/sft_distillation.sbatch: the kernel cache of a kernel_mode "fast" run, without Slurm."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
JOB = (ROOT / "slurm/sft_distillation.sbatch").read_text()
HELPERS = JOB[JOB.index("restore_kernel_cache() {"):JOB.index("KERNEL_MODE=$(")]
SELECT = JOB[JOB.index("KERNEL_MODE=$("):JOB.index('"$PY" -m sft.distillation.train')]
cluster_tools = pytest.mark.skipif(
    sys.platform != "linux" or not all(map(shutil.which, ("timeout", "zstd", "sha256sum"))),
    reason="the job's helpers need coreutils and zstd")


def run(script, env):
    return subprocess.run(["bash", "-euo", "pipefail", "-c", script], env=env, capture_output=True,
                          text=True, timeout=120, check=False)


def files(directory):
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())


@cluster_tools
def test_segments_of_a_run_share_its_kernel_cache(tmp_path):
    cache, archive = tmp_path / "cache", tmp_path / "run/kernel_cache.tar.zst"
    archive.parent.mkdir()
    env = {**os.environ, "KERNEL_CACHE": str(cache), "KERNEL_ARCHIVE": str(archive)}
    result = run(HELPERS + "restore_kernel_cache\n", env)
    assert result.returncode == 0, result.stderr
    assert files(cache) == [] and sorted(p.name for p in cache.iterdir()) == ["inductor", "triton"]
    (cache / "triton/0abc").mkdir()
    (cache / "triton/0abc/kernel.autotune.json").write_text("{}")
    (cache / "inductor/entry").write_text("graph")
    # The save outside /cache/users keeps the directory; the job's own path is removed.
    result = run(HELPERS + "save_kernel_cache\n", env)
    assert result.returncode == 0 and "Saved the run's kernel cache" in result.stdout, result.stderr
    shutil.rmtree(cache)
    result = run(HELPERS + "restore_kernel_cache\n", env)
    assert "Restored the run's kernel cache" in result.stdout, result.stderr
    assert files(cache) == ["inductor/entry", "triton/0abc/kernel.autotune.json"]
    # A truncated archive leaves training with an empty cache, not a failed job.
    archive.write_bytes(archive.read_bytes()[:20])
    result = run(HELPERS + "restore_kernel_cache\n", env)
    assert result.returncode == 0 and "empty kernel cache" in result.stderr
    assert files(cache) == []


@pytest.mark.skipif(not shutil.which("sha256sum"), reason="the job hashes the run path with coreutils")
@pytest.mark.parametrize("mode", [None, "reference", "fast"])
def test_only_fast_runs_use_the_run_cache(tmp_path, mode):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({} if mode is None else {"kernel_mode": mode}))
    result_dir = tmp_path / "run/attempt002/training"
    script = (
        "restore_kernel_cache() { echo restore; }\nsave_kernel_cache() { :; }\n" + SELECT
        + 'echo "mode=$KERNEL_MODE triton=$TRITON_CACHE_DIR archive=${KERNEL_ARCHIVE:-}"\n'
    )
    env = {**os.environ, "PY": sys.executable, "CONFIG": str(config), "RESULT": str(result_dir),
           "TRITON_CACHE_DIR": "/scratch/job/triton"}
    result = run(script, env)
    assert result.returncode == 0, result.stderr
    if mode == "fast":
        assert result.stdout.splitlines()[0] == "restore"
        last = result.stdout.splitlines()[-1]
        assert f"archive={tmp_path}/run/kernel_cache.tar.zst" in last
        assert f"/cache/users/{os.getuid()}/arc-distillation/compile/run-" in last
        # One path per run: every segment unpacks at the same place.
        other = run(script, {**env, "RESULT": str(tmp_path / "run/attempt003/training")})
        assert other.stdout.splitlines()[-1] == last
    else:
        assert result.stdout.strip() == "mode=reference triton=/scratch/job/triton archive="
