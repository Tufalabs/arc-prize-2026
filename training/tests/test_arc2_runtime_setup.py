"""ARC-2 environment isolation, frozen data, and batch stop behavior."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import pytest

from training.source_snapshot import create_source_snapshot, materialize_source_snapshot

ROOT = Path(__file__).resolve().parents[1]
HARNESS_WORKER = ROOT / "environments/arc2-worker-vllm-fork"


def test_frozen_source_retains_arc2_data_without_importable_harness(
    tmp_path: Path,
) -> None:
    """Dataset identity survives staging while conflicting harness code stays isolated."""
    repo = tmp_path / "repo"
    data = repo / "arc2-platypus/packages/taaf/src/taaf/data"
    data.mkdir(parents=True)
    (data.parent / "__init__.py").write_text("raise AssertionError('wrong harness')\n")
    for name in ("arc2-train", "arc2-eval"):
        (data / f"{name}.npz").write_bytes(name.encode())
    archive = create_source_snapshot(live_repo_root=repo, run_root=tmp_path / "run")
    staged = materialize_source_snapshot(
        archive_path=archive, workspace_root=tmp_path / "scratch"
    )
    staged_data = staged / data.relative_to(repo)
    assert (staged_data / "arc2-train.npz").read_bytes() == b"arc2-train"
    assert (staged_data / "arc2-eval.npz").read_bytes() == b"arc2-eval"
    assert not (staged_data.parent / "__init__.py").exists()


def test_harness_worker_lock_uses_only_pinned_repository_sources() -> None:
    """The ARC-2 worker beside harness engines builds from the gitlinks and the fork."""
    worker = tomllib.loads((HARNESS_WORKER / "uv.lock").read_text())
    sources = {package["name"]: package["source"] for package in worker["package"]}
    for name in ("duck", "platypus", "taaf"):
        assert sources[name] == {"editable": f"../../arc2-platypus/packages/{name}"}
    assert sources["ullm"] == {"directory": "../../platypus/packages/ullm"}
    git_sources = {
        name: source["git"] for name, source in sources.items() if "git" in source
    }
    assert list(git_sources) == ["vllm"]
    assert git_sources["vllm"].startswith(
        "https://github.com/Tufalabs/arc-prize-2026-vllm?rev="
    )
    # The worker never starts an engine: no prebuilt kernel wheel.
    assert "flashinfer-jit-cache" not in sources


@pytest.mark.skipif(
    not (ROOT / "platypus/uv.lock").exists(),
    reason="needs the platypus submodule checkout",
)
def test_harness_worker_lock_agrees_with_the_harness_lock() -> None:
    """EngineProxy refuses a worker whose tokenizer or image stack differs from the engine's."""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "scripts/env_checks.py"),
            "lock-agrees",
            str(HARNESS_WORKER / "uv.lock"),
            str(ROOT / "platypus/uv.lock"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_batch_term_stops_and_reaps_trainer_and_server_groups(tmp_path: Path) -> None:
    """The batch's actual TERM trap interrupts wait and cleans both owned groups."""
    source = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    cleanup = source[
        source.index("__arc3_stop_servers() {") : source.index(
            "for (( i = 0; i < NUM_SERVERS; i++ )); do",
            source.index("__arc3_stop_servers() {"),
        )
    ]
    term_trap = next(
        line
        for line in source.splitlines()
        if line.startswith("trap '") and line.endswith(" TERM")
    )
    child = tmp_path / "child.py"
    child.write_text(
        "import os, sys, time\nfrom pathlib import Path\n"
        "os.setsid()\nPath(sys.argv[1]).write_text(str(os.getpid()))\n"
        "while True: time.sleep(1)\n"
    )
    script = (
        "SERVER_PIDS=()\n__arc3_got_endsig=0\n"
        "__arc3_cleanup_build() { :; }\n__arc3_maybe_relaunch() { :; }\n"
        + term_trap
        + "\n"
        + cleanup
        + "\n"
        '"$TEST_PYTHON" "$TEST_CHILD" "$TEST_DIR/server.pid" &\nSERVER_PIDS+=($!)\n'
        '"$TEST_PYTHON" "$TEST_CHILD" "$TEST_DIR/trainer.pid" &\nTRAINER_PID=$!\n'
        'wait "$TRAINER_PID"\n'
    )
    env = dict(
        os.environ,
        TEST_PYTHON=sys.executable,
        TEST_CHILD=str(child),
        TEST_DIR=str(tmp_path),
    )
    process = subprocess.Popen(["bash", "-euo", "pipefail", "-c", script], env=env)
    child_pids: list[int] = []
    try:
        deadline = time.monotonic() + 5
        while not all(
            (tmp_path / f"{name}.pid").exists() for name in ("server", "trainer")
        ):
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child_pids = [
            int((tmp_path / f"{name}.pid").read_text())
            for name in ("server", "trainer")
        ]
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        for pid in child_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f"batch left child {pid} alive")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        for pid in child_pids:
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
