from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from training import launch_frozen
from training.source_snapshot import FrozenRunPaths


ROOT = Path(__file__).resolve().parents[1]


def _payload(mode):
    payload = json.loads(
        (ROOT / "configs/ppo_2gpu_arc3_platypus_lora_smoke.json").read_text()
    )
    if mode == "full_weight":
        payload["training"].update(
            {
                "weight_mode": mode,
                "device": "cuda",
                "device_map_gpus": [],
                "worker_device_map_gpus": [],
                "full_weight": {"device_ids": [2, 4, 6]},
            }
        )
    return payload


@pytest.mark.parametrize("mode", ["lora", "full_weight"])
@pytest.mark.parametrize("resume", [False, True])
def test_frozen_launch_uses_selected_policy_mode_and_only_its_gpus(
    monkeypatch, tmp_path, mode, resume
):
    run_root = tmp_path / "run"
    source = run_root / "source"
    source.mkdir(parents=True)
    config_path = run_root / "frozen_config.json"
    config_path.write_text(json.dumps(_payload(mode)))
    paths = FrozenRunPaths(
        live_repo_root=tmp_path / "live",
        run_root=run_root,
        source_archive_path=run_root / "source.tar",
        source_root=source,
        frozen_config_path=config_path,
        snapshot_metadata_path=run_root / "source_snapshot.json",
        runtime_fingerprint_path=run_root / "runtime_fingerprint.json",
    )
    captured = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    monkeypatch.setenv("SERVER_WEIGHT_MODE", "opposite-mode-must-not-control-launch")
    monkeypatch.setattr(
        launch_frozen.os, "chdir", lambda path: captured.update(cwd=path)
    )

    def execute(executable, command, env):
        captured.update(executable=executable, command=command, env=env)

    monkeypatch.setattr(launch_frozen.os, "execvpe", execute)
    launch_frozen._exec_training_from_snapshot(
        paths=paths,
        run_dir=str(run_root) if resume else None,
        argv0_python="/python",
    )
    command = captured["command"]
    env = captured["env"]
    assert captured["cwd"] == source
    assert env["ARC3_RUN_DIR"] == str(run_root)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(source)
    assert command[command.index("--config") + 1] == str(config_path)
    if resume:
        assert command[-2:] == ["--run-dir", str(run_root)]
    else:
        assert "--run-dir" not in command
    if mode == "full_weight":
        assert command[1:3] == ["-m", "training.full_weight_launch"]
        assert {
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=3",
            "--max-restarts=0",
        }.issubset(command)
        assert (
            command[command.index("--module") + 1] == "training.full_weight_entrypoint"
        )
        assert env["CUDA_VISIBLE_DEVICES"] == "2,4,6"
    else:
        assert command[1:3] == ["-m", "training.cli"]
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"


@pytest.mark.parametrize("mode", ["lora", "full_weight"])
def test_batch_trainer_dispatch_uses_torchrun_only_for_full_weight(tmp_path, mode):
    batch = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    dispatch = batch[batch.index("# --- training (same venv") :]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    capture = tmp_path / "dispatch.json"
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['CAPTURE']).write_text(json.dumps({'argv': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}))\n"
    )
    python.chmod(0o755)
    setsid = bin_dir / "setsid"
    setsid.write_text('#!/bin/bash\nexec "$@"\n')
    setsid.chmod(0o755)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", dispatch],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TRAIN_VENV": str(tmp_path),
            "CAPTURE": str(capture),
            "SERVER_WEIGHT_MODE": mode,
            "FULL_WEIGHT_TRAINER_GPUS": "2,4,6",
            "ARC3_VISIBLE_GPUS": "0,1,2,3,4,5,6,7",
            "DEFAULT_VISIBLE_GPUS": "0,1",
            "ARC3_FROZEN_CONFIG_PATH": str(tmp_path / "frozen_config.json"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    launched = json.loads(capture.read_text())
    command = launched["argv"]
    assert command[-2:] == ["--config", str(tmp_path / "frozen_config.json")]
    if mode == "full_weight":
        assert command[:2] == ["-m", "training.full_weight_launch"]
        assert "--nproc-per-node=3" in command
        assert "--max-restarts=0" in command
        assert launched["gpus"] == "2,4,6"
    else:
        assert command[:2] == ["-m", "training.cli"]
        assert launched["gpus"] == "0,1,2,3,4,5,6,7"


def test_frozen_batch_shape_exports_full_weight_mode_and_checks_all_trainer_gpus(
    tmp_path,
):
    batch = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    start = batch.index("# Validate the effective allocation")
    shape = batch[start : batch.index("# --- vLLM server(s)", start)]
    config = tmp_path / "frozen_config.json"
    config.write_text(json.dumps(_payload("full_weight")))
    script = (
        shape
        + '\n"$TRAIN_VENV/bin/python" -c \'import os,json,sys; print(json.dumps({"SERVER_WEIGHT_MODE": os.environ.get("SERVER_WEIGHT_MODE"), "FULL_WEIGHT_TRAINER_GPUS": sys.argv[1]}))\' "$FULL_WEIGHT_TRAINER_GPUS"\n'
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ARC3_", "SERVER_"))
    }
    env.update(
        TRAIN_VENV=sys.prefix, ARC3_FROZEN_CONFIG_PATH=str(config), NUM_SERVERS="1"
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "SERVER_WEIGHT_MODE": "full_weight",
        "FULL_WEIGHT_TRAINER_GPUS": "2,4,6",
    }
    env["ARC3_SERVER_GPUS"] = "4"
    rejected = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env=env,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert rejected.returncode != 0
    assert "claimed twice" in rejected.stderr


@pytest.mark.parametrize("trainer_exit", [0, 75, 3])
def test_batch_trainer_dispatch_propagates_the_trainer_exit_code(
    tmp_path, trainer_exit
):
    """A segment yield (75) must reach the wrapper as the launcher's own exit."""
    batch = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    dispatch = batch[batch.index("# --- training (same venv") :]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    python = bin_dir / "python"
    python.write_text(
        f"#!{sys.executable}\nimport os, sys\nsys.exit(int(os.environ['TRAINER_EXIT']))\n"
    )
    python.chmod(0o755)
    setsid = bin_dir / "setsid"
    setsid.write_text('#!/bin/bash\nexec "$@"\n')
    setsid.chmod(0o755)
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", dispatch],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TRAIN_VENV": str(tmp_path),
            "TRAINER_EXIT": str(trainer_exit),
            "SERVER_WEIGHT_MODE": "lora",
            "FULL_WEIGHT_TRAINER_GPUS": "2",
            "ARC3_VISIBLE_GPUS": "0,1",
            "DEFAULT_VISIBLE_GPUS": "0,1",
            "ARC3_FROZEN_CONFIG_PATH": str(tmp_path / "frozen_config.json"),
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == trainer_exit, result.stderr
    assert ("yielded at a safe boundary" in result.stdout) == (trainer_exit == 75)


@pytest.mark.parametrize(
    ("trainer_rc", "endsig", "expect_chain"),
    [(75, 0, True), (0, 0, False), (0, 1, True), (3, 0, False)],
)
def test_auto_relaunch_chains_on_a_segment_yield(
    tmp_path, trainer_rc, endsig, expect_chain
):
    batch = (ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    limit_start = batch.index("__arc3_limit_seconds() {")
    limit_end = batch.index("\n}\n", limit_start) + 3
    relaunch_start = batch.index("__arc3_maybe_relaunch() {")
    relaunch_end = batch.index("\n}\n", relaunch_start) + 3
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "scontrol").write_text(
        "#!/bin/bash\n"
        "echo 'JobId=42 JobState=TIMEOUT TimeLimit=12:00:00 "
        "TresPerNode=gres/gpu:b200:2 CpusPerTres=gres/gpu:8 MemPerTres=gres/gpu:96G QOS=b200-pool'\n"
    )
    (bin_dir / "sbatch").write_text(
        '#!/bin/bash\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\n'
        'printf "%s" "${ARC3_SEGMENT_DEADLINE_EPOCH:-}" > "$DEADLINE_CAPTURE"\n'
        "echo 4242\n"
    )
    for helper in ("scontrol", "sbatch"):
        (bin_dir / helper).chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "platypus_debug_smoke.sbatch").write_text("#!/bin/bash\n")
    capture = tmp_path / "sbatch.args"
    script = (
        f"REPO_ROOT={tmp_path}\nRUN_DIR={run_dir}\n"
        f"__arc3_got_endsig={endsig}\n__arc3_script_started_at=$(date +%s)\n"
        f"TRAINER_RC={trainer_rc}\n"
        + batch[limit_start:limit_end]
        + batch[relaunch_start:relaunch_end]
        + "__arc3_maybe_relaunch\n"
    )
    result = subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "ARC3_AUTO_RELAUNCH": "1",
            "SLURM_JOB_ID": "42",
            "SBATCH_CAPTURE": str(capture),
            "DEADLINE_CAPTURE": str(tmp_path / "deadline"),
            "ARC3_SEGMENT_DEADLINE_EPOCH": "1",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert capture.exists() == expect_chain, result.stdout
    if expect_chain:
        assert (tmp_path / "deadline").read_text() == ""
        next_allocation = subprocess.run(
            [
                "bash",
                "-euo",
                "pipefail",
                "-c",
                "__arc3_script_started_at=1000\n"
                + batch[limit_start:relaunch_start]
                + '\nprintf "%s" "$ARC3_SEGMENT_DEADLINE_EPOCH"\n',
            ],
            env={
                **os.environ,
                "PATH": f"{bin_dir}:{os.environ['PATH']}",
                "SLURM_JOB_ID": "4242",
                "ARC3_SEGMENT_DEADLINE_EPOCH": (tmp_path / "deadline").read_text(),
            },
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert next_allocation.returncode == 0, next_allocation.stderr
        assert next_allocation.stdout.splitlines()[-1] == str(1000 + 12 * 3600 - 300)
        args = capture.read_text().splitlines()
        assert "--dependency=afterany:42" in args
        assert "--time=12:00:00" in args
        assert "--qos=b200-pool" in args
        assert args[-1] == str(tmp_path / "scripts" / "platypus_debug_smoke.sbatch")
    if trainer_rc == 75:
        assert "yielded its segment" in result.stdout


@pytest.mark.parametrize(
    "outcome,expected_exit", [("complete", 0), ("yield", 75), ("peer_failure", 1)]
)
def test_real_torchrun_preserves_only_clean_segment_yields(
    tmp_path, outcome, expected_exit
):
    """Use real launcher processes, including a peer failing after rank zero yields."""
    worker = tmp_path / "worker.py"
    worker.write_text(
        "import os, sys\n"
        "from training import full_weight_entrypoint as entry\n"
        "rank = int(os.environ['RANK'])\n"
        f"outcome = {outcome!r}\n"
        "entry.run = lambda _, **kwargs: 75 if rank == 0 and outcome != 'complete' else 0\n"
        "sys.argv = ['worker', '--config', 'unused.json']\n"
        "entry.main()\n"
        "if rank == 1 and outcome == 'peer_failure': sys.exit(3)\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "training.full_weight_launch",
            "--standalone",
            "--nnodes=1",
            "--nproc-per-node=2",
            "--max-restarts=0",
            str(worker),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == expected_exit, result.stderr
