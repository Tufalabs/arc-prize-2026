"""Exercise batch allocation checks and cleanup without starting GPU processes."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
JOB_PATH = REPO_ROOT / "scripts/platypus_debug_smoke.sbatch"
JOB = JOB_PATH.read_text()


def _section(start: str, end: str) -> str:
    return JOB[JOB.index(start) : JOB.index(end, JOB.index(start))]


def _shell(script: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-euo", "pipefail", "-c", script],
        cwd=REPO_ROOT,
        env={
            **{
                key: value
                for key, value in os.environ.items()
                if not key.startswith(("ARC3_", "SERVER_"))
            },
            **env,
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


@pytest.fixture
def allocation(tmp_path: Path) -> tuple[dict, dict[str, str]]:
    config = json.loads(
        (REPO_ROOT / "configs/ppo_2gpu_arc3_games25_platypus.json").read_text()
    )
    config["inference"]["server_gpus"] = ["3"]
    config["inference"]["model_id"] = "fixture/served-alias"
    config["training"]["device"] = "cuda:0"
    config["training"]["device_map_gpus"] = [0]
    config["training"]["model_name"] = "fixture/configured-policy"
    config["training"]["model_revision"] = "pinned-revision"
    config["training"]["lora"]["r"] = 32
    return config, {
        "TRAIN_VENV": sys.prefix,
        "ARC3_FROZEN_CONFIG_PATH": str(tmp_path / "frozen_config.json"),
        "NUM_SERVERS": "1",
    }


def _validate(config: dict, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    Path(env["ARC3_FROZEN_CONFIG_PATH"]).write_text(json.dumps(config))
    return _shell(
        _section("# Validate the effective allocation", "# --- vLLM server(s)")
        + '\nprintf "%s\\n" "$SERVER_MODEL" "$SERVER_REVISION" "$SERVER_CONFIGURED_NAME" '
        '"$SERVER_MAX_LORA_RANK" "${SERVER_CONTEXT_WINDOWS[*]}" "${SERVER_GPUS[@]}"\n',
        env,
    )


@pytest.mark.parametrize(
    ("overrides", "expected_gpus"),
    [
        ({}, ["3"]),
        ({"ARC3_SERVER_GPUS": "2"}, ["2"]),
        (
            {"ARC3_SERVER_GPUS": "2,3", "ARC3_SERVER_TENSOR_PARALLEL": "2"},
            ["2,3"],
        ),
    ],
)
def test_effective_single_server_allocation_uses_config_and_overrides(
    allocation: tuple[dict, dict[str, str]],
    overrides: dict[str, str],
    expected_gpus: list[str],
) -> None:
    config, env = allocation
    result = _validate(config, {**env, **overrides})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "fixture/configured-policy",
        "pinned-revision",
        "fixture/served-alias",
        "32",
        "32768",
        *expected_gpus,
    ]


@pytest.mark.parametrize(
    ("gpus", "tp", "error"),
    [
        ("2,3", "1", "one distinct nonnegative GPU"),
        ("2,2", "2", "one distinct nonnegative GPU"),
        ("-1", "1", "one distinct nonnegative GPU"),
        ("2", "0", "one distinct nonnegative GPU"),
        ("0", "1", "claimed twice"),
    ],
)
def test_invalid_effective_allocation_fails_before_serving(
    allocation: tuple[dict, dict[str, str]], gpus: str, tp: str, error: str
) -> None:
    config, env = allocation
    result = _validate(
        config,
        {**env, "ARC3_SERVER_GPUS": gpus, "ARC3_SERVER_TENSOR_PARALLEL": tp},
    )
    assert result.returncode != 0
    assert error in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("claim", ["device", "replica"])
def test_override_cannot_overlap_implicit_device_or_worker_replica(
    allocation: tuple[dict, dict[str, str]], claim: str
) -> None:
    config, env = allocation
    if claim == "device":
        config["training"]["device_map_gpus"] = []
        config["training"]["device"] = "cuda:2"
    else:
        config["training"]["worker_device_map_gpus"] = [[2]]
    result = _validate(config, {**env, "ARC3_SERVER_GPUS": "2"})
    assert result.returncode != 0
    assert "GPUs [2] are claimed twice" in result.stderr


@pytest.mark.parametrize("eval_context", [None, 16384, 65536])
def test_eval_slots_keep_their_configured_gpus_and_context(
    allocation: tuple[dict, dict[str, str]],
    eval_context: int | None,
) -> None:
    config, env = allocation
    config["inference"]["server_gpus"] = ["2", "3"]
    config["official_eval"] = {
        "enabled": True,
        "games": ["ar25-0c556536"],
        "server_gpus": ["4", "5"],
        "context_window_tokens": eval_context,
    }
    # The single-server override cannot silently change one slot in a
    # multi-server allocation, including a dedicated eval slot.
    result = _validate(config, {**env, "NUM_SERVERS": "4", "ARC3_SERVER_GPUS": "0"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-5:] == [
        f"32768 32768 {eval_context or 32768} {eval_context or 32768}",
        "2", "3", "4", "5",
    ]


@pytest.mark.parametrize("mode", ["owned", "outside", "traversal", "symlink"])
def test_build_cleanup_only_removes_owned_job_local_directory(
    tmp_path: Path, mode: str
) -> None:
    local = tmp_path / "job-local"
    local.mkdir()
    owned = local / "arc3-venv-build-123.abcd"
    owned.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    marker = unrelated / "keep"
    marker.touch()
    if mode == "owned":
        build = owned
    elif mode == "outside":
        build = unrelated
    elif mode == "traversal":
        build = owned / ".." / ".." / "unrelated"
    else:
        build = local / "arc3-venv-build-link"
        build.symlink_to(unrelated, target_is_directory=True)
    result = _shell(
        _section("__arc3_cleanup_build()", "trap '__arc3_cleanup_build;")
        + '\n__arc3_cleanup_build\n[ -z "$BUILD_SRC" ]\n',
        {"TMPDIR": str(local), "BUILD_SRC": str(build)},
    )
    assert marker.exists()
    if mode == "owned":
        assert result.returncode == 0, result.stderr
        assert not owned.exists()
    else:
        assert result.returncode != 0
        assert "Refusing to remove unexpected build directory" in result.stderr
        assert owned.exists()


def test_server_shutdown_has_one_shared_grace_period() -> None:
    # Stub the process syscalls and advance Bash's elapsed-time counter on
    # each sleep. Two unresponsive servers must share one 30s grace period.
    script = """
SERVER_PIDS=(100001 100002)
kill() { if [ "$1" != -0 ]; then printf '%s\\n' "$*"; fi; }
sleep() { SECONDS=$((SECONDS + 10)); printf 'sleep\\n'; }
wait() { :; }
"""
    result = _shell(
        script
        + _section("__arc3_stop_servers()", "trap '__arc3_stop_servers;")
        + "\n__arc3_stop_servers\n",
        {},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "-TERM 100001",
        "-TERM 100002",
        "sleep",
        "sleep",
        "sleep",
        "-KILL -- -100001",
        "-KILL -- -100002",
    ]


def test_batch_script_is_valid_bash() -> None:
    subprocess.run(["bash", "-n", str(JOB_PATH)], check=True)
