import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from training.source_snapshot import prepare_frozen_run

REPO_ROOT = Path(__file__).resolve().parents[1]


def _free_server_port_block(slots: int) -> int:
    """A base port whose server slots, base + 100 * i, the launcher accepts.

    The job tail starts slot i on base + 100 * i, and
    launch_platypus_server.sh refuses a port it cannot bind on every
    interface with SO_REUSEADDR. A base picked on 127.0.0.1 alone left the
    other slot unchecked, and ephemeral ports are often taken on a busy node.
    """

    def accepted(port: int) -> bool:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("", port))
            except OSError:
                return False
            return True

    for _ in range(100):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            base = listener.getsockname()[1]
        ports = [base + 100 * slot for slot in range(slots)]
        if ports[-1] <= 65535 and all(map(accepted, ports)):
            return base
    raise RuntimeError("no free block of server ports")


@pytest.fixture
def launch_env(tmp_path: Path) -> dict[str, str]:
    live_repo = tmp_path / "live repo"
    # Config validation now imports context/Flash settings and harness support.
    # Copy the packages so the isolated snapshot has their transitive imports;
    # the GPU entrypoints below are still replaced with CPU stand-ins.
    for package in ("training", "harness_support"):
        shutil.copytree(
            REPO_ROOT / package, live_repo / package,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    for relative in (
        "scripts/launch_platypus_server.sh",
        "configs/ppo_2gpu_arc3_games25_platypus.json",
    ):
        target = live_repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, target)
    (live_repo / "training/vllm_runtime.py").write_text('VERSION = "frozen"\n')
    # The trainer and vLLM entrypoint are CPU stand-ins. Everything that
    # chooses the snapshot, launcher, cwd and Python import paths is real.
    (live_repo / "training/cli.py").write_text(
        "import json, os, sys, time\n"
        "from pathlib import Path\n"
        "from training import vllm_runtime\n"
        "run = Path(os.environ['ARC3_RUN_DIR'])\n"
        "deadline = time.monotonic() + 10\n"
        "while len(list(run.glob('server-*.json'))) < int(os.environ['NUM_SERVERS']):\n"
        "    if time.monotonic() > deadline:\n"
        "        raise RuntimeError('server did not start')\n"
        "    time.sleep(0.01)\n"
        "(run / 'trainer.json').write_text(json.dumps({\n"
        "    'version': vllm_runtime.VERSION, 'module': vllm_runtime.__file__,\n"
        "    'cwd': os.getcwd(), 'argv': sys.argv[1:],\n"
        "    'env': {key: os.environ[key] for key in (\n"
        "        'ARC3_RUN_DIR', 'ARC3_ORIGINAL_REPO_ROOT', 'ARC3_FROZEN_REPO_ROOT',\n"
        "        'ARC3_FROZEN_CONFIG_PATH', 'ARC3_RUNTIME_FINGERPRINT_PATH',\n"
        "        'FLA_TILELANG', 'FLA_CONV_BACKEND')},\n"
        "}))\n"
    )
    dependencies = tmp_path / "dependencies"
    for package in (
        "ullm",
        "ullm/backends",
    ):
        directory = dependencies / package
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").touch()
    (dependencies / "ullm/backends/config_local.py").write_text(
        "def set_env():\n    pass\n"
    )
    (live_repo / "training/native_vllm_server.py").write_text(
        "import json, os, sys\n"
        "from http.server import BaseHTTPRequestHandler, HTTPServer\n"
        "from pathlib import Path\n"
        "from training import vllm_runtime\n"
        "port = sys.argv[sys.argv.index('--port') + 1]\n"
        "name = sys.argv[sys.argv.index('--served-model-name') + 1]\n"
        "model = sys.argv[sys.argv.index('--model') + 1]\n"
        "(Path(os.environ['ARC3_RUN_DIR']) / f'server-{port}.json').write_text(json.dumps({\n"
        "    'version': vllm_runtime.VERSION, 'module': vllm_runtime.__file__,\n"
        "    'cwd': os.getcwd(), 'served_name': name, 'model': model,\n"
        "    'gpu': os.environ['CUDA_VISIBLE_DEVICES'],\n"
        "    'revision': os.environ['SERVER_REVISION'],\n"
        "    'lora_rank': sys.argv[sys.argv.index('--max-lora-rank') + 1],\n"
        "    'context_window': sys.argv[sys.argv.index('--max-model-len') + 1],\n"
        "}))\n"
        "key = os.environ['SERVER_API_KEY']\n"
        "assert '--api-key' not in sys.argv and key not in sys.argv\n"
        "class Handler(BaseHTTPRequestHandler):\n"
        "    def do_GET(self):\n"
        "        accepted = self.path == '/models' and self.headers.get('Authorization') == 'Bearer ' + key\n"
        "        self.send_response(200 if accepted else 401)\n"
        "        self.end_headers()\n"
        "    def log_message(self, *args):\n"
        "        pass\n"
        "HTTPServer(('127.0.0.1', int(port)), Handler).serve_forever()\n"
    )
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    # A symlink under a fake venv loses the real interpreter's site-packages.
    # The readiness helper uses requests, so delegate to the actual test env.
    (venv / "bin/python").write_text(
        "#!/bin/sh\nexec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    (venv / "bin/python").chmod(0o755)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = run_dir / "launch_config.json"
    payload = json.loads(
        (live_repo / "configs/ppo_2gpu_arc3_games25_platypus.json").read_text()
    )
    payload["inference"]["platypus_venv"] = str(venv)
    payload["inference"]["server_gpus"] = ["0"]
    payload["official_eval"] = {
        "enabled": True,
        "games": ["ar25-0c556536"],
        "server_gpus": ["1"],
        "context_window_tokens": 65536,
    }
    payload["inference"]["model_id"] = "fixture/served-alias"
    payload["training"]["device"] = "cuda:2"
    payload["training"]["device_map_gpus"] = [2]
    payload["training"]["model_name"] = "fixture/policy"
    payload["training"]["model_revision"] = "fixture-pinned-revision"
    payload["training"]["lora"]["r"] = 32
    config.write_text(json.dumps(payload))
    (tmp_path / "model").mkdir()
    port = _free_server_port_block(slots=2)
    return {
        **{
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("ARC3_", "SERVER_", "FLA_"))
        },
        "REPO_ROOT": str(live_repo),
        "RUN_DIR": str(run_dir),
        "LAUNCH_CONFIG": str(config),
        "TRAIN_VENV": str(venv),
        "PLATYPUS_VENV": str(venv),
        "ARC3_RL_CACHE_ROOT": str(tmp_path / "cache"),
        "ARC3_SERVER_MODEL_OVERRIDE": "../model",
        "PYTHONPATH": os.pathsep.join((str(live_repo), str(dependencies))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PATH": str(venv / "bin") + os.pathsep + os.environ["PATH"],
        "NUM_SERVERS": "2",
        "SERVER_PORT": str(port),
        "SERVER_API_KEY": "frozen-launch-fixture-key",
    }


def _prepare(env: dict[str, str]) -> Path:
    return prepare_frozen_run(
        config_path=env["LAUNCH_CONFIG"],
        live_repo_root=Path(env["REPO_ROOT"]),
        run_root=Path(env["RUN_DIR"]),
        training_python=Path(sys.executable),
        source_workspace=Path(env["ARC3_RL_CACHE_ROOT"]) / "source",
    ).source_root


@pytest.mark.parametrize("mode", ["new", "resume", "force"])
def test_job_server_and_trainer_share_one_snapshot(
    launch_env: dict[str, str], mode: str
) -> None:
    env = launch_env
    live_repo = Path(env["REPO_ROOT"])
    run_dir = Path(env["RUN_DIR"])
    old_archive = None
    if mode != "new":
        _prepare(env)
        old_archive = (run_dir / "source.tar").read_bytes()
        (live_repo / "training/vllm_runtime.py").write_text('VERSION = "live"\n')
        launcher = live_repo / "scripts/launch_platypus_server.sh"
        launcher.write_text(
            launcher.read_text().replace("Qwen/Qwen3.8-27B", "live-model")
        )
    if mode == "force":
        env["ARC3_FORCE_SNAPSHOT"] = "1"
    # Execute the real job's entire serving/training tail, bypassing only
    # cluster provisioning and venv/model setup. Both server slots run it.
    job = (REPO_ROOT / "scripts/platypus_debug_smoke.sbatch").read_text()
    tail = job[job.index("# Freeze once,") :]
    result = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            "SERVER_GPUS=(0 1)\n__arc3_maybe_relaunch() { :; }\n"
            "__arc3_cleanup_build() { :; }\n" + tail,
        ],
        cwd=live_repo,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    trainer = json.loads((run_dir / "trainer.json").read_text())
    frozen_root = Path(trainer["env"]["ARC3_FROZEN_REPO_ROOT"])
    expected_version = "live" if mode == "force" else "frozen"
    assert trainer["version"] == expected_version
    assert frozen_root != live_repo
    assert trainer["cwd"] == str(frozen_root)
    assert trainer["module"] == str(frozen_root / "training/vllm_runtime.py")
    assert (frozen_root / "training/server_endpoints.py").read_bytes() == (
        REPO_ROOT / "training/server_endpoints.py"
    ).read_bytes()
    assert env["SERVER_API_KEY"] not in result.stdout + result.stderr
    assert trainer["argv"] == ["--config", str(run_dir / "frozen_config.json")]
    assert trainer["env"] == {
        "ARC3_RUN_DIR": str(run_dir),
        "ARC3_ORIGINAL_REPO_ROOT": str(live_repo),
        "ARC3_FROZEN_REPO_ROOT": str(frozen_root),
        "ARC3_FROZEN_CONFIG_PATH": str(run_dir / "frozen_config.json"),
        "ARC3_RUNTIME_FINGERPRINT_PATH": str(run_dir / "runtime_fingerprint.json"),
        "FLA_TILELANG": "0",
        "FLA_CONV_BACKEND": "triton",
    }
    servers = list(run_dir.glob("server-*.json"))
    assert len(servers) == 2
    for report in servers:
        server = json.loads(report.read_text())
        assert server == {
            "version": expected_version,
            "module": trainer["module"],
            "cwd": trainer["cwd"],
            "served_name": "fixture/served-alias",
            "model": str((live_repo / "../model").resolve()),
            "gpu": "0" if report.stem == f"server-{env['SERVER_PORT']}" else "1",
            "revision": "fixture-pinned-revision",
            "lora_rank": "32",
            "context_window": "32768" if server["gpu"] == "0" else "65536",
        }
    if mode == "resume":
        assert (run_dir / "source.tar").read_bytes() == old_archive
    elif mode == "force":
        assert (run_dir / "source.tar").read_bytes() != old_archive
