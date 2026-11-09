"""Separate allocation routing and Slurm commands, without GPU processes."""

import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from flash_configs import RECIPE, write_arc3_flash_config
from training.config import ExperimentConfig
from training.inference import _per_server_configs
from training.job_shape import job_shape, separate_shape, write_launch_config

ROOT = Path(__file__).resolve().parents[1]
HET = (ROOT / "scripts/flash_rl_het.sh").read_text()


def without(*prefixes):
    """The test process's environment minus variables with these prefixes."""
    return {key: value for key, value in os.environ.items() if not key.startswith(prefixes)}


def stub_commands(tmp_path, sbatch="#!/bin/sh\necho submitted\n"):
    """sbatch and mkdir stubs for the submit helper; mkdir is its only filesystem
    action, and /data does not exist off the cluster."""
    commands = tmp_path / "bin"
    commands.mkdir(exist_ok=True)
    for name, text in (("sbatch", sbatch), ("mkdir", "#!/bin/sh\nexit 0\n")):
        (commands / name).write_text(text)
        (commands / name).chmod(0o755)
    return commands


def load(tmp_path, **inference):
    payload = json.loads(RECIPE.read_text())
    payload["inference"].update(inference)
    payload["training"]["flash"]["device_ids"] = [0, 1, 2, 3]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return ExperimentConfig.load(path)


def test_separate_gpu_claims_and_missing_runtime_endpoints(tmp_path):
    cfg = load(tmp_path, server_allocation="separate", server_gpus=["0", "1", "0", "1"])
    assert cfg.inference.base_urls == []
    with pytest.raises(ValueError, match="base_urls before training"):
        _per_server_configs(cfg)
    with pytest.raises(ValueError, match="must not repeat|disjoint"):
        load(tmp_path, server_gpus=["0", "1", "0", "1"])


@pytest.mark.parametrize("fields", [
    {"server_allocation": "other"},
    {"base_urls": "http://host:1234"},
    {"base_urls": ["http://host:1234"]},
    {"base_urls": [1, 2, 3, 4]},
    {"base_urls": ["http://host:1234"] * 4},
    {"base_urls": [f"file://host:{port}" for port in range(1234, 1238)]},
    {"server_gpus": ["-1", "1", "2", "3"]},
])
def test_invalid_fleet_config_is_rejected(tmp_path, fields):
    with pytest.raises(ValueError):
        load(tmp_path, **{"server_allocation": "separate", **fields})


def test_explicit_endpoints_include_server_zero_and_keep_logs(tmp_path):
    urls = [f"http://node{i // 2}:{50000 + 100*i}" for i in range(4)]
    logs = [str(tmp_path / f"server_{i}.log") for i in range(4)]
    cfg = load(tmp_path, server_allocation="separate", server_gpus=["0", "1", "0", "1"],
               base_urls=urls, server_log_paths=logs)
    children = _per_server_configs(cfg)
    assert [c.inference.base_url for c in children] == urls
    assert all(c.inference.server_log_paths == logs for c in children)
    assert cfg.inference.base_url != urls[0]


def test_shared_default_keeps_derived_urls():
    cfg = ExperimentConfig.load(RECIPE)
    assert cfg.inference.server_allocation == "shared"
    assert cfg.inference.base_urls == []
    assert [c.inference.base_url for c in _per_server_configs(cfg)] == [
        f"http://127.0.0.1:{58900 + 100*i}" for i in range(4)
    ]


@pytest.mark.parametrize(("nodes", "gpus"), [(["dgx01"], 4), (["dgx01"], 6),
                                               (["dgx01"], 8), (["dgx01", "dgx03"], 6)])
def test_separate_shape_and_config(tmp_path, nodes, gpus):
    original = ExperimentConfig.load(RECIPE)
    shape = separate_shape(original, nodes, gpus, 4, 50000, tmp_path)
    assert shape["SERVER_GPU_LIST"].split() == [str(g) for _ in nodes for g in range(gpus)]
    assert shape["TRAINER_GPUS"] == "0,1,2,3"
    assert shape["DEFAULT_VISIBLE_GPUS"] == "0,1,2,3"
    assert shape["SERVER_MODEL"] == original.training.flash.checkpoint_root
    assert original.training.flash.device_ids == [4, 5, 6, 7]
    target = tmp_path / "launch.json"
    write_launch_config(RECIPE, target, shape, 50000, "/cache/harness")
    cfg = ExperimentConfig.load(target)
    assert cfg.training.flash.device_ids == [0, 1, 2, 3]
    assert cfg.inference.server_allocation == "separate"
    assert cfg.inference.server_log_paths == [str(tmp_path / f"server_{i}.log") for i in range(len(nodes)*gpus)]
    assert cfg.inference.base_urls == [f"http://{host}:{50000 + 100*(n*gpus+g)}"
                                       for n, host in enumerate(nodes) for g in range(gpus)]
    assert job_shape(cfg, {}) == shape


def test_learner_count_requires_explicit_rewrite(tmp_path):
    cfg = ExperimentConfig.load(RECIPE)
    with pytest.raises(ValueError, match="rewrite-learner-devices"):
        separate_shape(cfg, ["dgx01"], 6, 2, 50000, tmp_path)
    shape = separate_shape(cfg, ["dgx01"], 6, 2, 50000, tmp_path, rewrite_learner_devices=True)
    assert shape["TRAINER_GPUS"] == "0,1"
    with pytest.raises(ValueError, match="ports"):
        separate_shape(cfg, ["dgx01", "dgx02"], 8, 4, 60000, tmp_path)


@pytest.mark.parametrize(("nodes", "gpus", "groups", "components"), [(0, 4, 1, 1), (1, 6, 1, 2), (1, 8, 1, 2), (2, 6, 1, 2), (1, 6, 2, 3)])
def test_submit_helper_composes_each_component(tmp_path, nodes, gpus, groups, components):
    record = tmp_path / "submission.json"
    commands = stub_commands(tmp_path, f"#!{sys.executable}\nimport json,sys\nfrom pathlib import Path\n"
                                       f"Path({str(record)!r}).write_text(json.dumps(sys.argv[1:]))\n")
    env = {**os.environ, "PATH": f"{commands}:{os.environ['PATH']}",
           "ARC3_FLASH_VENV_TARBALL": "/data/trainer.tar.zst", "ARC3_HARNESS_VENV_TARBALL": "/data/harness.tar.zst",
           # The recipe trains ARC-2.
           "ARC3_ARC2_WORKER_VENV_TARBALL": "/data/arc2-worker.tar.zst"}
    result = subprocess.run(["bash", str(ROOT / "scripts/flash_rl_submit.sh"), "--learner-gpus", "4",
                             "--server-nodes", str(nodes), "--server-gpus-per-node", str(gpus),
                             "--server-groups", str(groups), "--exclude", "dgx02", "--run-dir", "/data/test-run",
                             "--config", str(RECIPE), "--dependency", "afterany:123", "--dry-run",
                             "--", "/data/fetch-only", "abc123"],
                            env=env, text=True, capture_output=True, check=True)
    args = json.loads(record.read_text())
    assert args.count(":") == components - 1
    assert args.count("--output=/data/test-run/slurm-%j.out") == components
    assert args.count("--exclude=dgx02") == components
    assert args.count("--signal=B:TERM@600") == components
    assert args.count("--dependency=afterany:123") == 1  # one job-level flag, before the first component
    assert args.index("--dependency=afterany:123") < args.index("--job-name=arc3-flash-rl")
    assert f"--gres=gpu:b200:{8 if nodes == 0 else 4}" in args
    assert args[-2:] == ["/data/fetch-only", "abc123"]
    assert "SERVER_API_KEY" not in result.stdout


def submit(tmp_path, config, **env):
    """Run the submit helper with stubbed sbatch/mkdir; config=None omits --config."""
    commands = stub_commands(tmp_path)
    environment = {**without("ARC3_"),
                   "PATH": f"{commands}:{os.environ['PATH']}", "ARC3_FLASH_VENV_TARBALL": "/data/trainer.tar.zst",
                   "ARC3_HARNESS_VENV_TARBALL": "/data/harness.tar.zst", **env}
    options = [] if config is None else ["--config", str(config)]
    return subprocess.run(["bash", str(ROOT / "scripts/flash_rl_submit.sh"), "--run-dir", "/data/test-run",
                           *options, "--", "/data/fetch-only", "abc123"],
                          env=environment, text=True, capture_output=True)


def test_submit_helper_requires_the_arc2_worker_for_arc2_configs(tmp_path):
    # No worker tarball, and no clone to resolve one from: the helper names the variable.
    missing = submit(tmp_path, RECIPE)
    assert missing.returncode == 2
    assert "Cannot resolve the environments for: arc2-worker" in missing.stderr
    assert "ARC3_ARC2_WORKER_VENV_TARBALL" in missing.stderr
    given = submit(tmp_path, RECIPE, ARC3_ARC2_WORKER_VENV_TARBALL="/data/arc2-worker.tar.zst")
    assert given.returncode == 0, given.stderr
    assert submit(tmp_path, write_arc3_flash_config(tmp_path)).returncode == 0
    # A config path only the job can resolve (inside its archived revision) is left to
    # the job when nothing has to be derived from it: server concurrency is the job's.
    unreadable = submit(tmp_path, "configs/not-here.json", ARC3_HARNESS_VENV_TARBALL="")
    assert unreadable.returncode == 2
    assert "Cannot read configs/not-here.json here" in unreadable.stderr
    assert submit(tmp_path, "configs/not-here.json").returncode == 0


def prepare_het(tmp_path, config, gpus=4, srun="#!/bin/sh\nexit 0\n", **extra):
    """flash_rl_het.sh's prepare_het for a learner and one server node (dgx05),
    each allocated GPUS GPUs; returns the process and the job shape it wrote.
    EXTRA adds to the job's environment."""
    commands = tmp_path / "bin"
    commands.mkdir(exist_ok=True)
    (commands / "srun").write_text(srun)
    (commands / "srun").chmod(0o755)
    script = HET + f'''
timeout() {{ shift; "$@"; }}
scontrol() {{
  case "$1 $2" in
    "show hostnames") echo "$3" | tr , '\\n' ;;
    "show job") echo "JobId=$4 TresPerNode=gres/gpu:b200:{gpus}" ;;
  esac
}}
run() {{ shift; shift; "$@"; }}
port_free() {{ printf '%s\\n' "$1" >> "$WORK/gateway-ports"; }}
prepare_het
cat "$WORK/shape"
'''
    env = {**without("ARC3_", "SERVER_", "SLURM_"),
           "PATH": f"{commands}:{os.environ['PATH']}", "ARC3_DRY_RUN": "1",
           "ARC3_FLASH_PYTHON": sys.executable, "SRC_CONFIG": str(config), "WORK": str(tmp_path),
           "LAUNCH_CONFIG": str(tmp_path / "launch_config.json"), "RUN_DIR": str(tmp_path),
           "WORKER_VENV": "/cache/arc2-worker", "PLATYPUS_VENV": "/cache/harness", "SLURM_JOB_ID": "123",
           "SLURM_HET_SIZE": "2", "SLURM_JOB_ID_HET_GROUP_0": "123", "SLURM_JOB_ID_HET_GROUP_1": "124",
           "SLURM_JOB_NODELIST_HET_GROUP_1": "dgx05", "CMAKE_BUILD_PARALLEL_LEVEL": "48",
           "PYTHONPATH": str(ROOT), **extra}
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], cwd=ROOT, env=env,
                            text=True, capture_output=True, check=False)
    shape = dict(line.split("=", 1) for line in result.stdout.splitlines()) if result.returncode == 0 else {}
    return result, shape


def test_het_preparation_writes_the_worker_venv_and_served_ceiling(tmp_path):
    result, shape = prepare_het(tmp_path, RECIPE)
    assert result.returncode == 0, result.stderr
    assert shape["SERVER_ENVIRONMENT"] == "arc2"
    assert shape["SERVER_MAX_NUM_SEQS"] == "40"  # ceil(128 / 4 servers) + 8
    assert shlex.split(shape["SERVER_CONTEXT_LIST"])[0].split() == ["262144"] * 4
    cfg = ExperimentConfig.load(tmp_path / "launch_config.json")
    assert cfg.inference.platypus_venv == "/cache/arc2-worker"
    assert cfg.inference.base_urls == [f"http://dgx05:{50123 + 100 * i}" for i in range(4)]
    assert cfg.training.flash.device_ids == [0, 1, 2, 3]


@pytest.mark.parametrize("busy", ["never", "first", "always"])
def test_het_preparation_probes_nodes_and_retries_the_whole_block(tmp_path, busy):
    # Run the remote Python probe with fake node-local sockets. Only dgx03's
    # second engine collides; every command and attempted bind is recorded.
    srun = f"#!{sys.executable}\n" + '''
import json, os, socket, sys
from pathlib import Path
args = sys.argv[1:]
node = next(arg.split("=", 1)[1] for arg in args if arg.startswith("--nodelist="))
code_index = args.index("-c") + 1
sys.argv = args[code_index:]
ports = []
class ProbeSocket:
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def setsockopt(self, *args):
        assert args == (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    def bind(self, address):
        assert address[0] == ""
        ports.append(address[1])
        busy = os.environ["PROBE_BUSY"]
        if node == "dgx03" and len(ports) == 2 and (
            busy == "always" or (busy == "first" and sys.argv[1] == "50123")
        ):
            raise OSError(98, "Address already in use")
socket.socket = ProbeSocket
socket.gethostname = lambda: node
assert not Path(os.environ["LAUNCH_CONFIG"]).exists()
for name in ("TMPDIR", "TEMPDIR", "SCRATCH_DIR", "SLURM_TMPDIR"):
    assert name not in os.environ
try:
    exec(sys.argv[0], {})
finally:
    with (Path(os.environ["WORK"]) / "probes.jsonl").open("a") as log:
        log.write(json.dumps({"args": args, "ports": ports}) + "\\n")
'''
    nodes = ["dgx01", "dgx03", "dgx05"]
    result, shape = prepare_het(
        tmp_path, RECIPE, gpus=2, srun=srun, PROBE_BUSY=busy,
        SLURM_HET_SIZE="3", SLURM_JOB_NODELIST_HET_GROUP_1="dgx01,dgx03",
        SLURM_JOB_NODELIST_HET_GROUP_2="dgx05", SLURM_JOB_ID_HET_GROUP_2="125",
        TMPDIR="/learner/tmp", TEMPDIR="/learner/tmp", SCRATCH_DIR="/learner/scratch",
        SLURM_TMPDIR="/learner/tmp",
    )
    attempts = {"never": 1, "first": 2, "always": 40}[busy]
    probes = [json.loads(line) for line in (tmp_path / "probes.jsonl").read_text().splitlines()]
    expected = [(attempt, node) for attempt in range(attempts)
                for node in range(2 if busy == "always" or (busy == "first" and attempt == 0) else 3)]
    assert len(probes) == len(expected)
    for probe, (attempt, node) in zip(probes, expected, strict=True):
        args = probe["args"]
        assert f"--het-group={1 if node < 2 else 2}" in args
        assert f"--nodelist={nodes[node]}" in args
        assert {"--nodes=1", "--ntasks=1", "--overlap"} <= set(args)
        assert f"--chdir={tmp_path}" in args
        assert probe["ports"] == [50123 + 101 * attempt + 100 * (node * 2 + gpu) for gpu in range(2)]
    assert (tmp_path / "gateway-ports").read_text().splitlines() == [
        str(50123 + 101 * attempt + 100 * i + 5000) for attempt in range(attempts) for i in range(6)
    ]
    if busy == "always":
        assert result.returncode == 1
        assert "No free learner gateway and server engine port block" in result.stderr
        assert not (tmp_path / "launch_config.json").exists()
    else:
        assert result.returncode == 0, result.stderr
        base = 50123 + 101 * (attempts - 1)
        urls = [f"http://{host}:{base + 100 * (node * 2 + gpu)}"
                for node, host in enumerate(nodes) for gpu in range(2)]
        assert shlex.split(shape["SERVER_URL_LIST"])[0].split() == urls
        cfg = ExperimentConfig.load(tmp_path / "launch_config.json")
        assert cfg.inference.base_urls == urls
        assert cfg.inference.base_url == urls[0]
    if busy != "never":
        assert "engine port 50423 is in use on dgx03" in result.stderr


def test_rollout_node_bind_failure_names_node_and_port():
    node = (ROOT / "scripts/flash_rollout_node.sh").read_text()
    snippet = node.split('run timeout 15s "$HARNESS_PYTHON" -c ', 1)[1].split(' "$SERVER_PORT"', 1)[0]
    (code,) = shlex.split(snippet)
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("", 0))
        listener.listen()
        port = listener.getsockname()[1]
        result = subprocess.run([sys.executable, "-c", code, str(port)],
                                text=True, capture_output=True, check=False)
        assert result.returncode == 1
        assert result.stderr == f"engine port {port} is in use on {socket.gethostname()}\n"
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            pass  # The listener is still serving.


@pytest.mark.parametrize(("gpus", "explicit", "seqs"), [
    (2, None, "72"),  # ceil(128 / 2) + 8
    (4, "32", "32"),  # an explicit value holding the share of 32
    (4, "31", None),  # below the share: refused before any server starts
])
def test_het_preparation_sizes_engines_for_the_whole_fleet(tmp_path, gpus, explicit, seqs):
    payload = json.loads(RECIPE.read_text())
    payload["training"]["flash"]["device_ids"] = [0, 1, 2, 3]
    config = tmp_path / "recipe.json"
    config.write_text(json.dumps(payload))
    result, shape = prepare_het(tmp_path, config, gpus=gpus,
                                **({} if explicit is None else {"SERVER_MAX_NUM_SEQS": explicit}))
    if seqs is None:
        assert result.returncode != 0
        assert "SERVER_MAX_NUM_SEQS=31 must be an integer of at least 32" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert shape["SERVER_MAX_NUM_SEQS"] == seqs


def test_het_servers_get_the_jobs_concurrency(tmp_path):
    """The rollout nodes take each engine's --max-num-seqs from the job's shape,
    not from the submitting environment (srun --export=ALL carries no derived value)."""
    script = HET + f'''
run() {{ shift; shift; "$@"; }}
print_command() {{ printf '%q ' "$@"; printf '\\n'; }}
RUN_DIR={shlex.quote(str(tmp_path))} SLURM_JOB_ID=123 SLURM_HET_SIZE=2 ARC3_FROZEN_REPO_ROOT={shlex.quote(str(ROOT))}
SERVER_CONTEXT_WINDOWS=(49152) SERVER_GPUS_PER_NODE=4 GROUP_NODE_COUNTS=(1) GROUP_JOB_IDS=(124)
SERVER_PORT=50123 SERVER_MAX_NUM_SEQS=40 SERVER_PIDS=() ARC3_DRY_RUN=0
launch_het_servers
wait
'''
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("setsid", "srun"):
        (commands / name).write_text("#!/bin/sh\nexit 0\n")
        (commands / name).chmod(0o755)
    env = {**without("ARC3_", "SERVER_", "SLURM_"), "PATH": f"{commands}:{os.environ['PATH']}"}
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], env=env,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    (command,) = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("srun ")]
    (export,) = [arg for arg in command if arg.startswith("--export=")]
    assert export.split(",")[1:] == ["ARC3_SERVER_NODE_OFFSET=0", "ARC3_SERVER_JOB_ID=124",
                                     "SERVER_MAX_NUM_SEQS=40"]


def test_slurm_gpu_parser_crosschecks_real_tres():
    script = '''
timeout() { shift; "$@"; }
scontrol() {
  echo 'JobId=122 TresPerNode=gres/gpu:b200:4'
  echo 'JobId=123 TresPerNode=gres/gpu:b200:6'
}
'''
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", HET + script + '\nallocation_gpus 123 ""\n'],
                            text=True, capture_output=True, check=True)
    assert result.stdout.strip() == "6"


def test_rollout_node_commands_have_offsets_and_no_secret():
    node = (ROOT / "scripts/flash_rollout_node.sh").read_text()
    snippet = node[node.index("export SERVER_HOST="):node.index('if [[ "${ARC3_DRY_RUN:-0}" != 1 ]]; then\n  deadline=')]
    prefix = '''
run() { :; }
NODE_INDEX=1 GPU_COUNT=6 BASE_PORT=50000 RUN_DIR=/data/run WORK=/scratch/work
ARC3_DRY_RUN=1 HARNESS_PYTHON=/cache/python PLATYPUS_VENV=/cache/harness
SERVER_MODEL=/checkpoint SERVER_WEIGHT_MODE=flash SERVER_SERVED_NAME=Flash
SERVER_MAX_MODEL_LEN=32768 SERVER_ENVIRONMENT=arc3 SERVER_MULTIMODAL_CONTEXT=current_grid
SERVER_SCORE_CENTERING_TOP_K=0 ARC3_RL_CACHE_ROOT=/cache/rl
SERVER_API_KEY=do-not-print-this-key SERVER_MAX_NUM_SEQS=40
'''
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", prefix + snippet],
                            text=True, capture_output=True, check=True)
    commands = [shlex.split(line) for line in result.stdout.splitlines()]
    assert len(commands) == 6
    for gpu, command in enumerate(commands):
        assert f"SERVER_GPU={gpu}" in command
        assert f"SERVER_PORT={50600 + 100*gpu}" in command
        assert f"SERVER_LOG=/data/run/server_{6+gpu}.log" in command
        assert "SERVER_HOST=0.0.0.0" in command
        assert "SERVER_MAX_NUM_SEQS=40" in command
    assert "do-not-print-this-key" not in result.stdout + result.stderr


def fake_nvidia_smi(directory, gpus):
    """A bin directory whose nvidia-smi -L lists GPUS GPUs (one with MIG devices,
    which are not GPUs), or fails as on a node without them (GPUS=None)."""
    commands = directory / "bin"
    commands.mkdir(exist_ok=True)
    if gpus is None:
        text = "#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n"
    else:
        listing = [f"GPU {i}: NVIDIA B200 (UUID: GPU-{i:08x})" for i in range(gpus)]
        listing[1:1] = ["  MIG 1g.23gb     Device  0: (UUID: MIG-0)"] if gpus else []
        text = "#!/bin/sh\n[ \"$1\" = -L ] || exit 2\n" + "".join(f"echo '{line}'\n" for line in listing)
    (commands / "nvidia-smi").write_text(text)
    (commands / "nvidia-smi").chmod(0o755)
    return commands


@pytest.mark.parametrize(("visible", "env", "refused"), [
    (4, {}, False),
    (8, {}, True),  # an unconfined node: index 0 would not be this step's first GPU
    (None, {}, True),
    (8, {"ARC3_DRY_RUN": "1"}, False),  # a dry run without GPUs checks nothing
    (8, {"ARC3_DRY_RUN": "1", "SLURM_GPUS_ON_NODE": "4"}, True),
])
def test_rollout_nodes_refuse_unconfined_gpus(tmp_path, visible, env, refused):
    """Each node numbers its engines' GPUs 0..N-1: right only when the step sees
    exactly its N allocated GPUs (kander's cgroup ConstrainDevices)."""
    node = (ROOT / "scripts/flash_rollout_node.sh").read_text()
    block = node[node.index("# SERVER_GPU=0..GPU_COUNT-1 below"):node.index('BASE_PORT="${SERVER_PORT:?}"')]
    commands = fake_nvidia_smi(tmp_path, visible)
    environment = {**without("ARC3_", "SERVER_", "SLURM_"), "PATH": f"{commands}:{os.environ['PATH']}", **env}
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", "GPU_COUNT=4\n" + block + "echo checked\n"],
                            env=environment, text=True, capture_output=True, check=False)
    if refused:
        assert result.returncode == 1
        assert f"shows this step {visible or 0} GPUs, not the 4 allocated" in result.stderr
        assert "GPU confinement (cgroup ConstrainDevices)" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "checked\n"


@pytest.mark.parametrize(("bound", "visible", "env", "refused"), [
    ("0,1,2,3", 4, {}, False),  # a heterogeneous learner on four of a node's GPUs
    ("0,1,2,3,4,5,6,7", 8, {}, False),  # the shared node
    ("0,1,2,3", 8, {}, True),  # unconfined: device 0 need not be this job's
    ("0,1,2,3", None, {}, True),
    ("0,1,2,3", 8, {"ARC3_DRY_RUN": "1"}, False),
    ("0,1,2,3", 8, {"ARC3_DRY_RUN": "1", "SLURM_GPUS_ON_NODE": "4"}, True),
])
def test_learner_refuses_unconfined_gpus(tmp_path, bound, visible, env, refused):
    """The job binds GPUs 0..N-1 (CUDA_VISIBLE_DEVICES): right only when the node
    shows it exactly those N (kander's cgroup ConstrainDevices)."""
    job = (ROOT / "scripts/flash_rl.sbatch").read_text()
    block = job[job.index("# These indices (the learner's device_ids"):job.index("export FLA_TILELANG")]
    commands = fake_nvidia_smi(tmp_path, visible)
    environment = {**without("ARC3_", "SERVER_", "SLURM_"), "PATH": f"{commands}:{os.environ['PATH']}", **env}
    result = subprocess.run(["bash", "-euo", "pipefail", "-c",
                             f"DEFAULT_VISIBLE_GPUS={bound}\n" + block + "echo checked\n"],
                            env=environment, text=True, capture_output=True, check=False)
    if refused:
        assert result.returncode == 1
        assert f"shows this job {visible or 0} GPUs, not the 4 it binds as 0,1,2,3" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout == "checked\n"


def test_scripts_are_valid_bash():
    for name in ("flash_rl.sbatch", "flash_rl_het.sh", "flash_rollout_node.sh", "flash_rl_submit.sh"):
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / name)], check=True)


def test_launchers_require_an_explicit_config(tmp_path):
    missing = submit(tmp_path, None)
    assert missing.returncode == 2
    assert "Set --config <recipe> or ARC3_CONFIG; there is no default" in missing.stderr
    assert submit(tmp_path, None, ARC3_CONFIG=str(write_arc3_flash_config(tmp_path))).returncode == 0
    job = (ROOT / "scripts/flash_rl.sbatch").read_text()
    tail = job[job.index("# CPU launch preparation:"):]
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", tail], env={**without("ARC3_"), "RUN_DIR": str(tmp_path)},
                            text=True, capture_output=True, check=False)
    assert result.returncode != 0
    assert "ARC3_CONFIG: set ARC3_CONFIG" in result.stderr


def test_publications_never_go_by_path_between_allocations():
    """Merged weights reach the rollout nodes over NCCL from the learner, so no
    launcher names a shared publication root, rollout-node markers or a push."""
    scripts = HET + (ROOT / "scripts/flash_rl.sbatch").read_text() + (ROOT / "scripts/flash_rollout_node.sh").read_text()
    for retired in ("ARC3_RUNTIME_ADAPTER_ROOT", "ARC3_ROLLOUT_NODES", "runtime_adapters", "flash_publication_nodes"):
        assert retired not in scripts


def test_het_jobs_check_gpu_claims_after_renumbering(tmp_path):
    """A two-GPU learner configured as device_ids [0, 1] beside the recipe's
    server_gpus 0-3 collides on one node. A heterogeneous job replaces both
    with its own allocations' indices, so it checks the claims after that."""
    payload = json.loads(RECIPE.read_text())
    payload["training"]["flash"]["device_ids"] = [0, 1]
    config = tmp_path / "toy.json"
    config.write_text(json.dumps(payload))
    shared = subprocess.run([sys.executable, "-m", "training.job_shape", str(config), "--flash-job"],
                            cwd=ROOT, env={**without("ARC3_"), "PYTHONPATH": str(ROOT)},
                            text=True, capture_output=True, check=False)
    assert shared.returncode != 0
    assert "GPUs [0, 1] are claimed twice" in shared.stderr
    result, shape = prepare_het(tmp_path, config, gpus=2)
    assert result.returncode == 0, result.stderr
    assert shape["TRAINER_GPUS"] == "0,1"
    assert shlex.split(shape["SERVER_GPU_LIST"]) == ["0 1"]
    cfg = ExperimentConfig.load(tmp_path / "launch_config.json")
    assert cfg.training.flash.device_ids == [0, 1]
    assert cfg.inference.server_gpus == ["0", "1"]
    assert cfg.inference.server_allocation == "separate"
    assert cfg.inference.base_urls == [f"http://dgx05:{50123 + 100 * i}" for i in range(2)]


def test_relaunch_het_names_ntasks_on_every_component(tmp_path):
    """Kander's submit plugin refuses a GPU job without an explicit --ntasks; the
    chained segment of run epmean-20261011 was refused that way on 2026-10-12.
    Every component of the relaunch names --ntasks (the learner's 1, each
    server group's node count), beside the initial submission's arguments."""
    function = HET[HET.index("relaunch_het() {"):HET.index("\n}\n", HET.index("relaunch_het() {")) + 3]
    script = (
        "timeout() { shift; \"$@\"; }\n"
        "env() { while [[ \"$1\" == -u ]]; do shift 2; done; \"$@\"; }\n"
        "sbatch() { printf '%s\\n' \"$*\"; }\n"
        f"RUN_DIR={tmp_path}\nREPO_ROOT=/source\nSOURCE_REPO=/fetch-only\nREVISION=abcdef\n"
        "SLURM_JOB_ID=123\nSLURM_HET_SIZE=3\nLEARNER_GPUS=4\nSERVER_GPUS_PER_NODE=2\n"
        "GROUP_NODE_COUNTS=(2 1)\nARC3_EXCLUDE=dgx06\n"
        + function
        + 'relaunch_het 12:00:00\n'
    )
    result = subprocess.run(["bash", "-euo", "pipefail", "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    components = result.stdout.strip().split(" : ")
    assert len(components) == 3
    assert "--nodes=1 --ntasks=1 --gres=gpu:b200:4" in components[0]
    assert "--nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:b200:2" in components[1]
    assert "--nodes=1 --ntasks=1 --ntasks-per-node=1 --gres=gpu:b200:2" in components[2]
    assert all("--exclude=dgx06" in part and "--time=12:00:00" in part for part in components)
    assert components[0].startswith("--parsable --dependency=afterany:123")
