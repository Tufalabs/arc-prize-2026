"""Submit a snapshotted arc-training drafter step to Slurm.

    python -m drafter.jobs NAME [--gpus 4] [--after ID[:ID...]] -- drafter.train ARGS

Training, evaluation and export use spec_decode's own locked environment.
Extraction/online checks need --runtime-python (or
DRAFTER_SERVING_PYTHON): an existing, qualified serving interpreter. Each
job installs THIS package into a private environment layered over it; the
harness supplies the engine, never the drafter training implementation.
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

_SERVING_MODULES = {"drafter.extract", "drafter.online"}
_TARGET_ENV = ("DRAFTER_TARGET", "DRAFTER_BASE", "DRAFTER_MAX_TOKENS", "DRAFTER_ROOT")
_WORKER = '''\
import importlib.metadata
import json
import os
import runpy
import sys
from pathlib import Path

os.environ.update({env!r})
import drafter
expected = Path(__file__).resolve().parent / "source" / "drafter"
if Path(drafter.__file__).resolve().parent != expected:
    raise RuntimeError(f"Wrong drafter source: {{drafter.__file__}}, expected {{expected}}")
versions = {{}}
for name in ("torch", "transformers", "drafter", "duck", "ullm", "vllm"):
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
(Path(__file__).resolve().parent / "runtime.json").write_text(json.dumps({{
    "python": sys.executable, "drafter_source": str(expected), "versions": versions,
}}, indent=2))
sys.argv = {argv!r}
runpy.run_module({module!r}, run_name="__main__", alter_sys=True)
'''


def snapshot(source: Path, destination: Path) -> dict[str, str]:
    """Freeze the package, lock and launch scripts, including uncommitted edits."""
    destination.mkdir()
    for name in ("pyproject.toml", "uv.lock"):
        shutil.copy2(source / name, destination / name)
    for name in ("drafter", "tools"):
        shutil.copytree(source / name, destination / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return {
        str(path.relative_to(destination)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(destination.rglob("*")) if path.is_file()
    }


def git_output(source: Path, *args: str) -> Optional[str]:
    """Record unavailable git metadata as null, rather than a false revision."""
    try:
        result = subprocess.run(["git", "-C", str(source), *args], capture_output=True, text=True)
    except OSError:
        return None
    return result.stdout.rstrip("\n") if result.returncode == 0 else None


def submit(
    name: str,
    module: str,
    args: list[str],
    *,
    node: Optional[str],
    minutes: int,
    gpus: int,
    after: Optional[str] = None,
    runtime_python: Optional[str] = None,
) -> Path:
    """Snapshot arc-training before queueing a job; preserve its source manifest."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("job name must contain only letters, numbers, _, - or .")
    if not re.fullmatch(r"drafter\.[a-z_][a-z0-9_]*", module):
        raise ValueError("expected a drafter module after --")
    if minutes <= 0 or gpus < 0:
        raise ValueError("minutes must be positive and gpus nonnegative")
    if after and not re.fullmatch(r"[0-9]+(?::[0-9]+)*", after):
        raise ValueError("--after must contain colon-separated Slurm job IDs")
    source = Path(__file__).resolve().parents[1]
    if not (source / "pyproject.toml").is_file():
        raise ValueError("submit from an editable arc-training/spec_decode checkout")
    if not (source / "drafter" / f"{module.split('.')[-1]}.py").is_file():
        raise ValueError(f"unknown module: {module}")
    if module in _SERVING_MODULES:
        runtime_python = runtime_python or os.environ.get("DRAFTER_SERVING_PYTHON")
        if not runtime_python:
            raise ValueError("serving steps require --runtime-python or DRAFTER_SERVING_PYTHON")
    # A node-local runtime may not exist on the submitting/login host.
    # run_job.sh checks executability on the allocated worker.
    if runtime_python and not Path(runtime_python).is_absolute():
        raise ValueError("--runtime-python must be an absolute path accessible on the worker")
    results = Path(os.environ.get("DRAFTER_ROOT") or f"/data/shared/research/arc_3_results/{getpass.getuser()}/drafter")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = (results / "jobs" / f"{stamp}_{name}").resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    hashes = snapshot(source, job_dir / "source")
    revision = git_output(source, "rev-parse", "HEAD")
    status = git_output(source, "status", "--porcelain")
    provenance = {
        "repository": "arc-training", "source": str(source), "revision": revision,
        "dirty": None if status is None else bool(status), "git_status": status,
        "files_sha256": hashes, "runtime_python": runtime_python, "module": module, "args": args,
    }
    (job_dir / "source_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    env = {key: os.environ[key] for key in _TARGET_ENV if os.environ.get(key)}
    (job_dir / "run_in_worker.py").write_text(_WORKER.format(env=env, argv=[module, *args], module=module))
    script = "#!/bin/bash\nset -euo pipefail\nexec " + shlex.join([
        "bash", str(job_dir / "source/tools/run_job.sh"), str(job_dir), runtime_python or "",
    ]) + "\n"
    (job_dir / "sbatch_script.sh").write_text(script)
    command = [
        "sbatch", "--parsable", "--account=research", "--partition=dev", "--nodes=1", "--ntasks=1",
        f"--job-name=drafter_{name}", f"--time={minutes // 60:02d}:{minutes % 60:02d}:00",
        f"--output={job_dir}/stdout.log", f"--error={job_dir}/stderr.log",
    ]
    if gpus:
        command += [f"--gres=gpu:b200:{gpus}", "--cpus-per-gpu=16", "--mem-per-gpu=160G"]
    else:
        command += ["--cpus-per-task=8", "--mem=48G"]
    if node:
        command += [f"--nodelist={node}"]
    if after:
        command += [f"--dependency=afterok:{after}", "--kill-on-invalid-dep=yes"]
    (job_dir / "submission.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, input=script, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"sbatch failed: {result.stderr.strip()}")
    job_id = result.stdout.strip().split(";")[0]
    (job_dir / "job_id").write_text(job_id + "\n")
    print(f"submitted {job_id}\njob_dir={job_dir}", flush=True)
    return job_dir


def main(argv: Optional[list[str]] = None) -> None:
    """Parse the job resources separately from the module's arguments."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        raise SystemExit("usage: python -m drafter.jobs NAME [options] -- MODULE [ARGS...]")
    split = argv.index("--")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name")
    parser.add_argument("--node", default=None)
    parser.add_argument("--minutes", type=int, default=20)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--after", default=None, help="ID[:ID...]: wait for successful jobs")
    parser.add_argument("--runtime-python", help="qualified serving interpreter, accessible on the worker")
    args = parser.parse_args(argv[:split])
    command = argv[split + 1 :]
    if not command:
        raise SystemExit("no module after --")
    submit(args.name, command[0], command[1:], node=args.node, minutes=args.minutes,
           gpus=args.gpus, after=args.after, runtime_python=args.runtime_python)


if __name__ == "__main__":
    main()
