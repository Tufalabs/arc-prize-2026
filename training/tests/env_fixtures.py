"""A throwaway checkout of the launch tooling, fake environment artifacts and a
recording sbatch, for the environment resolver and the submit helpers.

The checkout commits the real scripts and Flash recipe beside stand-in locks and
submodule pins, so scripts/build_env.sbatch --print-plan resolves it without a
network; artifacts are empty tarballs with manifests (and receipts) shaped as
build_env.sbatch writes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GIT = [
    "git",
    "-c",
    "user.name=test",
    "-c",
    "user.email=test@example.com",
    "-c",
    "commit.gpgsign=false",
]
SCRIPTS = (
    "flash_rl_submit.sh",
    "flash_rl.sbatch",
    "build_envs.sh",
    "resolve_envs.py",
    "build_env.sbatch",
)
RECIPE = "configs/flash_8gpu_arc2_grpo_tokmean.json"
PROJECT = "environments/arc2-worker-vllm-fork"
PIN, ARC2_PIN = "1" * 40, "2" * 40
LOCK = "version = 1\n"
PROJECT_LOCK = 'version = 1\n\n[[package]]\nname = "duck"\nversion = "0.1.0"\n'
NAMES = {
    "trainer": "trainer-flash-distillation",
    "harness": "harness",
    "arc2-worker": "arc2-worker-vllm-fork",
}
VARIABLES = {
    "trainer": "ARC3_FLASH_VENV_TARBALL",
    "harness": "ARC3_HARNESS_VENV_TARBALL",
    "arc2-worker": "ARC3_ARC2_WORKER_VENV_TARBALL",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        [*GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def make_checkout(path: Path, builder: str | None = None) -> str:
    """Commit the tooling into a new repository at PATH (BUILDER replaces
    scripts/build_env.sbatch); returns the commit."""
    (path / "scripts").mkdir(parents=True)
    git(path, "init", "-q")
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, path / "scripts" / name)
    if builder is not None:
        (path / "scripts/build_env.sbatch").write_text(builder)
    (path / "configs").mkdir()
    shutil.copy2(ROOT / RECIPE, path / RECIPE)
    (path / PROJECT).mkdir(parents=True)
    (path / PROJECT / "pyproject.toml").write_text("[project]\n")
    (path / PROJECT / "uv.lock").write_text(PROJECT_LOCK)
    (path / "uv.lock").write_text(LOCK)
    (path / "pyproject.toml").write_text("[project]\n")
    (path / ".gitignore").write_text("__pycache__/\n")
    (path / ".gitmodules").write_text(
        '[submodule "platypus"]\n\tpath = platypus\n\turl = git@github.com:Example/harness.git\n'
        '[submodule "arc2-platypus"]\n\tpath = arc2-platypus\n\turl = git@github.com:Example/arc2.git\n'
    )
    git(path, "add", "-A")
    git(path, "update-index", "--add", "--cacheinfo", f"160000,{PIN},platypus")
    git(
        path, "update-index", "--add", "--cacheinfo", f"160000,{ARC2_PIN},arc2-platypus"
    )
    git(path, "commit", "-qm", "fixture")
    for submodule in (
        "platypus",
        "arc2-platypus",
    ):  # as a clone leaves uninitialized submodules
        (path / submodule).mkdir()
    return git(path, "rev-parse", "HEAD")


def plan(
    checkout: Path, recipe: str, revision: str, builder: Path | None = None
) -> dict[str, str]:
    """build_env.sbatch --print-plan for the checkout (its own builder by default)."""
    builder = builder or checkout / "scripts/build_env.sbatch"
    result = subprocess.run(
        [
            "bash",
            str(builder),
            recipe,
            revision,
            "--repo",
            str(checkout),
            "--print-plan",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def source(recipe: str, python: str, **changes) -> dict:
    """The provenance build_env.sbatch records for a build of the checkout's
    revision (its "fields"), with a few fields that are not identity."""
    common = {
        "builder": "scripts/build_env.sbatch",
        "recipe": recipe,
        "revision": "f" * 40,
        "uv_lock_sha256": sha256(LOCK),
        "requirements_sha256": "0" * 64,
        "python_source": python,
        "slurm_job": "100",
        "versions": {"torch": "2.13.0"},
    }
    harness = {
        "harness_revision": PIN,
        "harness_revision_source": "submodule pin",
        "harness_submodule_pin": PIN,
    }
    fields = {
        "trainer": {
            "extras": ["flash-distillation"],
            "harness_packages": ["ullm"],
            "forbidden_modules": ["vllm", "flashinfer", "platypus", "duck"],
            **harness,
        },
        "harness": {"package": "duck", "vllm_fork_revision": "3" * 40, **harness},
        "arc2-worker": {
            "project": PROJECT,
            "project_uv_lock_sha256": sha256(PROJECT_LOCK),
            "arc2_harness_revision": ARC2_PIN,
            "arc2_harness_submodule_pin": ARC2_PIN,
            **harness,
        },
    }[recipe]
    return {**common, **fields, **changes}


def write_artifact(
    root: Path,
    recipe: str,
    key: str,
    record_source: dict,
    created_at: str = "2026-10-01T00:00:00+00:00",
) -> Path:
    """An artifact as pack_venv.sh publishes it: tarball, then manifest."""
    root.mkdir(parents=True, exist_ok=True)
    name = NAMES[recipe]
    tarball = root / f"{name}-{key}.tar.zst"
    tarball.write_bytes(b"")
    manifest = {
        "name": name,
        "key": key,
        "recipe": "venv-pack-v2",
        "tarball": str(tarball),
        "created_at": created_at,
        "source": record_source,
    }
    (root / f"{name}-{key}.json").write_text(json.dumps(manifest))
    return tarball


def write_receipt(
    root: Path,
    recipe: str,
    key: str,
    record_source: dict,
    job: str,
    created_at: str = "2026-10-02T00:00:00+00:00",
) -> Path:
    name = NAMES[recipe]
    (root / "receipts").mkdir(parents=True, exist_ok=True)
    path = root / "receipts" / f"{name}-{key}-{job}.json"
    path.write_text(
        json.dumps(
            {
                "name": name,
                "key": key,
                "tarball": f"{name}-{key}.tar.zst",
                "created_at": created_at,
                "slurm_job": job,
                "source": record_source,
            }
        )
    )
    return path


def write_all(root: Path, python: str) -> dict[str, Path]:
    """One matching artifact per recipe, keyed 0..., 1..., 2..."""
    return {
        recipe: write_artifact(root, recipe, str(index) * 16, source(recipe, python))
        for index, recipe in enumerate(NAMES)
    }


def recording_commands(directory: Path, record: Path) -> Path:
    """A bin directory whose sbatch appends {argv, script, env} to RECORD (one
    JSON line per call) and prints a job id (4242, 4243, ...), and whose mkdir,
    given a path outside DIRECTORY, only logs its arguments to DIRECTORY/mkdir.log
    (/data does not exist off the cluster)."""
    commands = directory / "bin"
    commands.mkdir(exist_ok=True)
    (commands / "sbatch").write_text(f"""#!{sys.executable}
import json, os, sys
argv = sys.argv[1:]
script = next((arg for arg in argv if arg.startswith("/dev/fd/") or arg.endswith((".sbatch", ".sh"))), None)
text = open(script).read() if script and os.path.exists(script) else None
env = {{key: value for key, value in os.environ.items() if key.startswith(("ARC3_", "SERVER_"))}}
previous = len(open({str(record)!r}).readlines()) if os.path.exists({str(record)!r}) else 0
with open({str(record)!r}, "a") as handle:
    handle.write(json.dumps({{"argv": argv, "script": text, "env": env}}) + "\\n")
print(4242 + previous)
""")
    (commands / "mkdir").write_text(f"""#!/bin/bash
for arg; do
  [[ "$arg" == -* || "$arg" == {str(directory)!r}/* ]] || {{ echo "$*" >> {str(directory / "mkdir.log")!r}; exit 0; }}
done
exec {shutil.which("mkdir")} "$@"
""")
    for name in ("sbatch", "mkdir"):
        (commands / name).chmod(0o755)
    return commands


def calls(record: Path) -> list[dict]:
    return (
        [json.loads(line) for line in record.read_text().splitlines()]
        if record.exists()
        else []
    )


def tool_env(commands: Path, **extra: str) -> dict[str, str]:
    """The test environment without launcher settings, with the recording commands first on PATH."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("ARC3_", "SERVER_", "SLURM_", "ARC_TRAINING_ENVS"))
    }
    env.update(PATH=f"{commands}:{env['PATH']}", USER="tester", **extra)
    return env
