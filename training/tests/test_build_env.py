"""Environment tooling without Slurm, /data or a network: build_env.sbatch's
argument parsing and --print-plan, env_checks.py on a throwaway venv, and
pack_venv.sh's key/manifest behaviour (GNU userland, i.e. Linux CI)."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import packaging
import pytest
from env_fixtures import make_checkout

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "scripts/build_env.sbatch"
PACK = ROOT / "scripts/pack_venv.sh"
CHECKS = ROOT / "scripts/env_checks.py"
FORK = "077920f36a0e4d83f303a59f12118d9970215c47"
HARNESS_URL = "ssh://git@github.com/Example/harness.git"
ARC2_URL = "git@github.com:Example/arc2-harness.git"
PROJECT = "environments/arc2-worker-vllm-fork"
GIT = ["git", "-c", "user.name=test", "-c", "user.email=test@example.com", "-c", "commit.gpgsign=false"]
PY = f"python{sys.version_info.major}.{sys.version_info.minor}"


def git(repo, *args):
    return subprocess.run([*GIT, "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def commit_files(repo, files):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    git(repo, "add", "--", *files)
    git(repo, "commit", "-qm", "fixture")
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture(scope="module")
def repos(tmp_path_factory):
    """A harness repository (two commits), an ARC-2 harness repository and a
    training repository pinning the first harness commit and the ARC-2 one."""
    root = tmp_path_factory.mktemp("repos")
    harness, training, arc2 = root / "harness", root / "training", root / "arc2"
    for repo in (harness, training, arc2):
        repo.mkdir()
        git(repo, "init", "-q")
    lock = ('version = 1\n\n[[package]]\nname = "vllm"\nversion = "0.29.1"\n'
            f'source = {{ git = "https://github.com/Example/vllm-fork?rev={FORK}#{FORK}" }}\n\n'
            '[[package]]\nname = "duck"\nversion = "0.1.0"\n')
    pin = commit_files(harness, {
        "uv.lock": lock,
        "packages/ullm/pyproject.toml": '[project]\nname = "ullm"\n',
        "packages/duck/pyproject.toml": '[project]\nname = "duck"\n',
    })
    later = commit_files(harness, {"packages/platypus/pyproject.toml": '[project]\nname = "platypus"\n'})
    arc2_pin = commit_files(arc2, {f"packages/{name}/pyproject.toml": f'[project]\nname = "{name}"\n'
                                   for name in ("duck", "platypus", "taaf")})
    git(training, "update-index", "--add", "--cacheinfo", f"160000,{pin},platypus")
    git(training, "update-index", "--add", "--cacheinfo", f"160000,{arc2_pin},arc2-platypus")
    modules = (f'[submodule "platypus"]\n\tpath = platypus\n\turl = {HARNESS_URL}\n'
               f'[submodule "arc2-platypus"]\n\tpath = arc2-platypus\n\turl = {ARC2_URL}\n')
    revision = commit_files(training, {".gitmodules": modules, "uv.lock": "version = 1\n",
                                       "pyproject.toml": "[project]\n",
                                       f"{PROJECT}/pyproject.toml": "[project]\n", f"{PROJECT}/uv.lock": lock})
    return {"harness": harness, "training": training, "pin": pin, "later": later, "revision": revision,
            "arc2": arc2, "arc2_pin": arc2_pin,
            "lock_sha": hashlib.sha256(b"version = 1\n").hexdigest(),
            "harness_lock_sha": hashlib.sha256(lock.encode()).hexdigest()}


def build_env(*args, user="envtest", script=BUILD):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    env["USER"] = user
    return subprocess.run(["bash", str(script), *args], capture_output=True, text=True, env=env, timeout=60)


def plan(*args):
    result = build_env(*args, "--print-plan")
    assert result.returncode == 0, result.stderr
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_trainer_plan_defaults(repos):
    fields = plan("trainer", repos["revision"], "--repo", str(repos["training"]))
    assert fields["recipe"] == "trainer"
    assert fields["revision"] == repos["revision"]
    assert fields["uv_lock_sha256"] == repos["lock_sha"]
    assert fields["extras"] == "flash-distillation"
    assert fields["harness_packages"] == "ullm"
    assert fields["forbid"] == "vllm flashinfer platypus duck"
    assert fields["harness_submodule_pin"] == fields["harness_revision"] == repos["pin"]
    assert fields["harness_revision_source"] == "submodule pin"
    assert fields["harness_repository"] == HARNESS_URL
    assert fields["harness_source"] == f"fresh clone of {HARNESS_URL} into job scratch"
    assert fields["name"] == "trainer-flash-distillation"
    assert fields["out"] == "/data/personal/envtest/arc-training-envs"
    assert fields["artifact"] == "/data/personal/envtest/arc-training-envs/trainer-flash-distillation-<key>.tar.zst"
    assert fields["stage"] == "yes"
    assert "--extra flash-distillation --no-emit-project --no-emit-workspace" in fields["build"]
    assert "uv pip install --no-deps --compile-bytecode <harness>/packages/ullm" in fields["build"]
    assert "imports: ullm ullm.media" in fields["checks"]
    assert "vllm_fork_revision" not in fields


def test_trainer_plan_options_replace_defaults(repos):
    fields = plan("trainer", repos["revision"], "--repo", str(repos["training"]), "--extra", "sft-train",
                  "--extra", "fa4", "--no-harness-packages", "--forbid", "vllm", "--forbid", "flash_attn",
                  "--out", "/data/shared/envs/", "--no-stage")
    assert fields["name"] == "trainer-fa4-sft-train"
    assert fields["extras"] == "fa4 sft-train"
    assert fields["harness_packages"] == ""
    assert fields["forbid"] == "vllm flash_attn"
    assert fields["artifact"] == "/data/shared/envs/trainer-fa4-sft-train-<key>.tar.zst"
    assert fields["stage"] == "no"
    assert not any(key.startswith("harness_") and key != "harness_packages" for key in fields)
    assert "--no-deps" not in fields["build"]


def test_harness_plan_reads_the_pinned_lock_from_a_clone(repos):
    common = ("harness", repos["revision"], "--repo", str(repos["training"]),
              "--harness-clone", str(repos["harness"]))
    fields = plan(*common)
    assert fields["name"] == "harness"
    assert fields["package"] == "duck"
    assert fields["harness_revision"] == repos["pin"]
    assert fields["harness_source"] == str(repos["harness"])
    assert fields["harness_uv_lock_sha256"] == repos["harness_lock_sha"]
    assert fields["vllm_fork_revision"] == FORK
    assert fields["build"] == "uv sync --locked --no-dev --no-editable --compile-bytecode --package duck"
    assert "native_vllm_server --help only with a GPU" in fields["checks"]
    assert "extras" not in fields and "forbid" not in fields
    override = plan(*common, "--harness-revision", repos["later"], "--package", "platypus")
    assert override["name"] == "harness-platypus"
    assert override["harness_revision"] == repos["later"]
    assert override["harness_submodule_pin"] == repos["pin"]
    assert override["harness_revision_source"] == "--harness-revision"


def test_arc2_worker_plan_pins_both_submodules(repos):
    fields = plan("arc2-worker", repos["revision"], "--repo", str(repos["training"]))
    assert fields["recipe"] == "arc2-worker"
    assert fields["name"] == "arc2-worker-vllm-fork"
    assert fields["artifact"] == "/data/personal/envtest/arc-training-envs/arc2-worker-vllm-fork-<key>.tar.zst"
    assert fields["project"] == PROJECT
    assert fields["project_uv_lock_sha256"] == repos["harness_lock_sha"]  # the fixture reuses one lock text
    assert fields["vllm_fork_revision"] == FORK  # read from the project lock at the revision
    assert fields["harness_submodule_pin"] == fields["harness_revision"] == repos["pin"]
    assert fields["harness_revision_source"] == "submodule pin"
    assert fields["harness_uv_lock_sha256"] == "read in the job"
    assert fields["arc2_harness_submodule_pin"] == repos["arc2_pin"]
    assert fields["arc2_harness_repository"] == ARC2_URL
    assert fields["arc2_harness_source"] == f"fresh clone of {ARC2_URL} into job scratch"
    assert fields["build"].startswith(f"uv sync --locked --no-dev --no-editable --compile-bytecode in {PROJECT}")
    assert "agrees with the harness lock" in fields["checks"]
    assert "scripts/platypus_rollout_worker.py --help" in fields["checks"]
    assert not {"extras", "forbid", "package", "harness_packages"} & set(fields)
    cloned = plan("arc2-worker", repos["revision"], "--repo", str(repos["training"]),
                  "--harness-clone", str(repos["harness"]), "--arc2-harness-clone", str(repos["arc2"]))
    assert cloned["harness_uv_lock_sha256"] == repos["harness_lock_sha"]
    assert cloned["arc2_harness_source"] == str(repos["arc2"])
    assert cloned["vllm_fork_revision"] == FORK


def test_arc2_worker_plan_rejects_missing_inputs(repos, tmp_path):
    common = ("arc2-worker", repos["revision"], "--repo", str(repos["training"]), "--print-plan")
    result = build_env(*common, "--arc2-harness-clone", str(repos["harness"]))
    assert result.returncode == 2
    assert f"{repos['arc2_pin']} is not in {repos['harness']}" in result.stderr
    # A training revision without the project (the harness fixture's first state).
    bare = tmp_path / "training"
    subprocess.run(["git", "clone", "-q", str(repos["training"]), str(bare)], check=True)
    git(bare, "rm", "-q", "-r", PROJECT)
    git(bare, "commit", "-qm", "no project")
    result = build_env("arc2-worker", git(bare, "rev-parse", "HEAD"), "--repo", str(bare), "--print-plan")
    assert result.returncode == 2
    assert f"has no {PROJECT}/pyproject.toml and uv.lock" in result.stderr


def test_harness_plan_without_clone_resolves_in_the_job(repos):
    fields = plan("harness", repos["revision"], "--repo", str(repos["training"]))
    assert fields["harness_uv_lock_sha256"] == "read in the job"
    assert fields["vllm_fork_revision"] == "read in the job"


@pytest.mark.parametrize(("args", "error"), [
    ([], "first argument must be a recipe"),
    (["serve", "HEAD"], "first argument must be a recipe"),
    (["trainer"], "missing <revision>"),
    (["trainer", "{rev}", "--bogus"], "unknown option --bogus"),
    (["trainer", "{rev}", "--extra"], "--extra needs a value"),
    (["trainer", "{rev}", "--extra", "--no-stage"], "--extra needs a value"),
    (["harness", "{rev}", "--extra", "x"], "--extra applies to the trainer recipe"),
    (["trainer", "{rev}", "--package", "duck"], "--package applies to the harness recipe"),
    (["trainer", "{rev}", "--harness-revision", "0" * 40], "--harness-revision applies to the harness recipe"),
    (["harness", "{rev}", "--harness-revision", "abc123"], "full 40-hex commit"),
    (["arc2-worker", "{rev}", "--harness-revision", "0" * 40], "--harness-revision applies to the harness recipe"),
    (["arc2-worker", "{rev}", "--package", "duck"], "--package applies to the harness recipe"),
    (["arc2-worker", "{rev}", "--extra", "x"], "--extra applies to the trainer recipe"),
    (["harness", "{rev}", "--arc2-harness-clone", "/x"], "--arc2-harness-clone applies to the arc2-worker recipe"),
    (["arc2-worker", "{rev}", "--arc2-harness-clone", "relative"], "paths must be absolute"),
    (["trainer", "{rev}", "--forbid", "ullm"], "--harness-package ullm is also forbidden"),
    (["trainer", "{rev}", "--extra", "Flash_Distillation"], "bad name"),
    (["trainer", "{rev}", "--extra", "a", "--extra", "a"], "an --extra is repeated"),
    (["trainer", "{rev}", "--forbid", "not-a-module"], "bad module name"),
    (["trainer", "{rev}", "--out", "/tmp/envs"], "--out must be a directory under /data"),
    (["trainer", "{rev}", "--python", "relative/python3"], "paths must be absolute"),
    (["trainer", "f" * 40], "is not a commit in"),
    (["trainer", "{rev}", "--harness-clone", "{harness}", "--harness-package", "nosuch"], "no packages/nosuch"),
])
def test_plan_rejects_bad_arguments(repos, args, error):
    args = [arg.format(rev=repos["revision"], harness=repos["harness"]) for arg in args]
    result = build_env(*args, "--repo", str(repos["training"]), "--print-plan")
    assert result.returncode == 2
    assert error in result.stderr
    assert result.stdout == ""


def test_harness_clone_must_hold_the_revision(repos, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    result = build_env("harness", repos["revision"], "--repo", str(repos["training"]),
                       "--harness-clone", str(empty), "--print-plan")
    assert result.returncode == 2
    assert f"{repos['pin']} is not in {empty}" in result.stderr


def test_default_repository_must_exist(tmp_path):
    # A copy outside any checkout, as Slurm runs a submitted build.
    copy = tmp_path / "build_env.sbatch"
    shutil.copy2(BUILD, copy)
    result = build_env("trainer", "HEAD", "--print-plan", user="envtest-nobody", script=copy)
    assert result.returncode == 2
    assert "fetch-only clone /data/personal/envtest-nobody/arc-training-src not found" in result.stderr
    assert "git clone --no-checkout" in result.stderr


def test_plan_from_a_checkout_reads_that_checkout(tmp_path):
    checkout = tmp_path / "arc-training"
    revision = make_checkout(checkout)
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM_")}
    result = subprocess.run(["bash", "scripts/build_env.sbatch", "trainer", "HEAD", "--print-plan"],
                            cwd=checkout, capture_output=True, text=True, env={**env, "USER": "envtest"},
                            timeout=60)
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert fields["repo"] == str(checkout.resolve())
    assert fields["revision"] == revision
    assert f"planning the checkout holding this script, {checkout.resolve()}" in result.stderr


def test_building_requires_a_slurm_job(repos):
    result = build_env("trainer", repos["revision"], "--repo", str(repos["training"]))
    assert result.returncode == 2
    assert "submit with sbatch" in result.stderr


def test_help_and_shell_syntax():
    result = subprocess.run(["bash", str(BUILD), "--help"], capture_output=True, text=True, check=True)
    assert "sbatch scripts/build_env.sbatch trainer <revision>" in result.stdout
    for script in (BUILD, PACK, ROOT / "scripts/stage_venv.sh"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    header = BUILD.read_text().split("\nset -euo pipefail", 1)[0]
    assert "#SBATCH --qos=dev" in header and "--gres" not in header
    assert "#SBATCH --output=/data/personal/%u/arc-training-envs/slurm-%j.out" in header


# ---- env_checks.py on a throwaway venv -------------------------------------

def make_venv(path):
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(path)], check=True)
    return next(path.glob("lib/python3.*/site-packages"))


def add_dist(site, name, version, origin=None, files=None, version_attr=None):
    """A minimal installed distribution: METADATA, RECORD, optional direct_url.json and module."""
    module = site / name.replace("-", "_")
    module.mkdir(exist_ok=True)
    (module / "__init__.py").write_text(f"__version__ = {version_attr or version!r}\n")
    info = site / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir()
    (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    records = files or {f"{module.name}/__init__.py": "sha256=module,10"}
    (info / "RECORD").write_text("".join(f"{path},{digest}\n" for path, digest in records.items())
                                 + f"{info.name}/METADATA,,\n{info.name}/RECORD,,\n")
    if origin is not None:
        (info / "direct_url.json").write_text(json.dumps(origin))
    return info


@pytest.fixture
def venv(tmp_path):
    site = make_venv(tmp_path / "venv")
    # Marker evaluation uses the environment's packaging; a plain copy is
    # importable without being an installed distribution.
    shutil.copytree(Path(packaging.__file__).parent, site / "packaging")
    add_dist(site, "alpha", "1.0")
    add_dist(site, "beta", "0.1.0", {"url": "file:///scratch/build/packages/beta", "dir_info": {}})
    add_dist(site, "gamma", "2.0.dev3+gabcdef1234", {"url": "https://github.com/Example/gamma",
                                                     "vcs_info": {"vcs": "git", "commit_id": "abcdef1234" * 4}})
    (tmp_path / "packages/beta").mkdir(parents=True)
    (tmp_path / "packages/beta/pyproject.toml").write_text('[project]\nname = "beta"\n')
    return tmp_path


REQUIREMENTS = """# This file was autogenerated by uv via the following command:
alpha==1.0 \\
    --hash=sha256:0000 \\
    --hash=sha256:1111
    # via
    #   beta
colorama==0.4.6 ; sys_platform == 'win32' \\
    --hash=sha256:2222
gamma @ git+https://github.com/Example/gamma@{commit}
./packages/beta
"""


def checks(root, *args):
    return subprocess.run([str(root / "venv/bin/python"), "-I", "-B", str(CHECKS), *args],
                          capture_output=True, text=True, cwd=root, timeout=60)


def write_requirements(root, commit="abcdef1234" * 4, extra=""):
    path = root / "requirements.txt"
    path.write_text(REQUIREMENTS.format(commit=commit) + extra)
    return path


def test_lock_check_accepts_the_exact_export(venv):
    result = checks(venv, "lock", str(write_requirements(venv)), "--base", str(venv), "--import-local")
    assert result.returncode == 0, result.stderr
    assert "3 distributions (1 pinned, 1 by URL, 1 local) installed exactly as exported" in result.stdout
    assert "import beta ok" in result.stdout


@pytest.mark.parametrize(("change", "error"), [
    ("version", "alpha: installed 1.0, lock export 1.1"),
    ("commit", "gamma: installed from commit abcdef1234"),
    ("missing", "delta: in the lock export, not installed"),
    ("extra", "epsilon 3.0: installed, not in the lock export"),
    ("editable", "beta: editable install"),
])
def test_lock_check_reports_differences(venv, change, error):
    site = next(venv.glob("venv/lib/python3.*/site-packages"))
    requirements = write_requirements(venv, commit="1" * 40 if change == "commit" else "abcdef1234" * 4,
                                      extra="delta==1.0\n" if change == "missing" else "")
    if change == "version":
        requirements.write_text(requirements.read_text().replace("alpha==1.0", "alpha==1.1"))
    if change == "extra":
        add_dist(site, "epsilon", "3.0")
    if change == "editable":
        (site / "beta-0.1.0.dist-info/direct_url.json").write_text(
            json.dumps({"url": "file:///x", "dir_info": {"editable": True}}))
    result = checks(venv, "lock", str(requirements), "--base", str(venv))
    assert result.returncode != 0
    assert error in result.stderr


def test_lock_check_allows_named_additions(venv):
    site = next(venv.glob("venv/lib/python3.*/site-packages"))
    add_dist(site, "ullm", "0.1.0", {"url": "file:///scratch/harness/packages/ullm", "dir_info": {}})
    requirements = str(write_requirements(venv))
    assert checks(venv, "lock", requirements, "--base", str(venv)).returncode != 0
    result = checks(venv, "lock", requirements, "--base", str(venv), "--allow-extra", "ullm")
    assert result.returncode == 0, result.stderr
    assert "allowed additions: ullm" in result.stdout


def test_absent_imports_and_commit_checks(venv):
    site = next(venv.glob("venv/lib/python3.*/site-packages"))
    assert checks(venv, "absent", "vllm", "duck").returncode == 0
    add_dist(site, "vllm-flash-attn", "1.0")
    result = checks(venv, "absent", "vllm", "alpha")
    assert result.returncode != 0
    assert "module alpha is importable" in result.stderr
    assert "distribution vllm-flash-attn is installed" in result.stderr
    assert "import alpha ok: 1.0" in checks(venv, "imports", "alpha", "beta").stdout
    assert checks(venv, "imports", "nonexistent_module").returncode != 0
    assert checks(venv, "reports-commit", "gamma", "abcdef1234" * 4).returncode == 0
    mismatch = checks(venv, "reports-commit", "gamma", "1" * 40)
    assert mismatch.returncode != 0 and "does not name commit" in mismatch.stderr


LOCK = """version = 1

[[package]]
name = "torch"
version = "{torch}"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "triton"
version = "3.7.1"
source = {{ registry = "https://pypi.org/simple" }}

[[package]]
name = "vllm"
version = "0.29.1"
source = {{ git = "https://github.com/Example/vllm-fork?rev={commit}#{commit}" }}

[[package]]
name = "duck"
version = "{duck}"
source = {{ {duck_source} }}
"""


def write_lock(path, torch="2.13.0", commit=FORK, duck="0.1.0", duck_source='editable = "packages/duck"', extra=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(LOCK.format(torch=torch, commit=commit, duck=duck, duck_source=duck_source) + extra)
    return str(path)


def test_lock_agreement(venv):
    reference = write_lock(venv / "harness/uv.lock", extra=(
        '\n[[package]]\nname = "triton"\nversion = "3.6.0"\nsource = { registry = "https://pypi.org/simple" }\n'))
    agreeing = write_lock(venv / "worker/uv.lock", duck="9.9", duck_source='directory = "../arc2/packages/duck"',
                          extra='\n[[package]]\nname = "taaf"\nversion = "0.1.0"\nsource = { registry = "x" }\n')
    result = checks(venv, "lock-agrees", agreeing, reference)
    assert result.returncode == 0, result.stderr
    assert "3 shared distributions at the reference's versions; only in worker: taaf" in result.stdout
    for change, error in (({"torch": "2.11.0"}, "torch: 2.11.0; reference 2.13.0"),
                          ({"commit": "1" * 40}, f"vllm: 0.29.1 @ {'1' * 40}; reference 0.29.1 @ {FORK}")):
        result = checks(venv, "lock-agrees", write_lock(venv / "drift/uv.lock", **change), reference)
        assert result.returncode != 0
        assert error in result.stderr


def test_versions_lists_installed_and_missing(venv):
    result = checks(venv, "versions", "alpha", "Gamma", "nonexistent")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["alpha 1.0", "gamma 2.0.dev3+gabcdef1234", "nonexistent -"]


def test_provenance_records_fields_versions_and_commits(venv):
    out = venv / "provenance.json"
    result = checks(venv, "provenance", str(out), "revision=abc", "extras:=[\"flash-distillation\"]",
                    "builder_matches_revision:=true")
    assert result.returncode == 0, result.stderr
    record = json.loads(out.read_text())
    assert record["revision"] == "abc"
    assert record["extras"] == ["flash-distillation"]
    assert record["builder_matches_revision"] is True
    assert record["versions"] == {"alpha": "1.0", "beta": "0.1.0", "gamma": "2.0.dev3+gabcdef1234"}
    assert record["git_commits"] == {"gamma": "abcdef1234" * 4}


# ---- pack_venv.sh key and manifest -------------------------------------------

linux_only = pytest.mark.skipif(sys.platform != "linux" or shutil.which("zstd") is None,
                                reason="pack_venv.sh runs on cluster nodes (GNU find/stat/sed, zstd)")


def packable_venv(scratch, name, base, local_digest="sha256=beta,10", alpha="1.0"):
    """A venv like uv builds: bin/python links to a standalone base interpreter."""
    venv = scratch / name
    site = make_venv(venv)
    for link in venv.glob("bin/python*"):
        link.unlink()
    (venv / "bin/python").symlink_to(base / f"bin/{PY}")
    (venv / "bin/tool").write_text(f"#!{venv}/bin/python\nprint('tool')\n")  # a build-path shebang
    add_dist(site, "alpha", alpha)
    add_dist(site, "beta", "0.1.0", {"url": f"file://{scratch}/{name}-src/beta", "dir_info": {}},
             files={"beta/__init__.py": local_digest, "../../../bin/tool": f"sha256={name},20"})
    return venv


@pytest.fixture
def pack_root(tmp_path):
    """Fake /scratch and /data roots and a standalone 'base interpreter' (a wrapper)."""
    base = tmp_path / "base"
    (base / "bin").mkdir(parents=True)
    (base / f"lib/{PY}").mkdir(parents=True)
    (base / f"lib/{PY}/os.py").write_text("")
    wrapper = base / f"bin/{PY}"
    wrapper.write_text(f"#!/bin/sh\nexec {sys._base_executable} \"$@\"\n")
    wrapper.chmod(0o755)
    (base / "bin/python3").symlink_to(PY)
    for sub in ("scratch", "data"):
        (tmp_path / sub).mkdir()
    return tmp_path


def pack(root, venv, name="tiny", **env):
    environment = {**{k: v for k, v in os.environ.items() if not k.startswith("SLURM_")},
                   "PACK_VENV_WORK_PREFIX": f"{root}/scratch/", "PACK_VENV_OUT_PREFIX": f"{root}/data/",
                   "SLURM_CPUS_PER_TASK": "2", **env}
    return subprocess.run(["bash", str(PACK), "pack", str(venv), name, str(venv), str(root / "data/envs")],
                          capture_output=True, text=True, env=environment, timeout=120)


@linux_only
def test_pack_publishes_keyed_artifact_once(pack_root):
    scratch = pack_root / "scratch"
    first = pack(pack_root, packable_venv(scratch, "a", pack_root / "base"))
    assert first.returncode == 0, first.stderr
    tarball = Path(first.stdout.strip())
    match = re.fullmatch(rf"{pack_root}/data/envs/tiny-([0-9a-f]{{16}})\.tar\.zst", str(tarball))
    assert match, tarball
    manifest = json.loads(tarball.with_suffix("").with_suffix(".json").read_text())
    assert manifest["name"] == "tiny" and manifest["key"] == match.group(1)
    assert manifest["recipe"] == "venv-pack-v2"
    assert manifest["tarball_sha256"] == hashlib.sha256(tarball.read_bytes()).hexdigest()
    assert manifest["tarball_bytes"] == tarball.stat().st_size
    assert manifest["distributions"] == 2
    assert manifest["python"]["version"] == ".".join(map(str, sys.version_info[:3]))
    assert {"./bin/tool", "./pyvenv.cfg"} <= set(manifest["prefix_files"])
    assert manifest["stage"] == f"stage_venv {tarball} {match.group(1)} <VAR>"
    names = subprocess.run(["tar", "-I", "zstd", "-tf", str(tarball)], capture_output=True, text=True,
                           check=True).stdout.split()
    assert {f"./.python/bin/{PY}", "./.venv-pack/original-prefix", "./bin/tool"} <= set(names)
    before = (tarball.stat().st_mtime_ns, manifest["tarball_sha256"])

    # Same distributions built elsewhere (other build path, other local-source path): same key, untouched,
    # after checking the published tarball against its manifest.
    again = pack(pack_root, packable_venv(scratch, "b", pack_root / "base"))
    assert again.returncode == 0, again.stderr
    assert again.stdout.strip() == str(tarball)
    assert "already published" in again.stderr
    assert f"{tarball} matches its manifest (sha256" in again.stderr
    assert (tarball.stat().st_mtime_ns, hashlib.sha256(tarball.read_bytes()).hexdigest()) == before

    # New code in a local distribution with an unchanged version, or a new index version: new keys.
    code = pack(pack_root, packable_venv(scratch, "c", pack_root / "base", local_digest="sha256=changed,11"))
    version = pack(pack_root, packable_venv(scratch, "d", pack_root / "base", alpha="1.1"))
    keys = {Path(result.stdout.strip()).name for result in (first, code, version)}
    assert len(keys) == 3, (code.stderr, version.stderr)


@linux_only
def test_pack_refuses_a_damaged_published_artifact(pack_root):
    """A build whose key exists checks the published bytes (its staged check may
    run on an older copy already on the node) and never repairs them in place."""
    scratch = pack_root / "scratch"
    tarball = Path(pack(pack_root, packable_venv(scratch, "a", pack_root / "base")).stdout.strip())
    manifest = tarball.with_name(tarball.name.removesuffix(".tar.zst") + ".json")
    published = tarball.read_bytes()
    damaged = bytes([published[0] ^ 1]) + published[1:]  # same size, other bytes
    tarball.write_bytes(damaged)
    result = pack(pack_root, packable_venv(scratch, "b", pack_root / "base"))
    assert result.returncode == 4
    assert result.stdout == ""
    assert f"{tarball} does not match its manifest (sha256 {hashlib.sha256(damaged).hexdigest()}" in result.stderr
    assert f"move it and {manifest} aside, then rebuild" in result.stderr
    assert tarball.read_bytes() == damaged and manifest.is_file()
    tarball.write_bytes(published[:-1])  # truncated: refused on size alone
    result = pack(pack_root, packable_venv(scratch, "c", pack_root / "base"))
    assert result.returncode == 4
    assert f"size {len(published) - 1} B, manifest {len(published)} B" in result.stderr


@linux_only
def test_pack_waits_for_a_concurrent_publication(pack_root):
    scratch = pack_root / "scratch"
    first = pack(pack_root, packable_venv(scratch, "a", pack_root / "base"))
    tarball = Path(first.stdout.strip())
    stem = tarball.name.removesuffix(".tar.zst")
    for path in (tarball, tarball.with_name(f"{stem}.json")):
        path.unlink()
    lock = tarball.with_name(f".{stem}.publishing")
    lock.mkdir()
    blocked = pack(pack_root, packable_venv(scratch, "b", pack_root / "base"), PACK_VENV_LOCK_WAIT="0")
    assert blocked.returncode == 3
    assert f"remove {lock}" in blocked.stderr
    assert not tarball.exists()
    lock.rmdir()
    assert pack(pack_root, packable_venv(scratch, "c", pack_root / "base")).stdout.strip() == str(tarball)
    assert not lock.exists()
