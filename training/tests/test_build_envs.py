"""scripts/build_envs.sh: one command submits the three environment builds for a
revision, skipping recipes whose artifact already matches. sbatch is a recorder;
the builder is build_env.sbatch's real planner minus its /data guard on --out."""

from __future__ import annotations

import subprocess

import pytest
from env_fixtures import (
    NAMES,
    PIN,
    ROOT,
    calls,
    make_checkout,
    plan,
    recording_commands,
    source,
    tool_env,
    write_all,
    write_artifact,
)

# The next step for a smoke run, beside the full recipe.
SMOKE = (
    'Or, for a smoke run first, derive a short config as training/FLASH_RL.md shows ("The recipe '
    'and smoke configs") and pass its absolute path to --config.'
)

# build_env.sbatch as the test checkout's builder: log the call, plan without --out.
BUILDER = """#!/bin/bash
printf '%s\\n' "$*" >> "$PLAN_LOG"
args=()
while (( $# )); do
  if [[ "$1" == --out ]]; then shift 2; continue; fi
  args+=("$1"); shift
done
exec bash "$REAL_BUILDER" "${args[@]}"
"""


@pytest.fixture
def setup(tmp_path):
    checkout = tmp_path / "arc-training"
    revision = make_checkout(checkout, builder=BUILDER)
    record = tmp_path / "sbatch.jsonl"
    envs = tmp_path / "envs"
    env = tool_env(
        recording_commands(tmp_path, record),
        ARC_TRAINING_ENVS=str(envs),
        PLAN_LOG=str(tmp_path / "plans.log"),
        REAL_BUILDER=str(ROOT / "scripts/build_env.sbatch"),
    )
    python = plan(
        checkout, "trainer", revision, builder=ROOT / "scripts/build_env.sbatch"
    )["python"]
    return checkout, revision, record, envs, env, python


def build_envs(checkout, env, *args):
    return subprocess.run(
        ["bash", str(checkout / "scripts/build_envs.sh"), *args],
        cwd=checkout.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_submits_the_three_builds_for_head(setup, tmp_path):
    checkout, revision, record, envs, env, _ = setup
    result = build_envs(checkout, env)
    assert result.returncode == 0, result.stderr
    assert envs.is_dir()
    submitted = calls(record)
    assert [call["argv"][-6] for call in submitted] == list(NAMES)
    for job, (recipe, call) in enumerate(zip(NAMES, submitted), start=4242):
        argv = call["argv"]
        assert argv[:3] == [
            "--parsable",
            f"--job-name=build-env-{recipe}",
            f"--output={envs}/slurm-%j.out",
        ]
        assert argv[3].startswith("/dev/fd/")  # the revision's own build_env.sbatch
        assert call["script"] == BUILDER
        assert argv[4:] == [
            recipe,
            revision,
            "--repo",
            str(checkout),
            "--out",
            str(envs),
        ]
        assert f"build_envs: {recipe}: job {job}" in result.stdout
    # Every recipe was planned against the real --out before anything was submitted.
    planned = (tmp_path / "plans.log").read_text().splitlines()
    assert planned[:3] == [
        f"{recipe} {revision} --repo {checkout} --out {envs} --print-plan"
        for recipe in NAMES
    ]
    assert f"revision {revision} of {checkout}; artifacts in {envs}" in result.stdout
    assert (
        "Check progress: sacct -X -j 4242,4243,4244 -o JobID,JobName%22,State,Elapsed"
        in result.stdout
    )
    logs = " ".join(f"{envs}/slurm-{job}.out" for job in (4242, 4243, 4244))
    assert f"grep -h BUILD_ENV_RESULT {logs}" in result.stdout
    assert (
        "submit the full recipe: "
        f"{checkout}/scripts/flash_rl_submit.sh --config configs/flash_8gpu_arc2_grpo_tokmean.json"
        in result.stdout
    )
    assert SMOKE in result.stdout
    assert "--envs" not in result.stdout  # the default root needs no option
    assert f"-- {checkout}" not in result.stdout  # HEAD of the submit helper's checkout


def commit(checkout, name):
    (checkout / name).write_text("later\n")
    for args in (["add", name], ["commit", "-qm", name]):
        subprocess.run(
            ["git", "-C", str(checkout), "-c", "user.name=t", "-c", "user.email=t@e"]
            + args,
            check=True,
        )


def test_the_next_step_runs_the_revision_that_was_built(setup, tmp_path):
    """flash_rl_submit.sh runs its checkout's HEAD by default: another revision
    or clone is named after --, as its full commit."""
    checkout, revision, record, envs, env, python = setup
    commit(checkout, "later.txt")
    submit = f"{checkout}/scripts/flash_rl_submit.sh --config configs/flash_8gpu_arc2_grpo_tokmean.json"
    result = build_envs(checkout, env, "--revision", revision[:10])
    assert result.returncode == 0, result.stderr
    assert [call["argv"][5] for call in calls(record)] == [revision] * 3
    assert (
        f"Once all have COMPLETED, submit the full recipe: {submit} -- {checkout} {revision}\n"
        in result.stdout
    )
    other = tmp_path / "other-clone"
    subprocess.run(["git", "clone", "-q", str(checkout), str(other)], check=True)
    head = subprocess.run(
        ["git", "-C", str(other), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    write_all(envs, python)  # the later commit changed no lock or pin
    result = build_envs(checkout, env, "--repo", str(other))
    assert result.returncode == 0, result.stderr
    assert (
        f"nothing to build. Submit the full recipe: {submit} -- {other} {head}\n"
        in result.stdout
    )


def test_check_reports_without_submitting(setup):
    checkout, revision, record, envs, env, python = setup
    result = build_envs(checkout, env, "--check", "--", "--exclude=dgx02")
    assert result.returncode == 3
    assert not envs.exists()  # read-only: not even the root is created
    for recipe, name in NAMES.items():
        assert f"build_envs: {recipe} is not built" in result.stdout
        assert f"there is no {name}-<key> artifact in {envs}" in result.stderr
    assert (
        "build_envs: not built: trainer harness arc2-worker; the same command without "
        "--check submits their builds" in result.stdout
    )
    tarballs = write_all(envs, python)
    # The harness artifact was built for another harness pin.
    write_artifact(
        envs, "harness", "1" * 16, source("harness", python, harness_revision="9" * 40)
    )
    result = build_envs(checkout, env, "--check")
    assert result.returncode == 3
    for recipe in ("trainer", "arc2-worker"):
        assert f"build_envs: {recipe} is built: {tarballs[recipe]}" in result.stdout
    assert "build_envs: harness is not built" in result.stdout
    assert f"no harness environment in {envs} matches {revision}" in result.stderr
    assert (
        f"the nearest is key {'1' * 16} (harness-{'1' * 16}.json), which differs in "
        f"harness_revision (recorded {'9' * 40}, this revision {PIN})"
    ) in result.stderr
    assert "build_envs: not built: harness;" in result.stdout
    write_all(envs, python)
    result = build_envs(checkout, env, "--check")
    assert result.returncode == 0, result.stderr
    assert "build_envs: all three are built. Submit the full recipe: " in result.stdout
    assert calls(record) == []
    both = build_envs(checkout, env, "--check", "--force")
    assert both.returncode == 2
    assert "--force builds and --check only reports" in both.stderr


def test_matching_artifacts_are_not_rebuilt(setup):
    checkout, revision, record, envs, env, python = setup
    tarballs = write_all(envs, python)
    tarballs.pop("arc2-worker").unlink()
    result = build_envs(checkout, env)
    assert result.returncode == 0, result.stderr
    assert [call["argv"][4] for call in calls(record)] == ["arc2-worker"]
    for recipe, tarball in tarballs.items():
        assert f"build_envs: {recipe} is up to date: {tarball}" in result.stdout
    write_all(envs, python)
    record.unlink()
    result = build_envs(checkout, env, "--revision", revision)
    assert result.returncode == 0, result.stderr
    assert calls(record) == []
    assert "nothing to build. Submit the full recipe: " in result.stdout
    assert SMOKE in result.stdout
    forced = build_envs(checkout, env, "--force")
    assert forced.returncode == 0, forced.stderr
    assert [call["argv"][4] for call in calls(record)] == list(NAMES)


def test_options_reach_the_builds(setup, tmp_path):
    checkout, revision, record, _, env, python = setup
    elsewhere = tmp_path / "shared-envs"
    write_all(elsewhere, python)
    (elsewhere / f"{NAMES['harness']}-{'1' * 16}.json").write_text(
        "{}"
    )  # no longer matches
    result = build_envs(
        checkout,
        env,
        "--out",
        f"{elsewhere}/",
        "--revision",
        "HEAD",
        "--",
        "--exclude=dgx02",
    )
    assert result.returncode == 0, result.stderr
    (call,) = calls(record)
    assert call["argv"][:2] == ["--parsable", "--job-name=build-env-harness"]
    assert "--exclude=dgx02" in call["argv"]
    assert call["argv"][-6:] == [
        "harness",
        revision,
        "--repo",
        str(checkout),
        "--out",
        str(elsewhere),
    ]
    assert f"--envs {elsewhere}" in result.stdout


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (["--revision", "no-such-ref"], "no-such-ref is not a commit in"),
        (["--bogus"], "unknown option --bogus"),
    ],
)
def test_bad_requests_submit_nothing(setup, args, error):
    checkout, _, record, _, env, _ = setup
    result = build_envs(checkout, env, *args)
    assert result.returncode == 2
    assert error in result.stderr
    assert calls(record) == []


def test_a_plan_failure_submits_nothing(setup):
    checkout, _, record, _, env, _ = setup
    (checkout / "uv.lock").unlink()
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@e",
            "commit",
            "-qam",
            "no lock",
        ],
        check=True,
    )
    result = build_envs(checkout, env)
    assert result.returncode == 2
    assert "cannot plan trainer" in result.stderr
    assert calls(record) == []


def test_help_and_syntax():
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/build_envs.sh"), "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "scripts/build_envs.sh [--revision REV] [--out DIR]" in result.stdout
    assert "need not be pushed" in result.stdout
    assert "--check         submit nothing" in result.stdout
    assert "set -euo" not in result.stdout
