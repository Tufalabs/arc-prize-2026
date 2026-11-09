"""scripts/flash_rl_submit.sh with only --config: HEAD of its own checkout, the
environments built for it, a fresh run directory, and server concurrency left to
the job, which derives it from the recipe for the allocation submitted. A
recording sbatch stands in for Slurm; nothing is allocated."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from env_fixtures import (
    RECIPE,
    ROOT,
    VARIABLES,
    calls,
    make_checkout,
    plan,
    recording_commands,
    tool_env,
    write_all,
)
from flash_configs import write_arc3_flash_config

from training.config import ExperimentConfig
from training.job_shape import job_shape, separate_shape


@pytest.fixture
def setup(tmp_path):
    checkout = tmp_path / "arc-training"
    revision = make_checkout(checkout)
    envs = tmp_path / "envs"
    tarballs = write_all(envs, plan(checkout, "trainer", revision)["python"])
    record = tmp_path / "sbatch.jsonl"
    env = tool_env(recording_commands(tmp_path, record), ARC_TRAINING_ENVS=str(envs))
    return checkout, revision, envs, tarballs, record, env


def submit(checkout, env, *args, cwd=None):
    return subprocess.run(
        ["bash", str(checkout / "scripts/flash_rl_submit.sh"), *args],
        cwd=cwd or checkout.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def submitted(record):
    (call,) = calls(record)
    return call["argv"], call["env"]


def job_concurrency(argv, config):
    """Each engine's --max-num-seqs as the job derives it from CONFIG for the
    allocation in ARGV: the shared node's four servers, or every server node's
    GPUs (components after the learner's, each --nodes x --gres GPUs)."""
    cfg = ExperimentConfig.load(config, gpu_claims=False)
    components = []
    for arg in argv:
        if arg == ":" or not components:
            components.append({})
        if arg.startswith(("--nodes=", "--gres=")):
            components[-1][arg.split("=")[0]] = int(arg.rsplit(":", 1)[-1].split("=")[-1])
    if len(components) == 1:
        return job_shape(cfg, {})["SERVER_MAX_NUM_SEQS"]
    nodes = [f"node{i}" for group in components[1:] for i in range(group["--nodes"])]
    gpus = {group["--gres"] for group in components[1:]}
    assert len(gpus) == 1
    shape = separate_shape(cfg, nodes, gpus.pop(), 4, 50000, Path("/data/run"),
                           rewrite_learner_devices=True)
    return shape["SERVER_MAX_NUM_SEQS"]


# The drafting engine's KV pool on one B200 (training/FLASH_RL.md, "Speculative
# decoding"): 3,967,385 tokens in attention blocks of 928 tokens.
DRAFTING_POOL_TOKENS, DRAFTING_BLOCK_TOKENS = 3967385, 928


def test_recipe_slots_fit_the_drafting_pool_at_their_longest():
    """The recipe's engines take a server's 32 attempts plus 8 slots, which the
    drafting engine's pool holds even with every slot at the window plus a reply
    (98,304 tokens, 106 blocks): 40 x 106 of 4,275 blocks."""
    cfg = ExperimentConfig.load(ROOT / RECIPE)
    assert cfg.inference.speculative_decoding.enabled
    slots = int(job_shape(cfg, {})["SERVER_MAX_NUM_SEQS"])
    assert slots == 40
    longest = cfg.inference.agent_context_window + cfg.inference.agent_max_completion_tokens
    blocks = -(-longest // DRAFTING_BLOCK_TOKENS)
    assert slots * blocks <= DRAFTING_POOL_TOKENS // DRAFTING_BLOCK_TOKENS


def test_config_alone_runs_head_with_its_environments(setup):
    checkout, revision, envs, tarballs, record, env = setup
    # From outside the checkout: the relative config is read from the revision.
    result = submit(checkout, env, "--config", RECIPE)
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert argv[-3:] == [
        str(checkout / "scripts/flash_rl.sbatch"),
        str(checkout),
        revision,
    ]
    for recipe, variable in VARIABLES.items():
        assert job_env[variable] == str(tarballs[recipe])
    # Server concurrency is the job's: 128 rollouts over 4 servers, + 8.
    assert "SERVER_MAX_NUM_SEQS" not in job_env
    assert job_concurrency(argv, checkout / RECIPE) == "40"
    assert job_env["ARC3_CONFIG"] == RECIPE
    run_dir = job_env["ARC3_RUN_DIR"]
    assert re.fullmatch(
        r"/data/personal/tester/arc-training-runs/flash_8gpu_arc2_grpo_tokmean-\d{8}-\d{4}/lora",
        run_dir,
    ), run_dir
    assert f"--output={run_dir}/slurm-%j.out" in argv
    assert f"flash_rl_submit: run directory {run_dir}" in result.stderr
    assert f"flash_rl_submit: source {checkout} at {revision}" in result.stderr
    assert (
        "flash_rl_submit: SERVER_MAX_NUM_SEQS unset: each segment derives it from the run's config"
        in result.stderr
    )


@pytest.mark.parametrize(
    ("shape", "seqs"),
    [
        ([], 40),
        (["--server-nodes", "0"], 40),
        (["--server-gpus-per-node", "6"], 30),  # ceil(128 / 6) + 8
        (["--server-nodes", "2", "--server-gpus-per-node", "8"], 16),
        (["--server-groups", "2", "--server-gpus-per-node", "6"], 19),  # 12 servers
    ],
)
def test_server_concurrency_follows_rollouts_per_server(setup, shape, seqs):
    checkout, _, _, _, record, env = setup
    result = submit(checkout, env, "--config", RECIPE, *shape)
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert "SERVER_MAX_NUM_SEQS" not in job_env
    assert job_concurrency(argv, checkout / RECIPE) == str(seqs)


def test_concurrency_override_and_episode_cap(setup, tmp_path):
    checkout, _, _, _, record, env = setup
    result = submit(checkout, {**env, "SERVER_MAX_NUM_SEQS": "48"}, "--config", RECIPE)
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert job_env["SERVER_MAX_NUM_SEQS"] == "48"
    assert "SERVER_MAX_NUM_SEQS=48 (from the environment, for every segment)" in result.stderr
    # The job keeps an explicit value that holds a server's share, and refuses one below it.
    cfg = ExperimentConfig.load(checkout / RECIPE)
    assert job_shape(cfg, job_env)["SERVER_MAX_NUM_SEQS"] == "48"
    with pytest.raises(ValueError, match="SERVER_MAX_NUM_SEQS=31 must be an integer of at least 32"):
        job_shape(cfg, {"SERVER_MAX_NUM_SEQS": "31"})
    bad = submit(checkout, {**env, "SERVER_MAX_NUM_SEQS": "lots"}, "--config", RECIPE)
    assert bad.returncode == 2
    assert "SERVER_MAX_NUM_SEQS must be a positive integer" in bad.stderr
    payload = json.loads((checkout / RECIPE).read_text())
    payload["inference"]["max_concurrent_episodes"] = 16
    capped = tmp_path / "capped.json"
    capped.write_text(json.dumps(payload))
    record.unlink()
    result = submit(checkout, env, "--config", str(capped))
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert job_concurrency(argv, capped) == "24"
    assert "/arc-training-runs/capped-" in job_env["ARC3_RUN_DIR"]


def test_test_only_leaves_the_run_directory_to_the_real_submission(setup, tmp_path):
    """Slurm checks a --test-only request without the run directory. Creating it
    would make a real submission in the same minute refuse the default one."""
    checkout, _, _, _, record, env = setup
    result = submit(checkout, env, "--config", RECIPE, "--test-only")
    assert result.returncode == 0, result.stderr
    assert "--test-only" in submitted(record)[0]
    assert not (tmp_path / "mkdir.log").exists()
    record.unlink()
    result = submit(checkout, env, "--config", RECIPE)
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert "--test-only" not in argv
    log = (tmp_path / "mkdir.log").read_text().splitlines()
    assert log == [f"-p {job_env['ARC3_RUN_DIR']}"]


def test_uncommitted_changes_are_refused_unless_allowed(setup):
    checkout, revision, _, _, record, env = setup
    config = checkout / RECIPE
    config.write_text(
        config.read_text().replace('"num_iterations": 10000', '"num_iterations": 2')
    )
    (checkout / "notes.txt").write_text("scratch\n")
    result = submit(checkout, env, "--config", RECIPE)
    assert result.returncode == 2
    assert "has uncommitted changes, and the job runs its HEAD commit" in result.stderr
    assert f" M {RECIPE}" in result.stderr and "?? notes.txt" in result.stderr
    assert calls(record) == []
    allowed = submit(checkout, env, "--config", RECIPE, "--allow-dirty")
    assert allowed.returncode == 0, allowed.stderr
    assert (
        "--allow-dirty: running HEAD without the uncommitted changes" in allowed.stderr
    )
    assert submitted(record)[0][-1] == revision


def test_missing_environments_name_the_build_command(setup, tmp_path):
    checkout, revision, envs, tarballs, record, env = setup
    tarballs["harness"].unlink()
    result = submit(checkout, env, "--config", RECIPE)
    assert result.returncode == 2
    assert f"no harness environment in {envs} matches {revision}" in result.stderr
    assert f"there is no harness-<key> artifact in {envs}" in result.stderr
    assert (
        f"  {checkout}/scripts/build_envs.sh --revision {revision} --out {envs}\n"
        in result.stderr
    )
    assert calls(record) == []
    # The command builds away from the nodes the submission excludes.
    excluded = submit(checkout, env, "--config", RECIPE, "--exclude", "dgx02")
    assert excluded.returncode == 2
    assert (
        f"  {checkout}/scripts/build_envs.sh --revision {revision} --out {envs} "
        "-- --exclude=dgx02\n" in excluded.stderr
    )
    # An explicit variable wins, and is all the harness needs.
    explicit = submit(
        checkout,
        {
            **env,
            "ARC3_HARNESS_VENV_TARBALL": "/data/explicit/harness-0123456789abcdef.tar.zst",
        },
        "--config",
        RECIPE,
    )
    assert explicit.returncode == 0, explicit.stderr
    job_env = submitted(record)[1]
    assert (
        job_env["ARC3_HARNESS_VENV_TARBALL"]
        == "/data/explicit/harness-0123456789abcdef.tar.zst"
    )
    assert job_env["ARC3_FLASH_VENV_TARBALL"] == str(tarballs["trainer"])
    # Another root, given by --envs, and another clone, given explicitly.
    other = tmp_path / "other-clone"
    subprocess.run(["git", "clone", "-q", str(checkout), str(other)], check=True)
    result = submit(
        checkout,
        env,
        "--config",
        RECIPE,
        "--envs",
        str(tmp_path / "empty"),
        "--",
        str(other),
        revision,
    )
    assert result.returncode == 2
    assert f"--out {tmp_path / 'empty'} --repo {other}\n" in result.stderr


def test_the_job_gets_the_full_commit(setup):
    """An abbreviation or a ref is resolved once, here: the job archives the
    commit the environments were resolved for."""
    checkout, revision, _, _, record, env = setup
    for given in (revision[:8], "HEAD"):
        record.unlink(missing_ok=True)
        result = submit(checkout, env, "--config", RECIPE, "--", str(checkout), given)
        assert result.returncode == 0, result.stderr
        assert submitted(record)[0][-2:] == [str(checkout), revision]
        assert f"flash_rl_submit: source {checkout} at {revision}" in result.stderr
    missing = submit(checkout, env, "--config", RECIPE, "--", str(checkout), "f" * 12)
    assert missing.returncode == 2
    assert f"{'f' * 12} is not a commit in {checkout}" in missing.stderr


def test_arc3_configs_need_no_worker(setup, tmp_path):
    checkout, _, _, tarballs, record, env = setup
    tarballs["arc2-worker"].unlink()
    result = submit(checkout, env, "--config", str(write_arc3_flash_config(tmp_path)))
    assert result.returncode == 0, result.stderr
    job_env = submitted(record)[1]
    assert "ARC3_ARC2_WORKER_VENV_TARBALL" not in job_env
    assert job_env["ARC3_HARNESS_VENV_TARBALL"] == str(tarballs["harness"])


def test_a_config_missing_from_the_revision_is_refused(setup):
    checkout, revision, _, _, record, env = setup
    result = submit(checkout, env, "--config", "configs/not-committed.json")
    assert result.returncode == 2
    assert (
        f"Cannot read configs/not-committed.json here (a relative path is read from {revision}"
        in result.stderr
    )
    assert calls(record) == []


def test_a_copy_outside_a_checkout_needs_an_explicit_source(setup, tmp_path):
    checkout, revision, _, _, record, env = setup
    tools = tmp_path / "tools"
    tools.mkdir()
    shutil.copy2(
        checkout / "scripts/flash_rl_submit.sh", tools / "H_flash_rl_submit.sh"
    )
    command = ["bash", str(tools / "H_flash_rl_submit.sh"), "--config", RECIPE]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 2
    assert (
        "is not at the root of a training repository: pass -- <repo> <revision>"
        in result.stderr
    )
    explicit = {
        **env,
        **{
            variable: f"/data/{recipe}-0123456789abcdef.tar.zst"
            for recipe, variable in VARIABLES.items()
        },
    }
    result = subprocess.run(
        [*command, "--", str(checkout), revision],
        env=explicit,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    argv, job_env = submitted(record)
    assert argv[-3:] == [str(tools / "H_flash_rl.sbatch"), str(checkout), revision]
    assert "SERVER_MAX_NUM_SEQS" not in job_env
