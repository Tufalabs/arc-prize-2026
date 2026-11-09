"""Qualify one experiment of the tool-call-aligned recorded-logprob study on four B200s, then train it.

The three experiments share PR 121's frozen mixed corpus (the same training and validation rows,
with this branch's `teacher_logprob_targets`), PR 121's trainer runtime, harness capsule and
recipe; each differs from `configs/flash_sft/reap307-kimi-mixed-cosine.json` only in the data
files, the W&B run identity and the recorded-logprob loss (weight, CE/KL exclusivity). The
launcher refuses any other difference. Modeled on scripts/flash_mixed_cosine120/train.py.
"""
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-toolcalls-20261030")
BASELINE_ARTIFACTS = Path(
    "/data/shared/research/arc_3_results/shared/reap307-kimi-mixed-cekl-cosine120-20261030/artifacts"
)
BASELINE_RECIPE = "configs/flash_sft/reap307-kimi-mixed-cosine.json"
EXPERIMENT = os.environ["TOOLCALLS_EXPERIMENT"]
PUBLIC = ROOT / "experiments" / EXPERIMENT
ATTEMPT = PUBLIC / "runs" / ("job-" + os.environ["SLURM_JOB_ID"])
SCRATCH = Path("/scratch/jobs") / os.environ["SLURM_JOB_ID"] / "toolcalls-training"
SOURCE = SCRATCH / "source"
PRODUCTION = PUBLIC / "training"
TESTS = [
    "tests/test_flash_sft.py", "tests/test_flash_sft_logprobs.py", "tests/test_flash_sft_logprobs_sharded.py",
    "tests/test_flash_sft_cosine120.py", "tests/test_kimi_conversion_diagnostics.py",
    "tests/test_mixed_sft_provenance.py", "tests/logprob",
]
# The only keys an experiment may change against PR 121's recipe. offload_every only places memory (see
# training.flash_sft.PLACEMENT_KEYS); the launcher may probe it and a checkpoint resumes under any value.
ALLOWED = {"rows", "validation", "wandb", "recorded_logprobs", "offload_every"}
PLACEMENT_KEYS = ("offload_every", "offload_chunk_mib")
# Descending offload_every values to try (the first whose preflight fits the longest rows wins); empty = as frozen.
# Given by TOOLCALLS_OFFLOAD_CANDIDATES, or by a one-shot request file <experiment dir>/OFFLOAD_CANDIDATES that a
# pending job picks up at start and consumes.
OFFLOAD_REQUEST = PUBLIC / "OFFLOAD_CANDIDATES"


def offload_candidates(env=None, request=None, job_id=None):
    """Descending offload_every values to try, from TOOLCALLS_OFFLOAD_CANDIDATES or the one-shot request file.

    A blank variable counts as unset: a continuation exports it blank (so the first job's probe is not inherited
    through --export=ALL), and a request file placed while that continuation was pending must still be honored."""
    env = os.environ if env is None else env
    request = OFFLOAD_REQUEST if request is None else Path(request)
    job_id = env.get("SLURM_JOB_ID", "unknown") if job_id is None else job_id
    text = env.get("TOOLCALLS_OFFLOAD_CANDIDATES") or ""
    if not text.strip() and request.exists():
        text = request.read_text()
        request.rename(request.with_name(f"{request.name}.used-job-{job_id}"))
    return [int(x) for x in text.replace("\n", ",").split(",") if x.strip()]


def config_sha256(config):
    """training.flash_sft.config_sha256: the identity every checkpoint's state.json records."""
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def remember_identity(public, config):
    """Keep every configuration production trains under, by its hash, before the run starts: a checkpoint
    written under a moved placement stays resumable even if that attempt is interrupted before the launcher
    updates train-config.json."""
    path = Path(public) / "identities" / (config_sha256(config) + ".json")
    if not path.exists():
        save(path, config)
    return path


def resume_identity(production, public, config_path):
    """The configuration the latest checkpoint was written under: train-config.json when it still matches, else
    the remembered identity with the checkpoint's hash (or an earlier frozen config). Raises when none hashes to
    the checkpoint, rather than letting the trainer reject the resume as another configuration."""
    production, public = Path(production), Path(public)
    latest = read(production / "checkpoints/latest.json")
    sha = read(production / "checkpoints" / latest["path"] / "state.json")["config_sha256"]
    candidates = [Path(config_path), public / "identities" / f"{sha}.json",
                  *sorted(public.glob("train-config.before-job-*.json"))]
    for path in candidates:
        if path.exists() and config_sha256(read(path)) == sha:
            return read(path)
    raise RuntimeError(f"no known configuration hashes to the latest checkpoint's {sha}: {[str(c) for c in candidates]}")
# Compiled-kernel caches (inductor graphs and autotune results, Triton binaries) persist per user on the node, so a
# continuation or another arm landing on the same node does not recompile every sequence length: a process with
# empty caches spends 25-35 s per update compiling for each new length (measured 29-46 s cold vs 8-13 s warm on
# 28k-42k-token rows, 298 s vs 91 s for the cold validation). The caches must NOT live on NFS: Triton's temp-file +
# rename protocol is not safe across nodes there (observed: "Stale file handle" in one job and a cache entry whose
# .json was not yet visible in another, both crashing a rank and stalling the others until the NCCL timeout).
CACHE_ROOT = Path(os.environ.get("TOOLCALLS_CACHE_ROOT", f"/cache/users/{os.getuid()}/flash-sft"))
ALLOWED_VALIDATION = {"data", "data_sha256"}
ALLOWED_WANDB = {"name", "id"}
ALLOWED_LOSS = {"weight", "ce_positions"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def run(command, phase, env):
    print(json.dumps({"utc": dt.datetime.now(dt.UTC).isoformat(), "phase": phase, "status": "started"}), flush=True)
    with (ATTEMPT / (phase + ".log")).open("w") as f:
        subprocess.run([str(x) for x in command], cwd=SOURCE, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(json.dumps({"phase": phase, "status": "passed"}), flush=True)


def differences(actual, expected):
    return {k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k)}


def placement_free(config):
    return {k: v for k, v in config.items() if k not in PLACEMENT_KEYS}


def count_updates(metrics):
    return sum(1 for line in metrics.open() if '"train"' in line) if metrics.exists() else 0


def check_recipe(config, baseline, ready):
    """The experiment is PR 121's recipe on PR 121's rows; only data files, W&B identity and loss differ."""
    assert differences(config, baseline) <= ALLOWED, differences(config, baseline)
    assert differences(config["validation"], baseline["validation"]) <= ALLOWED_VALIDATION
    assert differences(config["wandb"], baseline["wandb"]) <= ALLOWED_WANDB
    assert differences(config["recorded_logprobs"], baseline["recorded_logprobs"]) <= ALLOWED_LOSS
    assert config["wandb"]["id"] and config["wandb"]["id"] != baseline["wandb"]["id"]
    assert config["rows_render_files"] == baseline["rows_render_files"]
    # The rows are the targets job's output: PR 121's deduplicated rows with this branch's targets.
    assert ready["status"] == "ready"
    for block, split in ((config["rows"], "rows"), (config["validation"], "validation-probe")):
        assert block["data"] == ready[split]["data"], (block["data"], ready[split]["data"])
        assert block["data_sha256"] == ready[split]["data_sha256"] == digest(block["data"]), split
    assert ready["rows"]["rows"] == 1934 and ready["validation-probe"]["rows"] == 20
    assert ready["tokenizer_sha256"] == config["recorded_logprobs"]["tokenizer_sha256"]
    # One row per rank, four ranks: the schedule PR 121 froze (484 updates per pass, 968 in all).
    updates = config["passes"] * math.ceil(ready["rows"]["rows"] / 4)
    assert updates == 968 and config["optimizer"]["decay"]["decay_until"] == updates // config["passes"] - 1
    return config["recorded_logprobs"]["weight"], updates


def accept(directory, config_path, expected=None):
    """Finite CE/KL accounting under the config's weight, frozen routers, synchronized adapters, a full checkpoint."""
    state = read(directory / "RUN_STATE.json")
    config = read(config_path)
    weight = config["recorded_logprobs"]["weight"]
    assert state["world_size"] == 4
    assert expected is None or state["update"] == expected
    assert state["config_sha256"] == config_sha256(config)
    records = [json.loads(line) for line in (directory / "metrics.jsonl").open()]
    updates = [r for r in records if "train" in r]
    assert updates and updates[-1]["update"] == state["update"]
    for record in updates:
        metrics = record["train"]
        assert all(math.isfinite(metrics[k]) for k in ["loss_per_token", "ce_loss", "teacher_kl", "grad_norm"])
        assert metrics["teacher_kl"] >= -1e-6
        assert math.isclose(metrics["loss_per_token"], metrics["ce_loss"] + weight * metrics["teacher_kl"],
                            rel_tol=2e-5, abs_tol=2e-6)
    validation = updates[-1]["validation"]
    assert set(validation["by_game"]) == {"pathfinder-0001", "taps-0001", "cd82-fb555c5d", "dc22-fdcac232",
                                          "escape_velocity-0001"}
    assert math.isclose(validation["ce_plus_kl_loss"], validation["ce_loss"] + weight * validation["teacher_kl"],
                        rel_tol=2e-5, abs_tol=2e-6)
    parameters = read(directory / "PARAMETERS_PASSED.json")
    assert parameters["status"] == "passed" and parameters["update"] == state["update"]
    assert parameters["updated_adapter_groups"] == ["attention", "routed", "shared"]
    assert parameters["router_tensors"] == 48
    assert all(parameters[k] for k in ["trainable_replicas_equal", "frozen_routers_unchanged",
                                        "optimizer_contains_only_adapters", "head_frozen"])
    latest = read(directory / "checkpoints/latest.json")
    checkpoint = directory / "checkpoints" / latest["path"]
    assert latest["update"] == state["update"]
    assert all((checkpoint / name).is_file() for name in ["adapters.pt", "optimizer.pt", "state.json",
               *[f"optimizer-rank{i}.pt" for i in range(4)], *[f"rng-rank{i}.pt" for i in range(4)]])
    return {"status": "passed", **state, "validation": validation, "checkpoint": str(checkpoint),
            "parameters": parameters}


def main():
    ATTEMPT.mkdir(parents=True, exist_ok=False)
    SCRATCH.mkdir(parents=True)
    assert int(os.environ["SLURM_GPUS_ON_NODE"]) == 4
    identity = read(ROOT / "artifacts/training-identity.json")
    assert digest(ROOT / "artifacts/source.tar") == identity["source_sha256"]
    for p in [SOURCE, SCRATCH / "runtime", SCRATCH / "harness"]:
        p.mkdir()
    subprocess.run(["tar", "-xf", str(ROOT / "artifacts/source.tar"), "-C", str(SOURCE)], check=True)
    # PR 121's sealed runtime, harness capsule and pytest dependencies, verified against the
    # checksums committed with the source.
    expected = read(SOURCE / "scripts/flash_mixed_cosine120/artifacts.json")["files"]
    for name in ("trainer-runtime.tar.zst", "harness-source.tar.gz"):
        assert digest(BASELINE_ARTIFACTS / name) == expected[name], name
    ready = read(ROOT / "data/DATA_READY.json")
    config_path = PUBLIC / "train-config.json"
    resume = bool(os.environ.get("TOOLCALLS_RESUME"))
    source_config = read(SOURCE / "configs/flash_sft" / (EXPERIMENT + ".json"))
    if resume:
        # The frozen config rules; the capsule may only differ in memory placement (an earlier probe moved it).
        assert placement_free(read(config_path)) == placement_free(source_config)
        # The latest checkpoint names the configuration to resume under. An interrupted attempt may have written
        # it under a moved placement that train-config.json never recorded: adopt that identity now.
        identity_config = resume_identity(PRODUCTION, PUBLIC, config_path)
        assert placement_free(identity_config) == placement_free(source_config)
        if read(config_path) != identity_config:
            save(PUBLIC / f"train-config.before-job-{os.environ['SLURM_JOB_ID']}.json", read(config_path))
            save(config_path, identity_config)
    else:
        # A fresh start may follow an attempt that failed before production (its config is frozen already).
        assert not PRODUCTION.exists(), "a fresh experiment must not overwrite an earlier training directory"
        if config_path.exists():
            assert placement_free(read(config_path)) == placement_free(source_config), "frozen config differs from the capsule"
        else:
            save(config_path, source_config)
    weight, updates = check_recipe(read(config_path), read(SOURCE / BASELINE_RECIPE), ready)
    candidates = offload_candidates() or [read(config_path)["offload_every"]]
    env = dict(os.environ, PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", PYTHONPATH=str(SOURCE) + ":" + str(BASELINE_ARTIFACTS / "test-deps"),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", GLOO_SOCKET_IFNAME="lo", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", WANDB_DIR=str(PUBLIC / "wandb"), TRITON_CACHE_DIR=str(CACHE_ROOT / "triton"),
               TORCHINDUCTOR_CACHE_DIR=str(CACHE_ROOT / "inductor"), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               FLASH_SFT_PINNED_TELEMETRY="1")
    for name in ("triton", "inductor"):
        (CACHE_ROOT / name).mkdir(parents=True, exist_ok=True)
    env.pop("WANDB_MODE", None)  # production syncs online through ~/.netrc
    run(["tar", "--zstd", "-xf", BASELINE_ARTIFACTS / "trainer-runtime.tar.zst", "-C", SCRATCH / "runtime"],
        "unpack-runtime", env)
    run(["tar", "-xzf", BASELINE_ARTIFACTS / "harness-source.tar.gz", "-C", SCRATCH / "harness"], "unpack-harness", env)
    run([sys.executable, SOURCE / "scripts/flash_mixed_cosine120/bootstrap.py", SCRATCH / "runtime", SCRATCH / "harness",
         SOURCE / "scripts/flash_mixed_cosine120/artifacts.json"], "install-current-harness", env)
    python = SCRATCH / "runtime/bin/python"
    env["FLASH_MIXED_HARNESS_IDENTITY"] = str(SCRATCH / "runtime/HARNESS_IMPORTED.json")
    save(ATTEMPT / "SOURCE.json", dict(identity, experiment=EXPERIMENT, weight=weight,
                                      ce_positions=read(config_path)["recorded_logprobs"].get("ce_positions", "all")))
    (PUBLIC / "wandb").mkdir(exist_ok=True)
    run([python, "-m", "pytest", *TESTS, "-q", "--tb=short"], "trainer-environment-tests", dict(env, WANDB_MODE="disabled"))
    run([python, "-m", "sft.distillation.staging", "--source", read(config_path)["model"], "--destination",
         SCRATCH / "model"], "stage-model", env)
    env["FLASH_SFT_STAGED_MODEL"] = str(SCRATCH / "model")
    command = [python, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4",
               "--module", "training.flash_sft"]
    if not resume:
        sys.path.insert(0, str(SOURCE))
        from training.flash_sft import RowIndex

        config = read(config_path)
        index = RowIndex(config["rows"]["data"], config["rows"]["data_sha256"], max_length=config["max_length"])
        longest = sorted(range(len(index)), key=lambda i: index.lengths[i], reverse=True)
        chosen = []
        for want_kl in [False, True]:
            chosen.append(next(i for i in longest if bool(index.row(i)["teacher_logprob_targets"]["events"]) == want_kl))
        for i in longest:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) == 4:
                break
        assert len(chosen) == 4 and len(set(chosen)) == 4
        qualification_rows = ATTEMPT / "qualification-rows.jsonl"
        with qualification_rows.open("w") as f:
            for i in chosen:
                f.write(json.dumps(index.row(i), separators=(",", ":")) + "\n")
        config["rows"] = {"data": str(qualification_rows), "data_sha256": digest(qualification_rows)}
        # Four rows make one update per pass; keep the production schedule's length so the
        # decay-range guard accepts the unchanged decay. --max-updates=1 runs the first update only.
        config["passes"] = updates
        config["wandb"] = None
        # The qualification update runs the four longest rows: the first offload_every whose preflight fits wins.
        probe = {"candidates": candidates, "failed": []}
        for every in candidates:
            config["offload_every"] = every
            qualification_config = ATTEMPT / f"qualification-config-offload{every}.json"
            save(qualification_config, config)
            try:
                run([*command, "--config", qualification_config, "--run-dir", ATTEMPT / f"qualification-offload{every}",
                     "--max-updates", "1", "--verify-parameters"], f"mixed-gpu-qualification-offload{every}",
                    dict(env, WANDB_MODE="disabled"))
            except subprocess.CalledProcessError:
                probe["failed"].append(every)
                if every == candidates[-1]:
                    raise
                continue
            qualified = accept(ATTEMPT / f"qualification-offload{every}", qualification_config, 1)
            assert qualified["validation"]["kl_coverage"] > 0
            probe["chosen"] = every
            save(ATTEMPT / "QUALIFICATION_PASSED.json", dict(qualified, offload_probe=probe))
            break
        frozen = read(config_path)
        if frozen["offload_every"] != probe["chosen"]:
            save(PUBLIC / f"train-config.before-job-{os.environ['SLURM_JOB_ID']}.json", frozen)
            save(config_path, dict(frozen, offload_every=probe["chosen"]))
        save(ATTEMPT / "OFFLOAD_PROBE.json", probe)
    else:
        assert (PRODUCTION / "checkpoints/latest.json").exists()
    save(ATTEMPT / "RUNNING.json", {"status": "running", "fresh_adapters": not resume, "target_updates": updates})
    if resume and candidates != [read(config_path)["offload_every"]]:
        # Probe on resume: the checkpoint was written under the frozen config (the resume identity); a candidate that
        # fails before its first update (preflight on the longest rows does not fit) hands over to the next one.
        probe = {"candidates": candidates, "failed": []}
        frozen = read(config_path)
        before = count_updates(PRODUCTION / "metrics.jsonl")
        for every in candidates:
            variant = ATTEMPT / f"config-offload{every}.json"
            save(variant, dict(frozen, offload_every=every))
            remember_identity(PUBLIC, dict(frozen, offload_every=every))
            try:
                run([*command, "--config", variant, "--run-dir", PRODUCTION, "--resume-identity", config_path,
                     "--time-budget-seconds", "72000", "--stop-file", PUBLIC / "STOP", "--verify-parameters"],
                    f"production-offload{every}", env)
            except subprocess.CalledProcessError:
                if count_updates(PRODUCTION / "metrics.jsonl") > before or every == candidates[-1]:
                    raise  # the run itself failed after training, or every candidate is exhausted
                probe["failed"].append(every)
                continue
            probe["chosen"] = every
            break
        save(ATTEMPT / "OFFLOAD_PROBE.json", probe)
        if frozen["offload_every"] != probe["chosen"]:
            save(PUBLIC / f"train-config.before-job-{os.environ['SLURM_JOB_ID']}.json", frozen)
            save(config_path, dict(frozen, offload_every=probe["chosen"]))
    else:
        remember_identity(PUBLIC, read(config_path))
        run([*command, "--config", config_path, "--run-dir", PRODUCTION, *([] if resume else ["--max-updates", "20"]),
             *(["--resume-identity", config_path] if resume else []),
             "--time-budget-seconds", "72000", "--stop-file", PUBLIC / "STOP", "--verify-parameters"], "production", env)
    result = accept(PRODUCTION, config_path)
    assert result["total_updates"] == updates
    save(ATTEMPT / "SEGMENT_PASSED.json", result)
    (ATTEMPT / "RUNNING.json").unlink()
    if result["complete"]:
        save(PUBLIC / "TRAINING_PASSED.json", dict(result, source=identity))
    elif (PUBLIC / "STOP").exists():
        save(PUBLIC / "STOPPED.json", result)
    else:
        response = subprocess.check_output(
            ["sbatch", "--parsable", "--dependency=afterok:" + os.environ["SLURM_JOB_ID"],
             "--export=ALL,TOOLCALLS_RESUME=1,TOOLCALLS_OFFLOAD_CANDIDATES=,TOOLCALLS_EXPERIMENT=" + EXPERIMENT,
             "--job-name", EXPERIMENT,
             "--output", str(PUBLIC / "training-job-%j.log"), str(ROOT / "artifacts/train.sbatch")], text=True).strip()
        job = response.split(";")[0]
        assert job.isdigit()
        save(ATTEMPT / "CONTINUATION.json", {"job_id": job, "completed_updates": result["update"]})
        save(PUBLIC / "LATEST_TRAINING_JOB.json", {"job_id": job, "completed_updates": result["update"]})


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if ATTEMPT.exists():
            (ATTEMPT / "RUNNING.json").unlink(missing_ok=True)
            save(ATTEMPT / "FAILED.json", {"type": type(error).__name__, "message": str(error)})
        raise
