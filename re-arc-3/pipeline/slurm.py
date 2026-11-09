from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .commons import JSONValue, read_json, utc_now, worker_status_path, write_json
from .config import PipelineRunConfig
from .idea_loading import load_idea_specs, load_official_description_pair_specs
from .prompt_building import load_spec_generation_prompt
from .spec_generation import generate_structured_response
from .worker import run_worker

SlurmRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def _run_slurm_command(cmd: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(cmd), cwd=str(cwd), check=False, text=True, capture_output=True)


def python_bin_for_root(root: Path) -> Path:
    python_bin = root / ".venv" / "bin" / "python"
    if python_bin.exists():
        return python_bin
    return Path(sys.executable)


def _slurm_logs_dir(*, root_dir: Path, run_id: str) -> Path:
    return root_dir / "pipeline" / "runs" / run_id / "slurm"


def submit_slurm_array(
    *,
    root_dir: Path,
    run_id: str,
    task_count: int,
    partition: str,
    time_limit: str,
    mem: str,
    ntasks: int,
    cpus_per_task: int,
    runner: SlurmRunner = _run_slurm_command,
) -> dict[str, str | int]:
    if task_count <= 0:
        raise ValueError("task_count must be positive")
    python_bin = python_bin_for_root(root_dir)
    logs_dir = _slurm_logs_dir(root_dir=root_dir, run_id=run_id)
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_pattern = str(logs_dir / "%A_%a.out")
    stderr_pattern = str(logs_dir / "%A_%a.err")
    wrap_cmd = f'cd "{root_dir}" && exec "{python_bin}" -m pipeline.slurm --run-id "{run_id}"'
    cmd = [
        "sbatch",
        f"--partition={partition}",
        f"--time={time_limit}",
        f"--mem={mem}",
        f"--ntasks={ntasks}",
        f"--cpus-per-task={cpus_per_task}",
        f"--array=0-{task_count - 1}",
        f"--output={stdout_pattern}",
        f"--error={stderr_pattern}",
        f"--wrap={wrap_cmd}",
    ]
    result = runner(cmd, root_dir)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"sbatch failed with exit code {result.returncode}: {detail}")
    output = (result.stdout or "").strip()
    match = re.search(r"\b(\d+)\b", output)
    if not match:
        raise RuntimeError(f"Could not parse Slurm job id from sbatch output: {output!r}")
    return {
        "slurm_job_id": match.group(1),
        "slurm_submit_stdout": output,
        "slurm_array_range": f"0-{task_count - 1}",
        "slurm_wrap_command": wrap_cmd,
        "slurm_logs_dir": str(logs_dir),
        "slurm_stdout_pattern": stdout_pattern,
        "slurm_stderr_pattern": stderr_pattern,
    }


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "run_meta.json"


def _run_status_path(run_dir: Path) -> Path:
    return run_dir / "run_status.json"


def _resolve_repo_path(root_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    resolved = root_dir / path
    if resolved.exists():
        return resolved

    legacy_prompt_names = {
        "implementation_codex_prompt.md": "codex_check_visual_prompt.md",
        "spec_generation_instructions.md": "spec_prompt.md",
    }
    parts = list(path.parts)
    if "prompts_and_schemas" in parts:
        remapped_parts = ["prompts" if part == "prompts_and_schemas" else part for part in parts]
        remapped_path = root_dir / Path(*remapped_parts)
        if remapped_path.exists():
            return remapped_path
        remapped_name = legacy_prompt_names.get(remapped_path.name)
        if remapped_name is not None:
            remapped_with_alias = remapped_path.with_name(remapped_name)
            if remapped_with_alias.exists():
                return remapped_with_alias

    return resolved


def _slurm_states_for_job(*, job_id: str, root_dir: Path, runner: SlurmRunner) -> list[str]:
    result = runner(["squeue", "-h", "-j", job_id, "-o", "%T"], root_dir)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"squeue failed with exit code {result.returncode}: {detail}")
    return [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]


def cancel_slurm_job(*, job_id: str, root_dir: Path, runner: SlurmRunner = _run_slurm_command) -> None:
    result = runner(["scancel", job_id], root_dir)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"scancel failed with exit code {result.returncode}: {detail}")


def _aggregate_run_status(
    *, run_id: str, run_dir: Path, job_id: str, idea_ids: Sequence[str], slurm_states: Sequence[str]
) -> dict[str, JSONValue]:
    workers_dir = run_dir / "workers"
    counts: dict[str, int] = {}
    workers: list[dict[str, JSONValue]] = []
    for index, idea_id in enumerate(idea_ids):
        status_path = worker_status_path(workers_dir=workers_dir, idea_id=idea_id)
        worker_root = workers_dir / idea_id
        slurm_stdout_path = _slurm_logs_dir(root_dir=run_dir.parents[2], run_id=run_id) / f"{job_id}_{index}.out"
        slurm_stderr_path = _slurm_logs_dir(root_dir=run_dir.parents[2], run_id=run_id) / f"{job_id}_{index}.err"
        if status_path.exists():
            raw_status = read_json(status_path)
            status_payload = raw_status if isinstance(raw_status, dict) else {"status": "invalid"}
            worker_status = str(status_payload.get("status") or "unknown")
            worker_entry: dict[str, JSONValue] = {
                "idea_id": idea_id,
                "worker_root": str(worker_root),
                "worker_status_path": str(status_path),
                "slurm_log_path": str(slurm_stdout_path),
                "slurm_error_log_path": str(slurm_stderr_path),
                **{str(key): value for key, value in status_payload.items()},
            }
        else:
            worker_status = "pending"
            worker_entry = {
                "idea_id": idea_id,
                "status": worker_status,
                "worker_root": str(worker_root),
                "worker_status_path": str(status_path),
                "slurm_log_path": str(slurm_stdout_path),
                "slurm_error_log_path": str(slurm_stderr_path),
            }
        counts[worker_status] = counts.get(worker_status, 0) + 1
        workers.append(worker_entry)

    scheduler_counts: dict[str, int] = {}
    for state in slurm_states:
        scheduler_counts[state] = scheduler_counts.get(state, 0) + 1

    payload: dict[str, JSONValue] = {
        "run_id": run_id,
        "slurm_job_id": job_id,
        "updated_at": utc_now(),
        "active": bool(slurm_states),
        "slurm_states": list(slurm_states),
        "slurm_state_counts": scheduler_counts,
        "worker_state_counts": counts,
        "workers": workers,
    }
    return payload


def _slurm_terminal_state(*, job_id: str, root_dir: Path, runner: SlurmRunner) -> str | None:
    result = runner(["scontrol", "show", "job", job_id], root_dir)
    if result.returncode != 0:
        return None
    match = re.search(r"\bJobState=([A-Z_]+)\b", result.stdout or "")
    if not match:
        return None
    return match.group(1)


def _run_status_signature(payload: dict[str, JSONValue]) -> dict[str, JSONValue]:
    workers = payload.get("workers")
    worker_statuses: list[dict[str, str]] = []
    if isinstance(workers, list):
        for worker in workers:
            if not isinstance(worker, dict):
                continue
            worker_statuses.append(
                {"idea_id": str(worker.get("idea_id") or ""), "status": str(worker.get("status") or "")}
            )
    return {
        "active": bool(payload.get("active")),
        "worker_state_counts": payload.get("worker_state_counts"),
        "workers": worker_statuses,
    }


def _append_run_status_history(*, status_path: Path, payload: dict[str, JSONValue]) -> None:
    history: list[dict[str, JSONValue]] = []
    if status_path.exists():
        raw_history = read_json(status_path)
        if isinstance(raw_history, list):
            history = [entry for entry in raw_history if isinstance(entry, dict)]
    if history and _run_status_signature(history[-1]) == _run_status_signature(payload):
        return
    history.append(payload)
    write_json(status_path, history)


def monitor_slurm_run(
    *,
    run_id: str,
    root_dir: Path,
    job_id: str,
    idea_ids: Sequence[str],
    runner: SlurmRunner = _run_slurm_command,
    poll_interval_seconds: float = 10.0,
) -> dict[str, JSONValue]:
    run_dir = root_dir / "pipeline" / "runs" / run_id
    status_path = _run_status_path(run_dir)
    while True:
        slurm_states = _slurm_states_for_job(job_id=job_id, root_dir=root_dir, runner=runner)
        payload = _aggregate_run_status(
            run_id=run_id, run_dir=run_dir, job_id=job_id, idea_ids=idea_ids, slurm_states=slurm_states
        )
        if not slurm_states:
            terminal_state = _slurm_terminal_state(job_id=job_id, root_dir=root_dir, runner=runner)
            if terminal_state is not None:
                payload["slurm_terminal_state"] = terminal_state
        _append_run_status_history(status_path=status_path, payload=payload)
        if not slurm_states:
            if str(payload.get("slurm_terminal_state") or "") in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY"}:
                logs_dir = _slurm_logs_dir(root_dir=root_dir, run_id=run_id)
                raise RuntimeError(
                    f"Slurm job {job_id} finished with state {payload['slurm_terminal_state']}. "
                    f"See logs under {logs_dir}."
                )
            return payload
        time.sleep(poll_interval_seconds)


def run_slurm_task(*, run_id: str, root_dir: Path | None = None) -> dict[str, object]:
    resolved_root = root_dir or Path.cwd()
    raw_array_index = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if not raw_array_index:
        raise RuntimeError("SLURM_ARRAY_TASK_ID is required for Slurm task execution.")
    array_index = int(raw_array_index)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    run_dir = resolved_root / "pipeline" / "runs" / run_id
    raw_meta = json.loads(_run_meta_path(run_dir).read_text(encoding="utf-8"))
    if not isinstance(raw_meta, dict):
        raise RuntimeError(f"Run metadata at {_run_meta_path(run_dir)} was not a JSON object.")
    idea_count = raw_meta.get("idea_count")
    if not isinstance(idea_count, int):
        raise RuntimeError(f"Run metadata for {run_id} did not define a valid idea_count.")

    raw_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    config = PipelineRunConfig.model_validate(raw_config)
    if config.idea_json is not None:
        idea_path = _resolve_repo_path(resolved_root, config.idea_json)
        ideas = load_idea_specs(idea_path)
    else:
        official_description_paths = [
            _resolve_repo_path(resolved_root, path_value) for path_value in config.official_description_paths
        ]
        ideas = load_official_description_pair_specs(
            official_description_paths, root=resolved_root, num_levels=config.official_description_num_levels
        )
    if len(ideas) != idea_count:
        raise RuntimeError(f"Idea count mismatch for run {run_id}: metadata has {idea_count}, file has {len(ideas)}.")
    if array_index < 0 or array_index >= len(ideas):
        raise RuntimeError(f"Array index {array_index} is out of range for run {run_id} with {len(ideas)} ideas.")
    idea = ideas[array_index]

    specs_dir = run_dir / "specs"
    prompts_dir = run_dir / "prompts"
    workers_dir = run_dir / "workers"
    specs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    workers_dir.mkdir(parents=True, exist_ok=True)
    worker_root = workers_dir / idea.idea_id
    worker_root.mkdir(parents=True, exist_ok=True)

    template = load_spec_generation_prompt(
        instructions_path=_resolve_repo_path(resolved_root, config.spec_instructions_path)
    )
    implementation_prompt_path = _resolve_repo_path(resolved_root, config.implementation_prompt_path)

    output, worker_package = run_worker(
        api_key=api_key,
        config=config,
        run_id=run_id,
        idea=idea,
        root_dir=resolved_root,
        specs_dir=specs_dir,
        prompts_dir=prompts_dir,
        workers_dir=workers_dir,
        status_log_path=None,
        template=template,
        spec_provider=generate_structured_response,
        implementation_prompt_path=implementation_prompt_path,
    )
    return {
        "run_id": run_id,
        "array_index": array_index,
        "idea_id": idea.idea_id,
        "output": output,
        "worker_package": worker_package,
        "slurm_log_path": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute one pipeline Slurm task.")
    parser.add_argument("--run-id", required=True, help="Pipeline v2 run id.")
    args = parser.parse_args()
    run_slurm_task(run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
