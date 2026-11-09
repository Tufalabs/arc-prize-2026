from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

from tqdm import tqdm  # type: ignore[import-untyped]

from .commons import JSONValue, utc_now, write_json, write_text


@dataclass(frozen=True)
class BatchJob:
    index: int
    variables: dict[str, str]


def _path_safe(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-") or "job"


class _FormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        available = ", ".join(sorted(self)) or "(none)"
        raise KeyError(f"Missing prompt variable {key!r}. Available variables: {available}.")


def _load_jobs(path: Path) -> list[BatchJob]:
    suffix = path.suffix.lower()
    if suffix != ".csv":
        raise ValueError(f"Unsupported jobs file format for {path}. Use .csv.")

    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=",")
        rows = list(reader)

    jobs: list[BatchJob] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ValueError(f"Job {index} in {path} is not an object.")
        variables = {str(key): str(value) for key, value in row.items() if value is not None}
        if "game" not in variables or "folder" not in variables:
            raise ValueError(f"Job {index} in {path} must define at least 'game' and 'folder'.")
        jobs.append(BatchJob(index=index, variables=variables))
    if not jobs:
        raise ValueError(f"No jobs found in {path}.")
    return jobs


def _build_prompt(template: str, variables: Mapping[str, str]) -> str:
    return template.format_map(_FormatDict(dict(variables)))


def _run_codex_command(
    cmd: Sequence[str], cwd: Path, log_path: Path, output_path: Path
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(cmd), cwd=str(cwd), check=False, text=True, capture_output=True)
    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    log_path.write_text(stdout_text + stderr_text, encoding="utf-8")
    output_path.write_text(stdout_text, encoding="utf-8")
    return result


def _run_one_job(
    *,
    job: BatchJob,
    prompt_template: str,
    output_dir: Path,
    base_dir: Path,
    codex_bin: str,
    reasoning_effort: str | None,
    codex_runner: Callable[[Sequence[str], Path, Path, Path], subprocess.CompletedProcess[str]],
) -> dict[str, JSONValue]:
    folder_value = job.variables["folder"]
    target_dir = Path(folder_value)
    if not target_dir.is_absolute():
        target_dir = (base_dir / target_dir).resolve()
    prompt_text = _build_prompt(prompt_template, job.variables)
    run_slug = _path_safe(job.variables.get("game", f"job-{job.index}"))
    run_dir = output_dir / f"{job.index:03d}-{run_slug}"
    prompt_path = run_dir / "prompt.txt"
    log_path = run_dir / "stdout.log"
    message_path = run_dir / "last_message.txt"
    metadata_path = run_dir / "run.json"

    write_text(prompt_path, prompt_text)
    started_at = utc_now()
    started_monotonic = perf_counter()
    cmd = [
        codex_bin,
        "exec",
        "--cd",
        str(target_dir),
        "--dangerously-bypass-approvals-and-sandbox",
        *(["-c", f'model_reasoning_effort="{reasoning_effort}"'] if reasoning_effort else []),
        "-o",
        str(message_path),
        prompt_text,
    ]
    result = codex_runner(cmd, target_dir, log_path, message_path)
    finished_at = utc_now()
    duration_seconds = perf_counter() - started_monotonic
    payload: dict[str, JSONValue] = {
        "job_index": job.index,
        "game": job.variables.get("game", ""),
        "folder": str(target_dir),
        "variables": dict(job.variables),
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": duration_seconds,
        "exit_code": result.returncode,
        "status": "completed" if result.returncode == 0 else "failed",
        "prompt_path": str(prompt_path),
        "log_path": str(log_path),
        "output_path": str(message_path),
        "command": list(cmd),
    }
    write_json(metadata_path, payload)
    return payload


def run_batch(
    *,
    jobs: Iterable[BatchJob],
    prompt_template: str,
    output_dir: Path,
    base_dir: Path,
    codex_bin: str = "codex",
    reasoning_effort: str | None = None,
    max_workers: int = 1,
    continue_on_error: bool = False,
    codex_runner: Callable[[Sequence[str], Path, Path, Path], subprocess.CompletedProcess[str]] | None = None,
) -> list[dict[str, JSONValue]]:
    if max_workers < 1:
        raise ValueError("max_workers must be at least 1.")
    resolved_runner = codex_runner or _run_codex_command
    output_dir.mkdir(parents=True, exist_ok=True)
    job_list = list(jobs)
    results_by_index: dict[int, dict[str, JSONValue]] = {}
    failed_payloads: list[dict[str, JSONValue]] = []
    progress = tqdm(total=len(job_list), desc="Codex jobs", unit="game", disable=not job_list)

    def _handle_result(payload: dict[str, JSONValue]) -> None:
        job_index = cast(int, payload["job_index"])
        results_by_index[job_index] = payload
        progress.update(1)
        if payload["exit_code"] != 0:
            failed_payloads.append(payload)

    try:
        if max_workers == 1:
            for job in job_list:
                payload = _run_one_job(
                    job=job,
                    prompt_template=prompt_template,
                    output_dir=output_dir,
                    base_dir=base_dir,
                    codex_bin=codex_bin,
                    reasoning_effort=reasoning_effort,
                    codex_runner=resolved_runner,
                )
                _handle_result(payload)
                if failed_payloads and not continue_on_error:
                    failed_payload = failed_payloads[0]
                    raise RuntimeError(
                        f"Codex failed for game {failed_payload.get('game', '')!r} "
                        f"with exit code {failed_payload['exit_code']}."
                    )
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                future_to_job = {
                    pool.submit(
                        _run_one_job,
                        job=job,
                        prompt_template=prompt_template,
                        output_dir=output_dir,
                        base_dir=base_dir,
                        codex_bin=codex_bin,
                        reasoning_effort=reasoning_effort,
                        codex_runner=resolved_runner,
                    ): job
                    for job in job_list
                }
                for future in as_completed(future_to_job):
                    payload = future.result()
                    _handle_result(payload)
    finally:
        progress.close()

    results = [results_by_index[job.index] for job in sorted(job_list, key=lambda item: item.index)]
    if failed_payloads and not continue_on_error:
        failed_payload = sorted(failed_payloads, key=lambda item: cast(int, item["job_index"]))[0]
        raise RuntimeError(
            f"Codex failed for game {failed_payload.get('game', '')!r} with exit code {failed_payload['exit_code']}."
        )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run codex exec repeatedly with a parameterized prompt.")
    parser.add_argument("--jobs", required=True, help="Path to a .csv jobs file.")
    parser.add_argument("--prompt-file", default=None, help="Path to a prompt template file.")
    parser.add_argument(
        "--prompt", default=None, help="Inline prompt template. Use placeholders like {game} and {folder}."
    )
    parser.add_argument("--output-dir", default=None, help="Directory where per-job logs and outputs are written.")
    parser.add_argument(
        "--base-dir", default=".", help="Base directory used to resolve relative folder values from the jobs file."
    )
    parser.add_argument("--codex-bin", default="codex", help="Codex executable name or full path.")
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        choices=["low", "medium", "high", "xhigh"],
        help="Optional Codex model_reasoning_effort override for each job.",
    )
    parser.add_argument("--max-workers", type=int, default=1, help="Number of Codex jobs to run in parallel.")
    parser.add_argument(
        "--continue-on-error", action="store_true", help="Keep processing later jobs after a codex failure."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.prompt) == bool(args.prompt_file):
        raise RuntimeError("Pass exactly one of --prompt or --prompt-file.")

    jobs_path = Path(args.jobs).resolve()
    jobs = _load_jobs(jobs_path)
    prompt_template = (
        Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file is not None else str(args.prompt)
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (Path("logs") / "codex-batch" / _path_safe(utc_now())).resolve()
    )
    results = run_batch(
        jobs=jobs,
        prompt_template=prompt_template,
        output_dir=output_dir,
        base_dir=Path(args.base_dir).resolve(),
        codex_bin=str(args.codex_bin),
        reasoning_effort=args.reasoning_effort,
        max_workers=int(args.max_workers),
        continue_on_error=bool(args.continue_on_error),
    )
    write_json(output_dir / "summary.json", {"jobs": results, "job_count": len(results)})
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
