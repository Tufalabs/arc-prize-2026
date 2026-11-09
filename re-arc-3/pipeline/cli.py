from __future__ import annotations

import argparse
import os
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

from .commons import JSONValue, repo_root, write_json, write_text
from .config import PipelineRunConfig, load_run_config
from .game_catalog import write_game_catalog
from .idea_loading import IdeaSpec, load_idea_specs, load_official_description_pair_specs
from .implementation_stage import ArtifactRunner, ClaudeRunner, CodexRunner, prepare_and_launch_worker_package
from .prompt_building import load_spec_generation_prompt
from .review import review_run
from .slurm import SlurmRunner, _run_slurm_command, cancel_slurm_job, monitor_slurm_run, submit_slurm_array
from .spec_generation import generate_structured_response
from .worker import _build_spec_prompt, run_worker

SpecProvider = Callable[..., str]
DEFAULT_MAX_WORKERS = 8


@dataclass(frozen=True)
class PreparedRunContext:
    root_dir: Path
    run_dir: Path
    resolved_status_log_path: Path
    idea_path: Path | None
    official_description_paths: list[Path]
    implementation_prompt_path: Path
    spec_instructions_path: Path
    specs_dir: Path
    prompts_dir: Path
    workers_dir: Path
    ideas: list[IdeaSpec]
    template: str


def _timestamp_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _path_safe(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-") or "run"


def _compose_run_id(*, timestamp_run_id: str, run_name: str | None) -> str:
    suffix = str(run_name or "").strip()
    if not suffix:
        return timestamp_run_id
    return f"{timestamp_run_id}-{_path_safe(suffix)}"


def _resolve_repo_path(root_dir: Path, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    resolved = root_dir / path
    if resolved.exists():
        return resolved

    # Keep older config paths working after prompt files moved from
    # `pipeline/prompts_and_schemas/` to `pipeline/prompts/`.
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate or review pipeline runs.")
    parser.add_argument(
        "command", nargs="?", choices=("run", "review", "catalog-games", "implement-specs"), default="run"
    )
    parser.add_argument("--config", default=None, help="Path to a YAML run config file.")
    parser.add_argument("--run-id", default=None, help="Pipeline v2 run id.")
    parser.add_argument("--run-name", default=None, help="Optional suffix appended to the timestamp-based run id.")
    parser.add_argument("--host", default=None, help="Replay UI host for `review`.")
    parser.add_argument("--port", type=int, default=None, help="Replay UI port for `review`.")
    parser.add_argument("--output", default=None, help="Optional output path for local catalog commands.")
    parser.add_argument("--spec-dir", default=None, help="Directory of Markdown specs for `implement-specs`.")
    parser.add_argument(
        "--implementation-prompt",
        default="pipeline/prompts/codex_check_visual_prompt.md",
        help="Prompt template for `implement-specs`.",
    )
    parser.add_argument(
        "--implementation-backend",
        choices=("codex", "claude"),
        default="codex",
        help="Implementation backend for `implement-specs`.",
    )
    parser.add_argument(
        "--codex-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        default=None,
        help="Optional Codex reasoning effort for `implement-specs`.",
    )
    return parser


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "run_meta.json"


def _is_human_model(model: str) -> bool:
    return model.strip().lower() == "human"


def _prepare_run_context(
    *, config: PipelineRunConfig, run_id: str, root: Path | None, runs_root: Path | None, status_log_path: Path | None
) -> PreparedRunContext:
    root_dir = root or repo_root()
    implementation_prompt_path = _resolve_repo_path(root_dir, config.implementation_prompt_path)
    spec_instructions_path = _resolve_repo_path(root_dir, config.spec_instructions_path)
    idea_path: Path | None = None
    official_description_paths: list[Path] = []
    if config.idea_json is not None:
        idea_path = _resolve_repo_path(root_dir, config.idea_json)
        ideas = load_idea_specs(idea_path)
    else:
        official_description_paths = [
            _resolve_repo_path(root_dir, path_value) for path_value in config.official_description_paths
        ]
        ideas = load_official_description_pair_specs(
            official_description_paths, root=root_dir, num_levels=config.official_description_num_levels
        )
    if not ideas:
        raise ValueError("Spec source did not produce any ideas.")
    template = load_spec_generation_prompt(instructions_path=spec_instructions_path)
    resolved_runs_root = runs_root or (root_dir / "pipeline" / "runs")
    run_dir = resolved_runs_root / run_id
    resolved_status_log_path = status_log_path or (run_dir / "status_log.jsonl")
    specs_dir = run_dir / "specs"
    prompts_dir = run_dir / "prompts"
    workers_dir = run_dir / "workers"
    specs_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    workers_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        _run_meta_path(run_dir),
        {
            "run_id": run_id,
            "idea_count": len(ideas),
            "source_mode": "idea_json" if idea_path is not None else "official_description_pairs",
        },
    )
    write_json(run_dir / "run_config.json", config.model_dump(mode="json"))
    return PreparedRunContext(
        root_dir=root_dir,
        run_dir=run_dir,
        resolved_status_log_path=resolved_status_log_path,
        idea_path=idea_path,
        official_description_paths=official_description_paths,
        implementation_prompt_path=implementation_prompt_path,
        spec_instructions_path=spec_instructions_path,
        specs_dir=specs_dir,
        prompts_dir=prompts_dir,
        workers_dir=workers_dir,
        ideas=ideas,
        template=template,
    )


def _run_pipeline_slurm(
    *, config: PipelineRunConfig, run_id: str, context: PreparedRunContext, slurm_runner: SlurmRunner | None
) -> dict[str, JSONValue]:
    resolved_slurm_runner = slurm_runner or _run_slurm_command
    slurm_payload = submit_slurm_array(
        root_dir=context.root_dir,
        run_id=run_id,
        task_count=len(context.ideas),
        partition=config.slurm_partition,
        time_limit=config.slurm_time,
        mem=config.slurm_mem,
        ntasks=config.slurm_ntasks,
        cpus_per_task=config.slurm_cpus_per_task,
        runner=resolved_slurm_runner,
    )
    run_payload: dict[str, JSONValue] = {
        "run_id": run_id,
        "idea_json": str(context.idea_path) if context.idea_path is not None else None,
        "official_description_paths": [str(path) for path in context.official_description_paths],
        "implementation_prompt": str(context.implementation_prompt_path),
        "spec_instructions_path": str(context.spec_instructions_path),
        "idea_count": len(context.ideas),
        "outputs": [],
        "worker_packages": [],
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "workers_dir": str(context.workers_dir.relative_to(context.root_dir)),
        "config": config.model_dump(mode="json"),
        "execution_backend": "slurm",
        "run_meta_path": str(_run_meta_path(context.run_dir).relative_to(context.root_dir)),
    }
    run_payload.update(slurm_payload)
    completed = False
    try:
        run_status = monitor_slurm_run(
            run_id=run_id,
            root_dir=context.root_dir,
            job_id=str(slurm_payload["slurm_job_id"]),
            idea_ids=[idea.idea_id for idea in context.ideas],
            runner=resolved_slurm_runner,
            poll_interval_seconds=0.1 if slurm_runner is not None else 10.0,
        )
        run_payload["run_status_path"] = str((context.run_dir / "run_status.json").relative_to(context.root_dir))
        run_payload["run_status"] = run_status
        write_json(context.run_dir / "run.json", run_payload)
        completed = True
        return run_payload
    finally:
        if not completed:
            cancel_slurm_job(
                job_id=str(slurm_payload["slurm_job_id"]), root_dir=context.root_dir, runner=resolved_slurm_runner
            )


def _run_pipeline_human_specs(
    *, config: PipelineRunConfig, run_id: str, context: PreparedRunContext
) -> dict[str, JSONValue]:
    outputs: list[dict[str, JSONValue]] = []
    for idea in context.ideas:
        prompt = _build_spec_prompt(template=context.template, idea=idea)
        prompt_path = context.prompts_dir / f"{idea.idea_id}.prompt.txt"
        write_text(prompt_path, prompt)
        outputs.append(
            {
                "idea_id": idea.idea_id,
                "prompt_path": str(prompt_path.relative_to(context.root_dir)),
                "suggested_spec_path": str((context.specs_dir / f"{idea.idea_id}.md").relative_to(context.root_dir)),
            }
        )

    run_payload: dict[str, JSONValue] = {
        "run_id": run_id,
        "idea_json": str(context.idea_path) if context.idea_path is not None else None,
        "official_description_paths": [str(path) for path in context.official_description_paths],
        "implementation_prompt": str(context.implementation_prompt_path),
        "spec_instructions_path": str(context.spec_instructions_path),
        "idea_count": len(context.ideas),
        "outputs": outputs,
        "worker_packages": [],
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "specs_dir": str(context.specs_dir.relative_to(context.root_dir)),
        "prompts_dir": str(context.prompts_dir.relative_to(context.root_dir)),
        "workers_dir": str(context.workers_dir.relative_to(context.root_dir)),
        "config": config.model_dump(mode="json"),
        "execution_backend": "human",
        "source_mode": "human_spec_prompts",
        "run_meta_path": str(_run_meta_path(context.run_dir).relative_to(context.root_dir)),
        "run_status": "waiting_for_human_specs",
        "next_command": (
            f"python -m pipeline.cli implement-specs --spec-dir "
            f"{context.specs_dir.relative_to(context.root_dir)} --run-name <name>"
        ),
    }
    write_json(context.run_dir / "run.json", run_payload)
    return run_payload


def _run_pipeline_local(
    *,
    api_key: str,
    config: PipelineRunConfig,
    run_id: str,
    context: PreparedRunContext,
    spec_provider: SpecProvider,
    codex_bin: str,
    codex_runner: CodexRunner | None,
    claude_bin: str,
    claude_runner: ClaudeRunner | None,
    artifact_runner: ArtifactRunner | None,
) -> dict[str, JSONValue]:
    max_workers = max(1, min(len(context.ideas), DEFAULT_MAX_WORKERS))
    outputs_by_index: dict[int, dict[str, str]] = {}
    worker_packages_by_index: dict[int, dict[str, JSONValue]] = {}

    def _run_idea(index: int, idea: IdeaSpec) -> tuple[int, dict[str, str], dict[str, JSONValue]]:
        output, worker_package = run_worker(
            api_key=api_key,
            config=config,
            run_id=run_id,
            idea=idea,
            root_dir=context.root_dir,
            specs_dir=context.specs_dir,
            prompts_dir=context.prompts_dir,
            workers_dir=context.workers_dir,
            status_log_path=context.resolved_status_log_path,
            template=context.template,
            spec_provider=spec_provider,
            codex_bin=codex_bin,
            codex_runner=codex_runner,
            claude_bin=claude_bin,
            claude_runner=claude_runner,
            artifact_runner=artifact_runner,
            implementation_prompt_path=context.implementation_prompt_path,
        )
        return index, output, worker_package

    def _failed_idea_result(index: int, exc: Exception) -> tuple[dict[str, str], dict[str, JSONValue]]:
        idea = context.ideas[index - 1]
        worker_root = context.workers_dir / idea.idea_id
        spec_path = context.specs_dir / f"{idea.idea_id}.md"
        prompt_path = context.prompts_dir / f"{idea.idea_id}.prompt.txt"
        status_path = worker_root / "status.json"
        error = str(exc)
        output = {
            "idea_id": idea.idea_id,
            "spec_path": str(spec_path.relative_to(context.root_dir)),
            "prompt_path": str(prompt_path.relative_to(context.root_dir)),
            "worker_status_path": str(status_path.relative_to(context.root_dir)),
            "status": "failed",
            "error": error,
        }
        worker_package: dict[str, JSONValue] = {
            "idea_id": idea.idea_id,
            "idea_title": idea.title,
            "idea_description": idea.description,
            "worker_root": str(worker_root),
            "worker_spec_path": str(spec_path),
            "worker_prompt_path": str(prompt_path),
            "worker_status_path": str(status_path),
            "status": "failed",
            "error": error,
        }
        return output, worker_package

    future_to_index = {}
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_index = {
            pool.submit(_run_idea, index, idea): index for index, idea in enumerate(context.ideas, start=1)
        }
        for future in as_completed(future_to_index):
            completed_index = future_to_index[future]
            try:
                completed_index, output, worker_package = future.result()
            except Exception as exc:
                output, worker_package = _failed_idea_result(completed_index, exc)
            outputs_by_index[completed_index] = output
            worker_packages_by_index[completed_index] = worker_package
    except KeyboardInterrupt:
        for future in future_to_index:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        partial_payload: dict[str, JSONValue] = {
            "run_id": run_id,
            "idea_json": str(context.idea_path) if context.idea_path is not None else None,
            "official_description_paths": [str(path) for path in context.official_description_paths],
            "implementation_prompt": str(context.implementation_prompt_path),
            "spec_instructions_path": str(context.spec_instructions_path),
            "idea_count": len(context.ideas),
            "outputs": [outputs_by_index[index] for index in sorted(outputs_by_index)],
            "worker_packages": [worker_packages_by_index[index] for index in sorted(worker_packages_by_index)],
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "workers_dir": str(context.workers_dir.relative_to(context.root_dir)),
            "config": config.model_dump(mode="json"),
            "execution_backend": "local",
            "interrupted": True,
            "run_meta_path": str(_run_meta_path(context.run_dir).relative_to(context.root_dir)),
        }
        write_json(context.run_dir / "run.json", partial_payload)
        raise
    else:
        pool.shutdown(wait=True)

    outputs = [outputs_by_index[index] for index in range(1, len(context.ideas) + 1)]
    worker_packages = [worker_packages_by_index[index] for index in range(1, len(context.ideas) + 1)]
    failed_count = sum(1 for package in worker_packages if str(package.get("status") or "") == "failed")

    run_payload: dict[str, JSONValue] = {
        "run_id": run_id,
        "idea_json": str(context.idea_path) if context.idea_path is not None else None,
        "official_description_paths": [str(path) for path in context.official_description_paths],
        "implementation_prompt": str(context.implementation_prompt_path),
        "spec_instructions_path": str(context.spec_instructions_path),
        "idea_count": len(context.ideas),
        "outputs": outputs,
        "worker_packages": worker_packages,
        "model": config.model,
        "reasoning_effort": config.reasoning_effort,
        "workers_dir": str(context.workers_dir.relative_to(context.root_dir)),
        "config": config.model_dump(mode="json"),
        "execution_backend": "local",
        "run_meta_path": str(_run_meta_path(context.run_dir).relative_to(context.root_dir)),
        "run_status": "completed_with_failures" if failed_count else "completed",
        "succeeded_count": len(context.ideas) - failed_count,
        "failed_count": failed_count,
    }
    write_json(context.run_dir / "run.json", run_payload)
    return run_payload


def _spec_id_from_path(spec_path: Path) -> str:
    return _path_safe(spec_path.stem.replace("_", "-")).lower()


def run_implementation_from_specs(
    *,
    run_id: str,
    spec_dir: Path,
    root: Path | None = None,
    runs_root: Path | None = None,
    implementation_prompt_path: Path | None = None,
    implementation_backend: Literal["codex", "claude"] = "codex",
    codex_reasoning_effort: str | None = None,
    codex_bin: str = "codex",
    codex_runner: CodexRunner | None = None,
    claude_bin: str = "claude",
    claude_runner: ClaudeRunner | None = None,
    artifact_runner: ArtifactRunner | None = None,
) -> dict[str, JSONValue]:
    root_dir = root or repo_root()
    resolved_spec_dir = spec_dir if spec_dir.is_absolute() else root_dir / spec_dir
    if not resolved_spec_dir.is_dir():
        raise RuntimeError(f"--spec-dir does not exist or is not a directory: {resolved_spec_dir}")
    spec_paths = sorted(path for path in resolved_spec_dir.glob("*.md") if path.is_file())
    if not spec_paths:
        raise RuntimeError(f"--spec-dir contains no Markdown specs: {resolved_spec_dir}")

    resolved_runs_root = runs_root or (root_dir / "pipeline" / "runs")
    run_dir = resolved_runs_root / run_id
    prompts_dir = run_dir / "prompts"
    workers_dir = run_dir / "workers"
    status_log_path = run_dir / "status_log.jsonl"
    resolved_implementation_prompt_path = implementation_prompt_path or (
        root_dir / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    )
    if not resolved_implementation_prompt_path.is_absolute():
        resolved_implementation_prompt_path = root_dir / resolved_implementation_prompt_path

    prompts_dir.mkdir(parents=True, exist_ok=True)
    workers_dir.mkdir(parents=True, exist_ok=True)
    write_json(_run_meta_path(run_dir), {"run_id": run_id, "idea_count": len(spec_paths), "source_mode": "spec_dir"})

    outputs_by_index: dict[int, dict[str, JSONValue]] = {}
    worker_packages_by_index: dict[int, dict[str, JSONValue]] = {}

    def _run_spec(index: int, spec_path: Path) -> tuple[int, dict[str, JSONValue], dict[str, JSONValue]]:
        idea_id = _spec_id_from_path(spec_path)
        worker_package = prepare_and_launch_worker_package(
            run_id=run_id,
            idea_id=idea_id,
            idea_title=spec_path.stem.replace("_", " ").replace("-", " ").title(),
            idea_description=f"Implementation from existing spec {spec_path.name}",
            spec_path=spec_path,
            worker_root=workers_dir / idea_id,
            prompts_dir=prompts_dir,
            status_log_path=status_log_path,
            source_root=root_dir,
            implementation_prompt_template_path=resolved_implementation_prompt_path,
            implementation_backend=implementation_backend,
            codex_bin=codex_bin,
            codex_runner=codex_runner,
            codex_reasoning_effort=codex_reasoning_effort,
            claude_bin=claude_bin,
            claude_runner=claude_runner,
            artifact_runner=artifact_runner,
        )
        output: dict[str, JSONValue] = {
            "idea_id": idea_id,
            "spec_path": str(spec_path.relative_to(root_dir)) if spec_path.is_relative_to(root_dir) else str(spec_path),
            "worker_status_path": str((workers_dir / idea_id / "status.json").relative_to(root_dir)),
        }
        return index, output, worker_package

    def _failed_spec_result(spec_path: Path, exc: Exception) -> tuple[dict[str, JSONValue], dict[str, JSONValue]]:
        idea_id = _spec_id_from_path(spec_path)
        worker_root = workers_dir / idea_id
        error = str(exc)
        output: dict[str, JSONValue] = {
            "idea_id": idea_id,
            "spec_path": str(spec_path.relative_to(root_dir)) if spec_path.is_relative_to(root_dir) else str(spec_path),
            "worker_status_path": str((worker_root / "status.json").relative_to(root_dir)),
            "status": "failed",
            "error": error,
        }
        worker_package: dict[str, JSONValue] = {
            "idea_id": idea_id,
            "idea_title": spec_path.stem.replace("_", " ").replace("-", " ").title(),
            "idea_description": f"Implementation from existing spec {spec_path.name}",
            "worker_root": str(worker_root),
            "worker_spec_path": str(spec_path),
            "worker_status_path": str(worker_root / "status.json"),
            "status": "failed",
            "error": error,
        }
        return output, worker_package

    max_workers = max(1, min(len(spec_paths), DEFAULT_MAX_WORKERS))
    future_to_index = {}
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_index = {
            pool.submit(_run_spec, index, spec_path): index for index, spec_path in enumerate(spec_paths, start=1)
        }
        for future in as_completed(future_to_index):
            completed_index = future_to_index[future]
            spec_path = spec_paths[completed_index - 1]
            try:
                completed_index, output, worker_package = future.result()
            except Exception as exc:
                output, worker_package = _failed_spec_result(spec_path, exc)
            outputs_by_index[completed_index] = output
            worker_packages_by_index[completed_index] = worker_package
    except KeyboardInterrupt:
        for future in future_to_index:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        partial_payload: dict[str, JSONValue] = {
            "run_id": run_id,
            "spec_dir": str(resolved_spec_dir),
            "implementation_prompt": str(resolved_implementation_prompt_path),
            "idea_count": len(spec_paths),
            "outputs": [outputs_by_index[index] for index in sorted(outputs_by_index)],
            "worker_packages": [worker_packages_by_index[index] for index in sorted(worker_packages_by_index)],
            "workers_dir": str(workers_dir.relative_to(root_dir)),
            "execution_backend": "local",
            "source_mode": "spec_dir",
            "interrupted": True,
            "run_meta_path": str(_run_meta_path(run_dir).relative_to(root_dir)),
        }
        write_json(run_dir / "run.json", partial_payload)
        raise
    else:
        pool.shutdown(wait=True)

    outputs = [outputs_by_index[index] for index in range(1, len(spec_paths) + 1)]
    worker_packages = [worker_packages_by_index[index] for index in range(1, len(spec_paths) + 1)]
    failed_count = sum(1 for package in worker_packages if str(package.get("status") or "") == "failed")
    run_payload: dict[str, JSONValue] = {
        "run_id": run_id,
        "spec_dir": str(resolved_spec_dir),
        "implementation_prompt": str(resolved_implementation_prompt_path),
        "idea_count": len(spec_paths),
        "outputs": outputs,
        "worker_packages": worker_packages,
        "workers_dir": str(workers_dir.relative_to(root_dir)),
        "execution_backend": "local",
        "source_mode": "spec_dir",
        "implementation_backend": implementation_backend,
        "run_meta_path": str(_run_meta_path(run_dir).relative_to(root_dir)),
        "run_status": "completed_with_failures" if failed_count else "completed",
        "succeeded_count": len(spec_paths) - failed_count,
        "failed_count": failed_count,
    }
    write_json(run_dir / "run.json", run_payload)
    return run_payload


def run_pipeline(
    *,
    api_key: str,
    config: PipelineRunConfig,
    run_id: str,
    root: Path | None = None,
    runs_root: Path | None = None,
    status_log_path: Path | None = None,
    spec_provider: SpecProvider = generate_structured_response,
    codex_bin: str = "codex",
    codex_runner: CodexRunner | None = None,
    claude_bin: str = "claude",
    claude_runner: ClaudeRunner | None = None,
    artifact_runner: ArtifactRunner | None = None,
    slurm_runner: SlurmRunner | None = None,
) -> dict[str, JSONValue]:
    context = _prepare_run_context(
        config=config, run_id=run_id, root=root, runs_root=runs_root, status_log_path=status_log_path
    )
    if _is_human_model(config.model):
        return _run_pipeline_human_specs(config=config, run_id=run_id, context=context)
    if config.executor == "slurm":
        return _run_pipeline_slurm(config=config, run_id=run_id, context=context, slurm_runner=slurm_runner)
    return _run_pipeline_local(
        api_key=api_key,
        config=config,
        run_id=run_id,
        context=context,
        spec_provider=spec_provider,
        codex_bin=codex_bin,
        codex_runner=codex_runner,
        claude_bin=claude_bin,
        claude_runner=claude_runner,
        artifact_runner=artifact_runner,
    )


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "catalog-games":
        root_dir = repo_root()
        output_path = _resolve_repo_path(root_dir, args.output or "pipeline/game_ideas/catalog.json")
        write_game_catalog(output_path=output_path, root=root_dir)
        print(output_path.relative_to(root_dir) if output_path.is_relative_to(root_dir) else output_path)
        return 0

    if args.command == "review":
        run_id = str(args.run_id or "").strip()
        if not run_id:
            raise RuntimeError("--run-id is required for `review`.")
        manifest = review_run(run_id=run_id, root=repo_root(), host=args.host, port=args.port)
        print(manifest["review_dir"])
        return 0

    if args.command == "implement-specs":
        if args.run_id and args.run_name:
            raise RuntimeError("--run-id and --run-name cannot be combined.")
        spec_dir = str(args.spec_dir or "").strip()
        if not spec_dir:
            raise RuntimeError("--spec-dir is required for `implement-specs`.")
        resolved_run_id = str(
            args.run_id or _compose_run_id(timestamp_run_id=_timestamp_run_id(), run_name=args.run_name)
        )
        implementation_prompt_path = _resolve_repo_path(repo_root(), str(args.implementation_prompt))
        run_payload = run_implementation_from_specs(
            run_id=resolved_run_id,
            spec_dir=_resolve_repo_path(repo_root(), spec_dir),
            root=repo_root(),
            implementation_prompt_path=implementation_prompt_path,
            implementation_backend=cast(Literal["codex", "claude"], str(args.implementation_backend)),
            codex_reasoning_effort=args.codex_reasoning_effort,
        )
        print(Path(str(run_payload["workers_dir"])).parent)
        return 0

    if not args.config:
        raise RuntimeError(f"--config is required for `{args.command}`.")
    if args.run_id and args.run_name:
        raise RuntimeError("--run-id and --run-name cannot be combined.")
    config_path = _resolve_repo_path(repo_root(), str(args.config))
    resolved_run_id = str(args.run_id or _compose_run_id(timestamp_run_id=_timestamp_run_id(), run_name=args.run_name))

    run_config = load_run_config(config_path)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key and not _is_human_model(run_config.model):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    run_payload = run_pipeline(
        api_key=api_key,
        config=run_config,
        run_id=resolved_run_id,
        root=repo_root(),
        spec_provider=generate_structured_response,
    )
    print(Path(str(run_payload["workers_dir"])).parent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
