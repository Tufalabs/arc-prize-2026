from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

from .commons import (
    JSONValue,
    format_duration,
    print_pipeline_log,
    read_json,
    utc_now,
    worker_status_path,
    write_text,
    write_worker_status,
)
from .config import PipelineRunConfig
from .idea_loading import IdeaSpec
from .implementation_stage import ArtifactRunner, ClaudeRunner, CodexRunner, prepare_and_launch_worker_package

SpecProvider = Callable[..., str]


def _build_spec_prompt(*, template: str, idea: IdeaSpec) -> str:
    if idea.source_official_descriptions:
        source_sections: list[str] = []
        for idx, source in enumerate(idea.source_official_descriptions, start=1):
            source_sections.append(
                "\n".join(
                    [f"## Official description {idx}: {source['name']}", f"Path: {source['path']}", "", source["text"]]
                )
            )

        provenance_lines = [
            "Generate a new game spec by recombining the official descriptions below.",
            "The generated spec must include a short provenance section naming the source official descriptions.",
            "The implementation instructions in the spec must require metadata.json to include:",
            '  "source_official_descriptions": [',
        ]
        for source in idea.source_official_descriptions:
            provenance = {"name": source["name"], "path": source["path"]}
            provenance_lines.append(f"    {json.dumps(provenance, ensure_ascii=True)},")
        provenance_lines.extend(
            ["  ]", "Do not make a clone of either source game. Reuse only high-level mechanic inspiration."]
        )
        return (
            f"{template.rstrip()}\n\n"
            "Official-description combination request:\n"
            + "\n".join(provenance_lines)
            + "\n\nSource official descriptions:\n"
            + "\n\n".join(source_sections)
            + "\n"
        )

    idea_json = json.dumps(idea.model_dump(mode="json"), indent=2, ensure_ascii=True)
    return f"{template.rstrip()}\n\nGame idea JSON:\n{idea_json}\n"


def run_worker(
    *,
    api_key: str,
    config: PipelineRunConfig,
    run_id: str,
    idea: IdeaSpec,
    root_dir: Path,
    specs_dir: Path,
    prompts_dir: Path,
    workers_dir: Path,
    status_log_path: Path | None,
    template: str,
    spec_provider: SpecProvider,
    codex_bin: str = "codex",
    codex_runner: CodexRunner | None = None,
    claude_bin: str = "claude",
    claude_runner: ClaudeRunner | None = None,
    artifact_runner: ArtifactRunner | None = None,
    implementation_prompt_path: Path,
) -> tuple[dict[str, str], dict[str, JSONValue]]:
    spec_path = specs_dir / f"{idea.idea_id}.md"
    prompt_path = prompts_dir / f"{idea.idea_id}.prompt.txt"
    status_path = worker_status_path(workers_dir=workers_dir, idea_id=idea.idea_id)
    spec_started_at = utc_now()
    spec_started_monotonic = perf_counter()
    write_worker_status(
        status_path=status_path,
        run_id=run_id,
        idea_id=idea.idea_id,
        status="now writing specs",
        status_log_path=status_log_path,
        spec_started_at=spec_started_at,
    )
    try:
        prompt = _build_spec_prompt(template=template, idea=idea)
        resume_response_id: str | None = None
        if config.spec_background and status_path.exists():
            existing_status = read_json(status_path)
            if isinstance(existing_status, dict):
                raw_response_id = existing_status.get("openai_response_id")
                if isinstance(raw_response_id, str) and raw_response_id.strip():
                    resume_response_id = raw_response_id.strip()

        def _record_response_id(response_id: str) -> None:
            write_worker_status(
                status_path=status_path,
                run_id=run_id,
                idea_id=idea.idea_id,
                status="waiting for background spec",
                status_log_path=status_log_path,
                spec_started_at=spec_started_at,
                extra={"openai_response_id": response_id},
            )

        spec_kwargs: dict[str, object] = {
            "api_key": api_key,
            "model": config.model,
            "prompt": prompt,
            "reasoning_effort": config.reasoning_effort,
        }
        if config.spec_background:
            spec_kwargs.update(
                {
                    "background": True,
                    "resume_response_id": resume_response_id,
                    "response_id_callback": _record_response_id,
                }
            )
        spec = spec_provider(**spec_kwargs)
        if idea.source_official_descriptions:
            source_payload = json.dumps(
                [{"name": source["name"], "path": source["path"]} for source in idea.source_official_descriptions],
                indent=2,
                ensure_ascii=True,
            )
            spec = (
                spec.rstrip()
                + "\n\n## Source official descriptions\n\n"
                + "This spec was generated from the following official game descriptions:\n\n"
                + f"```json\n{source_payload}\n```\n"
            )
        write_text(spec_path, spec.rstrip() + "\n")
        write_text(prompt_path, prompt)
        spec_finished_at = utc_now()
        spec_duration_seconds = perf_counter() - spec_started_monotonic
        write_worker_status(
            status_path=status_path,
            run_id=run_id,
            idea_id=idea.idea_id,
            status="done",
            status_log_path=status_log_path,
            spec_started_at=spec_started_at,
            spec_finished_at=spec_finished_at,
            spec_duration_seconds=spec_duration_seconds,
        )
        print_pipeline_log(
            f"[pipeline] Spec generation done for {idea.idea_id} in {format_duration(spec_duration_seconds)}"
        )
    except Exception as exc:
        write_worker_status(
            status_path=status_path,
            run_id=run_id,
            idea_id=idea.idea_id,
            status="failed",
            status_log_path=status_log_path,
            spec_started_at=spec_started_at,
            spec_finished_at=utc_now(),
            spec_duration_seconds=perf_counter() - spec_started_monotonic,
            error=str(exc),
        )
        raise

    worker_package = prepare_and_launch_worker_package(
        run_id=run_id,
        idea_id=idea.idea_id,
        idea_title=idea.title,
        idea_description=idea.description,
        spec_path=spec_path,
        worker_root=workers_dir / idea.idea_id,
        prompts_dir=prompts_dir,
        status_log_path=status_log_path,
        source_root=root_dir,
        implementation_prompt_template_path=implementation_prompt_path,
        implementation_backend=config.implementation_backend,
        codex_bin=codex_bin,
        codex_runner=codex_runner,
        codex_reasoning_effort=config.codex_reasoning_effort,
        claude_bin=claude_bin,
        claude_runner=claude_runner,
        artifact_runner=artifact_runner,
    )
    return (
        {
            "idea_id": idea.idea_id,
            "spec_path": str(spec_path.relative_to(root_dir)),
            "prompt_path": str(prompt_path.relative_to(root_dir)),
            "worker_status_path": str(status_path.relative_to(root_dir)),
        },
        worker_package,
    )
