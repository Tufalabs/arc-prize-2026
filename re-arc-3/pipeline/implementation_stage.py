from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter
from typing import Literal, cast

from .commons import (
    JSONValue,
    format_duration,
    print_pipeline_log,
    repo_root,
    utc_now,
    write_json,
    write_text,
    write_worker_status,
)
from .prompt_building import build_implementation_prompt, load_prompt_template, load_spec_text

WORKER_REPO_ITEMS: tuple[str, ...] = ("re_arc", "README.md", "config.env", "pyproject.toml", "uv.lock")
ImplementationBackend = Literal["codex", "claude"]
CodexRunner = Callable[[Sequence[str], Path, Path], subprocess.CompletedProcess[str]]
ClaudeRunner = Callable[[Sequence[str], Path, Path, Path], subprocess.CompletedProcess[str]]
ArtifactRunner = CodexRunner
DEFAULT_REFERENCE_GAME_SLUGS: tuple[str, ...] = ("ft09_close", "ls20_close")
PIPELINE_REFERENCE_GAMES_RELATIVE_DIR = Path("pipeline") / "reference_games"
WORKER_DOC_PATHS: tuple[Path, ...] = (Path("docs") / "arcengine_authoring_guide.md",)
WORKER_ARCENGINE_REFERENCE_PATH = Path("vendor") / "arcengine"
REFERENCE_SCREENSHOTS_RELATIVE_DIR = PIPELINE_REFERENCE_GAMES_RELATIVE_DIR / "screenshots"
DEFAULT_ARTIFACT_TIMEOUT_SECONDS = 900
WORKER_AGENTS_TEXT = """# Agent Notes For Pipeline Worker Repos

This is an isolated pipeline worker repo. Treat the current directory as the repo root.
Do not use `git rev-parse --show-toplevel`, because this worker may sit inside a parent checkout.

## Allowed Edit Scope

- `re_arc/environment_files/<slug>/<variant>/**`
- `re_arc/dsl/agents/<slug>.py`

Implement exactly one generated game and one matching DSL solver.

## Environment Score Contract

When editing files under `re_arc/environment_files/`:

1. Do not mutate `self._score` or `self._win_score` directly.
2. Use a game-specific variable for internal points if needed.
3. Use `self.next_level()` to represent solved-level progression.

Reason: `TransitionRewardEnv` derives reward from `levels_completed / win_levels`,
which are backed by `ARCBaseGame` score fields.

## Worker Validation

Do not run `make prepare`; worker repos do not include a Makefile.
Do not run `pytest -q tests/test_environment_score_contract.py`; that main-repo
test is intentionally not copied into worker repos.

Before finishing, run the worker-local validator from this repo root:

- `PY=.venv/bin/python; [ -x "$PY" ] || PY=python`
- `$PY -m pipeline.validate_generated_game --game <game_id> --files-only`

If validation fails, fix the implementation and rerun the command until it passes.
"""


def _run_command(
    cmd: Sequence[str],
    cwd: Path,
    log_path: Path,
    *,
    output_path: Path | None = None,
    timeout_seconds: float | None = None,
) -> subprocess.CompletedProcess[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            list(cmd), cwd=str(cwd), check=False, text=True, capture_output=True, timeout=timeout_seconds
        )
    except subprocess.TimeoutExpired as exc:
        timeout_message = (
            f"\nProcess timed out after {timeout_seconds:.0f} seconds.\n"
            if timeout_seconds is not None
            else "\nProcess timed out.\n"
        )
        stdout_text = (
            exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        )
        stderr_text = (
            exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        )
        log_path.write_text(stdout_text + stderr_text + timeout_message, encoding="utf-8")
        if output_path is not None:
            output_path.write_text(stdout_text, encoding="utf-8")
        raise

    stdout_text = result.stdout or ""
    stderr_text = result.stderr or ""
    log_path.write_text(stdout_text + stderr_text, encoding="utf-8")
    if output_path is not None:
        output_path.write_text(stdout_text, encoding="utf-8")
    return result


def _run_codex_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    return _run_command(cmd, cwd, log_path)


def _run_claude_command(
    cmd: Sequence[str], cwd: Path, log_path: Path, output_path: Path
) -> subprocess.CompletedProcess[str]:
    return _run_command(cmd, cwd, log_path, output_path=output_path)


def _run_artifact_command(cmd: Sequence[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    return _run_command(cmd, cwd, log_path, timeout_seconds=DEFAULT_ARTIFACT_TIMEOUT_SECONDS)


def _reference_game_slugs(source_root: Path) -> list[str]:
    environment_root = source_root / "re_arc" / "environment_files"
    agents_root = source_root / "re_arc" / "dsl" / "agents"
    environment_slugs = (
        {path.name for path in environment_root.iterdir() if path.is_dir()} if environment_root.exists() else set()
    )
    agent_slugs = (
        {path.stem for path in agents_root.glob("*.py") if path.is_file() and path.stem != "__init__"}
        if agents_root.exists()
        else set()
    )
    return sorted(environment_slugs & agent_slugs)


def _pipeline_reference_games_root(source_root: Path) -> Path:
    return source_root / PIPELINE_REFERENCE_GAMES_RELATIVE_DIR


def _default_reference_game_slugs(source_root: Path) -> list[str]:
    candidates = set(_reference_game_slugs(source_root)) | set(
        _reference_game_slugs(_pipeline_reference_games_root(source_root))
    )
    required_slugs = list(DEFAULT_REFERENCE_GAME_SLUGS)
    missing = [slug for slug in required_slugs if slug not in candidates]
    if missing:
        raise RuntimeError(f"Missing required default reference games: {missing!r}")
    return sorted(set(required_slugs))


def _copy_pipeline_reference_games(*, source_root: Path, worker_repo_dir: Path, keep_slugs: Sequence[str]) -> None:
    reference_root = _pipeline_reference_games_root(source_root)
    source_environment_root = reference_root / "re_arc" / "environment_files"
    source_agents_root = reference_root / "re_arc" / "dsl" / "agents"
    target_environment_root = worker_repo_dir / "re_arc" / "environment_files"
    target_agents_root = worker_repo_dir / "re_arc" / "dsl" / "agents"

    for slug in keep_slugs:
        source_env_dir = source_environment_root / slug
        target_env_dir = target_environment_root / slug
        if source_env_dir.exists():
            if target_env_dir.exists():
                shutil.rmtree(target_env_dir)
            target_env_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source_env_dir, target_env_dir)

        source_agent_path = source_agents_root / f"{slug}.py"
        target_agent_path = target_agents_root / f"{slug}.py"
        if source_agent_path.exists():
            target_agent_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_agent_path, target_agent_path)


def _copy_worker_context_docs(*, source_root: Path, worker_repo_dir: Path) -> None:
    for relative_path in WORKER_DOC_PATHS:
        source_path = source_root / relative_path
        if not source_path.exists():
            continue
        target_path = worker_repo_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def _copy_reference_screenshots(*, source_root: Path, worker_repo_dir: Path) -> None:
    source_path = source_root / REFERENCE_SCREENSHOTS_RELATIVE_DIR
    if not source_path.exists():
        return
    target_path = worker_repo_dir / REFERENCE_SCREENSHOTS_RELATIVE_DIR
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path)


def _copy_arcengine_reference(*, worker_repo_dir: Path) -> None:
    spec = find_spec("arcengine")
    if spec is None or spec.origin is None:
        return
    source_path = Path(spec.origin).resolve().parent
    if not source_path.is_dir():
        return
    target_path = worker_repo_dir / WORKER_ARCENGINE_REFERENCE_PATH
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".mypy_cache", ".pytest_cache")
    shutil.copytree(source_path, target_path, ignore=ignore)


def _write_worker_agents(*, worker_repo_dir: Path) -> None:
    write_text(worker_repo_dir / "AGENTS.md", WORKER_AGENTS_TEXT)


def _prune_reference_games(*, worker_repo_dir: Path, keep_slugs: Sequence[str]) -> None:
    keep = set(keep_slugs)

    environment_root = worker_repo_dir / "re_arc" / "environment_files"
    if environment_root.exists():
        for path in environment_root.iterdir():
            if not path.is_dir() or path.name in keep:
                continue
            shutil.rmtree(path)

    agents_root = worker_repo_dir / "re_arc" / "dsl" / "agents"
    if agents_root.exists():
        for path in agents_root.glob("*.py"):
            if not path.is_file() or path.stem == "__init__" or path.stem in keep:
                continue
            path.unlink()


def _write_worker_tests(*, worker_repo_dir: Path, reference_game_slugs: Sequence[str]) -> None:
    tests_root = worker_repo_dir / "tests"
    tests_root.mkdir(parents=True, exist_ok=True)
    reference_slugs_json = json.dumps(sorted({str(slug) for slug in reference_game_slugs}))
    template_root = repo_root() / "pipeline" / "worker_tests"
    for template_path in sorted(template_root.iterdir()):
        if not template_path.is_file():
            continue
        target_path = tests_root / template_path.name
        template_text = template_path.read_text(encoding="utf-8")
        if template_path.name == "_worker_game.py":
            template_text = template_text.replace("__REFERENCE_GAME_SLUGS_JSON__", reference_slugs_json)
        write_text(target_path, template_text)


def _write_worker_validation_tool(*, source_root: Path, worker_repo_dir: Path) -> None:
    source_pipeline_root = source_root / "pipeline"
    if not (source_pipeline_root / "validate_generated_game.py").exists():
        source_pipeline_root = repo_root() / "pipeline"
    target_pipeline_root = worker_repo_dir / "pipeline"
    target_pipeline_root.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "validate_generated_game.py"):
        shutil.copy2(source_pipeline_root / name, target_pipeline_root / name)


def _copy_worker_repo(*, source_root: Path, worker_repo_dir: Path) -> list[str]:
    if worker_repo_dir.exists():
        shutil.rmtree(worker_repo_dir)
    worker_repo_dir.mkdir(parents=True, exist_ok=True)
    for name in WORKER_REPO_ITEMS:
        source_path = source_root / name
        target_path = worker_repo_dir / name
        if source_path.is_dir():
            shutil.copytree(source_path, target_path)
        elif source_path.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
    selected_slugs = _default_reference_game_slugs(source_root)
    _copy_pipeline_reference_games(source_root=source_root, worker_repo_dir=worker_repo_dir, keep_slugs=selected_slugs)
    _copy_worker_context_docs(source_root=source_root, worker_repo_dir=worker_repo_dir)
    _copy_reference_screenshots(source_root=source_root, worker_repo_dir=worker_repo_dir)
    _copy_arcengine_reference(worker_repo_dir=worker_repo_dir)
    _write_worker_agents(worker_repo_dir=worker_repo_dir)
    _prune_reference_games(worker_repo_dir=worker_repo_dir, keep_slugs=selected_slugs)
    _write_worker_tests(worker_repo_dir=worker_repo_dir, reference_game_slugs=selected_slugs)
    _write_worker_validation_tool(source_root=source_root, worker_repo_dir=worker_repo_dir)
    return selected_slugs


def _implemented_game_record(*, worker_repo_dir: Path, reference_game_slugs: Sequence[str]) -> dict[str, str]:
    reference_slugs = set(reference_game_slugs)
    environment_root = worker_repo_dir / "re_arc" / "environment_files"
    agents_root = worker_repo_dir / "re_arc" / "dsl" / "agents"
    environment_slugs = (
        {path.name for path in environment_root.iterdir() if path.is_dir()} if environment_root.exists() else set()
    )
    agent_slugs = (
        {path.stem for path in agents_root.glob("*.py") if path.is_file() and path.stem != "__init__"}
        if agents_root.exists()
        else set()
    )
    candidate_slugs = sorted((environment_slugs & agent_slugs) - reference_slugs)
    if len(candidate_slugs) != 1:
        raise RuntimeError(
            "Expected exactly one generated game slug after implementation finished, "
            f"found {candidate_slugs!r} (reference slugs: {sorted(reference_slugs)!r})."
        )
    slug = candidate_slugs[0]
    metadata_paths = sorted((environment_root / slug).rglob("metadata.json"))
    if len(metadata_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one metadata.json for generated slug {slug!r}, found {len(metadata_paths)}."
        )
    payload = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
    game_id = str(payload.get("game_id") or "").strip()
    if not game_id:
        raise RuntimeError(f"Generated metadata at {metadata_paths[0]} did not define game_id.")
    return {"slug": slug, "game_id": game_id}


def _python_bin_for_source_root(source_root: Path) -> Path:
    python_bin = source_root / ".venv" / "bin" / "python"
    if python_bin.exists():
        return python_bin
    return Path(sys.executable)


def _generate_worker_artifacts(
    *, source_root: Path, worker_repo_dir: Path, game_id: str, artifact_runner: ArtifactRunner, artifact_log_path: Path
) -> dict[str, str]:
    artifacts_dir = worker_repo_dir / "artifacts"
    replay_dir = artifacts_dir / "replays"
    gif_path = replay_dir / f"{game_id}.gif"
    replay_path = replay_dir / f"{game_id}.replay.json"
    cmd = [
        str(_python_bin_for_source_root(source_root)),
        "-m",
        "re_arc",
        "--config",
        "config.env",
        "--game",
        game_id,
        "--policy",
        "dsl",
        "--max-actions",
        "1200",
        "--gif",
        str(gif_path),
        "--replay",
        str(replay_path),
    ]
    result = artifact_runner(cmd, worker_repo_dir, artifact_log_path)
    if result.returncode != 0:
        raise RuntimeError(f"artifact generation failed with exit code {result.returncode}")
    return {"dsl_gif_path": str(gif_path), "dsl_replay_path": str(replay_path)}


def prepare_worker_package(
    *,
    run_id: str,
    idea_id: str,
    idea_title: str = "",
    idea_description: str = "",
    spec_path: Path,
    worker_root: Path,
    prompts_dir: Path,
    status_log_path: Path | None = None,
    source_root: Path | None = None,
    implementation_prompt_template_path: Path | None = None,
) -> dict[str, JSONValue]:
    root = source_root or repo_root()
    template_path = implementation_prompt_template_path or (
        root / "pipeline" / "prompts" / "codex_check_visual_prompt.md"
    )
    worker_root.mkdir(parents=True, exist_ok=True)
    worker_repo_dir = worker_root / "repo"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    reference_game_slugs = _copy_worker_repo(source_root=root, worker_repo_dir=worker_repo_dir)

    worker_prompt_path = prompts_dir / f"{idea_id}.implementation_prompt.txt"
    worker_status_path = worker_root / "status.json"
    worker_idea_path = worker_root / "idea.json"

    spec_text = load_spec_text(spec_path)
    prompt_template = load_prompt_template(template_path)
    prompt_text = build_implementation_prompt(template=prompt_template, spec_text=spec_text)

    write_text(worker_prompt_path, prompt_text)
    write_json(worker_idea_path, {"idea_id": idea_id, "title": idea_title, "description": idea_description})
    write_worker_status(
        status_path=worker_status_path, run_id=run_id, idea_id=idea_id, status="queued", status_log_path=status_log_path
    )

    return {
        "idea_id": idea_id,
        "idea_title": idea_title,
        "idea_description": idea_description,
        "worker_root": str(worker_root),
        "worker_repo_path": str(worker_repo_dir),
        "worker_spec_path": str(spec_path),
        "worker_idea_path": str(worker_idea_path),
        "worker_prompt_path": str(worker_prompt_path),
        "worker_status_path": str(worker_status_path),
        "reference_game_slugs": reference_game_slugs,
    }


def prepare_and_launch_worker_package(
    *,
    run_id: str,
    idea_id: str,
    idea_title: str = "",
    idea_description: str = "",
    spec_path: Path,
    worker_root: Path,
    prompts_dir: Path,
    status_log_path: Path | None = None,
    source_root: Path | None = None,
    implementation_prompt_template_path: Path | None = None,
    implementation_backend: ImplementationBackend = "codex",
    codex_bin: str = "codex",
    codex_runner: CodexRunner | None = None,
    codex_reasoning_effort: str | None = None,
    claude_bin: str = "claude",
    claude_runner: ClaudeRunner | None = None,
    artifact_runner: ArtifactRunner | None = None,
) -> dict[str, JSONValue]:
    package = prepare_worker_package(
        run_id=run_id,
        idea_id=idea_id,
        idea_title=idea_title,
        idea_description=idea_description,
        spec_path=spec_path,
        worker_root=worker_root,
        prompts_dir=prompts_dir,
        status_log_path=status_log_path,
        source_root=source_root,
        implementation_prompt_template_path=implementation_prompt_template_path,
    )
    worker_repo_path = Path(str(package["worker_repo_path"]))
    worker_prompt_path = Path(str(package["worker_prompt_path"]))
    worker_status_path = Path(str(package["worker_status_path"]))
    implementation_root = worker_root / implementation_backend
    implementation_output_path = implementation_root / "last_message.txt"
    implementation_log_path = implementation_root / "stdout.log"
    artifact_log_path = worker_root / "artifacts" / "generation.log"
    prompt_text = worker_prompt_path.read_text(encoding="utf-8")
    implementation_started_at = utc_now()
    started_monotonic = perf_counter()
    raw_reference_game_slugs = package.get("reference_game_slugs", [])
    reference_game_slugs = (
        [str(slug) for slug in raw_reference_game_slugs] if isinstance(raw_reference_game_slugs, list) else []
    )
    resolved_codex_runner = codex_runner or _run_codex_command
    resolved_claude_runner = claude_runner or _run_claude_command
    resolved_artifact_runner = artifact_runner or _run_artifact_command

    write_worker_status(
        status_path=worker_status_path,
        run_id=run_id,
        idea_id=idea_id,
        status=f"launching {implementation_backend}",
        status_log_path=status_log_path,
        codex_started_at=implementation_started_at,
    )
    if implementation_backend == "codex":
        cmd = [
            codex_bin,
            "exec",
            "--cd",
            str(worker_repo_path),
            "--dangerously-bypass-approvals-and-sandbox",
            *(["-c", f'model_reasoning_effort="{codex_reasoning_effort}"'] if codex_reasoning_effort else []),
            "-o",
            str(implementation_output_path),
            prompt_text,
        ]
        result = resolved_codex_runner(cmd, worker_repo_path, implementation_log_path)
    else:
        cmd = [claude_bin, "--dangerously-skip-permissions", "-p", prompt_text]
        result = resolved_claude_runner(cmd, worker_repo_path, implementation_log_path, implementation_output_path)
    implementation_finished_at = utc_now()
    implementation_duration_seconds = perf_counter() - started_monotonic
    status = f"{implementation_backend} finished" if result.returncode == 0 else f"{implementation_backend} failed"
    error = (
        None if result.returncode == 0 else f"{implementation_backend} exec failed with exit code {result.returncode}"
    )
    write_worker_status(
        status_path=worker_status_path,
        run_id=run_id,
        idea_id=idea_id,
        status=status,
        status_log_path=status_log_path,
        codex_started_at=implementation_started_at,
        codex_finished_at=implementation_finished_at,
        codex_duration_seconds=implementation_duration_seconds,
        codex_exit_code=str(result.returncode),
        error=error,
    )
    outcome = "done" if result.returncode == 0 else "failed"
    print_pipeline_log(
        f"[pipeline] Implementation {outcome} for {idea_id} with {implementation_backend} "
        f"in {format_duration(implementation_duration_seconds)}"
    )
    if result.returncode != 0:
        raise RuntimeError(str(error))

    implemented_game = _implemented_game_record(
        worker_repo_dir=worker_repo_path, reference_game_slugs=reference_game_slugs
    )
    try:
        generated_artifacts = _generate_worker_artifacts(
            source_root=source_root or repo_root(),
            worker_repo_dir=worker_repo_path,
            game_id=implemented_game["game_id"],
            artifact_runner=resolved_artifact_runner,
            artifact_log_path=artifact_log_path,
        )
    except Exception as exc:
        write_worker_status(
            status_path=worker_status_path,
            run_id=run_id,
            idea_id=idea_id,
            status="artifact failed",
            status_log_path=status_log_path,
            codex_started_at=implementation_started_at,
            codex_finished_at=implementation_finished_at,
            codex_duration_seconds=implementation_duration_seconds,
            codex_exit_code=str(result.returncode),
            error=str(exc),
        )
        raise

    result_payload = cast(dict[str, JSONValue], dict(package))
    result_payload["implementation_backend"] = implementation_backend
    result_payload["implementation_output_path"] = str(implementation_output_path)
    result_payload["implementation_log_path"] = str(implementation_log_path)
    result_payload["implementation_exit_code"] = result.returncode
    if implementation_backend == "codex":
        result_payload["codex_output_path"] = str(implementation_output_path)
        result_payload["codex_log_path"] = str(implementation_log_path)
        result_payload["codex_exit_code"] = result.returncode
    else:
        result_payload["claude_output_path"] = str(implementation_output_path)
        result_payload["claude_log_path"] = str(implementation_log_path)
        result_payload["claude_exit_code"] = result.returncode
    result_payload["generated_slug"] = implemented_game["slug"]
    result_payload["game_id"] = implemented_game["game_id"]
    result_payload["artifact_log_path"] = str(artifact_log_path)
    result_payload.update(generated_artifacts)
    return result_payload
