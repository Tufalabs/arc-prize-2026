#!/usr/bin/env python3
"""Run, package, and publish reusable multi-game PRO-LONG evaluations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = ROOT / "configs/prolong/short-games-action7-v1.json"
RAW_ROOT = ROOT / "runs/prolong-panel"
COPY_FILES = ("workspace/logs.txt", "trace.jsonl", "codex.stderr.log", "result.json", "token_usage.json")
PROHIBITED_PUBLISHED_FILES = {"codex.events.jsonl", "manifest.json"}
SECRET_PATH_MARKERS = ("auth.json", ".codex/")
SECRET_TEXT_MARKERS = ("OPENAI_API_KEY", "CODEX_ACCESS_TOKEN")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt", ".md"}
REDACTIONS = (
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "<redacted-email>"),
    (re.compile(r"/(?:Users|home)/[^/\s\"']+"), "<redacted-home>"),
    (re.compile(r"/private/(?:tmp|var)/[^\s\"']+"), "<redacted-local-path>"),
    (re.compile(r'(?i)(["\'](?:HOME|PATH)["\']\s*:\s*["\'])[^"\']*(["\'])'), r"\1<redacted-env>\2"),
    (re.compile(r"(?im)\b(?:HOME|PATH)=[^\s]+"), "<redacted-env>"),
    (
        re.compile(r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "<redacted-token>",
    ),
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]{16,}"), "Bearer <redacted-token>"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), "<redacted-token>"),
)
REQUIRED_PANEL_KEYS = (
    "protocol",
    "description",
    "seed",
    "model",
    "reasoning_effort",
    "timeout_seconds",
    "watchdog_grace_seconds",
    "canary_timeout_seconds",
    "action_cap",
    "max_episode_actions",
    "analyzer_retries",
    "concurrency",
    "games",
)


def run_text(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True, check=False)
    if check and result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"Command failed: {' '.join(args)}"
        raise RuntimeError(detail)
    return result.stdout.strip()


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-.")
    if not cleaned:
        raise ValueError("Name must contain a letter or number.")
    return cleaned[:80]


def result_root(panel: dict[str, Any]) -> Path:
    return ROOT / "results/prolong" / safe_name(str(panel["protocol"]))


def load_panel(path: Path) -> dict[str, Any]:
    panel = json.loads(path.read_text(encoding="utf-8"))
    if panel.get("archived") is True:
        raise ValueError(f"Panel is archived provenance and cannot be evaluated: {path}")
    missing_keys = [key for key in REQUIRED_PANEL_KEYS if key not in panel]
    if missing_keys:
        raise ValueError(f"Panel is missing required keys: {', '.join(missing_keys)}")

    games = panel["games"]
    if not isinstance(games, list) or not games or len(games) != len(set(games)):
        raise ValueError(f"Panel must contain a non-empty unique games list: {path}")
    for game_id in games:
        try:
            slug, variant = str(game_id).rsplit("-", 1)
        except ValueError as error:
            raise ValueError(f"Invalid game ID {game_id!r}; expected <slug>-<variant>.") from error
        required = (
            ROOT / "re_arc/environment_files" / slug / variant / f"{slug}.py",
            ROOT / "re_arc/environment_files" / slug / variant / "manifest.json",
            ROOT / "re_arc/environment_files" / slug / variant / "metadata.json",
        )
        missing = [str(item.relative_to(ROOT)) for item in required if not item.is_file()]
        if missing:
            raise FileNotFoundError(f"{game_id} is incomplete: {', '.join(missing)}")

    positive_fields = (
        "timeout_seconds",
        "watchdog_grace_seconds",
        "canary_timeout_seconds",
        "action_cap",
        "max_episode_actions",
        "analyzer_retries",
        "concurrency",
    )
    invalid = [key for key in positive_fields if not isinstance(panel[key], (int, float)) or panel[key] <= 0]
    if invalid:
        raise ValueError(f"Panel values must be positive numbers: {', '.join(invalid)}")
    safe_name(str(panel["protocol"]))
    return panel


def timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def codex_preflight(*, allow_login: bool) -> tuple[str, str]:
    codex = shutil.which("codex")
    if codex is None:
        raise RuntimeError("Codex CLI is not installed. Install it, then rerun ./evaluate.")
    version = run_text(codex, "--version")
    status = subprocess.run([codex, "login", "status"], cwd=ROOT, text=True, capture_output=True, check=False)
    if status.returncode and allow_login:
        print("Codex is not logged in. Starting device login for your own account...", flush=True)
        subprocess.run([codex, "login", "--device-auth"], cwd=ROOT, check=True)
        status = subprocess.run([codex, "login", "status"], cwd=ROOT, text=True, capture_output=True, check=False)
    if status.returncode:
        raise RuntimeError("Codex login is required. Run `codex login --device-auth`, then rerun ./evaluate.")
    return codex, version


@dataclass
class Job:
    game_id: str
    process: subprocess.Popen[str]
    started: float
    stdout: TextIO
    stderr: TextIO


def command_for(panel: dict[str, Any], game_id: str, raw_dir: Path, codex: str) -> list[str]:
    return [
        str(ROOT / ".venv/bin/python"),
        "-m",
        "re_arc.prolong_player",
        game_id,
        "--seed",
        str(panel["seed"]),
        "--output-root",
        str(raw_dir),
        "--run-id",
        game_id,
        "--codex-bin",
        codex,
        "--model",
        str(panel["model"]),
        "--reasoning-effort",
        str(panel["reasoning_effort"]),
        "--timeout-seconds",
        str(panel["timeout_seconds"]),
        "--action-cap",
        str(panel["action_cap"]),
        "--max-episode-actions",
        str(panel["max_episode_actions"]),
        "--analyzer-retries",
        str(panel["analyzer_retries"]),
    ]


def launch(panel: dict[str, Any], game_id: str, raw_dir: Path, codex: str) -> Job:
    launcher = raw_dir / "launcher"
    launcher.mkdir(exist_ok=True)
    stdout = (launcher / f"{game_id}.stdout.log").open("w", encoding="utf-8")
    stderr = (launcher / f"{game_id}.stderr.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        command_for(panel, game_id, raw_dir, codex),
        cwd=ROOT,
        text=True,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    print(f"started {game_id} (pid {process.pid})", flush=True)
    return Job(game_id, process, time.monotonic(), stdout, stderr)


def stop_job(job: Job, sig: signal.Signals = signal.SIGINT) -> None:
    if job.process.poll() is None:
        try:
            os.killpg(job.process.pid, sig)
        except ProcessLookupError:
            pass


def close_job(job: Job) -> None:
    job.stdout.close()
    job.stderr.close()


def first_action_written(raw_dir: Path, game_id: str) -> bool:
    trace = raw_dir / game_id / "trace.jsonl"
    if not trace.exists():
        return False
    for line in trace.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "decision":
            return True
    return False


def supervise(panel: dict[str, Any], raw_dir: Path, codex: str, concurrency: int) -> dict[str, int]:
    games = list(panel["games"])
    pending = games[1:]
    active: dict[str, Job] = {}
    exits: dict[str, int] = {}
    interrupted = False

    def interrupt(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print("Stopping active games and preserving partial traces...", flush=True)
        for active_job in active.values():
            stop_job(active_job)

    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    canary = launch(panel, games[0], raw_dir, codex)
    active[canary.game_id] = canary
    canary_deadline = time.monotonic() + float(panel["canary_timeout_seconds"])
    while not interrupted and not first_action_written(raw_dir, canary.game_id):
        code = canary.process.poll()
        if code is not None:
            exits[canary.game_id] = code
            close_job(canary)
            active.pop(canary.game_id)
            raise RuntimeError(f"Canary {canary.game_id} exited before its first action (exit {code}).")
        if time.monotonic() >= canary_deadline:
            stop_job(canary)
            try:
                code = canary.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                stop_job(canary, signal.SIGKILL)
                code = canary.process.wait()
            exits[canary.game_id] = int(code)
            close_job(canary)
            active.pop(canary.game_id)
            raise RuntimeError(f"Canary {canary.game_id} produced no action within the canary deadline.")
        time.sleep(0.25)
    if not interrupted:
        print(f"canary passed: {canary.game_id} wrote its first action", flush=True)

    watchdog = float(panel["timeout_seconds"]) + float(panel["watchdog_grace_seconds"])
    while active or pending:
        if not interrupted:
            while pending and len(active) < concurrency:
                game_id = pending.pop(0)
                active[game_id] = launch(panel, game_id, raw_dir, codex)
        for game_id, job in list(active.items()):
            code = job.process.poll()
            if code is None and time.monotonic() - job.started > watchdog:
                print(f"watchdog interrupt: {game_id}", flush=True)
                stop_job(job)
                try:
                    code = job.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop_job(job, signal.SIGKILL)
                    code = job.process.wait()
            if code is not None:
                exits[game_id] = int(code)
                close_job(job)
                active.pop(game_id)
                print(f"finished {game_id} (exit {code})", flush=True)
        if interrupted and active:
            time.sleep(0.25)
            continue
        if interrupted:
            exits.update(dict.fromkeys(pending, 130))
            pending.clear()
        time.sleep(0.1)
    return exits


def copy_if_present(source: Path, destination: Path) -> None:
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix in TEXT_SUFFIXES:
            destination.write_text(
                sanitize_text(source.read_text(encoding="utf-8", errors="replace")), encoding="utf-8"
            )
        else:
            shutil.copy2(source, destination)


def sanitize_text(value: str) -> str:
    for pattern, replacement in REDACTIONS:
        value = pattern.sub(replacement, value)
    return value


def sanitize_bundle(path: Path) -> None:
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        if item.name in PROHIBITED_PUBLISHED_FILES:
            item.unlink()
        elif item.suffix in TEXT_SUFFIXES:
            sanitized = sanitize_text(item.read_text(encoding="utf-8", errors="replace"))
            item.write_text(sanitized, encoding="utf-8")


def result_row(game_id: str, game_dir: Path, exit_code: int) -> dict[str, Any]:
    result_path = game_dir / "result.json"
    if result_path.exists():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "game_id": game_id,
            "termination_reason": payload.get("termination_reason", "unknown"),
            "final_state": payload.get("final_state", "unknown"),
            "actions": payload.get("primitive_actions", 0),
            "duration_seconds": payload.get("duration_seconds", 0),
            "exit_code": exit_code,
        }
    return {
        "game_id": game_id,
        "termination_reason": "infrastructure_failure",
        "final_state": "unknown",
        "actions": 0,
        "duration_seconds": 0,
        "exit_code": exit_code,
    }


def package(
    panel: dict[str, Any], raw_dir: Path, submission_dir: Path, exits: dict[str, int], metadata: dict[str, Any]
) -> None:
    submission_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    for game_id in panel["games"]:
        raw_game = raw_dir / game_id
        output_game = submission_dir / "games" / game_id
        output_game.mkdir(parents=True)
        for relative in COPY_FILES:
            target_name = "logs.txt" if relative == "workspace/logs.txt" else relative
            copy_if_present(raw_game / relative, output_game / target_name)
        row = result_row(game_id, raw_game, exits.get(game_id, 130))
        rows.append(row)
        if not (output_game / "result.json").exists():
            (output_game / "failure.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")

    outcome_counts = Counter(str(row["final_state"]) for row in rows)
    summary = {
        "protocol": panel["protocol"],
        "description": panel["description"],
        "game_count": len(rows),
        "outcomes": dict(sorted(outcome_counts.items())),
        "games": rows,
    }
    (submission_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (submission_dir / "submission.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    counts = ", ".join(f"{state}: {count}" for state, count in sorted(outcome_counts.items()))
    lines = [
        f"# PRO-LONG results: {panel['description']}",
        "",
        f"Protocol: `{panel['protocol']}`  ",
        f"Evaluator branch: `{metadata['evaluator_branch']}`  ",
        f"Evaluator commit: `{metadata['evaluator_commit']}`  ",
        f"Codex: `{metadata['codex_version']}`  ",
        f"Model: `{metadata['model']}` at `{metadata['reasoning_effort']}` effort  ",
        f"Summary: {counts}",
        "",
        "| Game | Outcome | Reason | Actions | Runtime | Readable trace |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        game_id = row["game_id"]
        log = submission_dir / "games" / game_id / "logs.txt"
        trace_link = f"[logs.txt](games/{game_id}/logs.txt)" if log.exists() else "missing"
        lines.append(
            f"| `{game_id}` | {row['final_state']} | {row['termination_reason']} | "
            f"{row['actions']} | {float(row['duration_seconds']):.1f}s | {trace_link} |"
        )
    (submission_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sanitize_bundle(submission_dir)
    verify_bundle(submission_dir, expected_games=set(panel["games"]))


def verify_bundle(path: Path, *, expected_games: set[str] | None = None) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"Submission does not exist: {path}")
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Submission has no summary.json: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    actual_games = {row["game_id"] for row in summary["games"]}
    if expected_games is not None and actual_games != expected_games:
        raise RuntimeError(f"Submission game mismatch: expected {sorted(expected_games)}, got {sorted(actual_games)}")
    total = 0
    for item in path.rglob("*"):
        relative = item.relative_to(path).as_posix()
        if any(marker.lower() in relative.lower() for marker in SECRET_PATH_MARKERS):
            raise RuntimeError(f"Credential-like path rejected: {relative}")
        if not item.is_file():
            continue
        if item.name in PROHIBITED_PUBLISHED_FILES:
            raise RuntimeError(f"Raw evaluator artifact rejected: {relative}")
        size = item.stat().st_size
        total += size
        if size > 25 * 1024 * 1024:
            raise RuntimeError(f"File exceeds 25 MiB limit: {relative}")
        if item.suffix in TEXT_SUFFIXES:
            text = item.read_text(encoding="utf-8", errors="replace")
            for marker in SECRET_TEXT_MARKERS:
                if marker in text:
                    raise RuntimeError(f"Credential marker {marker} rejected in {relative}")
            for pattern, _replacement in REDACTIONS:
                if pattern.search(text):
                    raise RuntimeError(f"Sensitive value rejected in {relative}")
    if total > 300 * 1024 * 1024:
        raise RuntimeError(f"Submission exceeds 300 MiB ({total / 1024 / 1024:.1f} MiB).")


def relative_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def evaluate(args: argparse.Namespace) -> int:
    panel_path = Path(args.panel).resolve()
    panel = load_panel(panel_path)
    concurrency = int(args.concurrency or panel["concurrency"])
    if concurrency < 1:
        raise ValueError("Concurrency must be positive.")

    evaluator_commit = run_text("git", "rev-parse", "HEAD")
    evaluator_branch = run_text("git", "branch", "--show-current")
    if not evaluator_branch:
        raise RuntimeError("Evaluation must run on a named Git branch, not a detached HEAD.")
    if args.dry_run:
        codex = shutil.which("codex") or "codex"
        commands = [command_for(panel, game, Path("<run-directory>"), codex) for game in panel["games"]]
        print(
            json.dumps(
                {
                    "panel": relative_to_root(panel_path),
                    "protocol": panel["protocol"],
                    "game_count": len(panel["games"]),
                    "canary": panel["games"][0],
                    "concurrency": concurrency,
                    "result_root": relative_to_root(result_root(panel)),
                    "commands": commands,
                },
                indent=2,
            )
        )
        return 0

    codex, codex_version = codex_preflight(allow_login=True)
    tester = safe_name(args.tester or run_text("git", "config", "user.name", check=False) or "tester")
    submission_id = f"{tester}-{timestamp()}"
    raw_dir = RAW_ROOT / safe_name(str(panel["protocol"])) / submission_id
    submission_dir = result_root(panel) / submission_id
    raw_dir.mkdir(parents=True)
    metadata = {
        "protocol": panel["protocol"],
        "submission_id": submission_id,
        "tester": tester,
        "created_at": datetime.now(UTC).isoformat(),
        "evaluator_commit": evaluator_commit,
        "evaluator_branch": evaluator_branch,
        "panel_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "panel": relative_to_root(panel_path),
        "codex_version": codex_version,
        "model": panel["model"],
        "reasoning_effort": panel["reasoning_effort"],
        "seed": panel["seed"],
        "timeout_seconds": panel["timeout_seconds"],
        "action_cap": panel["action_cap"],
        "max_episode_actions": panel["max_episode_actions"],
        "concurrency": concurrency,
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    exits: dict[str, int] = {}
    try:
        exits = supervise(panel, raw_dir, codex, concurrency)
    finally:
        package(panel, raw_dir, submission_dir, exits, metadata)
    print(f"\nPackaged Markdown report and traces at {relative_to_root(submission_dir)}")
    if args.no_submit:
        print("Automatic submission disabled by --no-submit.")
        return 0
    print(f"Submitting results to {args.remote}/{evaluator_branch}...", flush=True)
    return submit(argparse.Namespace(submission=str(submission_dir), remote=args.remote, dry_run=False))


def candidate_submissions() -> list[Path]:
    return sorted(
        path
        for protocol_dir in (ROOT / "results/prolong").glob("*")
        if protocol_dir.is_dir()
        for path in protocol_dir.glob("*")
        if path.is_dir() and (path / "submission.json").is_file()
    )


def submit(args: argparse.Namespace) -> int:
    candidates = candidate_submissions()
    submission = Path(args.submission).resolve() if args.submission else (candidates[-1] if candidates else None)
    if submission is None:
        raise RuntimeError("No packaged submission found. Run ./evaluate first.")
    verify_bundle(submission)
    try:
        relative = submission.relative_to(ROOT)
    except ValueError as error:
        raise ValueError("Submission must be inside this repository.") from error
    metadata = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    branch = str(metadata["evaluator_branch"])
    current_branch = run_text("git", "branch", "--show-current")
    if current_branch != branch:
        raise RuntimeError(f"Results belong on {branch}; current branch is {current_branch or '(detached)'}.")
    tracked_changes = run_text("git", "status", "--porcelain", "--untracked-files=no")
    if tracked_changes:
        raise RuntimeError("Tracked files have local changes. Commit or stash them before submitting results.")
    if args.dry_run:
        print(json.dumps({"submission": str(relative), "branch": branch, "remote": args.remote}, indent=2))
        return 0

    run_text("git", "pull", "--rebase", args.remote, branch)
    run_text("git", "add", "--", str(relative))
    run_text("git", "commit", "-m", f"chore: add evaluation results {submission.name}")
    try:
        run_text("git", "push", "-u", args.remote, branch)
    except RuntimeError as error:
        print(f"Results were committed locally, but push failed:\n{error}", file=sys.stderr)
        print("Push the existing commit to a writable remote with:")
        print(f"  git push -u <remote> {branch}")
        return 1
    print(f"Pushed {relative} to {args.remote}/{branch}.")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Run and publish reusable PRO-LONG evaluation panels.")
    commands = root.add_subparsers(dest="command", required=True)
    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("--panel", default=str(DEFAULT_PANEL))
    evaluate_parser.add_argument("--concurrency", type=int)
    evaluate_parser.add_argument("--tester")
    evaluate_parser.add_argument("--remote", default="origin")
    evaluate_parser.add_argument("--no-submit", action="store_true")
    evaluate_parser.add_argument("--dry-run", action="store_true")
    evaluate_parser.set_defaults(func=evaluate)
    submit_parser = commands.add_parser("submit")
    submit_parser.add_argument("--submission")
    submit_parser.add_argument("--remote", default="origin")
    submit_parser.add_argument("--dry-run", action="store_true")
    submit_parser.set_defaults(func=submit)
    sanitize_parser = commands.add_parser("sanitize", help="redact a previously packaged submission")
    sanitize_parser.add_argument("submission")
    sanitize_parser.set_defaults(func=sanitize_command)
    return root


def sanitize_command(args: argparse.Namespace) -> int:
    submission = Path(args.submission).resolve()
    sanitize_bundle(submission)
    verify_bundle(submission)
    print(f"Sanitized {relative_to_root(submission)}")
    return 0


def main() -> int:
    args = parser().parse_args()
    try:
        return int(args.func(args))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
