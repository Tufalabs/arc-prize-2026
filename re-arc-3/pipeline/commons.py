from __future__ import annotations

import json
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | Mapping[str, "JSONValue"] | Sequence["JSONValue"]
_STATUS_LOG_LOCK = threading.Lock()
_TERMINAL_LOG_LOCK = threading.Lock()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"


def print_pipeline_log(message: str) -> None:
    with _TERMINAL_LOG_LOCK:
        print(message, flush=True)


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> JSONValue:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: JSONValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def worker_status_path(*, workers_dir: Path, idea_id: str) -> Path:
    return workers_dir / idea_id / "status.json"


def append_status_log(*, log_path: Path, payload: Mapping[str, JSONValue]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, sort_keys=True) + "\n"
    with _STATUS_LOG_LOCK:
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line)


def write_worker_status(
    *,
    status_path: Path,
    run_id: str,
    idea_id: str,
    status: str,
    status_log_path: Path | None = None,
    spec_started_at: str | None = None,
    spec_finished_at: str | None = None,
    spec_duration_seconds: float | None = None,
    codex_started_at: str | None = None,
    codex_finished_at: str | None = None,
    codex_duration_seconds: float | None = None,
    codex_exit_code: str | None = None,
    error: str | None = None,
    extra: Mapping[str, JSONValue] | None = None,
) -> None:
    payload: dict[str, JSONValue] = {}
    if status_path.exists():
        existing = read_json(status_path)
        if isinstance(existing, dict):
            payload.update({str(key): value for key, value in existing.items()})
    payload.update({"run_id": run_id, "idea_id": idea_id, "status": status, "updated_at": utc_now()})
    if spec_started_at is not None:
        payload["spec_started_at"] = spec_started_at
    if spec_finished_at is not None:
        payload["spec_finished_at"] = spec_finished_at
    if spec_duration_seconds is not None:
        payload["spec_duration_seconds"] = spec_duration_seconds
    if codex_started_at is not None:
        payload["codex_started_at"] = codex_started_at
    if codex_finished_at is not None:
        payload["codex_finished_at"] = codex_finished_at
    if codex_duration_seconds is not None:
        payload["codex_duration_seconds"] = codex_duration_seconds
    if codex_exit_code is not None:
        payload["codex_exit_code"] = codex_exit_code
    if error is not None:
        payload["error"] = error
    if extra is not None:
        payload.update({str(key): value for key, value in extra.items()})
    write_json(status_path, payload)
    if status_log_path is not None:
        append_status_log(log_path=status_log_path, payload={**payload, "worker_status_path": str(status_path)})
