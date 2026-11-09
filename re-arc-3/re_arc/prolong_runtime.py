from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from .dsl import is_terminal, unpack_step_result

DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_TIMEOUT_SECONDS = 1200.0
TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)
ACTION_ID_BY_NAME = {"RESET": 0, "UP": 1, "DOWN": 2, "LEFT": 3, "RIGHT": 4, "SPACE": 5, "CLICK": 6, "UNDO": 7}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: object, lock: threading.Lock | None = None) -> None:
    encoded = json.dumps(_json_safe(payload), separators=(",", ":")) + "\n"
    if lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
        return
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()


def _state_name(observation: Any) -> str:
    state = getattr(observation, "state", None)
    return str(getattr(state, "name", state)).upper()


def _level_index(observation: Any) -> int:
    try:
        return int(getattr(observation, "levels_completed", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _frame_grids(observation: Any) -> list[Any]:
    raw = getattr(observation, "frame", None)
    if raw is None:
        return []
    if isinstance(raw, list):
        return [frame for frame in raw if frame is not None]
    return [raw]


def _grid_dimensions(grid: Any) -> tuple[int, int]:
    shape = getattr(grid, "shape", None)
    if shape is not None and len(shape) >= 2:
        return int(shape[1]), int(shape[0])
    height = len(grid)
    width = len(grid[0]) if height else 0
    return int(width), int(height)


def _encode_grid(grid: Any) -> list[str]:
    width, height = _grid_dimensions(grid)
    if (width, height) != (64, 64):
        raise ValueError(f"Expected a 64x64 observation grid, received {width}x{height}.")
    rows: list[str] = []
    for y in range(height):
        values: list[str] = []
        for x in range(width):
            value = int(grid[y][x])
            if not 0 <= value <= 15:
                raise ValueError(f"Observation color id at ({x}, {y}) is outside 0 through 15: {value}.")
            values.append(format(value, "x"))
        rows.append("".join(values))
    return rows


def _grid_hash(grid: Any) -> str:
    return hashlib.sha256("\n".join(_encode_grid(grid)).encode("ascii")).hexdigest()


def _observation_payload(grids: Sequence[Any]) -> dict[str, Any]:
    return {"frame_hashes": [_grid_hash(grid) for grid in grids]}


def _available_action(env: Any, action_id: int) -> Any | None:
    for action in env.action_space:
        if int(getattr(action, "value", -1)) == int(action_id):
            return action
    return None


def _normalize_action_specs(payload: Any, *, maximum: int) -> tuple[str, list[dict[str, int | str]]]:
    if not isinstance(payload, dict):
        raise ValueError("The action request must be a JSON object.")
    reason = str(payload.get("reason") or "").strip()
    if not reason:
        raise ValueError("The action request must include a non-empty `reason`.")
    raw_actions = payload.get("actions")
    if not isinstance(raw_actions, list) or not raw_actions:
        raise ValueError("The action request must include a non-empty `actions` list.")

    expanded: list[dict[str, int | str]] = []
    for raw_action in raw_actions:
        if not isinstance(raw_action, dict):
            raise ValueError("Each action must be a JSON object.")
        name = str(raw_action.get("name") or "").strip().upper()
        if name not in ACTION_ID_BY_NAME:
            raise ValueError(f"Unknown human action: {name or '(empty)'}.")
        try:
            repeat = int(raw_action.get("repeat", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Action {name} has an invalid repeat count.") from exc
        if not 1 <= repeat <= maximum:
            raise ValueError(f"Action {name} repeat must be between 1 and {maximum}.")
        normalized: dict[str, int | str] = {"name": name}
        if name == "CLICK":
            try:
                x = int(raw_action["x"])
                y = int(raw_action["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("CLICK requires integer `x` and `y` coordinates.") from exc
            if not 0 <= x < 64 or not 0 <= y < 64:
                raise ValueError("CLICK coordinates must be between 0 and 63.")
            normalized.update({"x": x, "y": y})
        for _ in range(repeat):
            expanded.append(dict(normalized))
            if len(expanded) > maximum:
                raise ValueError(f"An action request may contain at most {maximum} primitive actions.")
    return reason, expanded


class ProlongGameSession:
    """Minimal trusted game bridge used only by the local PRO-LONG evaluator."""

    def __init__(
        self, *, env: Any, trace_path: Path, game_id: str, seed: int, max_batch_actions: int, max_episode_actions: int
    ):
        self.env = env
        self.trace_path = trace_path
        self.game_id = game_id
        self.seed = seed
        self.max_batch_actions = max_batch_actions
        self.max_episode_actions = max_episode_actions
        self.trace_lock = threading.Lock()
        self.decision_id = 0
        self.primitive_actions = 0
        self.episode_reward = 0.0
        self.finished = False
        self.last_transition_grids: list[Any] = []
        self.observation = self.env.reset()
        initial_grids = _frame_grids(self.observation)
        if not initial_grids:
            raise ValueError("The environment produced no initial observation frame.")
        _append_jsonl(
            self.trace_path,
            {
                "type": "run_started",
                "at": _utc_now(),
                "game_id": self.game_id,
                "seed": self.seed,
                "initial_observation": _observation_payload(initial_grids),
                "initial_level": _level_index(self.observation),
                "initial_state": _state_name(self.observation),
            },
            self.trace_lock,
        )

    def execute(self, payload: Any) -> dict[str, Any]:
        reason, requested_actions = _normalize_action_specs(payload, maximum=self.max_batch_actions)
        self.decision_id += 1
        before_level = _level_index(self.observation)
        before_state = _state_name(self.observation)
        all_grids: list[Any] = []
        primitive_events: list[dict[str, Any]] = []
        executed_actions: list[dict[str, int | str]] = []
        stopped_reason = "sequence_completed"
        automatic_resets = 0

        for action_spec in requested_actions:
            if self.finished:
                stopped_reason = "episode_ended"
                break
            if self.primitive_actions >= self.max_episode_actions:
                self.finished = True
                stopped_reason = "action_budget_exhausted"
                break

            action_name = cast(str, action_spec["name"])
            action = _available_action(self.env, ACTION_ID_BY_NAME[action_name])
            if action is None:
                stopped_reason = "action_unavailable"
                break
            action_data = {"x": int(action_spec["x"]), "y": int(action_spec["y"])} if action_name == "CLICK" else {}
            level_before = _level_index(self.observation)
            state_before = _state_name(self.observation)
            observation, reward, done, info = unpack_step_result(self.env.step(action, data=action_data))
            self.observation = observation
            self.primitive_actions += 1
            self.episode_reward += float(reward)
            executed_actions.append(action_spec)
            action_grids = _frame_grids(observation)
            all_grids.extend(action_grids)
            level_after = _level_index(observation)
            state_after = _state_name(observation)
            primitive_events.append(
                {
                    "primitive_id": self.primitive_actions,
                    "action": action_spec,
                    "reward": float(reward),
                    "level_before": level_before,
                    "level_after": level_after,
                    "state_before": state_before,
                    "state_after": state_after,
                    "done": bool(done),
                    "info": _json_safe(info),
                    "frame_count": len(action_grids),
                }
            )

            if state_after == "GAME_OVER":
                if self.primitive_actions >= self.max_episode_actions:
                    self.finished = True
                    stopped_reason = "action_budget_exhausted"
                    break
                reset_action = _available_action(self.env, ACTION_ID_BY_NAME["RESET"])
                if reset_action is None:
                    raise RuntimeError("The environment reached GAME_OVER but does not expose RESET.")
                reset_observation, reset_reward, reset_done, reset_info = unpack_step_result(
                    self.env.step(reset_action, data={})
                )
                self.observation = reset_observation
                self.primitive_actions += 1
                self.episode_reward += float(reset_reward)
                automatic_resets += 1
                reset_grids = _frame_grids(reset_observation)
                all_grids.extend(reset_grids)
                primitive_events.append(
                    {
                        "primitive_id": self.primitive_actions,
                        "action": {"name": "RESET"},
                        "automatic": True,
                        "reward": float(reset_reward),
                        "level_before": level_after,
                        "level_after": _level_index(reset_observation),
                        "state_before": state_after,
                        "state_after": _state_name(reset_observation),
                        "done": bool(reset_done),
                        "info": _json_safe(reset_info),
                        "frame_count": len(reset_grids),
                    }
                )
                if reset_done or is_terminal(reset_observation):
                    raise RuntimeError("Automatic RESET after GAME_OVER returned a terminal state.")
                if self.primitive_actions >= self.max_episode_actions:
                    self.finished = True
                    stopped_reason = "action_budget_exhausted"
                else:
                    stopped_reason = "game_over_reset"
                break
            if done or is_terminal(observation):
                self.finished = True
                stopped_reason = "episode_ended"
                break
            if bool((info or {}).get("level_reset", False)):
                stopped_reason = "reset"
                break
            if level_after != level_before:
                stopped_reason = "scene_transition"
                break

        self.last_transition_grids = list(all_grids)
        settled_grids = all_grids or _frame_grids(self.observation)
        trace_record = {
            "type": "decision",
            "at": _utc_now(),
            "decision_id": self.decision_id,
            "reason": reason,
            "requested_actions": requested_actions,
            "executed_actions": executed_actions,
            "automatic_resets": automatic_resets,
            "primitive_events": primitive_events,
            "observation_after": _observation_payload(settled_grids),
            "level_before": before_level,
            "level_after": _level_index(self.observation),
            "state_before": before_state,
            "state_after": _state_name(self.observation),
            "episode_reward": self.episode_reward,
            "stopped_reason": stopped_reason,
            "episode_ended": self.finished,
        }
        _append_jsonl(self.trace_path, trace_record, self.trace_lock)
        return {
            "decision_id": self.decision_id,
            "requested_actions": len(requested_actions),
            "executed_actions": len(executed_actions),
            "automatic_resets": automatic_resets,
            "stopped_reason": stopped_reason,
            "episode_ended": self.finished,
        }

    def finish_trace(self, *, codex_exit_code: int, termination_reason: str) -> None:
        _append_jsonl(
            self.trace_path,
            {
                "type": "run_finished",
                "at": _utc_now(),
                "codex_exit_code": codex_exit_code,
                "termination_reason": termination_reason,
                "primitive_actions": self.primitive_actions,
                "decisions": self.decision_id,
                "episode_reward": self.episode_reward,
                "final_level": _level_index(self.observation),
                "final_state": _state_name(self.observation),
                "episode_ended": self.finished,
            },
            self.trace_lock,
        )


def _safe_command_path() -> str:
    candidates: list[str] = []
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue
        resolved = Path(entry).expanduser()
        if ".venv" in resolved.parts or "re-arc-3" in resolved.parts:
            continue
        candidates.append(str(resolved))
    for fallback in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin", "/bin"):
        if fallback not in candidates and Path(fallback).exists():
            candidates.append(fallback)
    return os.pathsep.join(candidates)


def _codex_environment(workspace: Path) -> dict[str, str]:
    allowed_names = ("HOME", "USER", "LOGNAME", "SHELL", "TMPDIR", "LANG", "LC_ALL", "TERM")
    environment = {name: os.environ[name] for name in allowed_names if name in os.environ}
    environment["PATH"] = _safe_command_path()
    environment["PLAYER_WORKSPACE"] = str(workspace)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                yield record


def _normalized_usage(value: Any) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    usage = {field: int(value.get(field, 0) or 0) for field in TOKEN_FIELDS}
    if usage["total_tokens"] == 0:
        usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
    return usage


def _thread_id_from_events(codex_events_path: Path) -> str | None:
    for event in _iter_jsonl(codex_events_path):
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            return cast(str, event["thread_id"])
    return None


def build_token_usage_report(*, trace_path: Path, codex_events_path: Path) -> dict[str, Any]:
    decisions: list[dict[str, Any]] = []
    primitive_actions = 0
    for record in _iter_jsonl(trace_path):
        if record.get("type") != "decision":
            continue
        primitive_events = record.get("primitive_events")
        primitive_actions += len(primitive_events) if isinstance(primitive_events, list) else 0
        decisions.append(
            {
                "decision_id": record.get("decision_id"),
                "at": record.get("at"),
                "primitive_actions_after": primitive_actions,
            }
        )
    total: dict[str, int] = {}
    for event in _iter_jsonl(codex_events_path):
        if event.get("type") == "turn.completed":
            usage = _normalized_usage(event.get("usage"))
            if usage:
                total = usage
    return {
        "version": 1,
        "source": "codex.events.jsonl",
        "granularity": "decision",
        "total": total,
        "decisions": decisions,
        "tokens_after_last_decision": {},
    }


def _default_run_id(game_id: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    safe_game_id = re.sub(r"[^a-zA-Z0-9._-]+", "-", game_id).strip("-") or "game"
    return f"{timestamp}-{safe_game_id}-{uuid.uuid4().hex[:8]}"
