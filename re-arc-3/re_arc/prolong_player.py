from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .env_sampler import EnvSampler
from .prolong_runtime import (
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_TIMEOUT_SECONDS,
    ProlongGameSession,
    _codex_environment,
    _default_run_id,
    _encode_grid,
    _frame_grids,
    _level_index,
    _safe_command_path,
    _state_name,
    _thread_id_from_events,
    _utc_now,
    _write_json,
    build_token_usage_report,
)

CodexSandbox = Literal["workspace-write", "danger-full-access"]
DEFAULT_PROLONG_MAX_ACTIONS = 500

ACTION_TO_PLAYER = {
    "ACTION1": "UP",
    "ACTION2": "DOWN",
    "ACTION3": "LEFT",
    "ACTION4": "RIGHT",
    "ACTION5": "SPACE",
    "ACTION7": "UNDO",
    "RESET": "RESET",
}
PLAYER_TO_ACTION = {value: key for key, value in ACTION_TO_PLAYER.items()}
VALID_ACTIONS = frozenset({*ACTION_TO_PLAYER, "ACTION6"})
ACTION6_RE = re.compile(r"^ACTION6\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$")
ACTION_DESCRIPTIONS = {
    "RESET": "`RESET`: reset the current level; it still costs an action",
    "ACTION1": "`ACTION1`: up",
    "ACTION2": "`ACTION2`: down",
    "ACTION3": "`ACTION3`: left",
    "ACTION4": "`ACTION4`: right",
    "ACTION5": "`ACTION5`: space/interact",
    "ACTION6": "`ACTION6(x,y)`: click column x, row y, each from 0 through 63",
    "ACTION7": "`ACTION7`: undo",
}

PRO_LONG_SYSTEM_PROMPT = """# PRO-LONG Grid-Game Agent

You are a coding agent playing an unknown 64x64 grid game by writing action plans.

Your primary objective is to solve every level. Your secondary objective is to minimize total cumulative actions.

`./logs.txt` is a lossless, structured game log containing every action,
outcome, animation frame, settled board, and your prior analyses. Parse and
search it programmatically with grep, regular expressions, or Python. Do not
rely on visually scanning long grids when code can answer the question precisely.

The workspace persists across calls. `actions.json` is cleared before each
call; every other file you create persists. You may build parsers, notes,
tests, or executable world models when useful. Do not use the network or
inspect files outside this workspace.

Log markers:
- `[INITIAL BOARD STATE]`: the board before the first action.
- `[POST-ACTION BOARD STATE]`: observations caused by an action.
- `[frame i/N]` and `[settled]`: animation frames followed by the committed board.
- `[ANALYSIS]`: your briefing from the analyzer call that produced the following plan.
- A higher `Levels completed` value means a level was solved. `GAME_OVER` is followed by the runner's automatic reset.

Available actions:
{actions_section}

Respond with a concise strategic briefing containing a `[PLAN]` section. You
must also write `./actions.json` as
`{{"actions": ["ACTION4", "ACTION6(30,40)"]}}`. Write between 1 and
{action_cap} actions. Prefer one or two actions while testing a new hypothesis;
use longer plans only for a sequence already supported by the log. The runner
executes actions one at a time and asks you again when the plan ends or the
level/state changes.
"""

INITIAL_PROMPT = """Read the complete game history in `./logs.txt`
programmatically. This is the first analysis. Infer a useful first test or
proven move, then write `./actions.json`."""

RESUME_PROMPT = """Read `./logs.txt` programmatically, especially the newest
action sections and any score, level, state, or plan-adherence changes. Check
any files you previously saved, revise your model if evidence disagreed with
it, then write a new `./actions.json`."""

RETRY_PROMPT = """Your prior response did not create a valid
`./actions.json`. Write a JSON object with a non-empty `actions` list using only
the documented action strings."""


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _last_agent_message(path: Path) -> str:
    message = ""
    for event in _iter_jsonl(path):
        item = event.get("item")
        if event.get("type") == "item.completed" and isinstance(item, Mapping) and item.get("type") == "agent_message":
            text = item.get("text")
            if isinstance(text, str) and text.strip():
                message = text.strip()
    return message


def _action_from_entry(entry: Any) -> dict[str, int | str] | None:
    if isinstance(entry, str):
        value = entry.strip().upper()
        click = ACTION6_RE.fullmatch(value)
        if click:
            x, y = int(click.group(1)), int(click.group(2))
            if 0 <= x < 64 and 0 <= y < 64:
                return {"name": "CLICK", "x": x, "y": y}
            return None
        mapped = ACTION_TO_PLAYER.get(value)
        return {"name": mapped} if mapped is not None else None
    if not isinstance(entry, Mapping):
        return None
    name = str(entry.get("action") or entry.get("name") or "").strip().upper()
    if name in PLAYER_TO_ACTION:
        return {"name": name}
    if name == "CLICK" or name == "ACTION6":
        try:
            x, y = int(entry["x"]), int(entry["y"])
        except (KeyError, TypeError, ValueError):
            return None
        if 0 <= x < 64 and 0 <= y < 64:
            return {"name": "CLICK", "x": x, "y": y}
    return None


def parse_actions_json(path: Path, *, action_cap: int) -> list[dict[str, int | str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    entries = payload.get("actions") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        return []
    actions: list[dict[str, int | str]] = []
    for entry in entries[:action_cap]:
        action = _action_from_entry(entry)
        if action is not None:
            actions.append(action)
    return actions


def format_action(action: Mapping[str, int | str]) -> str:
    name = str(action["name"]).upper()
    if name == "CLICK":
        return f"ACTION6({int(action['x'])},{int(action['y'])})"
    return PLAYER_TO_ACTION[name]


@dataclass(frozen=True)
class AnalyzerResult:
    actions: tuple[dict[str, int | str], ...]
    analysis: str
    exit_code: int
    command: tuple[str, ...]


class ActionQueue:
    def __init__(self) -> None:
        self._actions: deque[dict[str, int | str]] = deque()
        self.plan_size = 0
        self.plan_index = 0

    def __bool__(self) -> bool:
        return bool(self._actions)

    def __len__(self) -> int:
        return len(self._actions)

    def load(self, actions: Sequence[dict[str, int | str]]) -> None:
        self._actions = deque(dict(action) for action in actions)
        self.plan_size = len(self._actions)
        self.plan_index = 0

    def clear(self) -> None:
        self._actions.clear()
        self.plan_size = 0
        self.plan_index = 0

    def pop(self) -> dict[str, int | str]:
        self.plan_index += 1
        return self._actions.popleft()


class FullInteractionLog:
    def __init__(self, path: Path):
        self.path = path

    def _append(self, text: str) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(text.rstrip() + "\n\n")
            stream.flush()

    @staticmethod
    def _boards(grids: Sequence[Any]) -> str:
        encoded = [_encode_grid(grid) for grid in grids]
        if not encoded:
            return "(no board returned)"
        if len(encoded) == 1:
            return "\n".join(encoded[0])
        sections: list[str] = []
        for index, rows in enumerate(encoded[:-1], start=1):
            sections.extend((f"[frame {index}/{len(encoded) - 1}]", "\n".join(rows)))
        sections.extend(("[settled]", "\n".join(encoded[-1])))
        return "\n".join(sections)

    def write_initial(self, *, grid: Any, level: int, state: str) -> None:
        self._append(
            f"{'=' * 80}\n"
            f"Action 0 | Level {level + 1} | Attempt 1 | Levels completed: {level} | State: {state}\n\n"
            f"[INITIAL BOARD STATE]\n{self._boards([grid])}"
        )

    def write_action(
        self,
        *,
        action_number: int,
        level_before: int,
        level_after: int,
        state_before: str,
        state_after: str,
        action: Mapping[str, int | str],
        analysis: str,
        plan_index: int,
        plan_size: int,
        stopped_reason: str,
        automatic_resets: int,
        grids: Sequence[Any],
    ) -> None:
        analysis_block = f"[ANALYSIS]\n{analysis}\n\n" if analysis else ""
        self._append(
            f"{'=' * 80}\n"
            f"Action {action_number} | Level {level_after + 1} | Levels completed: {level_after} | "
            f"State: {state_after}\n\n"
            f"{analysis_block}"
            f"[PLAN STEP] {plan_index}/{plan_size}\n"
            f"Tool Call: {format_action(action)}\n"
            f"Outcome: level {level_before + 1}->{level_after + 1}; state {state_before}->{state_after}; "
            f"stop={stopped_reason}; automatic_resets={automatic_resets}\n\n"
            f"[POST-ACTION BOARD STATE]\n{self._boards(grids)}"
        )


def build_prolong_codex_command(
    *,
    codex_bin: str,
    workspace: Path,
    prompt: str,
    model: str | None,
    reasoning_effort: str | None,
    sandbox_mode: CodexSandbox,
    resume_thread_id: str | None = None,
) -> list[str]:
    command = [
        codex_bin,
        "--ask-for-approval",
        "never",
        "exec",
        "--cd",
        str(workspace),
        "--sandbox",
        sandbox_mode,
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--json",
        "-c",
        'web_search="disabled"',
    ]
    if sandbox_mode == "workspace-write":
        command.extend(["-c", "sandbox_workspace_write.network_access=false"])
    command.extend(
        [
            "-c",
            'shell_environment_policy.inherit="none"',
            "-c",
            (
                "shell_environment_policy.set="
                f'{{PATH="{_safe_command_path()}",HOME="{workspace}",PYTHONDONTWRITEBYTECODE="1"}}'
            ),
        ]
    )
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
    if resume_thread_id:
        command.extend(["resume", resume_thread_id])
    command.append(prompt)
    return command


class LocalCodexAnalyzer:
    def __init__(
        self,
        *,
        workspace: Path,
        events_path: Path,
        stderr_path: Path,
        codex_bin: str,
        model: str | None,
        reasoning_effort: str | None,
        sandbox_mode: CodexSandbox,
        action_cap: int,
        available_actions: Sequence[str],
    ):
        self.workspace = workspace
        self.events_path = events_path
        self.stderr_path = stderr_path
        self.codex_bin = shutil.which(codex_bin) or codex_bin
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.sandbox_mode = sandbox_mode
        self.action_cap = action_cap
        self.calls = 0
        self.exit_codes: list[int] = []
        self.last_command: list[str] = []
        actions_section = "\n".join(
            f"- {ACTION_DESCRIPTIONS[action]}" for action in available_actions if action in ACTION_DESCRIPTIONS
        )
        system_prompt = PRO_LONG_SYSTEM_PROMPT.format(action_cap=action_cap, actions_section=actions_section)
        (workspace / "AGENTS.md").write_text(system_prompt, encoding="utf-8")

    def analyze(self, *, deadline: float, retry: bool = False) -> AnalyzerResult:
        actions_path = self.workspace / "actions.json"
        actions_path.unlink(missing_ok=True)
        prompt = RETRY_PROMPT if retry else (INITIAL_PROMPT if self.calls == 0 else RESUME_PROMPT)
        thread_id = _thread_id_from_events(self.events_path)
        command = build_prolong_codex_command(
            codex_bin=self.codex_bin,
            workspace=self.workspace,
            prompt=prompt,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            sandbox_mode=self.sandbox_mode,
            resume_thread_id=thread_id,
        )
        self.last_command = command
        with (
            self.events_path.open("a", encoding="utf-8") as events,
            self.stderr_path.open("a", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                command,
                cwd=self.workspace,
                env=_codex_environment(self.workspace),
                stdin=subprocess.DEVNULL,
                stdout=events,
                stderr=stderr,
                text=True,
            )
            while process.poll() is None:
                if time.monotonic() >= deadline:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise TimeoutError("PRO-LONG analyzer exceeded the run deadline.")
                time.sleep(0.05)
            exit_code = int(process.wait())
        self.calls += 1
        self.exit_codes.append(exit_code)
        return AnalyzerResult(
            actions=tuple(parse_actions_json(actions_path, action_cap=self.action_cap)),
            analysis=_last_agent_message(self.events_path),
            exit_code=exit_code,
            command=tuple(command),
        )


def _progress_invalidated(*, level_before: int, result: Mapping[str, Any]) -> bool:
    try:
        level_after = int(result.get("level_after", level_before))
    except (TypeError, ValueError):
        level_after = level_before
    return level_after != level_before or str(result.get("stopped_reason")) not in {"sequence_completed"}


def _trace_last_decision(path: Path) -> dict[str, Any]:
    latest: dict[str, Any] = {}
    for record in _iter_jsonl(path):
        if record.get("type") == "decision":
            latest = record
    return latest


def _trace_executed_actions(path: Path) -> list[dict[str, int | str]]:
    actions: list[dict[str, int | str]] = []
    for record in _iter_jsonl(path):
        if record.get("type") != "decision":
            continue
        executed = record.get("executed_actions")
        if not isinstance(executed, list):
            continue
        for entry in executed:
            action = _action_from_entry(entry)
            if action is not None:
                actions.append(action)
    return actions


def _last_frame_hash(record: Mapping[str, Any]) -> str | None:
    observation = record.get("observation_after")
    if not isinstance(observation, Mapping):
        return None
    hashes = observation.get("frame_hashes")
    if not isinstance(hashes, list) or not hashes:
        return None
    value = hashes[-1]
    return str(value) if value else None


def _available_actions(env: Any) -> list[str]:
    by_id = {
        0: "RESET",
        1: "ACTION1",
        2: "ACTION2",
        3: "ACTION3",
        4: "ACTION4",
        5: "ACTION5",
        6: "ACTION6",
        7: "ACTION7",
    }
    actions = {"RESET"}
    for action in env.action_space:
        name = by_id.get(int(getattr(action, "value", -1)))
        if name is not None:
            actions.add(name)
    return sorted(actions, key=lambda value: (value == "RESET", value))


def run_prolong_player(
    *,
    game_id: str,
    seed: int = 0,
    output_root: Path = Path("runs/pro-long-solver"),
    run_id: str | None = None,
    resume_from: Path | None = None,
    codex_bin: str = "codex",
    model: str | None = DEFAULT_MODEL,
    reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
    sandbox_mode: CodexSandbox = "workspace-write",
    action_cap: int = 20,
    max_episode_actions: int = DEFAULT_PROLONG_MAX_ACTIONS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    analyzer_retries: int = 3,
    env_factory: Callable[[], Any] | None = None,
    fps: int | None = None,
) -> dict[str, Any]:
    if action_cap <= 0 or max_episode_actions <= 0 or analyzer_retries <= 0:
        raise ValueError("Action limits and analyzer retries must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive when provided.")

    resolved_run_id = run_id or _default_run_id(game_id)
    run_dir = output_root.resolve() / resolved_run_id
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    source_dir = resume_from.resolve() if resume_from is not None else None
    if source_dir is not None:
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {source_dir}")
        for relative in ("trace.jsonl", "codex.events.jsonl", "workspace", "workspace/logs.txt"):
            if not (source_dir / relative).exists():
                raise FileNotFoundError(f"Resume artifact is missing: {source_dir / relative}")
    run_dir.mkdir(parents=True)
    workspace = run_dir / "workspace"
    trace_path = run_dir / "trace.jsonl"
    events_path = run_dir / "codex.events.jsonl"
    stderr_path = run_dir / "codex.stderr.log"
    log_path = workspace / "logs.txt"
    if source_dir is None:
        workspace.mkdir()
        (workspace / "scratch").mkdir()
    else:
        shutil.copytree(source_dir / "workspace", workspace)
        shutil.copyfile(source_dir / "codex.events.jsonl", events_path)
        source_stderr = source_dir / "codex.stderr.log"
        if source_stderr.exists():
            shutil.copyfile(source_stderr, stderr_path)

    env = env_factory() if env_factory else EnvSampler(augment=False, seed=seed).make(game_id, seed=seed)
    session = ProlongGameSession(
        env=env,
        trace_path=trace_path,
        game_id=game_id,
        seed=seed,
        max_batch_actions=1,
        max_episode_actions=max_episode_actions,
    )
    game_log = FullInteractionLog(log_path)
    initial_frames = _frame_grids(session.observation)
    replayed_actions = 0
    if source_dir is None:
        game_log.write_initial(
            grid=initial_frames[-1], level=_level_index(session.observation), state=_state_name(session.observation)
        )
    else:
        source_trace = source_dir / "trace.jsonl"
        source_last_decision = _trace_last_decision(source_trace)
        prior_actions = _trace_executed_actions(source_trace)
        replayed_actions = len(prior_actions)
        for action in prior_actions:
            session.execute({"reason": "Replay prior action for deterministic resume", "actions": [action]})
        replay_last_decision = _trace_last_decision(trace_path)
        expected_hash = _last_frame_hash(source_last_decision)
        actual_hash = _last_frame_hash(replay_last_decision)
        if expected_hash is None or actual_hash != expected_hash:
            raise RuntimeError(
                "Resume replay did not reconstruct the prior final frame: "
                f"expected {expected_hash or '(missing)'}, got {actual_hash or '(missing)'}"
            )
    analyzer = LocalCodexAnalyzer(
        workspace=workspace,
        events_path=events_path,
        stderr_path=stderr_path,
        codex_bin=codex_bin,
        model=model,
        reasoning_effort=reasoning_effort,
        sandbox_mode=sandbox_mode,
        action_cap=action_cap,
        available_actions=_available_actions(env),
    )
    queue = ActionQueue()
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    deadline = started_monotonic + timeout_seconds
    termination_reason = "runner_error:analyzer_exited_without_actions"
    current_analysis = ""
    plans = 0
    invalidated_plans = 0

    try:
        while not session.finished and session.primitive_actions < max_episode_actions:
            if time.monotonic() >= deadline:
                termination_reason = "timeout"
                break
            if not queue:
                loaded = False
                for retry_index in range(analyzer_retries):
                    analyzer_result = analyzer.analyze(deadline=deadline, retry=retry_index > 0)
                    if analyzer_result.exit_code == 0 and analyzer_result.actions:
                        queue.load(analyzer_result.actions)
                        current_analysis = analyzer_result.analysis
                        plans += 1
                        loaded = True
                        break
                if not loaded:
                    termination_reason = "analyzer_failed"
                    break

            action = queue.pop()
            plan_index, plan_size = queue.plan_index, queue.plan_size
            level_before = _level_index(session.observation)
            state_before = _state_name(session.observation)
            result = session.execute(
                {"reason": f"PRO-LONG queued plan step {plan_index}/{plan_size}", "actions": [action]}
            )
            decision = _trace_last_decision(trace_path)
            level_after = int(decision.get("level_after", _level_index(session.observation)))
            state_after = str(decision.get("state_after", _state_name(session.observation)))
            transition_grids = session.last_transition_grids or _frame_grids(session.observation)
            grid_offset = 0
            primitive_events = decision.get("primitive_events")
            if not isinstance(primitive_events, list):
                primitive_events = []
            for event_index, event in enumerate(primitive_events):
                if not isinstance(event, Mapping):
                    continue
                frame_count = int(event.get("frame_count", 0) or 0)
                event_grids = transition_grids[grid_offset : grid_offset + frame_count]
                grid_offset += frame_count
                logged_action = event.get("action")
                if not isinstance(logged_action, Mapping):
                    logged_action = action
                game_log.write_action(
                    action_number=int(event.get("primitive_id", session.primitive_actions)),
                    level_before=int(event.get("level_before", level_before)),
                    level_after=int(event.get("level_after", level_after)),
                    state_before=str(event.get("state_before", state_before)),
                    state_after=str(event.get("state_after", state_after)),
                    action=logged_action,
                    analysis=current_analysis if plan_index == 1 and event_index == 0 else "",
                    plan_index=plan_index,
                    plan_size=plan_size,
                    stopped_reason=str(result["stopped_reason"]),
                    automatic_resets=int(bool(event.get("automatic", False))),
                    grids=event_grids or _frame_grids(session.observation),
                )
            if queue and _progress_invalidated(
                level_before=level_before, result={**result, "level_after": level_after}
            ):
                queue.clear()
                invalidated_plans += 1

        if session.finished:
            termination_reason = "win" if _state_name(session.observation) == "WIN" else "action_budget_exhausted"
        elif session.primitive_actions >= max_episode_actions:
            termination_reason = "action_budget_exhausted"
    except TimeoutError:
        termination_reason = "timeout"
    finally:
        exit_code = analyzer.exit_codes[-1] if analyzer.exit_codes else -1
        session.finish_trace(codex_exit_code=exit_code, termination_reason=termination_reason)

    token_report = build_token_usage_report(trace_path=trace_path, codex_events_path=events_path)
    _write_json(run_dir / "token_usage.json", token_report)
    result_payload = {
        "run_id": resolved_run_id,
        "resume_from": str(source_dir) if source_dir is not None else None,
        "replayed_actions": replayed_actions,
        "game_id": game_id,
        "seed": seed,
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": time.monotonic() - started_monotonic,
        "harness": "pro-long-v1",
        "model": model or DEFAULT_MODEL,
        "reasoning_effort": reasoning_effort or DEFAULT_REASONING_EFFORT,
        "termination_reason": termination_reason,
        "episode_ended": session.finished,
        "final_state": _state_name(session.observation),
        "primitive_actions": session.primitive_actions,
        "decisions": session.decision_id,
        "plans": plans,
        "invalidated_plans": invalidated_plans,
        "analyzer_calls": analyzer.calls,
        "analyzer_exit_codes": analyzer.exit_codes,
        "episode_reward": session.episode_reward,
        "token_usage": token_report["total"],
        "run_dir": str(run_dir),
    }
    _write_json(run_dir / "result.json", result_payload)
    manifest = {
        "version": 1,
        "harness": "pro-long-v1",
        "run_id": resolved_run_id,
        "resume_from": str(source_dir) if source_dir is not None else None,
        "replayed_actions": replayed_actions,
        "game_id": game_id,
        "seed": seed,
        "model": model or DEFAULT_MODEL,
        "reasoning_effort": reasoning_effort or DEFAULT_REASONING_EFFORT,
        "sandbox_mode": sandbox_mode,
        "action_cap": action_cap,
        "max_episode_actions": max_episode_actions,
        "timeout_seconds": timeout_seconds,
        "analyzer_retries": analyzer_retries,
        "prompt_sha256": hashlib.sha256((workspace / "AGENTS.md").read_bytes()).hexdigest(),
        "last_command": analyzer.last_command,
        "artifacts": {
            "interaction_log": "workspace/logs.txt",
            "trace": "trace.jsonl",
            "codex_events": "codex.events.jsonl",
            "codex_stderr": "codex.stderr.log",
            "token_usage": "token_usage.json",
            "workspace": "workspace",
            "result": "result.json",
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    return result_payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the local PRO-LONG solver against one RE-ARC game.")
    parser.add_argument("game_id")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=Path("runs/pro-long-solver"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    parser.add_argument("--codex-sandbox", choices=("workspace-write", "danger-full-access"), default="workspace-write")
    parser.add_argument("--action-cap", type=int, default=20)
    parser.add_argument("--max-episode-actions", type=int, default=DEFAULT_PROLONG_MAX_ACTIONS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--analyzer-retries", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run_prolong_player(
        game_id=args.game_id,
        seed=args.seed,
        output_root=args.output_root,
        run_id=args.run_id,
        resume_from=args.resume_from,
        codex_bin=args.codex_bin,
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        sandbox_mode=args.codex_sandbox,
        action_cap=args.action_cap,
        max_episode_actions=args.max_episode_actions,
        timeout_seconds=args.timeout_seconds,
        analyzer_retries=args.analyzer_retries,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["final_state"] == "WIN" else 1


if __name__ == "__main__":
    raise SystemExit(main())
