"""Platypus run data adapted to ARC3-Training's replay-viewer payload.

Platypus keeps the same information in ``benchmark.json`` plus append-only
span streams and a content-addressed block store, so this module translates
between the two formats without creating another run artifact.  Everything
read here is infrastructure-owned surface — ``infra/*`` spans, ``llm/turn``
conversation artifacts, frame artifacts — so any agent's runs render.  The
one opt-in extra is the action-linkage convention
(:func:`_tool_result_action_number`): a host tool result that reports the
action it executed gets its transcript card linked to that action's frame,
and cards without it fall back to the move's own frame.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional, Union, cast

from platypus.chunks import action_display
from platypus.diagnostics import (
    BlockStore,
    SpanNode,
    archive_parts,
    build_span_tree,
    game_run_ids,
    job_block_store,
    read_game_events,
)

_ARC_PALETTE = [
    "rgb(255, 255, 255)",
    "rgb(204, 204, 204)",
    "rgb(153, 153, 153)",
    "rgb(102, 102, 102)",
    "rgb(51, 51, 51)",
    "rgb(0, 0, 0)",
    "rgb(229, 58, 163)",
    "rgb(255, 123, 204)",
    "rgb(249, 60, 49)",
    "rgb(30, 147, 255)",
    "rgb(136, 216, 241)",
    "rgb(255, 220, 0)",
    "rgb(255, 133, 27)",
    "rgb(146, 18, 49)",
    "rgb(79, 204, 48)",
    "rgb(163, 86, 214)",
]
_DISCOVERY_TTL_S = 30.0
_MAX_DISCOVERY_DEPTH = 3
_PRUNED_DIRS = {
    ".git",
    ".venv",
    "artifacts",
    "diag_blocks",
    "games",
    "movies",
    "traces",
    "viewer",
}
# A foreground viewer serves one root, so this cache stays one entry in practice.
_DISCOVERY_CACHE: dict[Path, tuple[float, list[Path]]] = {}
_DISCOVERY_LOCK = threading.Lock()
_ACTION_RESULT_NUMBER = re.compile(r"['\"]action_num['\"]\s*:\s*(\d+)")
_EXECUTED_ACTION_RESULT = re.compile(r"['\"]executed['\"]\s*:\s*(?:true|True)")


def _is_replayable_run(path: Path) -> bool:
    """Whether ``path`` is a platypus run this adapter can replay.

    A run whose diagnostics were archived (``platypus.diagnostics``, "Scratch
    and archive") has its streams in the job dir's zip parts, every one of
    which holds the run stream, and may have no ``games/`` dir at all.

    A directory the probe cannot stat (shared trees hold other users'
    permission-denied dirs) is simply not replayable — never an error.
    """
    try:
        if (path / "games").is_dir() and ((path / "benchmark.json").is_file() or (path / "run_spans.jsonl").is_file()):
            return True
        return bool(archive_parts(path))
    except OSError:
        return False


def _discover_uncached(root: Path) -> list[Path]:
    """Find replayable runs beneath ``root`` without descending into run payloads."""
    if _is_replayable_run(root):
        return [root]
    found: list[Path] = []

    def visit(directory: Path, depth: int) -> None:
        """Visit one shallow container in the experiment/run hierarchy."""
        if depth > _MAX_DISCOVERY_DEPTH:
            return
        try:
            children = list(directory.iterdir())
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            return
        for child in children:
            if child.name in _PRUNED_DIRS or child.name.startswith("."):
                continue
            try:
                is_dir = child.is_dir()
            except OSError:
                continue
            if not is_dir:
                continue
            if _is_replayable_run(child):
                found.append(child)
            else:
                visit(child, depth + 1)

    visit(root, 1)

    def recent_first(path: Path) -> tuple[int, str]:
        """Sort finished and live runs by benchmark freshness, then path."""
        marker = path / "benchmark.json"
        if not marker.is_file():
            marker = path / "run_spans.jsonl"
        if not marker.is_file():
            marker = next(reversed(archive_parts(path)), marker)
        try:
            mtime = marker.stat().st_mtime_ns
        except OSError:
            mtime = 0
        return (-mtime, str(path))

    return sorted(found, key=recent_first)


def list_run_dirs(root: Union[str, Path]) -> list[Path]:
    """Return discoverable runs, newest first, with a short TTL cache.

    Caching is operationally important: the viewer runs on a login node and
    the run root is commonly NFS.  Browser refreshes must not turn into a
    repeated metadata walk.
    """
    resolved = Path(root).resolve()
    now = time.monotonic()
    with _DISCOVERY_LOCK:
        cached = _DISCOVERY_CACHE.get(resolved)
        if cached is not None and now - cached[0] < _DISCOVERY_TTL_S:
            return list(cached[1])
        runs = _discover_uncached(resolved)
        _DISCOVERY_CACHE[resolved] = (now, runs)
        return list(runs)


def _run_label(root: Path, run: Path) -> str:
    """Stable selector label for a discovered run."""
    return run.name if run == root else run.relative_to(root).as_posix()


def resolve_run_dir(root: Union[str, Path], requested: Optional[str] = None) -> Path:
    """Resolve a viewer run selector only to a discovered run under ``root``."""
    resolved_root = Path(root).resolve()
    runs = list_run_dirs(resolved_root)
    if not runs:
        raise FileNotFoundError(f"No replayable runs found under {resolved_root}")
    requested = (requested or "").strip()
    if not requested:
        return runs[0]
    matches = {_run_label(resolved_root, run): run for run in runs}
    if requested not in matches:
        raise FileNotFoundError(f"Run {requested!r} not found under {resolved_root}")
    return matches[requested]


_FINISHED_LOCK = threading.Lock()
#: path -> (mtime_ns, size, finished). Keyed on what a rewrite changes, so a
#: file that has moved on is re-read and one that has not is answered from
#: the stat alone.
_FINISHED_CACHE: dict[str, tuple[int, int, bool]] = {}


def benchmark_finished(run_dir: Union[str, Path]) -> bool:
    """Whether the run's benchmark has ENDED, not merely been written.

    TAAF saves ``benchmark.json`` periodically from the first save on and
    stamps ``end_time`` immediately before the final one, so the file's
    existence says a run started and only the stamp says it stopped.

    One home for the rule because it had two: a listing that calls this per
    run dir per request, and a collector that decides from it whether a job
    can be read. Two copies of a predicate drift, and these already had.

    Answered from a ``(mtime, size)`` memo, since the file is multi-MB and
    both callers ask repeatedly — a listing on every request, a watch loop on
    every poll. A file mid-rewrite reads as unfinished and is re-read once it
    settles.
    """
    path = Path(run_dir) / "benchmark.json"
    try:
        stat = path.stat()
    except OSError:
        return False
    key = str(path)
    stamp = (stat.st_mtime_ns, stat.st_size)
    with _FINISHED_LOCK:
        cached = _FINISHED_CACHE.get(key)
        if cached is not None and cached[:2] == stamp:
            return cached[2]
    try:
        finished = json.loads(path.read_text()).get("end_time") is not None
    except (OSError, ValueError):
        return False  # mid-rewrite or unreadable: not cached, so the next call re-reads
    with _FINISHED_LOCK:
        _FINISHED_CACHE[key] = (*stamp, finished)
    return finished


def _benchmark(run_dir: Path) -> dict[str, Any]:
    """Read and validate the run's lightweight benchmark summary."""
    benchmark_path = run_dir / "benchmark.json"
    if not benchmark_path.is_file():
        return _live_benchmark(run_dir)
    decoded: Any = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError(f"benchmark is not an object: {run_dir / 'benchmark.json'}")
    payload = cast(dict[str, Any], decoded)
    raw_games: Any = payload.get("game_runs")
    if not isinstance(raw_games, list):
        raise ValueError(f"benchmark has no game_runs list: {run_dir / 'benchmark.json'}")
    games: list[dict[str, Any]] = []
    for index, raw_game in enumerate(cast(list[Any], raw_games)):
        if not isinstance(raw_game, dict):
            raise ValueError(f"Game index {index} is not an object in {run_dir / 'benchmark.json'}")
        games.append(cast(dict[str, Any], raw_game))
    payload["game_runs"] = games
    return payload


def _live_benchmark(run_dir: Path) -> dict[str, Any]:
    """Synthesize the summary absent until an active benchmark finishes."""
    records: list[dict[str, Any]] = []
    try:
        run_ids = game_run_ids(run_dir)
    except (NotADirectoryError, PermissionError):
        run_ids = []
    for run_id in run_ids:
        record = _live_game_record(run_dir, run_id)
        if record is not None:
            records.append(record)
    records.sort(key=lambda record: (str(record["game_id"]), str(record["run_id"])))
    return {"label": run_dir.name, "game_runs": records}


def _live_game_record(run_dir: Path, run_id: str) -> Optional[dict[str, Any]]:
    """Build temporary benchmark fields from one append-only game stream."""
    events = read_game_events(run_dir, run_id)
    roots = build_span_tree(events)
    root = next((node for node in roots if node.kind == "infra/game"), None)
    if root is None:
        return None
    metadata = _live_game_metadata(run_dir, root)
    moves = [node for node in root.children if node.kind == "infra/move"]
    actions = [action for move in moves for action in _action_events(move)]
    status = str(actions[-1].get("outcome") or "playing") if actions else "playing"
    levels_completed = max((int(action.get("levels_completed") or 0) for action in actions), default=0)
    action_counts: dict[int, int] = {}
    for action in actions:
        level = max(0, int(action.get("level", action.get("levels_completed", 0)) or 0))
        action_counts[level] = action_counts.get(level, 0) + 1
    last_level = max(action_counts, default=-1)
    actions_per_level = [action_counts.get(level, 0) for level in range(last_level + 1)]
    event_times = [float(event["t"]) for event in events if isinstance(event.get("t"), (int, float))]
    elapsed_s = max(event_times, default=root.t_open) - root.t_open
    return {
        "game_id": root.name,
        "run_id": run_id,
        "state": status,
        "number_of_levels": metadata.get("number_of_levels"),
        "levels_completed": levels_completed,
        "actions_per_level": actions_per_level,
        "base_actions_per_level": metadata.get("base_actions_per_level"),
        "final_score": None,
        "history": [{"action": action} for action in actions],
        "final_generated_tokens": root.incl_generated,
        "_live_elapsed_s": max(0.0, elapsed_s),
    }


def _live_game_metadata(run_dir: Path, root: SpanNode) -> dict[str, Any]:
    """Load game facts emitted before the first move of a live run."""
    artifact = next((item for item in reversed(root.artifacts) if item["name"] == "game_metadata"), None)
    if artifact is None:
        return {}
    try:
        payload = job_block_store(run_dir).load(artifact["block"], artifact["codec"])
    except (FileNotFoundError, OSError, ValueError):
        return {}
    return cast(dict[str, Any], payload) if isinstance(payload, dict) else {}


def _arc_score(
    levels_completed: int,
    actions_per_level: list[int],
    base_actions_per_level: list[int],
) -> Optional[float]:
    """Return the official score at one point in a game, if baselines exist."""
    if not base_actions_per_level:
        return None
    total_score = 0.0
    total_weights = 0
    max_weights = 0
    for level_index, baseline in enumerate(base_actions_per_level):
        weight = level_index + 1
        total_weights += weight
        actions = actions_per_level[level_index] if level_index < len(actions_per_level) else 0
        if level_index < levels_completed and actions > 0:
            level_score = min(115.0, (baseline / actions) ** 2 * 100)
        else:
            level_score = 0.0
        if level_score > 0:
            max_weights += weight
        total_score += level_score * weight
    return min(total_score / total_weights, max_weights / total_weights * 100)


def _integer_list(value: Any) -> list[int]:
    """Convert an untyped JSON list to integers, or return an empty list."""
    if not isinstance(value, list):
        return []
    try:
        return [int(item) for item in cast(list[Any], value)]
    except (TypeError, ValueError):
        return []


def _game_summary(game: dict[str, Any]) -> dict[str, Any]:
    """Translate one benchmark game record without reading its span stream."""
    raw_history: Any = game.get("history")
    history: list[Any] = cast(list[Any], raw_history) if isinstance(raw_history, list) else []
    actions_per_level = _integer_list(game.get("actions_per_level"))
    base_actions_per_level = _integer_list(game.get("base_actions_per_level"))
    run_id = str(game.get("run_id") or "")
    game_id = str(game.get("game_id") or run_id)
    levels_completed = int(game.get("levels_completed") or 0)
    total_levels = int(game.get("number_of_levels") or 0)
    status = str(game.get("state") or "unknown")
    final_score = game.get("final_score")
    arc_score = _arc_score(levels_completed, actions_per_level, base_actions_per_level)
    if arc_score is None:
        arc_score = final_score
    generated_tokens = int(game.get("final_generated_tokens") or 0)
    for record in history:
        if isinstance(record, dict):
            generated_tokens += int(cast(dict[str, Any], record).get("generated_tokens") or 0)
    rollout_return = float(levels_completed) / max(1, total_levels)
    summary: dict[str, Any] = {
        "game_id": game_id,
        "display_name": game_id,
        "status": status,
        "eventCount": len(history),
        "levels_completed": levels_completed,
        "total_levels": total_levels,
        "actions_per_level": list(actions_per_level),
        "final_score": final_score,
        "final_arc_score": final_score,
        "arc_agi_score": arc_score,
        "rollout_return": rollout_return,
        "generated_tokens": generated_tokens,
        "lastEvent": {
            "type": "result",
            "status": status,
            "action_num": len(history),
            "score": arc_score,
        },
    }
    return summary


def load_run_summary(root: Union[str, Path], requested: Optional[str] = None) -> dict[str, Any]:
    """Load the interactive viewer's run-picker and game summaries."""
    resolved_root = Path(root).resolve()
    run_dir = resolve_run_dir(resolved_root, requested)
    runs = list_run_dirs(resolved_root)
    label = _run_label(resolved_root, run_dir)
    benchmark = _benchmark(run_dir)
    games = cast(list[dict[str, Any]], benchmark["game_runs"])
    return {
        "run_name": label,
        "selected_run": label,
        "available_runs": [_run_label(resolved_root, run) for run in runs],
        "source": "platypus-spans",
        "arc_palette": _ARC_PALETTE,
        "games": [_game_summary(game) for game in games],
    }


def _game_record(run_dir: Path, game_index: int) -> dict[str, Any]:
    """Benchmark record at ``game_index``, with consistent missing-index errors."""
    games = cast(list[dict[str, Any]], _benchmark(run_dir)["game_runs"])
    if game_index < 0 or game_index >= len(games):
        raise FileNotFoundError(f"Game index {game_index} not found in {run_dir}")
    return games[game_index]


def _descendants(node: SpanNode) -> list[SpanNode]:
    """Return descendants in emission order."""
    out: list[SpanNode] = []
    for child in node.children:
        out.append(child)
        out.extend(_descendants(child))
    return out


def _artifact(node: SpanNode, name: str) -> Optional[dict[str, Any]]:
    """The last artifact named ``name`` on a span."""
    return next((artifact for artifact in reversed(node.artifacts) if artifact["name"] == name), None)


def _frame(store: BlockStore, node: SpanNode, name: str) -> Optional[list[list[int]]]:
    """Decode one PNG-grid artifact into the JSON board representation."""
    artifact = _artifact(node, name)
    if artifact is None:
        return None
    try:
        grid = store.load(artifact["block"], artifact["codec"])
    except (FileNotFoundError, OSError, ValueError):
        return None
    return grid.tolist()


def _action_events(move: SpanNode) -> list[dict[str, Any]]:
    """Action value maps emitted by one move."""
    return [event["values"] for event in move.events if event["kind"] == "infra/action"]


def _action_event_times(move: SpanNode) -> list[Optional[float]]:
    """Wall-clock timestamps parallel to one move's action values."""
    times: list[Optional[float]] = []
    for event in move.events:
        if event["kind"] != "infra/action":
            continue
        raw_time = event.get("t")
        times.append(float(raw_time) if isinstance(raw_time, (int, float)) else None)
    return times


def _action_level(action: dict[str, Any]) -> int:
    """One-based level containing an action and its immediate result frame."""
    return int(action.get("level", action.get("levels_completed", 0))) + 1


def _turn_nodes(move: SpanNode) -> list[SpanNode]:
    """All LLM turns nested beneath one move, in emission order."""
    return [node for node in _descendants(move) if node.kind == "llm/turn"]


def _turn_conversation_artifact(turn: SpanNode) -> Optional[dict[str, Any]]:
    """The captured conversation for one LLM turn, when present."""
    artifact = _artifact(turn, "conversation")
    return artifact if artifact is not None and artifact.get("codec") == "json" else None


def _span_elapsed_s(root: SpanNode, node: SpanNode) -> float:
    """Elapsed game time through the latest event represented by ``node``."""
    end_time = node.t_open + node.duration_s if node.duration_s is not None else node.t_open
    for event in node.events:
        raw_time = event.get("t")
        if isinstance(raw_time, (int, float)):
            end_time = max(end_time, float(raw_time))
    for child in node.children:
        end_time = max(end_time, root.t_open + _span_elapsed_s(root, child))
    return max(0.0, end_time - root.t_open)


def _is_completed_response_event(event: dict[str, Any]) -> bool:
    """Whether one diagnostic event represents a completed backend response.

    Cancellation can leave an empty ``LLMResponse`` shell in the conversation:
    it has no usage, stop reason, tool call, or synthetic marker.  Recording
    that shell is useful crash-tail evidence, but counting it as a response
    makes the viewer disagree with the transcript and backend traffic.
    """
    if event.get("kind") != "llm/response":
        return False
    raw_values = event.get("values")
    if not isinstance(raw_values, dict):
        return False
    values = cast(dict[str, Any], raw_values)
    if values.get("synthetic"):
        return False
    return any(
        (
            values.get("completion_tokens"),
            values.get("uncached_input_tokens"),
            values.get("cached_input_tokens"),
            values.get("stop_reason") is not None,
            values.get("n_tool_calls"),
        )
    )


def _text(content: Any, store: BlockStore) -> str:
    """Render canonical ULLM content as trace text, resolving block refs."""
    if not isinstance(content, dict):
        return str(content or "")
    content_object = cast(dict[str, Any], content)
    kind = str(content_object.get("c") or "")
    if kind == "str":
        value: Any = content_object.get("text", "")
        if isinstance(value, dict) and "$block" in value:
            reference = cast(dict[str, Any], value)
            try:
                return str(store.load(str(reference["$block"]), "json"))
            except (FileNotFoundError, OSError, ValueError):
                return f"[missing text block {reference['$block']}]"
        return str(cast(object, value))
    if kind == "list":
        return "\n".join(filter(None, (_text(item, store) for item in _items(content_object.get("elements")))))
    if kind == "image":
        return f"[image block {content_object.get('block', 'unknown')}]"
    if kind == "tool_call":
        return _tool_call_text(content_object)
    return ""


def _items(value: Any) -> list[Any]:
    """A JSON array narrowed at the untyped decode boundary."""
    return cast(list[Any], value) if isinstance(value, list) else []


def _objects(value: Any) -> list[dict[str, Any]]:
    """Object entries from a JSON array, with their value types retained."""
    return [cast(dict[str, Any], item) for item in _items(value) if isinstance(item, dict)]


def _tool_call_text(content: dict[str, Any]) -> str:
    """Readable tool-call body compatible with the imported trace cards."""
    name = str(content.get("tool_name") or "tool")
    arguments: Any = content.get("arguments")
    if not isinstance(arguments, dict):
        return str(arguments or "")
    argument_object = cast(dict[str, Any], arguments)
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in argument_object.items():
        rendered = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False)
        lines.extend((f"<parameter={key}>", rendered, "</parameter>"))
    lines.extend(("</function>", "</tool_call>"))
    return "\n".join(lines)


def _content_sections(
    content: Any, store: BlockStore, *, default_label: str, in_context: bool = True
) -> list[dict[str, Any]]:
    """Turn response content into assistant and tool-call sections."""
    content_object = cast(dict[str, Any], content) if isinstance(content, dict) else None
    if content_object is not None and content_object.get("c") == "list":
        sections: list[dict[str, Any]] = []
        for item in _items(content_object.get("elements")):
            sections.extend(_content_sections(item, store, default_label=default_label, in_context=in_context))
        return sections
    if content_object is not None and content_object.get("c") == "tool_call":
        is_tool = True
        label = f"TOOL CALL: {content_object.get('tool_name', 'tool')}"
    else:
        is_tool = False
        label = default_label
    text = _text(content, store).strip()
    if not text:
        return []
    return [{"label": label, "content": text, "kind": "tool" if is_tool else "reasoning", "inContext": in_context}]


def _element_sections(element: dict[str, Any], store: BlockStore) -> list[dict[str, Any]]:
    """Convert one canonical ULLM conversation element to viewer cards."""
    kind = element.get("kind")
    if kind == "user":
        text = _text(element.get("content"), store).strip()
        return [] if not text else [{"label": "USER PROMPT", "content": text, "kind": "meta", "inContext": True}]
    if kind == "tool_result":
        text = _text(element.get("content"), store).strip()
        label = f"TOOL RESULT: {element.get('tool_name', 'tool')}"
        if not text:
            return []
        section: dict[str, Any] = {"label": label, "content": text, "kind": "tool", "inContext": True}
        action_number = _tool_result_action_number(text)
        if action_number is not None:
            section["action_num"] = action_number
        return [section]
    if kind == "response":
        in_context = not bool(element.get("do_not_send_to_agent"))
        sections: list[dict[str, Any]] = []
        injected = element.get("response_injection")
        if isinstance(injected, str) and injected:
            sections.append({"label": "INJECTED PREFIX", "content": injected, "kind": "meta", "inContext": in_context})
        sections.extend(
            _content_sections(element.get("reasoning"), store, default_label="THINKING", in_context=in_context)
        )
        sections.extend(
            _content_sections(element.get("final"), store, default_label="ASSISTANT", in_context=in_context)
        )
        usage: Any = element.get("usage")
        if isinstance(usage, dict):
            _assign_generated_tokens(sections, cast(dict[str, Any], usage).get("completion_tokens"))
        return sections
    if kind == "turn":
        sections: list[dict[str, Any]] = []
        for child in _objects(element.get("elements")):
            sections.extend(_element_sections(child, store))
        return sections
    return []


def _tool_result_action_number(text: str) -> Optional[int]:
    """Executed action number explicitly reported by a host tool result."""
    if _EXECUTED_ACTION_RESULT.search(text) is None:
        return None
    match = _ACTION_RESULT_NUMBER.search(text)
    return int(match.group(1)) if match is not None else None


def _assign_generated_tokens(sections: list[dict[str, Any]], raw_total: Any) -> None:
    """Distribute one response's exact completion tokens across its visible cards.

    ULLM bills one total for reasoning, assistant text, and tool calls together.
    The viewer needs card-level prefixes, so allocation follows each card's
    visible word weight while preserving the exact response total.
    """
    try:
        total = max(0, int(raw_total))
    except (TypeError, ValueError):
        return
    if not sections:
        return
    weights = [max(1, len(str(section.get("content") or "").split())) for section in sections]
    weight_total = sum(weights)
    previous = 0
    cumulative = 0
    for section, weight in zip(sections, weights, strict=True):
        cumulative += weight
        boundary = round(total * cumulative / weight_total)
        section["generated_tokens"] = boundary - previous
        previous = boundary


def _retained_sections(elements: list[dict[str, Any]], store: BlockStore) -> list[dict[str, Any]]:
    """Cards for messages a compaction carried into a rebuilt context,
    flagged ``retained``: the turn showing them did not generate them."""
    sections: list[dict[str, Any]] = []
    for element in elements:
        for section in _element_sections(element, store):
            # Its tokens and actions belong to the turn that generated it.
            section.pop("generated_tokens", None)
            section.pop("action_num", None)
            section["retained"] = True
            sections.append(section)
    return sections


def _local_conversation_sections(
    conversation: dict[str, Any], store: BlockStore, *, include_system: bool, rebuilt: bool = False
) -> list[dict[str, Any]]:
    """Build the current turn's transcript cards from a captured request.

    Earlier messages were shown with the turns that produced them, so only
    the current turn is new — except in the first request of a context a
    compaction ``rebuilt``: the messages carried in ahead of the current
    turn (the summary, the retained rounds) appear nowhere else, so they are
    shown too, flagged ``retained`` because this turn did not generate them.
    """
    sections: list[dict[str, Any]] = []
    system: Any = conversation.get("system")
    if include_system and isinstance(system, dict):
        text = _text(cast(dict[str, Any], system).get("content"), store).strip()
        if text:
            sections.append({"label": "SYSTEM PROMPT", "content": text, "kind": "meta", "inContext": True})
    elements = _objects(conversation.get("elements"))
    if elements:
        last_turn = max(
            (index for index, element in enumerate(elements) if element.get("kind") == "turn"),
            default=len(elements) - 1,
        )
        start = last_turn
        if start > 0 and elements[start - 1].get("kind") == "user":
            start -= 1
        if rebuilt:
            sections.extend(_retained_sections(elements[:start], store))
        elements = elements[start : last_turn + 1]
    for element in elements:
        sections.extend(_element_sections(element, store))
    return sections


_COMPACTION_TURN_NAME = "compaction"
_COMPACTION_DISCARD_REASON = "context_compaction"


def _system_prompt_section(conversation: dict[str, Any], store: BlockStore) -> Optional[dict[str, Any]]:
    """The SYSTEM PROMPT card of a captured request, or None when it has none."""
    system: Any = conversation.get("system")
    if not isinstance(system, dict):
        return None
    system_object = cast(dict[str, Any], system)
    text = _text(system_object.get("content"), store).strip()
    if not text:
        return None
    return {
        "label": "SYSTEM PROMPT",
        "content": text,
        "kind": "meta",
        "inContext": True,
        # What the chat template renders ahead of the text.
        "tools": system_object.get("tools"),
        "reasoning_effort": system_object.get("reasoning_effort"),
    }


def _system_prompt_text(conversation: dict[str, Any], store: BlockStore) -> str:
    """Extract the system text used to recognize a rebuilt context."""
    section = _system_prompt_section(conversation, store)
    return str(section["content"]) if section is not None else ""


def _compaction_turns(turn: SpanNode, all_turns: list[SpanNode]) -> list[SpanNode]:
    """The compaction requests nested under one analyzer turn, in order."""
    return [
        child
        for child in _descendants(turn)
        if child.kind == "llm/turn" and child.name == _COMPACTION_TURN_NAME and child in all_turns
    ]


def _summary_compaction_outcomes(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Match each installed summary to its attempt using stream order.

    Success is emitted on the analyzer after the accepted child turn closes.
    Its merged kv cannot distinguish retries or multiple installs in one turn.
    """
    parents: dict[int, Optional[int]] = {}
    attempts: set[int] = set()
    latest: dict[int, int] = {}
    installed: dict[int, dict[str, Any]] = {}
    for record in records:
        if record["e"] == "open":
            parents[record["id"]] = record["parent"]
            if record["kind"] == "llm/turn" and record["name"] == _COMPACTION_TURN_NAME:
                attempts.add(record["id"])
        elif record["e"] == "close" and record["id"] in attempts:
            parent = parents.get(record["id"])
            while parent is not None:
                latest[parent] = record["id"]
                parent = parents.get(parent)
        elif record["e"] == "kv":
            values = record["values"]
            if values.get("compacted") and values.get("compaction_strategy") == "summarize":
                attempt = latest.pop(record["span"], None)
                if attempt is not None:
                    installed[attempt] = values
    return installed


def _discarded_compaction_rounds(turn: SpanNode, store: BlockStore) -> int:
    """How many oldest rounds of ``turn`` a compaction removed from the live prompt."""
    artifact = _artifact(turn, "discarded_rounds")
    if artifact is None or artifact.get("codec") != "json":
        return 0
    value: Any = store.load(artifact["block"], "json")
    if not isinstance(value, dict):
        return 0
    records = _objects(cast(dict[str, Any], value).get("records"))
    return sum(1 for record in records if record.get("reason") == _COMPACTION_DISCARD_REASON)


def _compaction_request_sections(
    conversation: dict[str, Any], store: BlockStore, *, include_previous_block: bool, rebuilt: bool = False
) -> list[dict[str, Any]]:
    """The cards of one compaction request: the rounds of the current move
    that the request folded in (when the live prompt had any), then the
    request itself and the model's reply (thinking + the memory call).
    A request that is the first in a context a compaction ``rebuilt`` (a
    second compaction in one turn) opens with what the rebuild carried in,
    as :func:`_local_conversation_sections` does."""
    elements = _objects(conversation.get("elements"))
    if not elements:
        return []
    user_indexes = [index for index, element in enumerate(elements) if element.get("kind") == "user"]
    if not user_indexes:
        return []
    start = user_indexes[-1]
    if include_previous_block and len(user_indexes) >= 2:
        start = user_indexes[-2]
    request_index = user_indexes[-1]
    sections = _retained_sections(elements[:start], store) if rebuilt else []
    for index in range(start, len(elements)):
        element_sections = _element_sections(elements[index], store)
        if index == request_index:
            for section in element_sections:
                if section.get("label") == "USER PROMPT":
                    section["label"] = "COMPACTION REQUEST"
        if index > request_index:
            # The memory tool's acknowledgement is recorded, but the context
            # is rebuilt before the model could read it: not a card.
            element_sections = [s for s in element_sections if s.get("label") != "TOOL RESULT: memory"]
        sections.extend(element_sections)
    return sections


def _boundary_section(number: int, kv: dict[str, Any]) -> dict[str, Any]:
    """The divider card between the context a compaction closed and the
    rebuilt one, carrying the compaction's outcome from the turn's kv."""
    before = kv.get("context_before")
    after = kv.get("context_after")
    compaction: dict[str, Any] = {"number": number, "tokens_before": before, "tokens_after": after}
    if kv.get("compaction_strategy") == "summarize":
        # The agent emits this outcome only after installing the summary;
        # this strategy never calls the legacy memory-dump tool.
        compaction["mode"] = "summarize"
        outcome = "summary installed"
        detail = "The fresh context below contains the summary and retained messages."
    else:
        ok = bool(kv.get("compaction_dump_ok", False))
        reason = str(kv.get("compaction_dump_reason") or "")
        compaction.update(dump_ok=ok, reason=reason)
        outcome = "notes saved" if ok else f"dump FAILED ({reason or 'no memory call'}); previous notes kept"
        detail = "Everything above was discarded from the model's view; below is the fresh context."
    sizes = ""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        sizes = f" · context {int(before):,} → {int(after):,} tokens"
    elif isinstance(after, (int, float)):
        sizes = f" · context after: {int(after):,} tokens"
    return {
        "label": f"COMPACTION #{number}",
        "content": f"Context rebuilt after compaction {number}: {outcome}{sizes}. {detail}",
        "kind": "boundary",
        "inContext": False,
        "compaction": compaction,
    }


class _ContextTracker:
    """Numbers the contexts (segments between compactions or level
    rebuilds) of one game and stamps every transcript card with its own."""

    def __init__(self) -> None:
        self.index = 0
        self.last_system: Optional[str] = None
        self.last_level: Optional[int] = None
        self.compactions = 0
        self.responses = 0
        self.contexts: list[dict[str, Any]] = [{"index": 0, "opened_by": "start"}]

    def stamp(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Associate each transcript card with the context containing it."""
        for section in sections:
            section["context"] = self.index
        return sections

    def close(self, *, reason: str, move_index: int, level: int, boundary: Optional[dict[str, Any]]) -> None:
        """Record a context boundary and start the following segment."""
        current = self.contexts[-1]
        current["closed_by"] = reason
        current["closed_at_move"] = move_index
        if boundary is not None:
            current["compaction"] = boundary["compaction"]
        self.index += 1
        self.contexts.append({"index": self.index, "opened_by": reason, "opened_at_move": move_index, "level": level})
        self.last_system = None


def _journal_evicted(values: dict[str, Any]) -> bool:
    """Ignore budget overruns that removed no history, including after earlier pruning."""
    if not values.get("compacted") or "blocks_evicted" not in values:
        return False
    evicted = values.get("history_evicted")
    if isinstance(evicted, bool):
        return evicted
    if values.get("blocks_evicted", 0) > 0:
        return True
    # Older diagnostics count pruned rounds across the entire turn. Equal
    # before/after sizes identify a no-op even when that count is positive.
    before, after = values.get("context_before"), values.get("context_after")
    return values.get("rounds_pruned", 0) > 0 and (before is None or before != after)


def _journal_elements(turn: SpanNode, conversation: dict[str, Any], store: BlockStore) -> list[dict[str, Any]]:
    """Restore oldest pruned rounds before the surviving current-turn elements."""
    elements = _objects(conversation.get("elements"))
    turns = [element for element in elements if element.get("kind") == "turn"]
    if not turns:
        return []
    restored: list[dict[str, Any]] = []
    artifact = _artifact(turn, "discarded_rounds")
    if artifact is not None and artifact.get("codec") == "json":
        discarded: Any = store.load(artifact["block"], "json")
        if isinstance(discarded, dict):
            records = _objects(discarded.get("records"))
            dropped_turns = _objects(discarded.get("conversation", {}).get("elements"))
            for record, dropped in zip(records, dropped_turns):
                if record.get("reason") == "context_compaction":
                    restored.extend(_objects(dropped.get("elements")))
    return restored + _objects(turns[-1].get("elements"))


def _journal_boundary(number: int, values: dict[str, Any], *, located: bool, pending: bool) -> dict[str, Any]:
    """Describe automatic eviction without implying that a summary was generated."""
    before, after = values.get("context_before"), values.get("context_after")
    size = f"Context after compaction: {int(after):,} tokens." if isinstance(after, (int, float)) else ""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        size = f"Context: {int(before):,} → {int(after):,} tokens."
    detail = "Older history was evicted; retained history and the persistent journal remain available. No summary was generated."
    if pending:
        detail += " The current turn's transcript is still being recorded."
    elif not located:
        detail += " This recording does not identify the exact response boundary; the marker follows the affected turn."
    return {
        "label": f"COMPACTION #{number}",
        "content": f"Automatic journal compaction. {size} {detail}",
        "kind": "boundary",
        "inContext": False,
        "compaction": {
            "number": number,
            "mode": "journal",
            "tokens_before": before,
            "tokens_after": after,
            "boundary_located": located,
            "pending_transcript": pending,
        },
    }


def _journal_response_positions(turn: SpanNode, visible_count: int) -> tuple[list[int], int]:
    """Map visible responses onto hook ordinals, including discarded cap retries."""
    events = [event for event in turn.events if event.get("kind") == "llm/response"]
    if not events:
        return list(range(visible_count)), visible_count
    positions = [
        i
        for i, event in enumerate(events)
        if event.get("values", {}).get("discarded_reason") in (None, "context_compaction")
    ]
    return positions, len(events)


def _journal_turn_sections(
    tracker: _ContextTracker,
    turn: SpanNode,
    conversation: dict[str, Any],
    updates: list[dict[str, Any]],
    store: BlockStore,
    *,
    move_index: int,
    level: int,
    pending: bool,
    response_offset: int,
) -> list[dict[str, Any]]:
    """Place each journal boundary before its first response, preserving evicted rounds."""
    elements = _journal_elements(turn, conversation, store)
    responses = [element for element in elements if element.get("kind") == "response"]
    response_positions, total_responses = _journal_response_positions(turn, len(responses))
    boundaries: dict[int, list[tuple[dict[str, Any], bool]]] = {}
    previous_position = 0
    for values in updates:
        before = values.get("compaction_before_response")
        ordinal = int(before) - response_offset - 1 if isinstance(before, int) else -1
        position = sum(p < ordinal for p in response_positions)
        located = (
            0 <= ordinal <= total_responses
            and len(response_positions) == len(responses)
            and previous_position <= position
        )
        if not located:
            # Older captures have no round ordinal. Exact served token counts
            # identify a boundary only when there is one unambiguous match.
            candidates = [
                i
                for i, response in enumerate(responses)
                if i >= previous_position
                and isinstance(response.get("usage"), dict)
                and response["usage"].get("prompt_tokens") == values.get("context_after")
            ]
            located = len(candidates) == 1
            position = candidates[0] if located else len(responses)
        boundaries.setdefault(position, []).append((values, located))
        previous_position = position
    sections: list[dict[str, Any]] = []
    user_elements = [e for e in _objects(conversation.get("elements")) if e.get("kind") == "user"]
    user = user_elements[-1] if user_elements else None
    if tracker.last_system is None:
        system = _system_prompt_section(conversation, store)
        if system is not None:
            sections.extend(tracker.stamp([system]))
        tracker.last_system = _system_prompt_text(conversation, store)
    if user is not None:
        sections.extend(tracker.stamp(_element_sections(user, store)))

    def emit_boundaries(position: int) -> None:
        for values, located in boundaries.get(position, []):
            tracker.compactions += 1
            boundary = _journal_boundary(tracker.compactions, values, located=located, pending=pending)
            sections.extend(tracker.stamp([boundary]))
            tracker.close(reason="compaction", move_index=move_index, level=level, boundary=boundary)
            raw_system = values.get("compaction_system_prompt")
            text = (
                _text({"c": "str", "text": raw_system}, store).strip()
                if raw_system is not None
                else _system_prompt_text(conversation, store)
            )
            if text:
                sections.extend(
                    tracker.stamp([{"label": "SYSTEM PROMPT", "content": text, "kind": "meta", "inContext": True}])
                )
                tracker.last_system = text
            if user is not None:
                sections.extend(tracker.stamp(_element_sections(user, store)))

    count = 0
    for element in elements:
        if element.get("kind") == "response":
            emit_boundaries(count)
            count += 1
        sections.extend(tracker.stamp(_element_sections(element, store)))
    emit_boundaries(count)
    tracker.last_level = level
    return sections


def _move_transcript_sections(
    tracker: _ContextTracker,
    *,
    move_index: int,
    level: int,
    turns: list[SpanNode],
    store: BlockStore,
    turn_elapsed: dict[int, float],
    summary_outcomes: Optional[dict[int, dict[str, Any]]] = None,
    journal_updates: Optional[dict[int, list[dict[str, Any]]]] = None,
    post_turn_updates: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Every transcript card of one move, in the order the model lived it:
    for a move in which a compaction fired, the rounds before the request,
    the request and its reply, the boundary, the rebuilt system prompt, then
    the rounds after. A system prompt is shown whenever the context was
    rebuilt (first move, after every compaction, at a new level)."""
    sections: list[dict[str, Any]] = []
    outer_turns = [turn for turn in turns if turn.name != _COMPACTION_TURN_NAME]
    claimed: set[int] = set()
    for position, turn in enumerate(outer_turns):
        compactions = [c for c in _compaction_turns(turn, turns) if c.span_id not in claimed]
        # A compaction span that is not nested under any analyzer turn of
        # this move rides with the last analyzer turn that precedes it.
        if position == len(outer_turns) - 1:
            compactions += [
                c
                for c in turns
                if c.name == _COMPACTION_TURN_NAME and c.span_id not in claimed and c not in compactions
            ]
        claimed.update(c.span_id for c in compactions)
        elapsed = turn_elapsed.get(turn.span_id)
        rebuilt = False
        if compactions:
            removed = _discarded_compaction_rounds(turn, store)
            without_previous = max(0, len(compactions) - removed)
            for number, compaction in enumerate(compactions):
                artifact = _turn_conversation_artifact(compaction)
                conversation: Any = store.load(artifact["block"], "json") if artifact is not None else None
                if isinstance(conversation, dict):
                    conversation_dict = cast(dict[str, Any], conversation)
                    if number > 0:
                        system_section = _system_prompt_section(conversation_dict, store)
                        if system_section is not None:
                            sections.extend(tracker.stamp([system_section]))
                    request_sections = _compaction_request_sections(
                        conversation_dict,
                        store,
                        include_previous_block=number >= without_previous,
                        rebuilt=rebuilt,
                    )
                    rebuilt = False
                    compaction_elapsed = turn_elapsed.get(compaction.span_id, elapsed)
                    for section in request_sections:
                        if compaction_elapsed is not None:
                            section["elapsed_s"] = compaction_elapsed
                    sections.extend(tracker.stamp(request_sections))
                outcome = {**turn.kv, **compaction.kv}
                if outcome.get("compaction_strategy") == "summarize":
                    installed = (summary_outcomes or {}).get(compaction.span_id)
                    if installed is None:
                        sections.extend(
                            tracker.stamp(
                                [
                                    {
                                        "label": "SUMMARY ATTEMPT",
                                        "content": "No context rebuild was recorded for this summary attempt.",
                                        "kind": "meta",
                                        "inContext": False,
                                    }
                                ]
                            )
                        )
                        continue
                    outcome = installed
                tracker.compactions += 1
                boundary = _boundary_section(tracker.compactions, outcome)
                sections.extend(tracker.stamp([boundary]))
                tracker.close(reason="compaction", move_index=move_index, level=level, boundary=boundary)
                rebuilt = True
        artifact = _turn_conversation_artifact(turn)
        value: Any = store.load(artifact["block"], "json") if artifact is not None else None
        conversation_dict = cast(dict[str, Any], value) if isinstance(value, dict) else {}
        response_offset = tracker.responses
        visible_count = sum(e.get("kind") == "response" for e in _journal_elements(turn, conversation_dict, store))
        # Response events survive best-effort transcript capture failures.
        tracker.responses += _journal_response_positions(turn, visible_count)[1]
        journal = list((journal_updates or {}).get(turn.span_id, []))
        if journal_updates is None and _journal_evicted(turn.kv):
            journal.append(turn.kv)
        if position == len(outer_turns) - 1:
            journal.extend(post_turn_updates or [])
        if journal and not compactions:
            if isinstance(value, dict) or artifact is None:
                journal_sections = _journal_turn_sections(
                    tracker,
                    turn,
                    conversation_dict,
                    journal,
                    store,
                    move_index=move_index,
                    level=level,
                    pending=artifact is None,
                    response_offset=response_offset,
                )
                for section in journal_sections:
                    if elapsed is not None:
                        section["elapsed_s"] = elapsed
                sections.extend(journal_sections)
            continue
        if not isinstance(value, dict):
            continue
        system_text = _system_prompt_text(conversation_dict, store)
        if tracker.last_system is not None and system_text != tracker.last_system:
            new_level = tracker.last_level is not None and level != tracker.last_level
            marker = {
                "label": "NEW LEVEL" if new_level else "CONTEXT REWRITE",
                "content": (
                    "Context rebuilt for a new level: the conversation was cleared and the notes in the system prompt describe the earlier levels."
                    if new_level
                    else "Context rebuilt within the same level: the system prompt changed after history or memory was refreshed."
                ),
                "kind": "boundary",
                "inContext": False,
            }
            sections.extend(tracker.stamp([marker]))
            tracker.close(
                reason="new level" if new_level else "context rewrite",
                move_index=move_index,
                level=level,
                boundary=None,
            )
        tracker.last_level = level
        turn_sections: list[dict[str, Any]] = []
        if tracker.last_system is None:
            system_section = _system_prompt_section(conversation_dict, store)
            if system_section is not None:
                turn_sections.append(system_section)
            tracker.last_system = system_text
        turn_sections.extend(
            _local_conversation_sections(conversation_dict, store, include_system=False, rebuilt=rebuilt)
        )
        for section in turn_sections:
            if elapsed is not None:
                section["elapsed_s"] = elapsed
        sections.extend(tracker.stamp(turn_sections))
    return sections


def _after_frame_artifacts(move: SpanNode) -> list[dict[str, Any]]:
    """Per-action result-frame artifacts in action order."""
    indexed: list[tuple[int, dict[str, Any]]] = []
    plain: list[dict[str, Any]] = []
    for artifact in move.artifacts:
        match = re.fullmatch(r"frame_after_(\d+)", artifact["name"])
        if match is not None:
            indexed.append((int(match.group(1)), artifact))
        elif artifact["name"] == "frame_after":
            plain.append(artifact)
    if indexed:
        return [artifact for _, artifact in sorted(indexed, key=lambda item: item[0])]
    return plain


def _artifact_frame(store: BlockStore, artifact: Optional[dict[str, Any]]) -> Optional[list[list[int]]]:
    """Decode a frame artifact, tolerating a live or partially copied run."""
    if artifact is None:
        return None
    try:
        return store.load(artifact["block"], artifact["codec"]).tolist()
    except (FileNotFoundError, OSError, ValueError):
        return None


def _fallback_elapsed_s(game: dict[str, Any]) -> Optional[float]:
    """Best monotonic elapsed time from benchmark bookkeeping."""
    candidates: list[Any] = [game.get("final_wallclock_seconds"), game.get("_live_elapsed_s")]
    raw_history: Any = game.get("history")
    if isinstance(raw_history, list):
        for raw_entry in cast(list[object], raw_history):
            if isinstance(raw_entry, dict):
                entry = cast(dict[str, Any], raw_entry)
                candidates.append(entry.get("wallclock_seconds"))
    elapsed: Optional[float] = None
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            elapsed = value if elapsed is None else max(elapsed, value)
    return elapsed


def _full_events(
    run_dir: Path, game: dict[str, Any]
) -> tuple[list[dict[str, Any]], Optional[float], list[dict[str, Any]]]:
    """Build the replay timeline, the exact hosted-game elapsed time, and the
    list of contexts (segments between compactions / level rebuilds)."""
    run_id = str(game.get("run_id") or "")
    stream_events = read_game_events(run_dir, run_id)
    roots = build_span_tree(stream_events)
    if not roots:
        return [], _fallback_elapsed_s(game), []
    root_elapsed = roots[0].duration_s
    elapsed_s = float(root_elapsed) if root_elapsed is not None and root_elapsed >= 0 else _fallback_elapsed_s(game)
    moves = [node for node in roots[0].children if node.kind == "infra/move"]
    store = job_block_store(run_dir)
    events: list[dict[str, Any]] = []
    event_index = 0
    action_number = 0
    state = "playing"
    score_actions_per_level: list[int] = []
    score_levels_completed = 0
    base_actions_per_level = _integer_list(game.get("base_actions_per_level"))
    first_actions = _action_events(moves[0]) if moves else []
    first_level = _action_level(first_actions[0]) if first_actions else 1
    if moves:
        initial_board = _frame(store, moves[0], "frame_at") or _frame(store, moves[0], "frame_before")
        initial: dict[str, Any] = {
            "type": "initial",
            "title": "Initial State",
            "event_index": event_index,
            "action_display": "RESET",
            "state": state,
            "level": first_level,
            "score": 0,
            "arc_agi_score": _arc_score(0, score_actions_per_level, base_actions_per_level),
            "elapsed_s": 0.0,
        }
        if initial_board is not None:
            initial["board"] = initial_board
        events.append(initial)
        event_index += 1

    total_actions = sum(len(_action_events(move)) for move in moves)
    total_levels = int(game.get("number_of_levels") or 0)
    previous_completed = 0
    tracker = _ContextTracker()
    summary_outcomes = _summary_compaction_outcomes(stream_events)
    journal_updates: dict[int, list[dict[str, Any]]] = {}
    for record in stream_events:
        values = record.get("values", {})
        if record.get("e") == "kv" and _journal_evicted(values):
            journal_updates.setdefault(record["span"], []).append(values)
    for move_index, move in enumerate(moves, start=1):
        actions = _action_events(move)
        action_times = _action_event_times(move)
        turns = _turn_nodes(move)
        before_board = _frame(store, move, "frame_at") or _frame(store, move, "frame_before")
        level = _action_level(actions[0]) if actions else min(max(1, total_levels), previous_completed + 1)
        analysis: dict[str, Any] = {
            "type": "analysis",
            "title": f"Turn {move_index}",
            "event_index": event_index,
            "analysis_step": move_index,
            "state": state,
            "level": level,
            "score": None,
            "arc_agi_score": _arc_score(
                score_levels_completed,
                score_actions_per_level,
                base_actions_per_level,
            ),
        }
        if before_board is not None:
            analysis["board"] = before_board
        # Inclusive totals of the TOP-level turns only: a compaction request
        # is a turn nested under the analyzer turn, and its parent's
        # inclusive total already holds it.
        nested_ids = {turn.span_id for turn in turns}
        analysis["generated_tokens"] = sum(turn.incl_generated for turn in turns if turn.parent_id not in nested_ids)
        analysis["response_count"] = sum(
            1 for turn in turns for event in turn.events if _is_completed_response_event(event)
        )
        turn_elapsed: dict[int, float] = {
            turn.span_id: _span_elapsed_s(roots[0], turn)
            for turn in turns
            if _turn_conversation_artifact(turn) is not None
        }
        turn_elapsed_values: list[float] = list(turn_elapsed.values())
        sections = _move_transcript_sections(
            tracker,
            move_index=move_index,
            level=level,
            turns=turns,
            store=store,
            summary_outcomes=summary_outcomes,
            turn_elapsed=turn_elapsed,
            journal_updates=journal_updates,
            post_turn_updates=[
                values
                for node in [move, *_descendants(move)]
                if node.kind != "llm/turn"
                for values in journal_updates.get(node.span_id, [])
            ],
        )
        if sections:
            analysis["transcript_sections"] = sections
        analysis["elapsed_s"] = max(turn_elapsed_values, default=_span_elapsed_s(roots[0], move))
        events.append(analysis)
        event_index += 1

        after_frames = _after_frame_artifacts(move)
        for action_index, action in enumerate(actions):
            action_number += 1
            action_time = action_times[action_index] if action_index < len(action_times) else None
            state = str(action.get("outcome") or state)
            completed = int(action.get("levels_completed", 0))
            completed_delta = max(0, completed - previous_completed)
            reward = float(completed_delta) / max(1, total_levels)
            is_final_action = action_number == total_actions
            score_level = max(0, int(action.get("level", completed) or 0))
            while len(score_actions_per_level) <= score_level:
                score_actions_per_level.append(0)
            score_actions_per_level[score_level] += 1
            score_levels_completed = max(score_levels_completed, completed)
            arc_score = _arc_score(
                score_levels_completed,
                score_actions_per_level,
                base_actions_per_level,
            )
            action_event: dict[str, Any] = {
                "type": "action",
                "title": f"Action {action_number}",
                "event_index": event_index,
                "analysis_step": move_index,
                "action_num": action_number,
                "action_name": str(action.get("name") or ""),
                "action_display": action_display(action.get("name"), action.get("row"), action.get("col")),
                "row": action.get("row"),
                "col": action.get("col"),
                "state": state,
                "level": _action_level(action),
                "levels_completed": completed,
                "reward": reward,
                "score": game.get("final_score") if is_final_action else None,
                "arc_agi_score": arc_score,
                "elapsed_s": (
                    max(0.0, action_time - roots[0].t_open)
                    if action_time is not None
                    else _span_elapsed_s(roots[0], move)
                ),
            }
            if is_final_action:
                action_event["final_arc_score"] = game.get("final_score")
            frame_artifact = after_frames[action_index] if action_index < len(after_frames) else None
            after_board = _artifact_frame(store, frame_artifact)
            if after_board is not None:
                action_event["board"] = after_board
            events.append(action_event)
            event_index += 1
            previous_completed = max(previous_completed, completed)
    return events, elapsed_s, tracker.contexts


def load_game_full(root: Union[str, Path], requested: Optional[str], game_index: int) -> dict[str, Any]:
    """Load one game as a reasoning-and-actions replay timeline."""
    resolved_root = Path(root).resolve()
    run_dir = resolve_run_dir(resolved_root, requested)
    game = _game_record(run_dir, game_index)
    payload = _game_summary(game)
    events, elapsed_s, contexts = _full_events(run_dir, game)
    payload.update(
        {
            "events": events,
            "eventCount": len(events),
            "lastEvent": events[-1] if events else payload.get("lastEvent"),
            "elapsed_s": elapsed_s,
            "contexts": contexts,
        }
    )
    return payload
