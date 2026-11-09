"""A platypus run as the replay viewer's payload: one game's transcript with
its contexts, the puzzle it was given, and the answers it stood on on the way.

Platypus keeps all of it in ``benchmark.json`` plus append-only span streams
and a content-addressed block store, so this module translates and creates no
artifact of its own. Everything read here is infrastructure-owned surface --
``infra/*`` spans, ``llm/turn`` conversation artifacts, the grid artifacts of
a move and of its submissions -- so any agent's runs render.

The transcript follows the model's own view: a context is the stretch from a
system prompt to the summary that closed it, and a rebuilt context is shown
whole -- the task, the summary, the rounds it kept -- before the rounds that
came after. The rounds it kept are marked ``retained``: they were generated,
and are counted, where they first appear.

An answer is placed where it came to stand. The duck submits from inside the
``python`` tool whenever the model's ``test_out`` has changed, and says
nothing to the model about it, so no card announces a submission; the
``infra/submit`` span carries the time, and the kernel's own execution events
within the turn say which tool call that was.

Ported from arc3-platypus's ``viewer_data``, whose payload was a game replay:
a board, actions and levels. A puzzle has none of the three.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any, cast

import taaf.grid
from platypus.diagnostics import (
    BlockStore,
    SpanNode,
    archive_parts,
    build_span_tree,
    game_run_ids,
    job_block_store,
    read_game_events,
)

_DISCOVERY_TTL_S = 30.0
_MAX_DISCOVERY_DEPTH = 3
_PRUNED_DIRS = {
    ".git",
    ".venv",
    "answers",
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


def list_run_dirs(root: str | Path) -> list[Path]:
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


def resolve_run_dir(root: str | Path, requested: str | None = None) -> Path:
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


def _live_game_record(run_dir: Path, run_id: str) -> dict[str, Any] | None:
    """Build temporary benchmark fields from one append-only game stream."""
    events = read_game_events(run_dir, run_id)
    roots = build_span_tree(events)
    root = next((node for node in roots if node.kind == "infra/game"), None)
    if root is None:
        return None
    event_times = [float(event["t"]) for event in events if isinstance(event.get("t"), (int, float))]
    elapsed_s = max(event_times, default=root.t_open) - root.t_open
    answers = [node for node in _descendants(root) if node.kind == "infra/submit"]
    return {
        "game_id": root.name,
        "run_id": run_id,
        "state": str(root.outcome or "finished") if root.closed else "playing",
        "final_score": None,
        "history": [{} for _ in answers],
        "final_generated_tokens": root.incl_generated,
        "_live_elapsed_s": max(0.0, elapsed_s),
    }


def _game_summary(game: dict[str, Any]) -> dict[str, Any]:
    """Translate one benchmark game record without reading its span stream.

    ``generated_tokens`` is the game's whole spend: TAAF books a game's tokens
    to the submission that followed them and the rest to the final figure.
    """
    raw_history: Any = game.get("history")
    history: list[Any] = cast(list[Any], raw_history) if isinstance(raw_history, list) else []
    run_id = str(game.get("run_id") or "")
    game_id = str(game.get("game_id") or run_id)
    status = str(game.get("state") or "unknown")
    final_score = game.get("final_score")
    generated_tokens = int(game.get("final_generated_tokens") or 0)
    for record in history:
        if isinstance(record, dict):
            generated_tokens += int(cast(dict[str, Any], record).get("generated_tokens") or 0)
    raw_pass: Any = game.get("pass_index")
    pass_index = int(raw_pass) if isinstance(raw_pass, int) else None
    display_name = game_id if pass_index is None else f"{game_id} · pass {pass_index + 1}"
    if isinstance(final_score, (int, float)):
        display_name += f" · {status} {float(final_score):g}"
    elif status != "unknown":
        display_name += f" · {status}"
    raw_result: Any = game.get("result")
    return {
        "game_id": game_id,
        "run_id": run_id,
        "display_name": display_name,
        "pass_index": pass_index,
        "status": status,
        "answer_count": len(history),
        "final_score": final_score,
        "generated_tokens": generated_tokens,
        "result": cast(dict[str, Any], raw_result) if isinstance(raw_result, dict) else None,
    }


def load_run_summary(root: str | Path, requested: str | None = None) -> dict[str, Any]:
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
        "arc_palette": list(taaf.grid.ARC_COLORS),
        # Where this run's image blocks are served (``platypus.viewer``, "URL
        # space = file space"): a transcript's image marker becomes a picture.
        "image_base": ("/" if run_dir == resolved_root else f"/{label}/") + "viewer/png/",
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


def _artifact(node: SpanNode, name: str) -> dict[str, Any] | None:
    """The last artifact named ``name`` on a span."""
    return next((artifact for artifact in reversed(node.artifacts) if artifact["name"] == name), None)


def _grid(store: BlockStore, artifact: dict[str, Any] | None) -> list[list[int]] | None:
    """Decode one PNG-grid artifact into JSON rows, tolerating a live or
    partially copied run."""
    if artifact is None:
        return None
    try:
        return store.load(artifact["block"], artifact["codec"]).tolist()
    except (FileNotFoundError, OSError, ValueError):
        return None


def _turn_nodes(move: SpanNode) -> list[SpanNode]:
    """All LLM turns nested beneath one move, in emission order."""
    return [node for node in _descendants(move) if node.kind == "llm/turn"]


def _turn_conversation_artifact(turn: SpanNode) -> dict[str, Any] | None:
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
        return [] if not text else [{"label": label, "content": text, "kind": "tool", "inContext": True}]
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


# The two records a compaction leaves under the turn it fired in (``duck.agent``): the request as it was
# served, and the rounds that left.
_COMPACTION_TURN_NAME = "compaction"
_SEGMENT_TURN_NAME = "compacted"


def _system_prompt_section(conversation: dict[str, Any], store: BlockStore) -> dict[str, Any] | None:
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


def _summary_compaction_outcomes(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Match each installed summary to its attempt using stream order.

    Success is emitted on the analyzer after the accepted child turn closes.
    Its merged kv cannot distinguish retries or multiple installs in one turn.
    """
    parents: dict[int, int | None] = {}
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


def _retained(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark cards as rounds a rebuilt context kept: generated, and counted,
    where they first appear."""
    for section in sections:
        section["retained"] = True
        section.pop("generated_tokens", None)
    return sections


def _live_turn_sections(
    element: dict[str, Any], store: BlockStore, *, shown: int, rebuilt: bool, offset: int
) -> tuple[list[dict[str, Any]], int, int]:
    """The cards of a live turn element, the turn a conversation ends with.

    Its first ``shown`` rounds were shown already: a rebuilt context opens
    with them, so there they come back as ``retained``; otherwise they are
    left out. The rounds after them are new, and each of their cards carries
    ``_tool_results``: how many of the analyzer turn's tool results stand at
    or before it, counted on from ``offset``. Returns the cards, the number of
    new tool results and the number of rounds in the element.
    """
    sections: list[dict[str, Any]] = []
    rounds = 0
    results = 0
    for child in _objects(element.get("elements")):
        if child.get("kind") == "response":
            rounds += 1
        if shown and rounds <= shown:
            if rebuilt:
                sections.extend(_retained(_element_sections(child, store)))
            continue
        if child.get("kind") == "tool_result":
            results += 1
        for section in _element_sections(child, store):
            section["_tool_results"] = offset + results
            sections.append(section)
    return sections, results, rounds


def _stretch_sections(
    elements: list[dict[str, Any]], store: BlockStore, *, rebuilt: bool, shown: int, offset: int
) -> tuple[list[dict[str, Any]], int, int]:
    """The cards of the part of a conversation not shown yet.

    A context that was just built is shown whole, as the model read it: the
    task, then after a summary the summary itself and the rounds it kept
    (``retained``). Otherwise only what is new: the user messages that opened
    this stretch, since everything before them was shown where it happened.
    Then the live turn's new rounds (:func:`_live_turn_sections`, whose counts
    are returned with the cards).
    """
    live = elements[-1] if elements and elements[-1].get("kind") == "turn" else None
    before = elements[:-1] if live is not None else elements
    sections: list[dict[str, Any]] = []
    if rebuilt:
        for element in before:
            cards = _element_sections(element, store)
            sections.extend(_retained(cards) if element.get("kind") == "turn" else cards)
    else:
        opening = len(before)
        while opening > 0 and before[opening - 1].get("kind") == "user":
            opening -= 1
        for element in before[opening:]:
            sections.extend(_element_sections(element, store))
    if live is None:
        return sections, 0, 0
    cards, results, rounds = _live_turn_sections(live, store, shown=shown, rebuilt=rebuilt, offset=offset)
    return sections + cards, results, rounds


def _boundary_section(number: int, kv: dict[str, Any]) -> dict[str, Any]:
    """The divider card between the context a summary closed and the rebuilt
    one, carrying the summary's outcome from the turn's kv."""
    before = kv.get("context_before")
    after = kv.get("context_after")
    compaction: dict[str, Any] = {
        "number": number,
        "mode": str(kv.get("compaction_strategy") or "summarize"),
        "tokens_before": before,
        "tokens_after": after,
        "summary_tokens": kv.get("summary_tokens"),
    }
    sizes = ""
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        sizes = f" · context {int(before):,} → {int(after):,} tokens"
    elif isinstance(after, (int, float)):
        sizes = f" · context after: {int(after):,} tokens"
    return {
        "label": f"COMPACTION #{number}",
        "content": (
            f"Context rebuilt after compaction {number}: summary installed{sizes}. "
            "The fresh context below is the task, the summary and the rounds it kept."
        ),
        "kind": "boundary",
        "inContext": False,
        "compaction": compaction,
    }


class _ContextTracker:
    """Numbers the contexts (the stretches between summaries, takes and other
    rebuilds) of one game and stamps every transcript card with its own."""

    def __init__(self) -> None:
        self.index = 0
        self.last_system: str | None = None
        self.last_take: int | None = None
        self.compactions = 0
        self.contexts: list[dict[str, Any]] = [{"index": 0, "opened_by": "start"}]

    def stamp(self, sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Associate each transcript card with the context containing it."""
        for section in sections:
            section["context"] = self.index
        return sections

    def close(self, *, reason: str, boundary: dict[str, Any] | None) -> None:
        """Record a context boundary and start the following segment."""
        current = self.contexts[-1]
        current["closed_by"] = reason
        if boundary is not None and "compaction" in boundary:
            current["compaction"] = boundary["compaction"]
        self.index += 1
        self.contexts.append({"index": self.index, "opened_by": reason})
        self.last_system = None


def _take_of(turn: SpanNode, by_id: dict[int, SpanNode]) -> SpanNode | None:
    """The take a turn was played in, for a game played in takes
    (``platypus.multi_take``); None for a game played in one."""
    parent = by_id.get(turn.parent_id) if turn.parent_id is not None else None
    while parent is not None:
        if parent.kind == "take":
            return parent
        parent = by_id.get(parent.parent_id) if parent.parent_id is not None else None
    return None


def _is_analyzer_turn(node: SpanNode) -> bool:
    """Whether a span is a turn of the agent's own, not one of the records a
    compaction leaves inside it."""
    return node.kind == "llm/turn" and node.name not in (_COMPACTION_TURN_NAME, _SEGMENT_TURN_NAME)


def _executions(turn: SpanNode) -> list[float]:
    """When the kernel ran the model's code during one analyzer turn, in order.

    The kernel's execution events keep the time they ran at, the same number
    of them for every tool call (the code, then the reading of the answer);
    a fresh kernel's own start-up is not a call and is left out. They are
    written out with whichever span is open at the time, so a turn's are
    spread over it and the compaction records inside it.
    """
    return sorted(
        float(event["t"])
        for node in [turn, *_descendants(turn)]
        if node.kind == "llm/turn"
        for event in node.events
        if event.get("kind") == "kernel/exec"
        and isinstance(event.get("t"), (int, float))
        and not event.get("values", {}).get("fresh_kernel")
    )


def _call_of(call_times: list[float], moment: float) -> int | None:
    """Which of a turn's tool calls was the last to have run by ``moment``,
    zero-based; None when the turn does not say. A submission is made from
    inside a call, right after its code ran, so this is the call an answer
    came from."""
    if not call_times:
        return None
    # The same clock, read a few milliseconds apart by the kernel's event and
    # by the submission that followed it.
    ran = [call for call, at in enumerate(call_times) if at <= moment + 0.05]
    return ran[-1] if ran else 0


def _call_times(executions: list[float], calls: int) -> list[float]:
    """When each of ``calls`` tool calls ran, given when the kernel ran.

    Exact where every call ran its code, two executions each. A call that
    was refused before it ran (a syntax error, a truncated reply) has a
    result and no execution, and a stretch with one spreads its executions
    over its calls in proportion, which can be a call out.
    """
    if not calls or not executions:
        return []
    per_call = len(executions) / calls
    return [executions[min(len(executions) - 1, math.ceil((call + 1) * per_call) - 1)] for call in range(calls)]


def _conversation(store: BlockStore, turn: SpanNode) -> dict[str, Any] | None:
    """One turn's captured request, when it has one."""
    artifact = _turn_conversation_artifact(turn)
    value: Any = store.load(artifact["block"], "json") if artifact is not None else None
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


class _TurnCards:
    """The cards of one analyzer turn as they are gathered, each given its
    context and its game time.

    A prompt, a request or a boundary is timed at the moment it is added for.
    A round's time is the kernel's to tell, so the rounds of each stretch —
    what the turn ran between two summaries — are held with the game time the
    stretch ended at, and :meth:`finish` times them.
    """

    def __init__(self, tracker: _ContextTracker) -> None:
        self.tracker = tracker
        self.cards: list[dict[str, Any]] = []
        self._stretches: list[tuple[list[dict[str, Any]], float]] = []

    def add(self, cards: list[dict[str, Any]], *, at: float, rounds_until: float | None = None) -> None:
        """Add cards in order: the rounds among them (the cards numbered by
        tool result) are a stretch ending by ``rounds_until``, and everything
        else stands ``at`` its moment."""
        rounds = [card for card in cards if "_tool_results" in card]
        for card in cards:
            if "_tool_results" not in card or rounds_until is None:
                card.setdefault("elapsed_s", at)
        if rounds and rounds_until is not None:
            self._stretches.append((rounds, rounds_until))
        self.cards.extend(self.tracker.stamp(cards))

    def finish(self, turn: SpanNode, executions: list[float]) -> list[float]:
        """Time the rounds, mark every card with its turn, and return the
        game time at which each of the turn's tool calls ran.

        ``executions`` is when the kernel ran, in game time. A stretch's are
        the ones up to the moment it ended, which keeps a call that cannot be
        placed exactly (:func:`_call_times`) inside its own stretch. A tool
        result stands when its call ran, a reply when the call it made ran,
        and a reply no call followed when its stretch ended.
        """
        call_times: list[float] = []
        taken = 0
        for rounds, until in self._stretches:
            calls = max(int(card["_tool_results"]) for card in rounds) - len(call_times)
            # The same clock, read a few milliseconds apart by the kernel's
            # event and by the span that followed it.
            window = [ran for ran in executions[taken:] if ran <= until + 0.05]
            taken += len(window)
            times = _call_times(window, calls) or [until] * calls
            first = len(call_times)
            call_times.extend(times)
            for card in rounds:
                made = int(card["_tool_results"])
                call = made - 1 if str(card.get("label") or "").startswith("TOOL RESULT") else made
                card["elapsed_s"] = min(until, call_times[call]) if first <= call < len(call_times) else until
        for card in self.cards:
            card["_turn"] = turn.span_id
        return call_times


def _move_transcript_sections(
    tracker: _ContextTracker,
    *,
    root: SpanNode,
    turns: list[SpanNode],
    by_id: dict[int, SpanNode],
    store: BlockStore,
    summary_outcomes: dict[int, dict[str, Any]],
    placement: dict[int, list[float]],
) -> list[dict[str, Any]]:
    """Every transcript card of one move, in the order the model lived it.

    A turn runs through contexts: each summary requested inside it closes
    one, and the duck leaves two records of that under the turn, in order —
    the request as it was served (``compaction``: the context the summary
    saw, the request, the reply) and the rounds the summary dropped
    (``compacted``; the rest it kept, and they open the next context). The
    turn's own conversation is its last context alone. So each context is
    read from the request that closed it, the rounds it inherited marked
    ``retained``; then come the request, its reply and the boundary; and the
    turn's own conversation gives what followed the last summary.

    A system prompt is shown whenever a context was built: at the start,
    after every summary, and at a new take. Each card carries ``_turn``, the
    analyzer turn it belongs to, and a round's cards ``_tool_results``, both
    for :func:`_stamp_answers`; ``placement`` receives, per analyzer turn,
    the game time at which each of its tool calls ran.
    """
    sections: list[dict[str, Any]] = []
    for turn in [node for node in turns if _is_analyzer_turn(node)]:
        gathered = _TurnCards(tracker)
        opened = max(0.0, turn.t_open - root.t_open)
        # Rounds of the live turn already shown, and tool results numbered.
        shown = 0
        results = 0

        take = _take_of(turn, by_id)
        if take is not None and tracker.last_take is not None and take.span_id != tracker.last_take:
            marker: dict[str, Any] = {
                "label": "NEW TAKE",
                "content": f"{take.name}: a conversation of its own, which knows nothing of the take before.",
                "kind": "boundary",
                "inContext": False,
            }
            gathered.add([marker], at=opened)
            tracker.close(reason="new take", boundary=None)
        if take is not None:
            tracker.last_take = take.span_id

        records = [
            node
            for node in _descendants(turn)
            if node.kind == "llm/turn" and node.name in (_COMPACTION_TURN_NAME, _SEGMENT_TURN_NAME)
        ]
        position = 0
        while position < len(records):
            record = records[position]
            position += 1
            requested = max(0.0, record.t_open - root.t_open)
            conversation = _conversation(store, record)
            elements = _objects(conversation.get("elements")) if conversation is not None else []
            if record.name == _SEGMENT_TURN_NAME:
                # Rounds that left with no summary (the prune strategy): this
                # record is the only place they are.
                if conversation is not None:
                    rebuilt = _open_context(tracker, gathered, conversation, store, at=opened)
                    cards, new_results, _ = _stretch_sections(elements, store, rebuilt=rebuilt, shown=0, offset=results)
                    results += new_results
                    gathered.add(cards, at=opened, rounds_until=requested)
                dropped = record.kv.get("rounds_dropped")
                count = f"{dropped} oldest rounds" if isinstance(dropped, int) else "oldest rounds"
                pruned: dict[str, Any] = {
                    "label": "ROUNDS PRUNED",
                    "content": f"The {count} left the context here, with no summary; the turn went on without them.",
                    "kind": "boundary",
                    "inContext": False,
                }
                gathered.add([pruned], at=requested)
                tracker.close(reason="rounds pruned", boundary=None)
                opened = requested
                continue
            answered = _span_elapsed_s(root, record)
            users = [index for index, element in enumerate(elements) if element.get("kind") == "user"]
            live_rounds = 0
            if conversation is not None and users:
                request = users[-1]
                context = elements[:request]
                if context and context[-1].get("kind") == "turn":
                    # The summary was asked for mid-turn: what the turn had
                    # run in this context is here, and nowhere else in order.
                    rebuilt = _open_context(tracker, gathered, conversation, store, at=opened)
                    cards, new_results, live_rounds = _stretch_sections(
                        context, store, rebuilt=rebuilt, shown=shown, offset=results
                    )
                    results += new_results
                    gathered.add(cards, at=opened, rounds_until=requested)
                request_cards = _element_sections(elements[request], store)
                for card in request_cards:
                    if card.get("label") == "USER PROMPT":
                        card["label"] = "COMPACTION REQUEST"
                for element in elements[request + 1 :]:
                    request_cards.extend(_element_sections(element, store))
                gathered.add(request_cards, at=answered)
            installed = summary_outcomes.get(record.span_id)
            if installed is None:
                attempt: dict[str, Any] = {
                    "label": "SUMMARY ATTEMPT",
                    "content": "No context rebuild was recorded for this summary attempt.",
                    "kind": "meta",
                    "inContext": False,
                }
                gathered.add([attempt], at=answered)
                # Nothing was rebuilt: the turn goes on in the same context.
                shown = max(shown, live_rounds)
                continue
            tracker.compactions += 1
            boundary = _boundary_section(tracker.compactions, installed)
            gathered.add([boundary], at=answered)
            tracker.close(reason="compaction", boundary=boundary)
            opened = answered
            segment = records[position] if position < len(records) else None
            if segment is not None and segment.name == _SEGMENT_TURN_NAME:
                position += 1
                dropped = segment.kv.get("rounds_dropped")
                shown = max(0, live_rounds - (dropped if isinstance(dropped, int) else live_rounds))
            else:
                # Nothing left the live turn: it had run no round in this
                # context (the summary came first), or the summary kept all.
                shown = live_rounds

        conversation = _conversation(store, turn)
        if conversation is not None:
            system_text = _system_prompt_text(conversation, store)
            if tracker.last_system is not None and system_text != tracker.last_system:
                rewrite: dict[str, Any] = {
                    "label": "CONTEXT REWRITE",
                    "content": "Context rebuilt: the system prompt changed.",
                    "kind": "boundary",
                    "inContext": False,
                }
                gathered.add([rewrite], at=opened)
                tracker.close(reason="context rewrite", boundary=None)
            rebuilt = _open_context(tracker, gathered, conversation, store, at=opened)
            cards, new_results, _ = _stretch_sections(
                _objects(conversation.get("elements")), store, rebuilt=rebuilt, shown=shown, offset=results
            )
            results += new_results
            gathered.add(cards, at=opened, rounds_until=_span_elapsed_s(root, turn))

        placement[turn.span_id] = gathered.finish(turn, [max(0.0, ran - root.t_open) for ran in _executions(turn)])
        sections.extend(gathered.cards)
    return sections


def _open_context(
    tracker: _ContextTracker, gathered: _TurnCards, conversation: dict[str, Any], store: BlockStore, *, at: float
) -> bool:
    """Show the system prompt if this conversation starts a context, and say
    whether it does."""
    if tracker.last_system is not None:
        return False
    system_section = _system_prompt_section(conversation, store)
    if system_section is not None:
        gathered.add([system_section], at=at)
    tracker.last_system = _system_prompt_text(conversation, store)
    return True


def _indexed_artifacts(node: SpanNode, prefix: str, suffix: str = "") -> dict[int, dict[str, Any]]:
    """The last artifact of each index among a span's ``<prefix><n><suffix>``."""
    found: dict[int, dict[str, Any]] = {}
    for artifact in node.artifacts:
        name = str(artifact["name"])
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        index = name[len(prefix) : len(name) - len(suffix)] if suffix else name[len(prefix) :]
        if index.isdigit():
            found[int(index)] = artifact
    return found


def _puzzle(store: BlockStore, moves: list[SpanNode]) -> dict[str, Any]:
    """What the game was given: the example pairs and the inputs it has to
    answer, from the grid artifacts of the move that presented them."""
    for move in moves:
        questions = _indexed_artifacts(move, "given_", "_question")
        asks = _indexed_artifacts(move, "ask_")
        if not questions and not asks:
            continue
        outputs = _indexed_artifacts(move, "given_", "_answer")
        return {
            "train": [
                {"input": _grid(store, questions[index]), "output": _grid(store, outputs.get(index))}
                for index in sorted(questions)
            ],
            "test": [{"input": _grid(store, asks[index])} for index in sorted(asks)],
        }
    return {"train": [], "test": []}


def _placeholders_at(turn: SpanNode | None, moment: float) -> int | None:
    """How many questions the answer submitted at ``moment`` filled in for:
    the count on the duck's ``answer`` event of that turn nearest in time.

    By time and not by the event's number: a game played in takes numbers its
    answers per take, and a take's answer that leaves the composition as it
    was has an event and no submission.
    """
    if turn is None:
        return None
    nearest: tuple[float, Any] | None = None
    for event in turn.events:
        at = event.get("t")
        if event.get("kind") != "answer" or not isinstance(at, (int, float)):
            continue
        distance = abs(float(at) - moment)
        if distance <= 1.0 and (nearest is None or distance < nearest[0]):
            nearest = (distance, event.get("values", {}).get("placeholders"))
    count = None if nearest is None else nearest[1]
    return count if isinstance(count, int) else None


def _answers(
    root: SpanNode,
    store: BlockStore,
    turns: list[SpanNode],
    by_id: dict[int, SpanNode],
    n_questions: int,
    placement: dict[int, list[float]],
) -> list[dict[str, Any]]:
    """Every submission of the game in order: when it was made, the grids it
    holds for each question, and where in the transcript it came to stand:
    ``_turn``, and ``_call``, the tool call of that turn it was submitted
    from (None when the turn does not say).

    The duck submits from inside the ``python`` tool, between a call's
    executions and its result, so the call is the last to have run by then
    (``placement``, from :func:`_move_transcript_sections`).
    """
    outer_turns = [turn for turn in turns if _is_analyzer_turn(turn)]
    answers: list[dict[str, Any]] = []
    for node in _descendants(root):
        if node.kind != "infra/submit":
            continue
        turn = by_id.get(node.parent_id) if node.parent_id is not None else None
        while turn is not None and not _is_analyzer_turn(turn):
            turn = by_id.get(turn.parent_id) if turn.parent_id is not None else None
        if turn is None:
            # Submitted outside any turn (a composition of takes): it stands
            # from the last turn that had opened by then.
            opened = [candidate for candidate in outer_turns if candidate.t_open <= node.t_open]
            turn = opened[-1] if opened else None
        predictions: list[dict[str, Any]] = []
        for question in range(n_questions):
            predictions.append(
                {
                    f"attempt_{attempt}": _grid(store, _artifact(node, f"ask_{question}_attempt_{attempt}"))
                    for attempt in (1, 2)
                }
            )
        placeholders = _placeholders_at(turn, node.t_open)
        answers.append(
            {
                "number": len(answers) + 1,
                "elapsed_s": max(0.0, node.t_open - root.t_open),
                "predictions": predictions,
                "placeholders": placeholders,
                "_turn": turn.span_id if turn is not None else None,
                "_call": (
                    _call_of(placement.get(turn.span_id, []), node.t_open - root.t_open) if turn is not None else None
                ),
            }
        )
    return answers


def _grid_size(grid: Any) -> str:
    """``rows x columns`` of a decoded grid, or a dash for one that is absent."""
    if not isinstance(grid, list) or not grid:
        return "-"
    rows = cast(list[Any], grid)
    first: Any = rows[0]
    return f"{len(rows)}x{len(cast(list[Any], first)) if isinstance(first, list) else 0}"


def _answer_marker(answer: dict[str, Any]) -> dict[str, Any]:
    """The divider card at the point an answer came to stand."""
    parts: list[str] = []
    for question, prediction in enumerate(cast(list[dict[str, Any]], answer["predictions"])):
        first, second = prediction.get("attempt_1"), prediction.get("attempt_2")
        sizes = _grid_size(first) if first == second else f"{_grid_size(first)} and {_grid_size(second)}"
        parts.append(f"test {question + 1}: {sizes}")
    placeholders = answer.get("placeholders")
    filler = f" · {placeholders} placeholder grids" if isinstance(placeholders, int) and placeholders else ""
    return {
        "label": f"ANSWER #{answer['number']}",
        "content": f"Answer {answer['number']} stands from here ({'; '.join(parts) or 'no grids'}){filler}.",
        "kind": "boundary",
        "inContext": False,
        "elapsed_s": answer["elapsed_s"],
        "answer": answer["number"],
    }


def _stamp_answers(
    sections: list[dict[str, Any]], answers: list[dict[str, Any]], turns: list[SpanNode]
) -> list[dict[str, Any]]:
    """Give every card ``answer``, the number of the answer standing when it
    was written (0 before the first), and put a marker card where each answer
    came to stand. Takes the placement marks off the cards and the answers."""
    order = {turn.span_id: position for position, turn in enumerate(turns)}
    keyed: list[tuple[tuple[float, float], dict[str, Any]]] = []
    for answer in answers:
        turn, call = answer.pop("_turn"), answer.pop("_call")
        # An answer whose call is unknown stands from the end of its turn.
        keyed.append(((float(order.get(turn, -1)), math.inf if call is None else float(call + 1)), answer))
    stamped: list[dict[str, Any]] = []
    standing = 0
    # A call's result is handed back after the submission made from inside
    # it, milliseconds after the kernel's own stamp for the call.
    submitted = 0.0
    for section in sections:
        turn = section.pop("_turn", None)
        tool_results = section.pop("_tool_results", None)
        if turn is not None and not section.get("retained"):
            here = (float(order.get(turn, -1)), float(tool_results or 0))
            while standing < len(keyed) and keyed[standing][0] <= here:
                marker = _answer_marker(keyed[standing][1])
                marker["context"] = section.get("context")
                submitted = max(submitted, float(marker["elapsed_s"]))
                standing += 1
                stamped.append(marker)
        if isinstance(section.get("elapsed_s"), (int, float)):
            section["elapsed_s"] = max(float(section["elapsed_s"]), submitted)
        section["answer"] = standing
        stamped.append(section)
    last_context = stamped[-1].get("context") if stamped else 0
    for _, answer in keyed[standing:]:
        marker = _answer_marker(answer)
        marker["context"] = last_context
        stamped.append(marker)
    return stamped


def _fallback_elapsed_s(game: dict[str, Any]) -> float | None:
    """Best monotonic elapsed time from benchmark bookkeeping."""
    candidates: list[Any] = [game.get("final_wallclock_seconds"), game.get("_live_elapsed_s")]
    raw_history: Any = game.get("history")
    if isinstance(raw_history, list):
        for raw_entry in cast(list[object], raw_history):
            if isinstance(raw_entry, dict):
                entry = cast(dict[str, Any], raw_entry)
                candidates.append(entry.get("wallclock_seconds"))
    elapsed: float | None = None
    for candidate in candidates:
        try:
            value = float(candidate)
        except (TypeError, ValueError):
            continue
        if value >= 0:
            elapsed = value if elapsed is None else max(elapsed, value)
    return elapsed


def _timeline(run_dir: Path, game: dict[str, Any]) -> dict[str, Any]:
    """One game's replay: its puzzle, its transcript cards with the contexts
    they fall in, and its answers."""
    run_id = str(game.get("run_id") or "")
    stream_events = read_game_events(run_dir, run_id)
    roots = build_span_tree(stream_events)
    empty: dict[str, Any] = {
        "puzzle": {"train": [], "test": []},
        "sections": [],
        "answers": [],
        "contexts": [],
        "elapsed_s": _fallback_elapsed_s(game),
        "response_count": 0,
    }
    if not roots:
        return empty
    root = roots[0]
    root_elapsed = root.duration_s
    elapsed_s = float(root_elapsed) if root_elapsed is not None and root_elapsed >= 0 else _fallback_elapsed_s(game)
    by_id = {node.span_id: node for node in [root, *_descendants(root)]}
    moves = [node for node in root.children if node.kind == "infra/move"]
    store = job_block_store(run_dir)
    puzzle = _puzzle(store, moves)
    tracker = _ContextTracker()
    summary_outcomes = _summary_compaction_outcomes(stream_events)
    placement: dict[int, list[float]] = {}
    sections: list[dict[str, Any]] = []
    all_turns: list[SpanNode] = []
    response_count = 0
    for move in moves:
        turns = _turn_nodes(move)
        all_turns.extend(turns)
        response_count += sum(1 for turn in turns for event in turn.events if _is_completed_response_event(event))
        sections.extend(
            _move_transcript_sections(
                tracker,
                root=root,
                turns=turns,
                by_id=by_id,
                store=store,
                summary_outcomes=summary_outcomes,
                placement=placement,
            )
        )
    answers = _answers(root, store, all_turns, by_id, len(cast(list[Any], puzzle["test"])), placement)
    return {
        "puzzle": puzzle,
        "sections": _stamp_answers(sections, answers, all_turns),
        "answers": answers,
        "contexts": tracker.contexts,
        "elapsed_s": elapsed_s,
        "response_count": response_count,
    }


def load_game_full(root: str | Path, requested: str | None, game_index: int) -> dict[str, Any]:
    """Load one game as a transcript-and-answers replay."""
    resolved_root = Path(root).resolve()
    run_dir = resolve_run_dir(resolved_root, requested)
    game = _game_record(run_dir, game_index)
    payload = _game_summary(game)
    payload.update(_timeline(run_dir, game))
    payload["answer_count"] = len(cast(list[Any], payload["answers"]))
    return payload
