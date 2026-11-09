"""A corpus of a seal run's framediff work, for searching it again offline.

Two kinds of case, each with its frames:

- **settled**: every decomposition the run made of a frame (a ``seal/frame_delta``
  record of event ``frame`` or ``test``): the target, each baseline offered with its
  kind, expression and handicap, and what the search returned -- the chosen baseline,
  every baseline's cost, the block, whether it was shown, and the limit it was held to.
- **animation**: every move whose animation keeps a frame in the kernel's ``history``
  (``seal.frame_delta.kept_animation``), whether or not a description was asked for:
  the board before the move and the frames kept, with each description the run made
  of it (stride, length, block).

``--transitions`` adds every move's settled frame with the frames it would be
explained from (the previous frame, and the level's start from a level's third
turn), described or not.

Frames come from the run's ``intermediate_states.pkl`` (the game as played) and, for
a run made since the seal kept them, from each record's own ``grids`` artifact, which
also holds the model's render. An older run kept no render: its baseline is listed
with no grid. A record is placed by the actions played before it, and the action
sequence of every game is checked against the state pickle first. A test failure or
a requested description made in a call that crossed a level cannot be placed on its
level, and is counted and left out, as is a description of a game's opening turn
made before that turn was built from its own first frame.

``--check N`` searches N settled cases and N requested descriptions again, with the
run's own framediff settings, and requires the recorded block: a cheap test that the
frames are the ones the run searched.

    uv run python packages/seal/tools/framediff_corpus.py <run_dir> <out_dir> [--transitions] [--check N]

``out_dir`` gets ``grids.npz`` (``grids``: every distinct frame once, int8, a
render's wildcards -1), ``cases.jsonl`` (one case a line, frames by index into
``grids``) and ``meta.json`` (the run, its framediff settings, the counts and the
checks). Levels are 1-based, as the seal numbers them; a turn is an index into that
level's ``history``.
"""

from __future__ import annotations

import argparse
import bisect
import dataclasses
import json
import multiprocessing
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from framediff import Decomposer, Source
from framediff.transforms import COLORS_NOTE
from platypus.diagnostics import (
    SpanNode,
    build_span_tree,
    job_block_store,
    read_game_events,
    resolve_blocks,
)
from seal.frame_delta import animation_start, kept_animation, unpack_grid

Grid = npt.NDArray[np.int8]

#: The record fields a case carries over as they are.
_SETTLED_FIELDS = ("event", "reason", "understood", "limit", "chosen", "cost", "shown", "timed_out", "duration_s")
_REQUEST_FIELDS = (
    "reason",
    "stride",
    "requested_stride",
    "built",
    "as_is",
    "chars",
    "cut",
    "shown",
    "timed_out",
    "duration_s",
)


def _walk(node: SpanNode) -> list[SpanNode]:
    """Every span under and including ``node``, in open order."""
    found = [node]
    for child in node.children:
        found.extend(_walk(child))
    return found


def read_stream(job_dir: str, run_id: str) -> dict[str, Any]:
    """One game's actions (name and time) and frame-delta records, with their artifacts resolved.

    A record keeps the start of the python call it followed, so a record made
    during a call that crossed a level can be told apart."""
    store = job_block_store(job_dir)
    actions: list[tuple[str, float]] = []
    records: list[dict[str, Any]] = []
    for root in build_span_tree(read_game_events(job_dir, run_id)):
        spans = _walk(root)
        calls = sorted(event["t"] for span in spans for event in span.events if event.get("kind") == "kernel/exec")
        for span in spans:
            actions.extend(
                (str(event["values"]["name"]), float(event["t"]))
                for event in span.events
                if event.get("kind") == "infra/action"
            )
            if span.kind != "seal/frame_delta":
                continue
            values = dict(span.kv)
            values.pop("text", None)
            record: dict[str, Any] = resolve_blocks(store, values)
            for artifact in span.artifacts:
                if artifact.get("name") in ("baselines", "grids") and artifact.get("codec") == "json":
                    record[str(artifact["name"])] = store.load(artifact["block"], "json")
            started = bisect.bisect_left(calls, span.t_open) - 1
            record["t"] = span.t_open
            record["call_t"] = calls[started] if started >= 0 else span.t_open
            records.append(record)
    actions.sort(key=lambda action: action[1])
    return {"run_id": run_id, "actions": actions, "records": records}


@dataclasses.dataclass
class _Grids:
    """Every distinct frame once, by its bytes."""

    index: dict[bytes, int] = dataclasses.field(default_factory=lambda: dict[bytes, int]())
    frames: list[Grid] = dataclasses.field(default_factory=lambda: list[Grid]())

    def add(self, grid: npt.ArrayLike) -> int:
        """The frame's index, added when new."""
        array = np.ascontiguousarray(grid, dtype=np.int8)
        key = bytes(array.shape) + array.tobytes()
        if key not in self.index:
            self.index[key] = len(self.frames)
            self.frames.append(array)
        return self.index[key]


class _Game:
    """One game as played: its settled frames, animations and levels, by state index."""

    def __init__(self, states: list[Any]) -> None:
        self.sent = [[np.asarray(frame, dtype=np.int8) for frame in state.raw.frame] for state in states]
        self.settled = [frames[-1] for frames in self.sent]
        self.completed = [int(state.raw.levels_completed) for state in states]
        self.actions = ["none" if s.previous_action is None else str(s.previous_action.id.name) for s in states]
        # A move that won a level, not the game, ends its turn on the win screen, the frame before the new level's
        # opening (``platypus.contract.partition``): the opening is the board the next move starts from.
        self.level_won = [bool(s.just_won_level) and not s.won for s in states]
        # The state each state's level opened on: the game's first, or the one a winning move led to.
        self.start: list[int] = []
        for index, completed in enumerate(self.completed):
            new = index == 0 or completed != self.completed[index - 1]
            self.start.append(index if new else self.start[index - 1])

    def kept(self, index: int) -> list[Grid]:
        """The animation frames the kernel's ``history`` keeps for the move that led to state ``index``, or
        for the game's first observation; for a move that won a level, those before its win screen."""
        sent = self.sent[index][:-1] if self.level_won[index] else self.sent[index]
        if not sent:
            return []
        before = self.settled[index - 1] if index else None
        return kept_animation(before, sent[:-1], sent[-1], lambda a, b: bool(np.array_equal(a, b)))


def _check_actions(game: _Game, actions: list[tuple[str, float]], run_id: str) -> None:
    """The run's actions, as its diagnostics saw them, are the state pickle's: the join rests on it."""
    played = [name for name, _ in actions]
    if played != game.actions[1:]:
        raise SystemExit(
            f"game {run_id}: its diagnostics' actions are not its states' ({len(played)} against "
            f"{len(game.actions) - 1}); the pickle and the streams do not describe the same play"
        )


def _settled_case(
    game: _Game, record: dict[str, Any], state: int, grids: _Grids, base: dict[str, Any]
) -> dict[str, Any] | None:
    """A decomposition record as a case, or None when it does not fit the level it was placed on."""
    start = game.start[state]
    turn = int(record["turn_index"])
    if record["event"] == "frame" and turn != state - start:
        return None
    if not 0 <= turn <= state - start:
        return None
    packed: dict[str, Any] = record.get("grids") or {}
    listed: list[dict[str, Any]] = record.get("baselines") or []
    # A failing test on a level's first frame has the render alone to explain it from.
    by_kind = {"level_start": game.settled[start], **({"previous": game.settled[start + turn - 1]} if turn else {})}
    target = unpack_grid(packed["target"]) if "target" in packed else game.settled[start + turn]
    baselines: list[dict[str, Any]] = []
    for baseline in listed:
        kind = str(baseline["kind"])
        grid = unpack_grid(packed[kind]) if kind in packed else by_kind.get(kind)
        baselines.append(
            {
                "kind": kind,
                "expression": baseline["expression"],
                "handicap": baseline["handicap"],
                "cost": baseline["cost"],
                "grid": None if grid is None else grids.add(grid),
            }
        )
    return {
        "kind": "settled",
        **base,
        "level": game.completed[state] + 1,
        "turn": turn,
        "target_expression": record["target"],
        **{field: record.get(field) for field in _SETTLED_FIELDS},
        "code": record.get("code"),
        "target": grids.add(target),
        "baselines": baselines,
    }


def _animation_cases(game: _Game, grids: _Grids, base: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    """Every move with kept animation frames, by (level, turn): the level it was played on, and its turn
    there, a winning move's included; and the game's opening turn, when its first observation carries one."""
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for index in range(len(game.settled)):
        frames = game.kept(index)
        if not frames:
            continue
        start = game.start[index - 1] if index else 0
        level, turn = game.completed[max(index - 1, 0)] + 1, index - start
        found[(level, turn)] = {
            "kind": "animation",
            **base,
            "level": level,
            "turn": turn,
            "action": game.actions[index],
            "won": bool(index) and game.completed[index] != game.completed[index - 1],
            "sent": len(game.sent[index]) - 1,
            "before_expression": animation_start(turn),
            "before": grids.add(game.settled[index - 1] if index else frames[0]),
            "frames": [grids.add(frame) for frame in frames],
            "requests": [],
        }
    return found


def _transition_cases(game: _Game, grids: _Grids, base: dict[str, Any]) -> list[dict[str, Any]]:
    """Every move's settled frame within its level, with the frames the seal would explain it from."""
    found: list[dict[str, Any]] = []
    for index in range(1, len(game.settled)):
        start = game.start[index]
        if start == index:
            continue  # a level's first frame: nothing in its level to explain it from
        turn = index - start
        found.append(
            {
                "kind": "transition",
                **base,
                "level": game.completed[index] + 1,
                "turn": turn,
                "action": game.actions[index],
                "target": grids.add(game.settled[index]),
                "previous": grids.add(game.settled[index - 1]),
                "level_start": grids.add(game.settled[start]) if turn >= 2 else None,
            }
        )
    return found


def _check(
    cases: list[dict[str, Any]], frames: list[Grid], decomposer: Decomposer, max_chars: int, count: int, seed: int
) -> dict[str, int]:
    """Search ``count`` settled cases with every baseline's frame, and ``count`` requested descriptions,
    again; any block that differs from the recorded one stops the extraction. The colours a copy names are
    left out of the comparison: a run before them recorded its blocks without them."""
    rng = random.Random(seed)
    settled = [
        case
        for case in cases
        if case["kind"] == "settled"
        and case["code"] is not None
        and all(b["grid"] is not None for b in case["baselines"])
    ]
    for case in rng.sample(settled, min(count, len(settled))):
        baselines: list[dict[str, Any]] = case["baselines"]
        sources = [Source(b["expression"], frames[int(b["grid"])], b["handicap"]) for b in baselines]
        found = dataclasses.replace(decomposer, max_bits=case["limit"]).explain(
            sources, frames[int(case["target"])], str(case["target_expression"])
        )
        if found.code is None or COLORS_NOTE.sub("", found.code) != case["code"]:
            raise SystemExit(
                f"a settled case does not search to its recorded block: {case['game']} {case['level']}/{case['turn']}"
            )
    requests = [(case, request) for case in cases if case["kind"] == "animation" for request in case["requests"]]
    requests = [
        (case, request) for case, request in requests if request["code"] is not None and not request["timed_out"]
    ]
    for case, request in rng.sample(requests, min(count, len(requests))):
        turn: int = case["turn"]
        stride: int | None = request["requested_stride"]
        kept: list[int] = case["frames"]
        found = decomposer.explain_animation(
            frames[int(case["before"])],
            [frames[frame] for frame in kept],
            str(case["before_expression"]),
            f"history[{turn}].animation_frames[{{k}}].grid",
            stride=stride,
            max_chars=max_chars if stride is None else None,
        )
        if COLORS_NOTE.sub("", found.code) != request["code"]:
            raise SystemExit(
                f"a description does not search to its recorded block: {case['game']} {case['level']}/{turn}"
            )
    return {"settled": min(count, len(settled)), "animations": min(count, len(requests))}


def _run_settings(run_dir: Path) -> tuple[Decomposer, dict[str, Any]]:
    """The run's framediff search and the seal's settings around it, from its pickled solver."""
    with (run_dir / "solver.pkl").open("rb") as handle:
        agent = pickle.load(handle).agent
    settings = {
        "decomposer": dataclasses.asdict(agent.decomposer),
        "max_bits_understood": agent.max_bits_understood,
        "animation_max_chars": agent.animation_max_chars,
        "animation_cut_chars": agent.animation_cut_chars,
    }
    return agent.decomposer, settings


def build(run_dir: Path, out_dir: Path, *, transitions: bool, check: int, workers: int) -> dict[str, Any]:
    """Extract the corpus of ``run_dir`` into ``out_dir``; returns its meta."""
    benchmark = json.loads((run_dir / "benchmark.json").read_text())
    with (run_dir / "intermediate_states.pkl").open("rb") as handle:
        states: list[list[Any]] = pickle.load(handle)
    runs: list[dict[str, Any]] = benchmark["game_runs"]
    if len(runs) != len(states):
        raise SystemExit(f"{len(runs)} game runs against {len(states)} games of states")
    # Spawned: an archive opened in this process must not be shared by forked readers.
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        streams = pool.starmap(read_stream, [(str(run_dir), str(run["run_id"])) for run in runs])

    grids = _Grids()
    cases: list[dict[str, Any]] = []
    left_out = {"crossed_a_level": 0, "not_on_its_level": 0, "opening_turn_from_the_newest_board": 0}
    for run, game_states, stream in zip(runs, states, streams):
        game = _Game(game_states)
        _check_actions(game, stream["actions"], str(run["run_id"]))
        base = {"game": run["run_id"], "game_id": run["game_id"], "pass": run["pass_index"]}
        times = [t for _, t in stream["actions"]]
        animations = _animation_cases(game, grids, base)
        settled: list[dict[str, Any]] = []
        for record in stream["records"]:
            state = bisect.bisect_left(times, record["t"])
            called = bisect.bisect_left(times, record["call_t"])
            if record.get("event") not in ("frame", "test", "animation") or "turn_index" not in record:
                continue
            if record.get("error") is not None:
                continue
            if record["event"] != "frame" and game.completed[called] != game.completed[state]:
                left_out["crossed_a_level"] += 1
                continue
            code = record.get("code")
            if record["event"] == "animation" and int(record["turn_index"]) == 0 and code is not None:
                if f"# Frame 0: the same as {animation_start(0)}" not in code:
                    # Built from whatever board was newest when it was asked, as the seal once did.
                    left_out["opening_turn_from_the_newest_board"] += 1
                    continue
            if record["event"] == "animation":
                key = (game.completed[state] + 1, int(record["turn_index"]))
                animation = animations.get(key)
                if animation is None or len(animation["frames"]) != record["frames"]:
                    left_out["not_on_its_level"] += 1
                    continue
                animation["requests"].append(
                    {**{field: record.get(field) for field in _REQUEST_FIELDS}, "code": record.get("code")}
                )
                continue
            if "chosen" not in record:
                continue  # predicted: no search ran
            case = _settled_case(game, record, state, grids, base)
            if case is None:
                left_out["not_on_its_level"] += 1
                continue
            settled.append(case)
        cases += settled + list(animations.values())
        if transitions:
            cases += _transition_cases(game, grids, base)

    decomposer, settings = _run_settings(run_dir)
    checked = _check(cases, grids.frames, decomposer, settings["animation_max_chars"], check, seed=0)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "grids.npz", grids=np.stack(grids.frames))
    with (out_dir / "cases.jsonl").open("w") as handle:
        for case in cases:
            handle.write(json.dumps(case) + "\n")
    tool_commit = subprocess.run(
        ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "run": str(run_dir),
        "label": benchmark.get("label"),
        "settings": settings,
        "tool_commit": tool_commit,
        "counts": {
            kind: sum(1 for case in cases if case["kind"] == kind) for kind in ("settled", "animation", "transition")
        }
        | {
            "settled_without_a_render": sum(
                1 for case in cases if case["kind"] == "settled" and any(b["grid"] is None for b in case["baselines"])
            ),
            "requests": sum(len(case["requests"]) for case in cases if case["kind"] == "animation"),
            "grids": len(grids.frames),
        },
        "left_out": left_out,
        "checked": checked,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--transitions", action="store_true", help="also every move's settled frame, described or not")
    parser.add_argument("--check", type=int, default=3, help="cases of each kind searched again (default 3)")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    meta = build(args.run_dir, args.out_dir, transitions=args.transitions, check=args.check, workers=args.workers)
    print(json.dumps({key: meta[key] for key in ("counts", "left_out", "checked")}, indent=1))


if __name__ == "__main__":
    main()
