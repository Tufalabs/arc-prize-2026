"""Named task sets, and the two ways bytes become :class:`GameSpec`s.

Our own packed ``.npz`` is the normal source and ships in the package, so a
run needs no network and no upstream checkout. The competition mount is the
other source, reachable only inside a Kaggle sandbox, and is a named set of
its own so that reading it is something we can deploy on purpose rather than
code whose first execution is an unobservable scoring rerun.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from taaf.game import GameSpec, Pair
from taaf.grid import Grid

DATA_DIR = Path(__file__).parent / "data"

#: Named sets built from our packed data, and the upstream directory each came from.
NPZ_SETS: dict[str, str] = {
    "arc2-train": "data/training",
    "arc2-eval": "data/evaluation",
}

#: `arc2-train` tasks a competent model solves reliably,
#: for exercising the HARNESS rather than measuring a solver.
#:
#: Why it exists: on real tasks a zero is ambiguous — the model could not
#: solve it, or the harness dropped the answer — and the ambiguity falls
#: hardest on paths that only some tasks reach. Multi-question tasks are the
#: worst case: 6.9% of `arc2-train` and 37.5% of `arc2-eval` have more than
#: one question, so a benchmark drawn from the head of the training set can
#: miss the loop over questions entirely. Here 6 of the 10 have two
#: questions and one has three, by construction.
#:
#: How it was curated (2026-08-22): the 30 smallest multi-question and 15
#: smallest single-question `arc2-train` tasks were played three times each
#: by the duck on `claude-haiku-4-5`; 38 of the 45 were correct on all three
#: passes. These 10 are drawn from those, cheapest first by median generated
#: tokens, keeping a spread of question counts and grid sizes and dropping
#: tasks whose token cost swung wildly between passes. A full pass costs
#: roughly 14k generated tokens.
#:
#: What it is NOT: a measurement. Every task here was selected for being
#: easy, so a score on it says nothing about ARC-AGI-2 ability, and the
#: curation is model-shaped — a different model has a different reliable
#: set. It is also small-grid by construction (the screen took the smallest
#: candidates), so nothing here would catch a defect that only shows up on a
#: 30x30 board.
EASY_TASK_IDS: tuple[str, ...] = (
    "6150a2bd",  # 1 question,  3x3
    "3c9b0459",  # 1 question,  3x3
    "3618c87e",  # 1 question,  5x5
    "44f52bb0",  # 2 questions, 3x3
    "d4469b4b",  # 2 questions, 5x5
    "dae9d2b5",  # 2 questions, 6x6
    "31d5ba1a",  # 2 questions, 6x6
    "6ea4a07e",  # 2 questions, 3x3
    "e9614598",  # 2 questions, 12x12
    "3cd86f4f",  # 3 questions, 12x12 — the only clean 3-question task
)

#: The `arc2-train` tasks Flash-Next did not solve on both of two passes, for
#: measuring on the part of the training set that is not solved every time.
#:
#: Why it exists: on the whole training set almost every task is solved on
#: both passes, so what separates configurations -- which task a second pass
#: rescues, what a pass costs before it goes wrong -- rests on a few dozen
#: tasks among a thousand; `arc2-eval` alone is 120 tasks.
#:
#: How it was chosen (2026-09-15): `arc2-train` x 2 passes, the duck on
#: Flash-Next e307 on one B200 at the cluster column's settings (arc2-platypus
#: `flash-arc2` 5c2a376: 130k tokens a game; results in
#: `arc_2_results/shared/training_set/duck_slurm_flash_train2_20260915_090800`).
#: 951 tasks were right on both passes; kept are the 31 right on one and the 18
#: right on neither, "right" meaning every test output. Two tasks go whose only
#: miss was a pass the harness crashed before it recorded an answer
#: (3ad05f52, 4ff4c9da), which says nothing of the task.
#:
#: What it is NOT: difficulty in general. It is one model's two samples, so a
#: task can land here or not on a third; and public training tasks may sit in
#: a model's training data.
HARD_TRAIN_TASK_IDS: tuple[str, ...] = (
    "09629e4f",
    "0d87d2a6",
    "1190bc91",
    "1acc24af",
    "1b8318e3",
    "1c56ad9f",
    "1e5d6875",
    "22425bda",
    "234bbc79",
    "2ccd9fef",
    "2dd70a9a",
    "31adaf00",
    "396d80d7",
    "4290ef0e",
    "456873bc",
    "50f325b5",
    "5e6bbc0b",
    "5ecac7f7",
    "6a1e5592",
    "84ba50d3",
    "891232d6",
    "8cb8642d",
    "8dab14c2",
    "8fbca751",
    "9473c6fb",
    "963c33f8",
    "9aec4887",
    "a64e4611",
    "a6953f00",
    "aa62e3f4",
    "b5bb5719",
    "b74ca5d1",
    "b7cb93ac",
    "b942fd60",
    "bf32578f",
    "c3fa4749",
    "d017b73f",
    "d6e50e54",
    "df9fd884",
    "e2092e0c",
    "e5062a87",
    "e73095fd",
    "f3b10344",
    "f8cc533f",
    "f8f52ecc",
    "f9d67f8b",
    "fe45cba4",
)

#: Named subsets of a named set: name -> (the set, the ids, in order).
DERIVED_SETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "arc2-train-easy": ("arc2-train", EASY_TASK_IDS),
    "arc2-train-hard": ("arc2-train", HARD_TRAIN_TASK_IDS),
}

#: The Kaggle competition this repo targets.
COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
COMPETITION_MOUNT = Path("/kaggle/input/competitions") / COMPETITION_SLUG
MOUNT_SET = "kaggle-mount"

DATASET_NAMES: tuple[str, ...] = (*NPZ_SETS, *DERIVED_SETS, MOUNT_SET)


def is_competition_rerun() -> bool:
    """Whether Kaggle is scoring this run against the hidden set."""
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes", "y", "on"}


# --- our packed form ---------------------------------------------------------


def pack(specs: Sequence[GameSpec], path: Path) -> None:
    """Write tasks as one flat int8 cell buffer plus an index.

    One array per grid would be ten times larger: an ``.npz`` is a zip, so
    every member carries its own header and its own compression stream and
    thousands of small grids never share their redundancy.
    """
    cells: list[npt.NDArray[np.int8]] = []
    index: list[tuple[int, int, int, int, int, int]] = []
    for t, spec in enumerate(specs):
        for p, pair in enumerate(spec.pairs):
            for role, grid in ((0, pair.question), (1, pair.answer)):
                if grid is None:
                    continue
                cells.append(grid.data.ravel())
                index.append((t, p, role, grid.shape[0], grid.shape[1], int(pair.given)))
    ids = "\n".join(spec.game_id for spec in specs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        cells=np.concatenate(cells) if cells else np.zeros(0, np.int8),
        index=np.array(index, dtype=np.int32).reshape(-1, 6),
        ids=np.frombuffer(ids.encode(), np.uint8),
    )


def unpack(path: Path) -> list[GameSpec]:
    """Inverse of :func:`pack`."""
    with np.load(path) as z:
        cells = z["cells"]
        index = z["index"]
        ids = bytes(z["ids"]).decode().split("\n") if z["ids"].size else []
    by_task: dict[int, dict[int, dict[str, Any]]] = {}
    offset = 0
    for t, p, role, rows, cols, given in index:
        n = int(rows) * int(cols)
        grid = Grid(cells[offset : offset + n].reshape(int(rows), int(cols)))
        offset += n
        slot = by_task.setdefault(int(t), {}).setdefault(int(p), {"given": bool(given)})
        slot["question" if role == 0 else "answer"] = grid
    specs: list[GameSpec] = []
    for t, game_id in enumerate(ids):
        pairs = by_task.get(t, {})
        specs.append(
            GameSpec(
                game_id=game_id,
                pairs=tuple(
                    Pair(question=pairs[p]["question"], answer=pairs[p].get("answer"), given=bool(pairs[p]["given"]))
                    for p in sorted(pairs)
                ),
            )
        )
    return specs


# --- upstream and competition JSON -------------------------------------------


def spec_from_task_json(game_id: str, task: dict[str, Any]) -> GameSpec:
    """One task in ARC's own JSON shape.

    A ``test`` pair carries no ``output`` in a competition challenges file;
    that absence is a withheld answer, not a malformed task.
    """
    pairs = [
        Pair(question=Grid.from_lists(p["input"]), answer=Grid.from_lists(p["output"]), given=True)
        for p in task["train"]
    ]
    pairs += [
        Pair(
            question=Grid.from_lists(p["input"]),
            answer=Grid.from_lists(p["output"]) if "output" in p else None,
            given=False,
        )
        for p in task["test"]
    ]
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def load_repo_dir(directory: Path) -> list[GameSpec]:
    """Read the upstream repo layout: one JSON per task, named by task id."""
    return [spec_from_task_json(f.stem, json.loads(f.read_text())) for f in sorted(directory.glob("*.json"))]


def load_challenges_file(path: Path, solutions: Path | None = None) -> list[GameSpec]:
    """Read the competition layout: one JSON per split, keyed by task id.

    ``solutions`` is supported for completeness and used by none of our run
    paths: our own data carries the same answers, and a simulated submission
    is not meant to score.
    """
    challenges: dict[str, Any] = json.loads(path.read_text())
    answers: dict[str, Any] = json.loads(solutions.read_text()) if solutions is not None else {}
    specs: list[GameSpec] = []
    for game_id in sorted(challenges):
        task = dict(challenges[game_id])
        if game_id in answers:
            task["test"] = [{**p, "output": a} for p, a in zip(task["test"], answers[game_id], strict=True)]
        specs.append(spec_from_task_json(game_id, task))
    return specs


def competition_challenges_path(mount: Path = COMPETITION_MOUNT) -> Path:
    """Which challenges file this run should read: the hidden set in a
    scoring rerun, the public evaluation set otherwise."""
    name = "arc-agi_test_challenges.json" if is_competition_rerun() else "arc-agi_evaluation_challenges.json"
    return mount / name


# --- the registry ------------------------------------------------------------


def load(name: str, *, mount: Path = COMPETITION_MOUNT) -> list[GameSpec]:
    """Load one named set: a packed set, a named subset of one, or the mount.

    Asking for the competition mount where none exists raises rather than
    falling back to our own data: a rehearsal of the mount path that quietly
    read something else would pass while proving nothing.
    """
    if name in DERIVED_SETS:
        parent, ids = DERIVED_SETS[name]
        by_id = {spec.game_id: spec for spec in load(parent, mount=mount)}
        missing = [game_id for game_id in ids if game_id not in by_id]
        if missing:
            raise LookupError(f"{name}: {missing} not in {parent} — the packed dataset is not what we expect")
        return [by_id[game_id] for game_id in ids]
    if name in NPZ_SETS:
        path = DATA_DIR / f"{name}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run `make fetch-data` to build the packed datasets.")
        return unpack(path)
    if name == MOUNT_SET:
        challenges = competition_challenges_path(mount)
        if not challenges.is_file():
            raise FileNotFoundError(
                f"{challenges} not found: {MOUNT_SET!r} is only available inside a Kaggle run with the "
                f"{COMPETITION_SLUG!r} competition attached."
            )
        return load_challenges_file(challenges)
    raise KeyError(f"unknown dataset {name!r}; known: {', '.join(DATASET_NAMES)}")


def combine(*sources: str | Iterable[GameSpec], mount: Path = COMPETITION_MOUNT) -> list[GameSpec]:
    """Concatenate sets, dropping later repeats of a task id.

    ``Benchmark.run`` rejects duplicate ``game_id``s outright, so an overlap
    left in place would fail the run rather than the load. Dropped ids are
    printed, because a silent dedup hides a dataset mistake.
    """
    seen: dict[str, GameSpec] = {}
    dropped: list[str] = []
    for source in sources:
        specs = load(source, mount=mount) if isinstance(source, str) else list(source)
        for spec in specs:
            if spec.game_id in seen:
                dropped.append(spec.game_id)
                continue
            seen[spec.game_id] = spec
    if dropped:
        print(f"taaf.datasets.combine: dropped {len(dropped)} duplicate id(s): {', '.join(sorted(set(dropped)))}")
    return list(seen.values())


def subset(specs: Sequence[GameSpec], *, ids: Iterable[str] | None = None, limit: int | None = None) -> list[GameSpec]:
    """Filter by id and/or take a prefix, preserving order."""
    out = list(specs)
    if ids is not None:
        wanted = set(ids)
        missing = wanted - {s.game_id for s in out}
        if missing:
            raise KeyError(f"ids not in this set: {', '.join(sorted(missing))}")
        out = [s for s in out if s.game_id in wanted]
    if limit is not None:
        out = out[:limit]
    return out


def sample(specs: Sequence[GameSpec], n: int, seed: int) -> list[GameSpec]:
    """A seeded random subset, in the original order so a run reads the same
    way twice."""
    rng = np.random.default_rng(seed)
    picked = sorted(int(i) for i in rng.choice(len(specs), size=min(n, len(specs)), replace=False))
    return [specs[i] for i in picked]
