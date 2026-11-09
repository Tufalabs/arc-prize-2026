"""Select and filter TAAF traces for SFT by game, tag, and a generic predicate.

A benchmark **run dir** (produced by ARC3-Inference) holds `benchmark.json` plus a
`traces/` folder. `benchmark.json.game_runs[]` is the join table: each entry has
`run_id` (== the trace folder name under `traces/`) and per-run metadata —
`game_id`, `final_score`, `levels_completed`, `state`, `number_of_levels`, … The
trace's own `trace.json` knows none of this, so scoring/game filters must come from
here.

Filtering has three independent knobs, all declarative in the SFT config:

- **games**   — include/exclude by `game_id`.
- **filter**  — a generic predicate over *any* `game_run` field (e.g. keep only
                `final_score > 0`, the "non-zero ARC score" case). See
                `matches_predicate`; nothing about the fields is hard-coded.
- **tags**    — require/exclude a game's tags. Tags are not in the trace or the
                benchmark; they live in the re-arc-3 `environment_files/*/
                metadata.json` registry (`game_id -> tags`, e.g. "official"). We
                read them the same way `re_arc/env_sampler.py` does.
"""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from sft.holdouts import validate_arc3_sft_training_games

# Generic scalar comparators. The predicate looks a field up by name on the
# game_run record, so these apply to whatever fields the benchmark emits.
_COMPARATORS: dict[str, Callable[[Any, Any], bool]] = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}
_MEMBERSHIP = ("in", "not_in")
_SUPPORTED_OPS = tuple(_COMPARATORS) + _MEMBERSHIP


# --- tag registry (mirrors re_arc/env_sampler.py) ----------------------------


def _normalize_id(value: str) -> str:
    return str(value).strip().lower()


def _normalize_tag(value: str) -> str:
    return str(value).strip().lower()


def load_game_tags(env_dir: str | Path) -> dict[str, set[str]]:
    """`game_id -> {tags}` from re-arc-3 `environment_files/*/metadata.json`.

    Matches `env_sampler._discover_game_metadata_from_dir`: every `metadata.json`
    under `env_dir` contributes its `game_id` and normalized `tags`.
    """
    base = Path(env_dir).expanduser()
    if not base.is_dir():
        raise FileNotFoundError(f"re_arc_env_dir is not a directory: {base}")
    tags_by_game: dict[str, set[str]] = {}
    for metadata_file in base.rglob("metadata.json"):
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # Skipping would make corruption indistinguishable from "no tags".
            raise ValueError(f"unreadable game metadata {metadata_file}: {exc}") from exc
        if not isinstance(metadata, dict):
            raise ValueError(f"{metadata_file}: expected a JSON object")
        game_id = metadata.get("game_id")
        if not game_id:
            continue
        raw = metadata.get("tags", [])
        if not isinstance(raw, list) or any(not isinstance(tag, str) for tag in raw):
            raise ValueError(f"{metadata_file}: expected `tags` to be a list of strings")
        # A game can have multiple environment versions. Every metadata file
        # contributes tags; replacing the previous set made selection depend
        # on filesystem traversal order when versions carried different tags.
        tags_by_game.setdefault(_normalize_id(game_id), set()).update(
            _normalize_tag(tag) for tag in raw if _normalize_tag(tag)
        )
    return tags_by_game


# --- generic field predicate -------------------------------------------------


def _match_condition(value: Any, op: str, target: Any) -> bool:
    if op in _COMPARATORS:
        if value is None:
            return False
        try:
            return _COMPARATORS[op](value, target)
        except TypeError as exc:
            # A str/None final_score silently dropping traces under a numeric
            # predicate is type skew in the run metadata — a data bug, not a
            # non-match.
            raise TypeError(f"predicate {op!r} cannot compare field value {value!r} with {target!r}") from exc
    if op == "in":
        return value in target if isinstance(target, (list, tuple, set)) else value == target
    if op == "not_in":
        return value not in target if isinstance(target, (list, tuple, set)) else value != target
    raise ValueError(f"unsupported predicate op {op!r}; use one of {list(_SUPPORTED_OPS)}")


def _match_field(record: dict[str, Any], name: str, condition: Any) -> bool:
    # Preserve the established predicate semantics: a missing field is treated
    # as None. In particular, negative predicates such as ``not_in`` continue
    # to include records that do not carry the optional field.
    value = record.get(name)
    if isinstance(condition, dict):
        # {">": 0, "<=": 10} → all ops must hold (implicit AND).
        return all(_match_condition(value, op, target) for op, target in condition.items())
    # Bare scalar → equality, e.g. {"state": "won"} or {"game_id": "ft09-0d8bbf25"}.
    return value == condition


def matches_predicate(record: dict[str, Any], predicate: dict[str, Any]) -> bool:
    """True iff `record` satisfies every field condition in `predicate` (AND).

    `predicate` maps a game_run field name to either a scalar (equality) or a
    `{op: value}` dict. Fields are read by name, so any benchmark field works.
    """
    return all(_match_field(record, name, cond) for name, cond in predicate.items())


def validate_predicate_fields(predicate: dict[str, Any], available: set[str]) -> None:
    """Guard against typos: every predicate field must exist in the data."""
    unknown = [name for name in predicate if name not in available]
    if unknown:
        raise ValueError(
            f"filter references unknown game_run field(s) {sorted(unknown)}; "
            f"available fields: {sorted(available)}"
        )
    for name, condition in predicate.items():
        if not isinstance(condition, dict):
            continue
        if not condition:
            raise ValueError(f"filter condition for {name!r} must contain at least one operator")
        unsupported = sorted(str(op) for op in condition if op not in _SUPPORTED_OPS)
        if unsupported:
            raise ValueError(
                f"filter condition for {name!r} uses unsupported operator(s) {unsupported}; "
                f"use one of {list(_SUPPORTED_OPS)}"
            )


# --- benchmark join ----------------------------------------------------------


def load_game_runs(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """`run_id -> game_run` from `<run_dir>/benchmark.json` (the join table)."""
    path = Path(run_dir) / "benchmark.json"
    if not path.is_file():
        raise FileNotFoundError(f"no benchmark.json in run dir: {run_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    runs = payload.get("game_runs")
    if not isinstance(runs, list):
        raise ValueError(f"{path}: expected a `game_runs` list")
    by_id: dict[str, dict[str, Any]] = {}
    for i, record in enumerate(runs):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: game_runs[{i}] must be a JSON object")
        run_id = record.get("run_id")
        if not run_id:
            raise ValueError(f"{path}: game_runs[{i}] has no run_id — malformed benchmark.json")
        game_id = record.get("game_id")
        if not isinstance(game_id, str) or not game_id.strip():
            raise ValueError(f"{path}: game_runs[{i}] has no game_id — malformed benchmark.json")
        normalized_run_id = str(run_id)
        if normalized_run_id in by_id:
            raise ValueError(f"{path}: duplicate run_id {normalized_run_id!r} in game_runs")
        by_id[normalized_run_id] = record
    return by_id


def _is_finalized_trace(trace_dir: Path) -> bool:
    return (trace_dir / "steps.jsonl").is_file() and (trace_dir / "_SUCCESS").is_file()


# --- resolution --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedTrace:
    trace_dir: Path
    run_id: str
    game_id: str
    tags: frozenset[str]
    label: str
    game_run: dict[str, Any]


@dataclass(slots=True)
class SourceSelection:
    path: Path
    selected: list[ResolvedTrace] = field(default_factory=list)
    total: int = 0  # game_runs considered
    dropped: dict[str, int] = field(default_factory=dict)  # reason -> count

    def _drop(self, reason: str) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1


def _packaged_dataset_path(name: str) -> Path:
    """Path of `re_arc/datasets/<name>.json` inside the installed re_arc package."""
    try:
        import re_arc
    except ImportError as exc:
        raise ValueError(
            f"`dataset` name {name!r} resolves from the re_arc package, which is not "
            "installed -- install the `sft-rearc` extra or set "
            "`re_arc_datasets_dir` in the config."
        ) from exc
    # A sys.modules stub (some tests plant one) has no __file__; treat it as
    # not installed rather than crashing on the attribute.
    origin = getattr(re_arc, "__file__", None)
    if origin is None:
        raise ValueError(
            f"`dataset` name {name!r} resolves from the re_arc package, which is not "
            "installed -- install the `sft-rearc` extra or set "
            "`re_arc_datasets_dir` in the config."
        )
    path = Path(origin).parent / "datasets" / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"re_arc package has no dataset named {name!r} (looked at {path}).")
    return path


def load_dataset_game_ids(
    datasets_dir: str | Path | None, names: str | list[str]
) -> set[str]:
    """`game_id`s of named re-arc-3 datasets (`re_arc/datasets/<name>.json`).

    Mirrors the ARC3-Inference `dataset` config knob: `"train"` (or
    `"train,eval"`, or a list) resolves to the installed `re_arc` package's
    `datasets/<name>.json` (the `sft-rearc` extra supplies re-arc-3, so the game
    lists have one source of truth and never go stale); `re_arc_datasets_dir`
    overrides the lookup dir when set. An entry that looks like a path (.json
    suffix or existing file) is loaded directly. Each file is either a bare
    list of game ids or an object with `game_ids` (the re-arc-3 dataset file
    format).
    """
    if isinstance(names, str):
        names = [part.strip() for part in names.split(",") if part.strip()]
    if not names:
        raise ValueError("dataset selection must name at least one dataset")
    ids: set[str] = set()
    for name in names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"dataset names must be non-empty strings, got {name!r}")
        path = Path(name)
        if path.suffix != ".json" and not path.is_file():
            if datasets_dir is not None:
                path = Path(datasets_dir) / f"{name}.json"
            else:
                path = _packaged_dataset_path(name)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            game_ids = payload
        elif isinstance(payload, dict):
            game_ids = payload.get("game_ids") or []
        else:
            raise ValueError(f"{path}: expected a list or an object with `game_ids`")
        if not game_ids:
            raise ValueError(f"{path}: dataset file has no game_ids")
        if not isinstance(game_ids, list) or any(
            not isinstance(game_id, str) or not game_id.strip() for game_id in game_ids
        ):
            raise ValueError(f"{path}: game_ids must be a list of non-empty strings")
        ids.update(_normalize_id(game_id) for game_id in game_ids)
    return ids


@dataclass(slots=True)
class DatasetSpecLike:
    """The selection-relevant view of a config `DatasetSpec` (kept decoupled)."""

    path: str
    games: list[str] = field(default_factory=list)
    exclude_games: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    filter: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    dataset: str | list[str] | None = None

    def _wants_metadata(self) -> bool:
        return bool(
            self.games
            or self.exclude_games
            or self.tags
            or self.exclude_tags
            or self.filter
            or self.dataset
        )


def resolve_source(
    spec: DatasetSpecLike,
    *,
    env_dir: str | Path | None = None,
    datasets_dir: str | Path | None = None,
) -> SourceSelection:
    """Resolve one source spec to its selected traces + drop stats.

    A `path` is a run dir when it holds `benchmark.json` (then traces live in
    `path/traces`); otherwise it is treated as a bare TAAF trace root/folder for
    back-compat — allowed only when the spec applies no game/tag/score filters,
    since those need the benchmark join.
    """
    root = Path(spec.path)
    sel = SourceSelection(path=root)
    has_benchmark = (root / "benchmark.json").is_file()

    if not has_benchmark:
        if spec._wants_metadata():
            raise FileNotFoundError(
                f"{root}: game/tag/score filters need benchmark.json, but none found. "
                "Point `path` at the run dir (containing benchmark.json + traces/)."
            )
        # Back-compat: unfiltered bare trace root. Import here (not at module top)
        # so the benchmark path stays free of the PIL/datasets adapter stack.
        from sft.taaf_adapter import discover_trace_dirs

        for trace_dir in discover_trace_dirs(root):
            sel.selected.append(
                ResolvedTrace(
                    trace_dir=trace_dir,
                    run_id=trace_dir.name,
                    game_id="",
                    tags=frozenset(),
                    label=spec.label or root.name,
                    game_run={},
                )
            )
        sel.total = len(sel.selected)
        return sel

    game_runs = load_game_runs(root)
    available = {key for record in game_runs.values() for key in record}
    validate_predicate_fields(spec.filter, available)

    include_games = {_normalize_id(g) for g in spec.games}
    if spec.dataset:
        # Keep the long-standing SFT behavior: explicit games add to the named
        # dataset selection. Existing configs use the two knobs as a union.
        include_games |= load_dataset_game_ids(datasets_dir, spec.dataset)
    exclude_games = {_normalize_id(g) for g in spec.exclude_games}
    want_tags = {_normalize_tag(t) for t in spec.tags}
    drop_tags = {_normalize_tag(t) for t in spec.exclude_tags}

    tags_by_game: dict[str, set[str]] = {}
    if want_tags or drop_tags:
        if env_dir is None:
            raise ValueError("tag filters require `re_arc_env_dir` to be set in the config")
        tags_by_game = load_game_tags(env_dir)

    traces_root = root / "traces"
    sel.total = len(game_runs)
    for run_id, record in game_runs.items():
        game_id = _normalize_id(record.get("game_id", ""))
        if include_games and game_id not in include_games:
            sel._drop("games")
            continue
        if game_id in exclude_games:
            sel._drop("exclude_games")
            continue
        game_tags = tags_by_game.get(game_id, set())
        if want_tags and game_tags.isdisjoint(want_tags):
            sel._drop("tags")
            continue
        if drop_tags and not game_tags.isdisjoint(drop_tags):
            sel._drop("exclude_tags")
            continue
        if spec.filter and not matches_predicate(record, spec.filter):
            sel._drop("filter")
            continue
        trace_dir = traces_root / run_id
        if not _is_finalized_trace(trace_dir):
            sel._drop("missing_trace")
            continue
        sel.selected.append(
            ResolvedTrace(
                trace_dir=trace_dir,
                run_id=run_id,
                game_id=record.get("game_id", ""),
                tags=frozenset(game_tags),
                label=spec.label or record.get("game_id", "") or root.name,
                game_run=record,
            )
        )
    return sel


def resolve_sources(
    specs: list[DatasetSpecLike],
    *,
    env_dir: str | Path | None = None,
    datasets_dir: str | Path | None = None,
    for_training: bool = False,
) -> tuple[list[ResolvedTrace], list[SourceSelection]]:
    """Resolve sources, refusing the shared holdout families when used for training."""
    reports = [resolve_source(spec, env_dir=env_dir, datasets_dir=datasets_dir) for spec in specs]
    traces = [trace for report in reports for trace in report.selected]
    if for_training:
        validate_arc3_sft_training_games(trace.game_id for trace in traces)
    return traces, reports
