"""Selection + filtering for the SFT trainer (game / tag / generic score predicate).

These exercise `sft.selection` and `sft.config` only, which are pure-stdlib (no
`datasets`/`pillow`/`trl`), so they run under CI's `--extra dev` install. Synthetic
run dirs stand in for a real benchmark run (`benchmark.json` + `traces/<run_id>/`).
"""

from __future__ import annotations

import importlib.util
import sys
import json
from pathlib import Path

import pytest

from sft.config import DatasetSpec, SFTRunConfig
from sft.selection import (
    DatasetSpecLike,
    load_dataset_game_ids,
    load_game_runs,
    load_game_tags,
    matches_predicate,
    resolve_source,
    resolve_sources,
    validate_predicate_fields,
)


def _write_run_dir(
    root: Path, game_runs: list[dict], *, make_traces: bool = True
) -> Path:
    """A benchmark run dir: benchmark.json + a finalized trace folder per run_id."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "benchmark.json").write_text(
        json.dumps({"game_runs": game_runs}), encoding="utf-8"
    )
    if make_traces:
        for record in game_runs:
            trace = root / "traces" / record["run_id"]
            trace.mkdir(parents=True, exist_ok=True)
            (trace / "steps.jsonl").write_text('{"step_index": 0}\n', encoding="utf-8")
            (trace / "_SUCCESS").write_text("", encoding="utf-8")
    return root


def _write_env_dir(base: Path, tags_by_game: dict[str, list[str]]) -> Path:
    """A re-arc-3 environment_files tree: <game>/<hash>/metadata.json with tags."""
    base.mkdir(parents=True, exist_ok=True)
    for i, (game_id, tags) in enumerate(tags_by_game.items()):
        meta_dir = base / game_id / f"{i:04d}"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "metadata.json").write_text(
            json.dumps({"game_id": game_id, "tags": tags}), encoding="utf-8"
        )
    return base


_RUNS = [
    {
        "run_id": "aaa",
        "game_id": "ft09-0d8bbf25",
        "final_score": 0.0,
        "state": "gave_up",
        "levels_completed": 0,
    },
    {
        "run_id": "bbb",
        "game_id": "ft09-0d8bbf25",
        "final_score": 12.5,
        "state": "gave_up",
        "levels_completed": 1,
    },
    {
        "run_id": "ccc",
        "game_id": "ar25-0c556536",
        "final_score": 40.0,
        "state": "won",
        "levels_completed": 6,
    },
]


# --- generic predicate (applies to any game_run field) -----------------------


def test_predicate_comparator_over_arbitrary_field() -> None:
    assert matches_predicate({"final_score": 12.5}, {"final_score": {">": 0}})
    assert not matches_predicate({"final_score": 0.0}, {"final_score": {">": 0}})
    # A completely different field works the same — nothing is hard-coded.
    assert matches_predicate({"levels_completed": 6}, {"levels_completed": {">=": 6}})


def test_predicate_bare_scalar_is_equality() -> None:
    assert matches_predicate({"state": "won"}, {"state": "won"})
    assert not matches_predicate({"state": "gave_up"}, {"state": "won"})


def test_predicate_membership_and_conjunction() -> None:
    record = {"state": "won", "final_score": 40.0}
    assert matches_predicate(record, {"state": {"in": ["won", "solved"]}})
    assert matches_predicate(record, {"state": "won", "final_score": {">": 10}})  # AND
    assert not matches_predicate(record, {"state": "won", "final_score": {">": 100}})


def test_predicate_missing_field_preserves_negative_filter_semantics() -> None:
    assert not matches_predicate({}, {"final_score": {">": 0}})  # field absent
    assert matches_predicate({}, {"state": {"not_in": ["lost"]}})
    assert matches_predicate({}, {"state": None})
    # Type skew in run metadata is a data bug: silently dropping such traces
    # under a numeric predicate would be indistinguishable from filtering.
    with pytest.raises(TypeError, match="cannot compare"):
        matches_predicate({"state": "won"}, {"state": {">": 0}})


def test_validate_predicate_fields_rejects_typos() -> None:
    with pytest.raises(ValueError, match="unknown game_run field"):
        validate_predicate_fields({"finall_score": {">": 0}}, {"final_score", "state"})


@pytest.mark.parametrize("condition", [{}, {"greater_than": 0}])
def test_validate_predicate_fields_rejects_invalid_conditions(condition: dict) -> None:
    with pytest.raises(ValueError, match="operator"):
        validate_predicate_fields({"final_score": condition}, {"final_score"})


# --- benchmark + tag loaders -------------------------------------------------


def test_load_game_runs_keyed_by_run_id(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    runs = load_game_runs(run_dir)
    assert set(runs) == {"aaa", "bbb", "ccc"}
    assert runs["ccc"]["game_id"] == "ar25-0c556536"


def test_load_game_runs_missing_benchmark(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_game_runs(tmp_path)


def test_load_game_runs_rejects_duplicate_run_ids(tmp_path: Path) -> None:
    run_dir = _write_run_dir(
        tmp_path / "run", [_RUNS[0], {**_RUNS[1], "run_id": "aaa"}]
    )
    with pytest.raises(ValueError, match="duplicate run_id"):
        load_game_runs(run_dir)


def test_load_game_tags_unions_multiple_environment_versions(tmp_path: Path) -> None:
    env = _write_env_dir(tmp_path / "env", {"ft09-0d8bbf25": ["Official"]})
    second = env / "ft09-0d8bbf25" / "new-version"
    second.mkdir(parents=True)
    (second / "metadata.json").write_text(
        json.dumps({"game_id": "FT09-0D8BBF25", "tags": ["click-only"]}),
        encoding="utf-8",
    )
    assert load_game_tags(env)["ft09-0d8bbf25"] == {"official", "click-only"}


def test_load_game_tags_rejects_non_list_tags(tmp_path: Path) -> None:
    metadata = tmp_path / "env" / "game" / "version" / "metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"game_id": "game", "tags": "official"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="list of strings"):
        load_game_tags(tmp_path / "env")


# --- source resolution -------------------------------------------------------


def test_resolve_source_score_filter_keeps_nonzero(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    spec = DatasetSpecLike(path=str(run_dir), filter={"final_score": {">": 0}})
    sel = resolve_source(spec)
    assert {t.run_id for t in sel.selected} == {"bbb", "ccc"}
    assert sel.dropped.get("filter") == 1  # the zero-score run


def test_resolve_source_game_include_filter(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    spec = DatasetSpecLike(path=str(run_dir), games=["ar25-0c556536"])
    sel = resolve_source(spec)
    assert {t.run_id for t in sel.selected} == {"ccc"}
    assert sel.dropped.get("games") == 2


def test_resolve_source_tag_filter_requires_env(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    spec = DatasetSpecLike(path=str(run_dir), tags=["official"])
    with pytest.raises(ValueError, match="re_arc_env_dir"):
        resolve_source(spec)


def test_resolve_source_tag_filter_selects_and_stamps(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    env = _write_env_dir(
        tmp_path / "env",
        {"ft09-0d8bbf25": ["official"], "ar25-0c556536": ["experimental"]},
    )
    spec = DatasetSpecLike(path=str(run_dir), tags=["official"])
    sel = resolve_source(spec, env_dir=env)
    assert {t.run_id for t in sel.selected} == {"aaa", "bbb"}
    assert all("official" in t.tags for t in sel.selected)
    assert sel.dropped.get("tags") == 1


def test_resolve_source_missing_trace_is_dropped(tmp_path: Path) -> None:
    # Two game_runs, but only build the trace folder for one of them.
    run_dir = _write_run_dir(tmp_path / "run", _RUNS, make_traces=False)
    trace = run_dir / "traces" / "ccc"
    trace.mkdir(parents=True)
    (trace / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    (trace / "_SUCCESS").write_text("", encoding="utf-8")
    sel = resolve_source(DatasetSpecLike(path=str(run_dir)))
    assert {t.run_id for t in sel.selected} == {"ccc"}
    assert sel.dropped.get("missing_trace") == 2


def test_resolve_source_preserves_label_and_game_run_provenance(tmp_path: Path) -> None:
    run_dir = _write_run_dir(tmp_path / "run", _RUNS)
    selected = resolve_source(
        DatasetSpecLike(
            path=str(run_dir),
            games=["ar25-0c556536"],
            label="my-eval",
        )
    ).selected

    assert len(selected) == 1
    assert selected[0].label == "my-eval"
    assert selected[0].game_run == _RUNS[2]


def test_resolve_source_filter_without_benchmark_errors(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(FileNotFoundError, match="benchmark.json"):
        resolve_source(
            DatasetSpecLike(path=str(bare), filter={"final_score": {">": 0}})
        )


def test_resolve_sources_flattens_and_reports(tmp_path: Path) -> None:
    run_a = _write_run_dir(tmp_path / "a", _RUNS[:2])
    run_b = _write_run_dir(tmp_path / "b", _RUNS[2:])
    traces, reports = resolve_sources(
        [DatasetSpecLike(path=str(run_a)), DatasetSpecLike(path=str(run_b))]
    )
    assert len(traces) == 3
    assert [r.total for r in reports] == [2, 1]


# --- config wiring -----------------------------------------------------------


def test_config_parses_datasets_and_eval() -> None:
    payload = {
        "datasets": [
            {
                "path": "/runs/train",
                "tags": ["official"],
                "filter": {"final_score": {">": 0}},
            }
        ],
        "eval_datasets": [{"path": "/runs/eval"}],
        "re_arc_env_dir": "/re-arc-3/env",
    }
    cfg = SFTRunConfig.from_dict(payload)
    assert isinstance(cfg.datasets[0], DatasetSpec)
    assert cfg.datasets[0].filter == {"final_score": {">": 0}}
    specs = cfg.train_specs()
    assert specs[0].tags == ["official"]
    assert cfg.eval_specs()[0].path == "/runs/eval"


def test_config_trace_root_backcompat_is_one_source() -> None:
    cfg = SFTRunConfig.from_dict({"trace_root": "/some/traces"})
    specs = cfg.train_specs()
    assert len(specs) == 1 and specs[0].path == "/some/traces"
    assert cfg.eval_specs() == []


def test_dataset_spec_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown dataset spec keys"):
        DatasetSpec.from_dict({"path": "/x", "tag": "typo-singular"})


def test_dataset_spec_label_roundtrips_to_selection() -> None:
    spec = DatasetSpec.from_dict({"path": "/x", "label": "curated-negative"})
    assert spec.label == "curated-negative"
    assert spec.to_selection_spec().label == "curated-negative"


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"eval_max_rows": 0}, "eval_max_rows"),
        ({"checkpoint_interval_minutes": 0}, "checkpoint_interval_minutes"),
        ({"gpus": 0}, "gpus"),
        ({"training_mode": "approximate"}, "training_mode"),
    ],
)
def test_config_rejects_invalid_runtime_values(payload: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        SFTRunConfig.from_dict(payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_config_load_rejects_non_finite_numbers(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "sft.json"
    path.write_text(f'{{"learning_rate": {constant}}}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON number"):
        SFTRunConfig.load(path)


def _write_datasets_dir(base: Path) -> Path:
    """A re-arc-3 datasets dir: one object-form and one bare-list dataset file."""
    d = base / "re_arc_datasets"
    d.mkdir()
    (d / "train.json").write_text(
        json.dumps({"name": "train", "game_ids": ["Alpha-0001", "beta-0001"]}),
        encoding="utf-8",
    )
    (d / "eval.json").write_text(json.dumps(["gamma-0001"]), encoding="utf-8")
    return d


def test_load_dataset_game_ids_names_lists_and_paths(tmp_path: Path) -> None:
    d = _write_datasets_dir(tmp_path)
    assert load_dataset_game_ids(d, "train") == {"alpha-0001", "beta-0001"}
    assert load_dataset_game_ids(d, "train,eval") == {
        "alpha-0001",
        "beta-0001",
        "gamma-0001",
    }
    assert load_dataset_game_ids(None, str(d / "eval.json")) == {"gamma-0001"}


def _re_arc_installed() -> bool:
    # test_rearc_env plants a spec-less SimpleNamespace stub in sys.modules,
    # which makes find_spec raise ValueError; probe defensively.
    stub = sys.modules.get("re_arc")
    if stub is not None:
        return getattr(stub, "__file__", None) is not None
    try:
        return importlib.util.find_spec("re_arc") is not None
    except ValueError:
        return False


def test_load_dataset_game_ids_package_fallback() -> None:
    # No datasets_dir -> the installed re_arc package's split files (one source
    # of truth; the checked-in copies are gone). Without re_arc (CI runs the
    # dev extra only), the error must point at the sft extra / override knob.
    if not _re_arc_installed():
        with pytest.raises(ValueError, match="re_arc package"):
            load_dataset_game_ids(None, "train")
        return
    train = load_dataset_game_ids(None, "train")
    eval_ids = load_dataset_game_ids(None, "eval")
    assert train and eval_ids
    assert not train & eval_ids  # the 70/30 split is disjoint
    with pytest.raises(ValueError, match="no dataset named"):
        load_dataset_game_ids(None, "no-such-split")


def test_resolve_source_dataset_include(tmp_path: Path) -> None:
    run = _write_run_dir(
        tmp_path / "run",
        [
            {"run_id": "r1", "game_id": "alpha-0001", "levels_completed": 2},
            {"run_id": "r2", "game_id": "gamma-0001", "levels_completed": 2},
        ],
    )
    d = _write_datasets_dir(tmp_path)
    sel = resolve_source(
        DatasetSpecLike(path=str(run), dataset="train"), datasets_dir=d
    )
    assert [t.game_id for t in sel.selected] == ["alpha-0001"]
    assert sel.dropped == {"games": 1}


def test_explicit_games_union_with_named_dataset(tmp_path: Path) -> None:
    run = _write_run_dir(
        tmp_path / "run",
        [
            {"run_id": "r1", "game_id": "alpha-0001", "levels_completed": 2},
            {"run_id": "r2", "game_id": "beta-0001", "levels_completed": 2},
            {"run_id": "r3", "game_id": "gamma-0001", "levels_completed": 2},
        ],
    )
    datasets_dir = _write_datasets_dir(tmp_path)
    selected = resolve_source(
        DatasetSpecLike(
            path=str(run),
            dataset="train",
            games=["beta-0001", "gamma-0001"],
        ),
        datasets_dir=datasets_dir,
    )
    assert [trace.game_id for trace in selected.selected] == [
        "alpha-0001",
        "beta-0001",
        "gamma-0001",
    ]


def test_config_dataset_knob_roundtrip(tmp_path: Path) -> None:
    spec = DatasetSpec(
        path="/run", dataset="train", filter={"levels_completed": {">=": 1}}
    )
    like = spec.to_selection_spec()
    assert like.dataset == "train"
    assert like._wants_metadata()
