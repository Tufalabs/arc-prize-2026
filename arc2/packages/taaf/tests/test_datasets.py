"""Tests for taaf.datasets — the packed form, the two JSON readers, the
registry, and the combinators."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import taaf.datasets
from taaf.game import GameSpec, Pair
from taaf.grid import Grid


def _spec(game_id: str, n_to_answer: int = 1, answers: bool = True) -> GameSpec:
    """A small task with a distinguishable shape per id."""
    pairs = [Pair(question=Grid(np.array([[1, 2]], np.int8)), answer=Grid(np.array([[3]], np.int8)), given=True)]
    pairs += [
        Pair(
            question=Grid(np.array([[4, 5, 6]], np.int8)),
            answer=Grid(np.array([[7], [8]], np.int8)) if answers else None,
            given=False,
        )
        for _ in range(n_to_answer)
    ]
    return GameSpec(game_id=game_id, pairs=tuple(pairs))


def test_pack_unpack_round_trip_preserves_everything(tmp_path: Path):
    """The packed form is what ships, so a lossy round-trip would corrupt
    every run rather than failing loudly."""
    specs = [_spec("a"), _spec("b", n_to_answer=3), _spec("c", answers=False)]
    path = tmp_path / "set.npz"
    taaf.datasets.pack(specs, path)
    assert taaf.datasets.unpack(path) == specs


def test_committed_datasets_match_their_upstream_shape():
    """The shipped artifacts are what every run reads and no test rebuilds
    them, so their headline numbers are asserted here."""
    train = taaf.datasets.load("arc2-train")
    ev = taaf.datasets.load("arc2-eval")
    assert len(train) == 1000
    assert len(ev) == 120
    assert sum(len(s.to_answer) for s in train) == 1076
    assert sum(len(s.to_answer) for s in ev) == 167
    assert all(s.answers_known for s in train + ev)


def test_eval_is_far_more_multi_test_than_train():
    """37.5% against 6.9% — the axis on which the two official metrics
    disagree, so a statistic calibrated on train misjudges eval."""
    train = taaf.datasets.load("arc2-train")
    ev = taaf.datasets.load("arc2-eval")
    assert sum(len(s.to_answer) > 1 for s in ev) / len(ev) > 0.35
    assert sum(len(s.to_answer) > 1 for s in train) / len(train) < 0.10


def test_repo_and_challenges_readers_agree(tmp_path: Path):
    """The competition mount and the upstream repo carry the same tasks in
    different layouts; the readers must not disagree about what a task is."""
    task = {
        "train": [{"input": [[1]], "output": [[2]]}],
        "test": [{"input": [[3]], "output": [[4]]}],
    }
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    (repo_dir / "zz.json").write_text(json.dumps(task))
    (tmp_path / "ch.json").write_text(json.dumps({"zz": task}))
    assert taaf.datasets.load_repo_dir(repo_dir) == taaf.datasets.load_challenges_file(tmp_path / "ch.json")


def test_challenges_file_without_outputs_withholds_answers(tmp_path: Path):
    """A competition challenges file has no `output` on its test pairs; that
    absence is a withheld answer, not a malformed task."""
    task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}
    (tmp_path / "ch.json").write_text(json.dumps({"zz": task}))
    (spec,) = taaf.datasets.load_challenges_file(tmp_path / "ch.json")
    assert spec.answers_known is False
    assert spec.to_answer[0].question == Grid(np.array([[3]], np.int8))


def test_challenges_file_can_take_a_solutions_sidecar(tmp_path: Path):
    """Supported for completeness; no run path uses it."""
    task = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[3]]}]}
    (tmp_path / "ch.json").write_text(json.dumps({"zz": task}))
    (tmp_path / "sol.json").write_text(json.dumps({"zz": [[[4]]]}))
    (spec,) = taaf.datasets.load_challenges_file(tmp_path / "ch.json", tmp_path / "sol.json")
    assert spec.answers_known is True
    assert spec.to_answer[0].answer == Grid(np.array([[4]], np.int8))


def test_mount_selects_by_rerun_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A scoring rerun reads the hidden set; anything else the public one.
    This is the branch that only ever runs where we get no logs back."""
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    assert taaf.datasets.competition_challenges_path(tmp_path).name == "arc-agi_evaluation_challenges.json"
    monkeypatch.setenv("KAGGLE_IS_COMPETITION_RERUN", "true")
    assert taaf.datasets.competition_challenges_path(tmp_path).name == "arc-agi_test_challenges.json"


def test_mount_without_a_mount_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Never a silent fall back to our own data: a rehearsal of the mount
    path that quietly read something else would pass while proving nothing."""
    monkeypatch.delenv("KAGGLE_IS_COMPETITION_RERUN", raising=False)
    with pytest.raises(FileNotFoundError, match="only available inside a Kaggle run"):
        taaf.datasets.load("kaggle-mount", mount=tmp_path / "absent")


def test_unknown_dataset_names_itself():
    """A typo should say what the options were."""
    with pytest.raises(KeyError, match="arc2-train"):
        taaf.datasets.load("arc2-trian")


def test_combine_dedups_by_id_and_says_so(capsys: pytest.CaptureFixture[str]):
    """Benchmark.run rejects duplicate game_ids outright, so an overlap left
    in place would fail the run rather than the load."""
    combined = taaf.datasets.combine([_spec("a"), _spec("b")], [_spec("b"), _spec("c")])
    assert [s.game_id for s in combined] == ["a", "b", "c"]
    assert "dropped 1 duplicate" in capsys.readouterr().out


def test_shipped_sets_do_not_overlap():
    """Today they are disjoint; if that ever changes, combine's dedup is what
    keeps a run startable."""
    train = {s.game_id for s in taaf.datasets.load("arc2-train")}
    ev = {s.game_id for s in taaf.datasets.load("arc2-eval")}
    assert not (train & ev)


def test_subset_and_sample():
    """Narrowing is by id or count; a missing id is an error, not a silent
    empty result."""
    specs = [_spec(c) for c in "abcde"]
    assert [s.game_id for s in taaf.datasets.subset(specs, ids=["b", "d"])] == ["b", "d"]
    assert [s.game_id for s in taaf.datasets.subset(specs, limit=2)] == ["a", "b"]
    with pytest.raises(KeyError, match="zz"):
        taaf.datasets.subset(specs, ids=["zz"])
    picked = taaf.datasets.sample(specs, 3, seed=0)
    assert len(picked) == 3
    assert picked == taaf.datasets.sample(specs, 3, seed=0)
    assert [s.game_id for s in picked] == sorted(s.game_id for s in picked)


def test_named_subsets_load_their_ids_in_order():
    """A named subset is its parent narrowed to the listed ids, in the listed
    order, so a run over it reads the same way twice."""
    train = {s.game_id: s for s in taaf.datasets.load("arc2-train")}
    for name, (parent, ids) in taaf.datasets.DERIVED_SETS.items():
        assert parent == "arc2-train"
        specs = taaf.datasets.load(name)
        assert [s.game_id for s in specs] == list(ids)
        assert all(s == train[s.game_id] for s in specs)
    assert len(taaf.datasets.load("arc2-train-hard")) == 47
    assert not set(taaf.datasets.EASY_TASK_IDS) & set(taaf.datasets.HARD_TRAIN_TASK_IDS)


def test_a_subset_naming_an_id_its_parent_lacks_refuses(monkeypatch: pytest.MonkeyPatch):
    """An id the packed set lacks means the data changed under the curation."""
    monkeypatch.setitem(taaf.datasets.DERIVED_SETS, "arc2-train-broken", ("arc2-train", ("00000000",)))
    with pytest.raises(LookupError, match="00000000"):
        taaf.datasets.load("arc2-train-broken")
