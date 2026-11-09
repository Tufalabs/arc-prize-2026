"""ARC-2 sections name one puzzle set or several and may list the games they
play from the union; the trainer's manifest records both, and the worker loads
the same union from TAAF."""

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import harness_support.arc2.trainer as arc2_support
from harness_support.arc2.dataset import DERIVED_DATASETS, dataset_manifest
from training.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]
TAAF = ROOT / "arc2-platypus/packages/taaf/src"
HAS_ARC2_DATA = (TAAF / "taaf/data/arc2-train.npz").exists()
needs_data = pytest.mark.skipif(not HAS_ARC2_DATA, reason="Requires the private ARC-2 submodule")
SETS = {
    "set-a": [f"a{index}" for index in range(4)],
    "set-b": [f"b{index}" for index in range(3)],
    "set-ab": ["a1", "b9"],
}


@pytest.fixture
def fake_sets(monkeypatch):
    """Exercise the selection without the private harness data."""

    def manifest(name):
        return {"name": name, "sha256": f"sha-{name}", "puzzle_ids": list(SETS[name]),
                "test_outputs": dict.fromkeys(SETS[name], 1)}

    monkeypatch.setattr(arc2_support, "dataset_manifest", manifest)
    return manifest


def _config(dataset, games, eval_dataset=None, eval_games=()):
    return SimpleNamespace(
        inference=SimpleNamespace(dataset=dataset, games=list(games), game=""),
        official_eval=SimpleNamespace(dataset=eval_dataset, games=list(eval_games)),
    )


@pytest.mark.parametrize("games", [[], SETS["set-a"]])
def test_one_whole_set_keeps_its_own_manifest(fake_sets, games):
    """The identity of every run before game lists, so they still resume."""
    config = _config("set-a", games)
    resolved = arc2_support.resolve_datasets(config)
    assert resolved == {"training": fake_sets("set-a"), "evaluation": None}
    assert config.inference.games == SETS["set-a"]
    assert config.inference.game == "a0"


def test_listed_games_are_drawn_from_the_union_in_their_order(fake_sets):
    games = ["b2", "a3", "a0"]
    config = _config(["set-a", "set-b"], games, eval_dataset="set-b")
    resolved = arc2_support.resolve_datasets(config)
    assert resolved["training"] == {
        "datasets": [fake_sets("set-a"), fake_sets("set-b")],
        "games": games,
    }
    assert config.inference.games == games and config.inference.game == "b2"
    # Evaluation stays separate, and a whole single set keeps its manifest.
    assert resolved["evaluation"] == fake_sets("set-b")
    assert config.official_eval.games == SETS["set-b"]
    identity = arc2_support.training_identity(
        SimpleNamespace(
            inference=SimpleNamespace(environment="arc2"),
            training=SimpleNamespace(
                advantage_estimator="grpo",
                malformed_call_penalty=0.05,
                max_episode_penalty=None,
            ),
        ),
        resolved,
    )
    assert identity["dataset_manifest"]["training"]["games"] == games


def test_several_sets_without_games_play_the_whole_union(fake_sets):
    config = _config(["set-b", "set-a"], [])
    resolved = arc2_support.resolve_datasets(config)
    union = SETS["set-b"] + SETS["set-a"]
    assert resolved["training"]["games"] == config.inference.games == union
    # A subset of one set is recorded with its games.
    config = _config("set-a", ["a2"])
    assert arc2_support.resolve_datasets(config)["training"] == {
        "datasets": [fake_sets("set-a")], "games": ["a2"]}


@pytest.mark.parametrize(
    ("dataset", "games", "message"),
    [
        (["set-a", "set-b"], ["a0", "c7", "b1", "d8"],
         r"ARC-2 games \['c7', 'd8'\] are not in set-a \+ set-b"),
        ("set-a", ["b0"], r"ARC-2 games \['b0'\] are not in set-a"),
        (["set-a", "set-b"], ["a0", "a0"], "ARC-2 games must be listed once each"),
        (["set-a", "set-ab"], [], r"share puzzles; name disjoint sets"),
    ],
)
def test_games_outside_the_union_are_refused(fake_sets, dataset, games, message):
    with pytest.raises(ValueError, match=message):
        arc2_support.resolve_datasets(_config(dataset, games))


def _write(tmp_path, **inference):
    payload = json.loads((ROOT / "configs/flash_8gpu_arc2_grpo_tokmean.json").read_text())
    payload["inference"].update(inference)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    return path


def test_the_config_takes_one_name_or_a_list(tmp_path):
    single = ExperimentConfig.load(_write(tmp_path, dataset="arc2-eval"))
    assert single.inference.dataset == "arc2-eval"
    assert single.to_dict()["inference"]["dataset"] == "arc2-eval"
    several = ExperimentConfig.load(
        _write(tmp_path, dataset=["arc2-eval", "arc2-train-hard"]))
    assert several.inference.dataset == ["arc2-eval", "arc2-train-hard"]
    assert several.to_dict()["inference"]["dataset"] == ["arc2-eval", "arc2-train-hard"]
    for dataset in (["arc2-eval", "arc2-eval"], [], ["arc2-eval", "arc2-test"], "arc2-test"):
        with pytest.raises(ValueError, match=r"inference.dataset must be 'arc2-train' or "
                           r"'arc2-eval' or 'arc2-train-hard', or a list of distinct ones"):
            ExperimentConfig.load(_write(tmp_path, dataset=dataset))


def _taaf(program: str, *args: str) -> str:
    """Run PROGRAM against the pinned harness's TAAF in its own import namespace."""
    result = subprocess.run(
        [sys.executable, "-c", program, *args],
        env={**os.environ, "PYTHONPATH": os.pathsep.join([str(TAAF), str(ROOT)])},
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout.splitlines()[-1]


@needs_data
def test_trainer_manifests_match_taaf_for_every_named_set():
    """The trainer reads the packed sets itself and copies the derived ones'
    ids; TAAF's own loader must give the same puzzles and test outputs."""
    names = ["arc2-train", "arc2-eval", *DERIVED_DATASETS]
    loaded = json.loads(_taaf(
        "import json, sys\n"
        "from taaf.datasets import DERIVED_SETS, load\n"
        "names = json.loads(sys.argv[1])\n"
        "print(json.dumps({'sets': {name: {task.game_id: len(task.to_answer) for task in load(name)}"
        " for name in names}, 'derived': {name: list(DERIVED_SETS[name][1]) for name in names"
        " if name in DERIVED_SETS}}))",
        json.dumps(names),
    ))
    for name in names:
        manifest = dataset_manifest(name)
        assert manifest["puzzle_ids"] == list(loaded["sets"][name]), name
        assert manifest["test_outputs"] == loaded["sets"][name], name
    for name, (source, ids) in DERIVED_DATASETS.items():
        assert list(ids) == loaded["derived"][name]
        assert dataset_manifest(name)["sha256"] == dataset_manifest(source)["sha256"]
    assert len(DERIVED_DATASETS["arc2-train-hard"][1]) == 47


@needs_data
def test_worker_loads_the_same_union_as_the_trainer():
    """harness_support/arc2/worker.py builds its games from this union."""
    counts = json.loads(_taaf(
        "import json\n"
        "from harness_support.arc2.worker import load_tasks\n"
        "print(json.dumps({'one': len(load_tasks('arc2-eval')),"
        " 'union': sorted(load_tasks(['arc2-eval', 'arc2-train-hard']))}))",
    ))
    assert counts["one"] == 120
    union = [*dataset_manifest("arc2-eval")["puzzle_ids"],
             *dataset_manifest("arc2-train-hard")["puzzle_ids"]]
    assert counts["union"] == sorted(union) and len(union) == 167
