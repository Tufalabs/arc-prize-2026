# type: ignore
"""Tests for the compare-runs scripts (``tools/compare_runs``).

They are shipped tooling rather than library code, but they read and
*delete* real run directories, so the destructive guards and the Kaggle
puller's in-place relabel are pinned here.
"""

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import taaf.benchmark
import taaf.game
import taaf.game_examples
import taaf.solver
import taaf.solver_examples

SCRIPTS = Path(__file__).resolve().parents[1] / "tools" / "compare_runs"


def _load(name: str):
    """Import a script by path, with its own directory importable so the
    ``_common`` sibling resolves the same way it does at the CLI."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class _MirrorsAllButAlpha(taaf.solver_examples.SolverFlipLR):
    """The reference solver, except that it answers ``alpha`` wrong: a run
    that differs from the reference run on one game, so the comparison's
    statistics have two runs with variation inside each and between them."""

    async def _answer(self, game: taaf.game.Game) -> None:
        view = game.query()
        name = "black" if view.game_id == "alpha" else "flip"
        game.execute_action(taaf.solver_examples.ANSWERS[name](view), generated_tokens=5)


async def _seed_run(
    job_dir: Path, label: str, n_passes: int = 1, solver: taaf.solver.Solver | None = None
) -> taaf.benchmark.Benchmark:
    """Three games — two the mirror solves, one it cannot — and one pass: a
    deterministic solver plays every pass alike, and the comparison's
    pass-level test warns on data with no variance, which the suite turns
    into an error."""
    bm = taaf.benchmark.Benchmark(
        label=label,
        games=[
            taaf.game_examples.flip_game("alpha"),
            taaf.game_examples.flip_game("beta"),
            taaf.game_examples.constant_game("gamma"),
        ],
        solver=solver if solver is not None else taaf.solver_examples.SolverFlipLR(report_tokens=5),
        n_passes=n_passes,
        job_dir=job_dir,
    )
    await bm.run()
    return bm


async def test_compare_runs_leaves_output_intact_with_one_finished_run(tmp_path: Path) -> None:
    """The renderer refuses fewer than two benchmarks, so proceeding would
    clear the previous comparison and then raise."""
    mod = _load("compare_runs")
    parent = tmp_path / "runs"
    await _seed_run(parent / "run_a", "a", n_passes=1)
    out = parent / "compare"
    out.mkdir(parents=True)
    (out / "index.html").write_text("previous comparison")

    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", str(parent), str(out)]
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 1
    assert (out / "index.html").read_text() == "previous comparison"


async def test_compare_runs_writes_a_comparison_over_the_finished_children(tmp_path: Path) -> None:
    """Two finished runs beside an in-flight one: the comparison covers the
    two, skips the third, and lands where the caller said."""
    mod = _load("compare_runs")
    parent = tmp_path / "runs"
    await _seed_run(parent / "run_a", "a")
    await _seed_run(parent / "run_b", "b", solver=_MirrorsAllButAlpha())
    unfinished = parent / "run_c"
    unfinished.mkdir()
    (unfinished / "benchmark.json").write_text(json.dumps({"label": "c", "game_runs": []}))

    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", str(parent)]
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 0
    index = (parent / "compare" / "index.html").read_text()
    assert "run_a" in index and "run_b" in index and "run_c" not in index


def test_compare_runs_guard_covers_legacy_pkl_run_trees(tmp_path: Path) -> None:
    """The clear guard must use the same run-dir test as the selector: a
    tree of legacy pkl-only runs is valid input, and deleting it is
    unrecoverable."""
    common = _load("_common")
    legacy = tmp_path / "out" / "old_run"
    legacy.mkdir(parents=True)
    (legacy / "benchmark.pkl").write_bytes(b"not really a pickle")

    empty = tmp_path / "empty"
    empty.mkdir()

    assert common.is_run_dir(legacy)
    assert common.holds_run_data(tmp_path / "out")
    assert not common.holds_run_data(empty)


def test_the_script_uses_the_shared_path_mapping() -> None:
    """The URL mapping and the run-dir test live in ``_common``; a copy in
    the script that drifted would delete real runs."""
    mod = _load("compare_runs")
    assert mod.url_to_fs_path.__module__ == "_common"
    assert mod.DEFAULT_ROOT is sys.modules["_common"].DEFAULT_ROOT


async def test_compare_runs_refuses_a_directory_it_did_not_write(tmp_path: Path) -> None:
    """The run-data guard is negative — it names the one shape that must not
    be deleted — so a mistyped out path, or a directory of analysis by hand,
    would reach the clear. Only a directory holding a previous comparison
    (an index.html at its root) or an empty one may be replaced."""
    mod = _load("compare_runs")
    parent = tmp_path / "runs"
    await _seed_run(parent / "run_a", "a")
    await _seed_run(parent / "run_b", "b", solver=_MirrorsAllButAlpha())
    precious = tmp_path / "by_hand"
    precious.mkdir()
    (precious / "notes.md").write_text("a whole afternoon")

    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", str(parent), str(precious)]
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 2
    assert (precious / "notes.md").read_text() == "a whole afternoon"


async def test_compare_runs_replaces_its_own_previous_output(tmp_path: Path) -> None:
    """The other half of the same rule: re-running over a comparison this
    tool wrote must still work, or the guard would make every out dir
    single-use."""
    mod = _load("compare_runs")
    parent = tmp_path / "runs"
    await _seed_run(parent / "run_a", "a")
    await _seed_run(parent / "run_b", "b", solver=_MirrorsAllButAlpha())
    out = parent / "compare"
    out.mkdir(parents=True)
    (out / "index.html").write_text("a previous comparison")
    (out / "per_game").mkdir()

    argv = sys.argv
    try:
        sys.argv = ["compare_runs.py", str(parent), str(out)]
        rc = mod.main()
    finally:
        sys.argv = argv
    assert rc == 0
    assert "run_a" in (out / "index.html").read_text()


def test_a_pull_into_a_directory_from_another_version_is_refused(tmp_path: Path) -> None:
    """``download`` keeps whatever is already under each name, so a reused
    directory would end up holding one version's bytes under another's
    label. The marker is checked before the pull, not merely written after."""
    pull = _load("pull_kaggle_run")
    out = tmp_path / "pulled"
    out.mkdir()
    stamp = "owner/slug version 5 scriptVersionId 999\n"
    (out / pull.MARKER).write_text("owner/slug version 3 scriptVersionId 111\n")

    refusal = pull.unusable_out_dir(out, stamp)
    assert refusal is not None and "different source" in refusal
    # The same version resumes: an interrupted pull must be restartable.
    (out / pull.MARKER).write_text(stamp)
    assert pull.unusable_out_dir(out, stamp) is None


def test_a_pull_into_an_unstamped_non_empty_directory_is_refused(tmp_path: Path) -> None:
    """Files this tool never stamped cannot be shown to belong to any
    version, and would be kept and labelled with the requested one. An empty
    or absent directory is fresh and fine."""
    pull = _load("pull_kaggle_run")
    stamp = "owner/slug version 5 scriptVersionId 999\n"

    used = tmp_path / "used"
    used.mkdir()
    (used / "benchmark.json").write_text("{}")
    assert pull.unusable_out_dir(used, stamp) is not None

    assert pull.unusable_out_dir(tmp_path / "fresh", stamp) is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert pull.unusable_out_dir(empty, stamp) is None


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


@pytest.mark.parametrize(
    "labels",
    [
        ['a"b', "c", "plain"],
        ["back\\slash", 'q"uote', "tab\there", ""],
    ],
)
def test_relabel_survives_successive_labels_with_json_specials(tmp_path: Path, labels: list[str]) -> None:
    """Each relabel must find exactly the value the previous one wrote, however
    that value had to be escaped -- the failure to pin is an escaped quote
    being read as the end of the string."""
    pull = _load("pull_kaggle_run")
    bench = _write(
        tmp_path / "benchmark.json", '{\n  "label": "demo_duck_kaggle",\n  "n_passes": 1,\n  "game_runs": []\n}'
    )
    for label in labels:
        pull.relabel(bench, label)
        data = json.loads(bench.read_text())
        assert data["label"] == label
        assert data["n_passes"] == 1 and data["game_runs"] == []


def test_relabel_finds_the_top_level_key_only(tmp_path: Path) -> None:
    """Whatever the file's key order or whitespace, and a ``label`` nested in a
    game run is not the one being rewritten."""
    pull = _load("pull_kaggle_run")
    bench = _write(
        tmp_path / "benchmark.json", '{"game_runs":[{"label":"inner","x":[1,2]}],"n":{"label":3},"label":"old"}'
    )
    pull.relabel(bench, "new")
    data = json.loads(bench.read_text())
    assert data["label"] == "new"
    assert data["game_runs"][0]["label"] == "inner" and data["n"]["label"] == 3


@pytest.mark.parametrize("text", ['{"n_passes": 1}', '{"label": 5}', "[]"])
def test_relabel_refuses_a_document_without_a_string_label(tmp_path: Path, text: str) -> None:
    pull = _load("pull_kaggle_run")
    bench = _write(tmp_path / "benchmark.json", text)
    with pytest.raises(SystemExit):
        pull.relabel(bench, "new")
    assert bench.read_text() == text, "a refused relabel must leave the file untouched"
