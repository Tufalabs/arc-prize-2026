"""The routing replay reads TAAF recordings, a bounded number of passes per game."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_route_stats.py"
spec = importlib.util.spec_from_file_location("prune_route_stats", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _recording(root: Path, games: dict[str, int]) -> Path:
    runs = []
    for game, passes in games.items():
        for index in range(passes):
            run_id = f"{game}-{index}"
            (root / "traces" / run_id).mkdir(parents=True)
            (root / "traces" / run_id / "steps.jsonl").write_text("{}\n")
            runs.append({"run_id": run_id, "game_id": f"{game}_{index}", "final_score": float(index)})
    (root / "benchmark.json").write_text(json.dumps({"game_runs": runs}))
    return root


def test_a_recording_yields_at_most_the_asked_passes_per_game(tmp_path):
    recording = _recording(tmp_path / "long_run", {"ar25-0c556536": 3, "ls20-9607627b": 2})
    picked = module.discover_recording(recording, passes=2)
    assert sorted(item["game_id"] for item in picked) == ["ar25-0c556536"] * 2 + ["ls20-9607627b"] * 2
    assert {item["source"] for item in picked} == {"long_run"}
    assert len(module.discover_recording(recording)) == 5


def test_a_scored_run_without_its_trace_is_refused(tmp_path):
    recording = _recording(tmp_path / "rec", {"ar25": 1})
    (recording / "traces" / "ar25-0" / "steps.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="without its trace"):
        module.discover_recording(recording)


def test_recording_arguments_take_an_optional_pass_count():
    assert module._recording_arg("/data/x/long_run:3") == (Path("/data/x/long_run"), 3)
    assert module._recording_arg("/data/x/long_run") == (Path("/data/x/long_run"), None)


def _sealed(root: Path, run_id: str, sealed: bool = True) -> None:
    trace = root / "traces" / run_id
    trace.mkdir(parents=True)
    (trace / "steps.jsonl").write_text("{}\n")
    if sealed:
        (trace / "_SUCCESS").write_text("")


def test_a_running_recording_yields_its_sealed_games_before_any_benchmark(tmp_path):
    recording = tmp_path / "running"
    _sealed(recording, "a")
    _sealed(recording, "b", sealed=False)  # still playing
    with pytest.raises(FileNotFoundError, match="benchmark.json"):
        module.discover_recording(recording)
    picked = module.discover_recording(recording, finished_only=True)
    assert [item["run_id"] for item in picked] == ["a"] and picked[0]["game_id"] == ""
    # A benchmark being rewritten reads as none; a saved one labels the games.
    (recording / "benchmark.json").write_text('{"game_runs": [')
    assert [item["run_id"] for item in module.discover_recording(recording, finished_only=True)] == ["a"]
    (recording / "benchmark.json").write_text(json.dumps({"game_runs": [{"run_id": "a", "game_id": "ls20-9607627b_1"}]}))
    assert module.discover_recording(recording, finished_only=True)[0]["game_id"] == "ls20-9607627b"
    assert not module.recording_done(recording)
    (recording / "benchmark.json").write_text(json.dumps({"game_runs": [], "end_time": "2026-09-30T16:00:00"}))
    assert module.recording_done(recording)


def test_every_run_belongs_to_exactly_one_shard_whatever_else_is_listed():
    run_ids = [f"run{index:03d}" for index in range(200)]
    owners = {run_id: [shard for shard in range(5) if module.assigned(run_id, shard, 5)] for run_id in run_ids}
    assert all(len(shards) == 1 for shards in owners.values())
    assert {shards[0] for shards in owners.values()} == set(range(5))


def test_a_claim_goes_to_one_worker_and_back_to_it_after_a_requeue(tmp_path):
    claim = tmp_path / "run.claim"
    assert module.take_claim(claim, "2830001")
    assert not module.take_claim(claim, "2830002")
    # A requeued job keeps its id and resumes the rollout it held.
    assert module.take_claim(claim, "2830001")
    assert claim.read_text() == "2830001"
