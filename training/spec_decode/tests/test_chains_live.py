"""Chains from a recording still playing: only sealed traces TAAF has listed."""

from __future__ import annotations

import json
from pathlib import Path

from drafter.chains import recording_chains


def _trace(recording: Path, run_id: str, sealed: bool) -> None:
    trace = recording / "traces" / run_id
    trace.mkdir(parents=True)
    step = {"step_index": 0, "input_token_ids": [1, 2, 3], "output_token_ids": [4, 5, 6]}
    (trace / "steps.jsonl").write_text(json.dumps(step) + "\n")
    if sealed:
        (trace / "_SUCCESS").write_text("")


def test_a_live_recording_yields_its_sealed_listed_traces(tmp_path):
    recording = tmp_path / "arc_2_results" / "shared" / "long_runs" / "run"
    _trace(recording, "done", sealed=True)
    _trace(recording, "writing", sealed=False)  # a game being sealed right now
    _trace(recording, "unlisted", sealed=True)  # sealed after the last save
    runs = [{"run_id": "done", "game_id": "c3fa4749"}, {"run_id": "writing", "game_id": "4c416de3"}]
    (recording / "benchmark.json").write_text(json.dumps({"game_runs": runs}))
    chains, dropped = recording_chains(recording)
    assert [(chain.trace, chain.game, chain.harness) for chain in chains] == [("done", "c3fa4749", "arc2")]
    assert dropped == {"not in benchmark.json": 1}


def test_a_recording_without_seals_yields_every_trace(tmp_path):
    recording = tmp_path / "arc_3_results" / "shared" / "rec"
    _trace(recording, "a", sealed=False)
    _trace(recording, "b", sealed=False)
    runs = [{"run_id": "a", "game_id": "ar25-0c556536_3"}, {"run_id": "b", "game_id": "ls20-9607627b"}]
    (recording / "benchmark.json").write_text(json.dumps({"game_runs": runs}))
    chains, _ = recording_chains(recording)
    assert [(chain.trace, chain.game) for chain in chains] == [("a", "ar25-0c556536"), ("b", "ls20-9607627b")]
