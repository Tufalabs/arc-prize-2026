"""Diagnostics end to end: infra spans + llm/turn spans through a real
benchmark run, in both process modes, with capture independence and the
minimal_diagnostics off-switch.

Runs on the cheap tier: ArithmeticStubBackend answers the agent's sums,
so streams carry real request-dependent traffic without a model. The
subprocess variant is the topology proof (the CHILD writes the game
stream); everything else is per-game behavior the process boundary
cannot influence and runs in-process."""

from __future__ import annotations

import json
from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.diagnostics import (
    SpanNode,
    archive_parts,
    blocks_dir,
    build_span_tree,
    game_stream_path,
    job_block_store,
    monitor_dir,
    monitor_files,
    read_game_events,
    read_run_events,
    run_stream_path,
)
from platypus.host_monitor import read_pressure
from platypus.reference import game_run_dir
from reference_agents import PromptedRandomAgent

N_GAMES = 2


def _benchmark(
    job_dir: Path, *, process_mode: str = "inprocess", capture: bool = False, scratch: Path | None = None
) -> taaf.benchmark.Benchmark:
    config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.ArithmeticStubBackend()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        capture=capture,
    )
    agent = PromptedRandomAgent(
        max_moves=4,
        llm_config=config,
        run_config=RunConfig(
            rng_seed=7,
            process_mode=process_mode,  # type: ignore[arg-type]
            diagnostics_scratch="" if scratch is None else str(scratch),
            diagnostics_archive_interval_s=0.05,
        ),
    )
    return taaf.benchmark.Benchmark(
        games=[ExampleGame(label=f"diaggame{i}") for i in range(N_GAMES)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
    )


def _game_tree(job_dir: Path, run_id: str) -> SpanNode:
    (root,) = build_span_tree(read_game_events(job_dir, run_id))
    return root


def _spans(node: SpanNode, kind: str) -> list[SpanNode]:
    found = [node] if node.kind == kind else []
    for child in node.children:
        found.extend(_spans(child, kind))
    return found


async def _run_and_check_structure(
    job_dir: Path, process_mode: str, scratch: Path | None = None
) -> taaf.benchmark.Benchmark:
    benchmark = _benchmark(job_dir, process_mode=process_mode, scratch=scratch)
    await benchmark.run()

    # Run-level stream: the infra/run span plus one game_hosted reference
    # per game, pointing at per-game streams that exist.
    run_events = read_run_events(job_dir)
    (run_root,) = build_span_tree(run_events)
    assert run_root.kind == "infra/run" and run_root.closed and run_root.outcome == "ok"
    hosted = [e for e in run_root.events if e["kind"] == "game_hosted"]
    assert {e["values"]["run_id"] for e in hosted} == {run.run_id for run in benchmark.game_runs}
    assert all(e["values"]["state"] in ("won", "gave_up") for e in hosted)

    for run in benchmark.game_runs:
        root = _game_tree(job_dir, run.run_id)
        assert root.kind == "infra/game" and root.name == run.game_id
        assert root.closed and root.outcome == run.state  # I12: terminal state stamped
        metadata_artifact = next(artifact for artifact in root.artifacts if artifact["name"] == "game_metadata")
        metadata = job_block_store(job_dir).load(metadata_artifact["block"], metadata_artifact["codec"])
        assert metadata == {
            "number_of_levels": run.number_of_levels,
            "base_actions_per_level": run.base_actions_per_level,
        }
        moves = [c for c in root.children if c.kind == "infra/move"]
        executed = [m for m in moves if any(e["kind"] == "infra/action" for e in m.events)]
        assert len(executed) == len(run.history)  # one action event per executed action
        actions = [e for move in executed for e in move.events if e["kind"] == "infra/action"]
        assert [a["values"]["name"] for a in actions] == [rec.action.id.name for rec in run.history]
        assert all(move.outcome and move.outcome.startswith("executed ") for move in executed)

        # Frame context on move spans: every executed move carries its
        # input and result frames; "before" starts at move 1 (move 0 has
        # no previous move).
        for i, move in enumerate(executed):
            names = {a["name"] for a in move.artifacts}
            assert {"frame_at", "frame_after"} <= names
            assert ("frame_before" in names) == (i > 0)

        # Level-transition frame semantics (block keys are content
        # hashes, so frame identity is exact): the winning move's "after"
        # is the WIN frame — the same frame the delivery span shows as
        # "at" (with the winning move's input as "before", and no
        # "after"); the next move's "before" is the win frame too.
        def frame_blocks(node: SpanNode) -> dict[str, str]:
            return {a["name"]: a["block"] for a in node.artifacts if a["name"].startswith("frame_")}

        for i, move in enumerate(moves):
            action = next((e for e in move.events if e["kind"] == "infra/action"), None)
            if action is None or action["values"]["outcome"] != "level_won":
                continue
            win_frame = frame_blocks(move)["frame_after"]
            delivery = moves[i + 1]
            assert delivery.outcome == "delivery" and delivery.name == "deliver level_won"
            delivery_frames = frame_blocks(delivery)
            assert delivery_frames["frame_at"] == win_frame
            assert delivery_frames["frame_before"] == frame_blocks(move)["frame_at"]
            assert "frame_after" not in delivery_frames
            if i + 2 < len(moves):
                assert frame_blocks(moves[i + 2])["frame_before"] == win_frame

        if run.state == "gave_up":  # the giving-up do_move gets its own stamped span
            give_up_move = moves[-1]
            assert give_up_move.outcome == "gave_up"
            names = {a["name"] for a in give_up_move.artifacts}
            assert "frame_at" in names and "frame_after" not in names  # no action, no result frame
            assert root.kv["give_up_reason"].startswith("reached max_moves")

        # llm/turn spans nest under the move that made them, carry spend,
        # and store a rehydratable conversation artifact.
        turns = _spans(root, "llm/turn")
        assert turns and all(t.self_generated > 0 and t.self_uncached > 0 for t in turns)
        assert all(any(a["name"] == "conversation" for a in t.artifacts) for t in turns)

        # The game span's inclusive totals equal what the game reported
        # into TAAF (the I1 numbers) — the per-span half of the three-way
        # reconciliation.
        reported_generated = sum(rec.generated_tokens for rec in run.history) + run.final_generated_tokens
        reported_uncached = sum(rec.uncached_input_tokens for rec in run.history) + run.final_uncached_input_tokens
        assert root.incl_generated == reported_generated > 0
        assert root.incl_uncached == reported_uncached > 0

    # The host monitors ran for the run: its last pressure interval was
    # written when they stopped, and no segment was left half-written.
    assert "monitor/pressure-0001.jsonl" in monitor_files(job_dir)
    assert all(not name.endswith(".part") for name in monitor_files(job_dir))
    assert list(read_pressure(job_dir)), "at least the closing interval"
    return benchmark


async def test_streams_inprocess(tmp_path: Path) -> None:
    await _run_and_check_structure(tmp_path / "job", "inprocess")


async def test_streams_written_by_subprocess_child(tmp_path: Path) -> None:
    await _run_and_check_structure(tmp_path / "job", "subprocess")


async def _check_archived(tmp_path: Path, process_mode: str) -> None:
    """Streams and blocks written to scratch end up archived in the job dir:
    every structure check above reads them back from the parts, nothing is
    loose in the job dir, and the run's own dir under scratch is gone — only
    that one: the root is the job's (``TAAF_LOCAL_JOB_ROOT`` holds its venvs)."""
    job_dir, scratch = tmp_path / "job", tmp_path / "scratch"
    benchmark = await _run_and_check_structure(job_dir, process_mode, scratch=scratch)
    assert archive_parts(job_dir), "no archive part in the job dir"
    assert not run_stream_path(job_dir).exists() and not blocks_dir(job_dir).exists()
    assert not monitor_dir(job_dir).exists(), "the monitors wrote to scratch and were archived"
    for run in benchmark.game_runs:
        assert not game_stream_path(game_run_dir(job_dir, run.run_id)).exists()
    assert not (scratch / "diagnostics").exists() and scratch.is_dir()


async def test_streams_archived_from_scratch_inprocess(tmp_path: Path) -> None:
    await _check_archived(tmp_path, "inprocess")


async def test_streams_archived_from_scratch_by_subprocess_children(tmp_path: Path) -> None:
    """Each child's pack is its own pid's, archived once the child exits."""
    await _check_archived(tmp_path, "subprocess")


async def test_capture_and_diagnostics_are_independent(tmp_path: Path) -> None:
    """Both sinks on: two separate file sets, no cross-references, and
    per-game token totals that AGREE (both derive from the same calls)."""
    job_dir = tmp_path / "job"
    benchmark = _benchmark(job_dir, capture=True)
    await benchmark.run()

    for run in benchmark.game_runs:
        game_dir = game_run_dir(job_dir, run.run_id)
        capture_records = [
            json.loads(line) for line in (game_dir / "capture" / "records.jsonl").read_text().splitlines()
        ]
        root = _game_tree(job_dir, run.run_id)
        turns = _spans(root, "llm/turn")

        captured_generated = sum(r["outcome"]["usage"]["completion_tokens"] for r in capture_records)
        captured_uncached = sum(
            r["outcome"]["usage"]["prompt_tokens"] - r["outcome"]["usage"]["cached_tokens"] for r in capture_records
        )
        assert sum(t.self_generated for t in turns) == captured_generated > 0
        assert sum(t.self_uncached for t in turns) == captured_uncached > 0

        # Independence: the span stream never references capture's files
        # (capture block hashes keyed by content vs diagnostics keys by
        # encoded bytes — spot-check no capture path strings leak).
        stream_text = game_stream_path(game_dir).read_text()
        assert "capture" not in stream_text


async def test_reconciliation_standalone(tmp_path: Path) -> None:
    """reconcile_job_dir agrees on a clean run and names the game when a
    stream's totals are tampered with."""
    from platypus.diagnostics import reconcile_job_dir

    job_dir = tmp_path / "job"
    benchmark = _benchmark(job_dir)
    await benchmark.run()
    assert reconcile_job_dir(job_dir) == []

    victim = benchmark.game_runs[0]
    stream = game_stream_path(game_run_dir(job_dir, victim.run_id))
    lines = stream.read_text().splitlines()
    tampered = [
        line.replace('"incl_generated":', '"incl_generated":9', 1) if '"e":"close"' in line else line for line in lines
    ]
    stream.write_text("\n".join(tampered) + "\n")
    mismatches = reconcile_job_dir(job_dir)
    assert len(mismatches) == 1 and victim.game_id in mismatches[0]


async def test_minimal_diagnostics_writes_nothing(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    benchmark = _benchmark(job_dir)
    await benchmark.run(minimal_diagnostics=True)
    assert not run_stream_path(job_dir).exists()
    assert not (job_dir / "diag_blocks").exists()
    assert not monitor_dir(job_dir).exists(), "the host monitors are diagnostics too"
    for run in benchmark.game_runs:
        assert not game_stream_path(game_run_dir(job_dir, run.run_id)).exists()
