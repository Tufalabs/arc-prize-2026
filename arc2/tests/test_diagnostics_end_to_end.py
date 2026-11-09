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

import ullm.backend
import ullm.backends.stub
import ullm.factory

import taaf.benchmark
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
from platypus.reference_agents import PromptedFlipAgent
from taaf.game_examples import flip_game

N_GAMES = 2


def _benchmark(
    job_dir: Path, *, process_mode: str = "inprocess", capture: bool = False, scratch: Path | None = None
) -> taaf.benchmark.Benchmark:
    config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.ArithmeticStubBackend()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        capture=capture,
    )
    agent = PromptedFlipAgent(
        answers=("flip",) * 4,
        llm_config=config,
        run_config=RunConfig(
            rng_seed=7,
            process_mode=process_mode,  # type: ignore[arg-type]
            diagnostics_scratch="" if scratch is None else str(scratch),
            diagnostics_archive_interval_s=0.05,
        ),
    )
    return taaf.benchmark.Benchmark(
        games=[flip_game(f"diaggame{i}") for i in range(N_GAMES)],
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
    assert all(e["values"]["state"] in ("won", "submitted") for e in hosted)

    for run in benchmark.game_runs:
        root = _game_tree(job_dir, run.run_id)
        assert root.kind == "infra/game" and root.name == run.game_id
        # The span says how many answers the agent submitted; whether they
        # were right is the scorer's business and never reaches this stream.
        assert root.closed and root.outcome == f"submitted {len(run.history)}"
        metadata_artifact = next(artifact for artifact in root.artifacts if artifact["name"] == "task_metadata")
        metadata = job_block_store(job_dir).load(metadata_artifact["block"], metadata_artifact["codec"])
        assert run.result is not None
        assert metadata["n_to_answer"] == len(run.result.pairs)
        moves = [c for c in root.children if c.kind == "infra/move"]
        (call,) = moves  # one do_move per game
        submits = [e for e in call.events if e["kind"] == "infra/submit"]
        assert len(submits) == len(run.history)  # one submit event per recorded answer
        assert [e["values"]["index"] for e in submits] == list(range(len(run.history)))
        # One infra/submit span per answer, somewhere under the call, carrying
        # the grids it submitted; and the call span holds the standing answer
        # as final_* artifacts, last wins — both are infra's, not the agent's.
        store = job_block_store(job_dir)
        answers = _spans(call, "infra/submit")
        assert [s.name for s in answers] == [f"answer {i + 1}" for i in range(len(run.history))]
        for span, record in zip(answers, run.history, strict=True):
            by_name = {a["name"]: a for a in span.artifacts}
            for q, attempts in enumerate(record.submission.attempts):
                for a, grid in enumerate(attempts):
                    artifact = by_name[f"ask_{q}_attempt_{a + 1}"]
                    assert (store.load(artifact["block"], artifact["codec"]) == grid.data).all()
        final = {a["name"]: a for a in call.artifacts if a["name"].startswith("final_")}
        for q, attempts in enumerate(run.history[-1].submission.attempts):
            for a, grid in enumerate(attempts):
                artifact = final[f"final_{q}_attempt_{a + 1}"]
                assert (store.load(artifact["block"], artifact["codec"]) == grid.data).all()

        # The task the agent was shown, as artifacts on the call span: one
        # image per demonstration grid and one per question, so a viewer
        # renders the task for any agent without one having to log it.
        names = {a["name"] for a in call.artifacts}
        assert {"ask_0", "given_0_question", "given_0_answer"} <= names

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
