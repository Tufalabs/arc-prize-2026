"""The viewer_html slice in a reference must reproduce across runs.

A reference exists so a change can be shown output-invariant. That only
works if the SAME conversation renders byte-identical HTML on the run that
wrote the reference and the run that replays it. Two things break that and
are handled: wall-clock durations (suppressed by deterministic rendering)
and the per-run uuid (replaced with a fixed token). Nothing else may leak.
"""

from __future__ import annotations

from pathlib import Path

import ullm.backend
import ullm.backends.stub
import ullm.factory

import taaf.benchmark
from platypus import LLMAgentSolver, LLMConfig, RunConfig
from platypus.reference import expected_from_run
from platypus.reference_agents import PromptedFlipAgent
from taaf.game_examples import flip_game


async def _run(job_dir: Path) -> taaf.benchmark.Benchmark:
    """One seeded stub-backed game with diagnostics into ``job_dir``."""
    agent = PromptedFlipAgent(
        answers=("flip",) * 3,
        attach_grid_image=True,  # image blocks: the pieces most likely to differ if storage changes
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.ArithmeticStubBackend()),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        ),
        run_config=RunConfig(rng_seed=3, process_mode="inprocess"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[flip_game("reprogame")], solver=LLMAgentSolver(agent=agent), n_passes=1, job_dir=job_dir
    )
    await benchmark.run()
    return benchmark


async def test_viewer_html_is_identical_across_two_runs(tmp_path: Path) -> None:
    """Two runs of the same seeded conversation differ only by run_id and
    durations; the captured HTML must be equal once those are neutralized."""
    (run_a,) = (await _run(tmp_path / "a")).game_runs
    (run_b,) = (await _run(tmp_path / "b")).game_runs
    a = expected_from_run(run_a, tmp_path / "a")
    b = expected_from_run(run_b, tmp_path / "b")
    assert "viewer_html" in a and "viewer_html" in b, "an image-carrying stub run should produce viewer_html"
    assert a["viewer_html"] == b["viewer_html"], (
        "viewer_html differs between two identical runs — something beyond run_id/duration leaked. "
        "First differing page: "
        + next((k for k in a["viewer_html"] if a["viewer_html"].get(k) != b["viewer_html"].get(k)), "?")
    )
    # And it actually rendered the conversation, not an empty shell.
    assert {"index", "game", "move_first", "move_last"} <= a["viewer_html"].keys()
    assert "RUN_ID" in a["viewer_html"]["move_first"]  # the uuid was neutralized in hrefs
    assert '<span class="chunk-time">' not in a["viewer_html"]["move_first"]  # duration span not rendered
    assert "final 0 attempt 1" in a["viewer_html"]["move_first"]  # the answers are part of what is pinned


async def test_no_job_dir_means_no_viewer_html(tmp_path: Path) -> None:
    """Omitted cleanly when there is nothing to render from — the property
    that lets references without HTML stay valid."""
    (run,) = (await _run(tmp_path / "c")).game_runs
    assert "viewer_html" not in expected_from_run(run)  # no job_dir
