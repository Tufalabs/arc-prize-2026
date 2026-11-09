"""The duck's production engine comes up and plays, on a B200.

Nothing else covers normal production startup: the rest of the gpu tier runs
backend mechanics on a small model, and everything about the presets that the
cheap tier can reach is config-level only. This boots Flash-Next as its
cluster column serves it -- the e307 NVFP4 prune in-process on the fork, its
PLE table on the host and the CPU KV offload connector beside it -- once, and
keeps it past its first run (``LLMAgentSolver.keep_backend``), since a second
engine in one process wedges its startup. The boot is put through the backend
matrix before the first game (``LLMConfig.verify_backend``), so a
configuration the engine refuses fails here by name rather than as lost
games.

Three runs share the engine. The curated easy set first, asserting a mean
score above 80: the set was built so a competent model solves all of it (its
curation bar was 3/3 on every task), so a shortfall reads as a harness or
serving defect first and a model failure second -- which is what makes a
threshold meaningful on this set and on no other. Then the same set traced
under a window small enough to compact, once pruning and once summarizing,
asking that every step nests on the served history and every fresh render
sits on a declared boundary.

Deliberately absent: any assertion about ``verify_serving_bug``. Its verdict
is a statistical read of boot-varying engine behaviour, so a test of it in
either direction is flaky by construction; the check is opt-in and untested
by policy.

Carries the ``b200`` mark as well as ``gpu``, which is what routes it to a
DGX node: ``make test-gpu-srun`` selects ``gpu and not b200``,
``make test-duck-engine-srun`` selects ``gpu and b200``, and ``test-full``
runs both.

The traced runs' job directories stay after the run, in
``.gpu-test-logs/duck_engine/<column>`` under the checkout the column ran from
(on the GPU stage, shared storage), so a failed game's trace can be read.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

#: Host RAM the engine holds at start, on top of weights and engine: the
#: offload tier the Flash-Next cluster cell claims eagerly (96 GiB) and the
#: PLE table pinned on the host (~48 GiB). A node sized only for the card
#: fails at boot.
HOST_GIB_REQUIRED = 144

#: The engine the boot serves: the one with the largest footprint here, so
#: a node that boots it boots the others.
ENGINE = "qwen3.8-flash-next"


def _kept_jobs(request: pytest.FixtureRequest, column: str) -> Path:
    """Where a column's runs write their job directories: in the checkout,
    not pytest's tmp_path, which a SLURM job keeps in scratch that goes with
    it. Emptied at the start of each run."""
    rootpath: Path = request.config.rootpath
    root = rootpath / ".gpu-test-logs" / "duck_engine" / column
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    return root


def _require_b200(request: pytest.FixtureRequest) -> None:
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        if "gpu" in (request.config.option.markexpr or ""):
            pytest.fail("gpu tests selected but no GPU is visible")
        pytest.skip("no GPU available")
    if torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3:
        pytest.skip("the duck's engine needs a B200-class card")

    import psutil

    if psutil.virtual_memory().available < HOST_GIB_REQUIRED * 1024**3:
        pytest.skip(
            f"the engine holds {HOST_GIB_REQUIRED} GiB of host RAM at start: the offload tier and the PLE table"
        )


# A window small enough that an easy task compacts: eviction and pruning are
# the duck's rewrites of served history, and only a game that reaches them
# exercises their declarations. The budget is window minus reply minus 512,
# 5120 here. The easy set's opening prompts on Flash's processor — system,
# tool schemas (the file tools' among them), the legend and one picture per
# pair and question — measure 2639 to 3486 tokens (B200, job 2918455), so a
# game compacts once its rounds add 1.6k to 2.5k. A 7680 budget compacted 1
# game of 10 in each arm, leaving both arms to one game's variance; one with
# that room compacted 4 pruning and 2 summarizing (B200, job 2613820). The
# floor is the largest opening prompt: a trim cannot remove it, and it must
# still leave a reply and the summary room; the target sits just above it.
# The game cap is the 150k these arms were sized at, not the cell's 500k: a
# game that compaction keeps short of an answer plays on, and at 500k one
# easy game ran 1,151 requests and the pruning arm alone took 52 minutes of
# the column's 90 (job 2918455).
_COMPACTING: dict[str, Any] = {
    "reply_cap_tokens": 4096,
    "context_window_tokens": 9728,
    "compact_to_tokens": 3584,
    "max_tokens_per_game": 150_000,
}

# The summarizing arm's rebuilt history holds more than the opening prompt:
# the summary, which the thorough request lets fill its cap (a sixteenth of
# the window, doubled), and the listing that closes it, a few hundred tokens
# that grow with ``candidate.py``. At 9728 the rebuild filled the 5120 budget
# and 9 of 10 games ended on a rebuild that no longer fit (job 2919078).
# 12288 leaves the largest opening, a 1536-token summary and a 600-token
# listing about 2.1k of rounds before the next summary.
_SUMMARIZING: dict[str, Any] = {**_COMPACTING, "context_window_tokens": 12288}


async def _play_easy(agent: Any, label: str, job_dir: Path | None, kept: Any) -> Any:
    """The easy set under the solver, on the kept engine; returns the benchmark."""
    from platypus import LLMAgentSolver
    from taaf import standard_benchmarks

    benchmark = standard_benchmarks.make_benchmark_easy(LLMAgentSolver(agent=agent, keep_backend=kept), label=label)
    benchmark.job_dir = job_dir
    await benchmark.run()
    return benchmark


def _assert_easy_set_played(benchmark: Any) -> None:
    """The first run: every game played, and the easy set's bar."""
    from taaf import datasets

    runs = benchmark.game_runs
    assert len(runs) == len(datasets.EASY_TASK_IDS)

    crashed = [run.game_id for run in runs if run.state == "crashed"]
    assert not crashed, f"the engine came up but these games crashed: {crashed}"

    generated = sum(record.generated_tokens for run in runs for record in run.history)
    assert generated > 0, "no game generated a token — the engine served nothing"

    # None (gave up with nothing standing) counts as 0: an unscored game is
    # a failed game for this assertion, not a skipped one.
    scores = [run.final_score or 0.0 for run in runs]
    solved = sum(1 for score in scores if score >= 100)
    mean_score = sum(scores) / len(scores)
    # Reaches a human only because the gpu dispatcher runs pytest with `-s`:
    # captured output is discarded for a passing test, stderr included.
    print(
        f"\n[duck-engine] {solved}/{len(runs)} solved, mean score {mean_score:.1f}, "
        f"{generated} generated tokens, states={sorted({run.state for run in runs})}"
    )
    assert mean_score > 80.00, (
        f"mean score {mean_score:.2f} on the curated easy set — every task here was screened to be "
        f"reliably solvable, so look for a harness or serving defect before blaming the model"
    )


def _assert_steps_nest_across_compaction(benchmark: Any, job_dir: Path, arm: str) -> None:
    """A traced game's steps nest: every step's prompt ids extend the
    previous prompt + sampled ids, except on a declared boundary. Under the
    preset's exact history the backend continues the served ids by tokens
    — images included, on this engine — and raises on an undeclared
    rewrite; the collector refuses a non-nesting step within an epoch. So
    the assertions below are the measured fact on top of the enforced one:
    every fresh render after the first sits on a bumped epoch, no game
    breaks more often than it declares, and the compacting window compacted.
    A declared boundary need not break: the turn manager declares a
    malformed-output retry's rollback on the request after the one it
    re-sampled, and that request extends the served ids untouched (job
    2918455). Declaring a rewrite that did not happen costs a fresh render;
    a rewrite left undeclared is the defect.

    The summarize arm's compaction is a request the duck renders WITHOUT
    thinking on a backend that has it on, the one shape where the backend's
    reasoning parser can disagree with the template; under exact history the
    disagreement is a refused witness on every summary, which crashes its
    game. So both arms ask the same: no crash, and a game that compacted."""
    import taaf.trace

    runs = benchmark.game_runs
    crashed = [(run.game_id, run.solver_note) for run in runs if run.state == "crashed"]
    assert not crashed, f"a crash under exact history is an undeclared rewrite until proven otherwise: {crashed}"

    folders = [p for p in (job_dir / "traces").iterdir() if (p / "_SUCCESS").exists()]
    assert len(folders) == len(runs), f"{len(folders)} traces for {len(runs)} games"
    compacted = 0
    for folder in folders:
        trace = taaf.trace.load_trace(folder)
        continued = [bool(step.extra.get("continued")) for step in trace.steps]
        epochs = [int(step.extra.get("history_epoch") or 0) for step in trace.steps]
        # A fresh render after the first step is legitimate only on a
        # declared boundary: a bumped epoch, which is a compaction here --
        # a prune trim or a summary alike -- or a malformed-output retry.
        boundaries = [i for i, c in enumerate(continued) if not c and i > 0]
        assert all(epochs[i] > epochs[i - 1] for i in boundaries), (trace.run_id, continued, epochs)
        breaks = trace.extra["nesting_breaks"]
        assert breaks <= len(boundaries), (trace.run_id, trace.extra, continued)
        # The pictures ride the first prompt and are continued from there;
        # a later entry whose opening block was evicted carries none, since
        # the kernel holds the puzzle and the duck does not re-send them.
        assert trace.steps[0].media, "the task's pictures must land as media on the opening step"
        # A break is a compaction's: a retry's boundary extends the served ids.
        if breaks:
            compacted += 1
    print(f"\n[duck-{arm}] {compacted}/{len(folders)} games compacted at least once, all boundaries declared")
    assert compacted, "the compacting window must have compacted at least one game, or this proves nothing"


@pytest.mark.gpu
@pytest.mark.b200
async def test_duck_engine_starts_plays_and_nests_across_compaction(request: pytest.FixtureRequest) -> None:
    """One boot of Flash-Next: the easy set through the solver's own start,
    then the easy set traced under a compacting window, pruning and then
    summarizing, against the same live engine handed on through the holder."""
    _require_b200(request)
    jobs = _kept_jobs(request, "duck_engine")
    from duck import DuckAgent, presets
    from transformers import AutoTokenizer

    import platypus.llm_solver

    kept = platypus.llm_solver.KeptBackend()
    first = DuckAgent.build(engine=ENGINE)
    factory = first.llm_config.backend_factory
    tokenizer = AutoTokenizer.from_pretrained(factory.engine_args.model)  # type: ignore[union-attr]
    assert tokenizer.encode("<tool_call>") == [presets.QWEN_TOOL_CALL_TOKEN_ID]
    # The backend matrix against the engine this boot serves, before its first
    # game; the runs below play on the same engine and do not repeat it.
    first.llm_config.verify_backend = True
    try:
        _assert_easy_set_played(await _play_easy(first, "duck_engine_easy", None, kept))
        assert kept.backend is not None and kept.factory is not None
        for arm, strategy, window in (("nests", "prune", _COMPACTING), ("summarizes", "summarize", _SUMMARIZING)):
            agent = DuckAgent.build(engine=ENGINE, compaction_strategy=strategy, **window)
            agent.llm_config.traces = True
            assert agent.llm_config.exact_history, "the columns promise exact history"
            job_dir = jobs / arm
            # The filled holder: this run serves from the kept engine and closes nothing.
            benchmark = await _play_easy(agent, f"duck_engine_{arm}", job_dir, kept)
            _assert_steps_nest_across_compaction(benchmark, job_dir, arm)
    finally:
        await kept.close()
