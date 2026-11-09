"""The production engines come up and play, on a B200.

Nothing else covers production startup: the GPU matrix runs backend
mechanics on a small model. One column, one engine boot:

- ``duck_direct_boot``: the 27B FP8 column (``DuckAgent.build(engine="qwen3.8-27b")``)
  on the fork in-process, started through the solver's normal path and
  kept alive past its first game (``LLMAgentSolver.keep_backend``). The
  first game must beat taps level 1: a smoke of the whole production loop.
  Three more games then run against the same live engine and check that a
  traced game's steps nest (requirements U5): plain, with the grid image
  riding every step, and under a window small enough to compact.

Flash-Next has no column behind vLLM's OpenAI-compatible server: the build
refuses it there, since that client names no conversation threads for the
host tier's park mode (``ullm.backends.deployment.build``).

Each boot is put through the backend matrix before its first game
(``LLMConfig.verify_backend``), so a configuration the engine refuses fails
here by name rather than as a lost game.

Deliberately absent: any assertion about the ``verify_serving_bug``
check. Its verdict is a statistical read of boot-varying engine behavior,
so no deterministic assertion about it exists in either direction; the
check is opt-in and untested by policy (see
EngineFactory.verify_serving_bug).

Run via ``make test-duck-engine-srun`` (needs a B200), not the standard
GPU matrix columns. The traced games' job directories stay after the run, in
``.gpu-test-logs/duck_engine/<column>`` under the checkout the column ran from
(on the GPU stage, shared storage), so a failed game's trace can be read.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest


def _kept_jobs(request: pytest.FixtureRequest, column: str) -> Path:
    """Where a column's games write their job directories: in the checkout,
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
            pytest.fail("gpu tests selected but no GPU is visible; run under `make test-duck-engine-srun`")
        pytest.skip("no GPU available")
    if torch.cuda.get_device_properties(0).total_memory < 100 * 1024**3:
        pytest.skip("the production engines need a B200-class card")
    if "direct_boot" not in (request.config.option.keyword or ""):
        pytest.skip("the production engine needs a GPU to itself: its own column selects it")


# A window small enough that a level-1 taps game summarizes within a few
# steps (the first prompt is ~3k tokens and each step adds one to two, and the
# budget is the window less the reply cap and the margin, 5,632): the summary
# and the history rebuilt around it are the duck's rewrites of served history,
# declared twice a summary, and only a game that reaches them exercises those
# declarations. Summarize is what ships (2026-09-26); the window still leaves
# the rebuilt request, the system prompt and a summary of at most twice
# 8,192 // 16, room under the budget.
_COMPACTING = {"reply_cap_tokens": 2048, "context_window_tokens": 8192}
# The fewest steps a game needs before the compacting window can have fired.
_STEPS_TO_COMPACT = 6
# The nesting variants the direct column runs against its one live engine:
# (multimodal_context, overrides).
_NESTING_VARIANTS: list[tuple[str, dict[str, Any]]] = [("", {}), ("current_grid", {}), ("", _COMPACTING)]


async def _play_taps(agent: Any, job_dir: Path | None, *, keep_backend: Any = None) -> Any:
    """One taps game to level 1 under the solver; returns the run. A
    ``KeptBackend`` holder receives the engine when given."""
    import taaf.benchmark
    import taaf.game_api

    from platypus import LLMAgentSolver

    solver = LLMAgentSolver(agent=agent, keep_backend=keep_backend)
    benchmark = taaf.benchmark.Benchmark(
        # stop_at_level=1: the game counts as won the instant one level is
        # completed, so a win assertion is exactly "beat level 1".
        games=[taaf.game_api.GameAPI(env_name="taps-0001", stop_at_level=1)],
        solver=solver,
        job_dir=job_dir,
    )
    await benchmark.run()
    (run,) = benchmark.game_runs
    return run


def _assert_steps_nest(job_dir: Path, run: Any, *, multimodal_context: str, overrides: dict[str, Any]) -> None:
    """A traced taps game's steps nest (requirements U5): every step's
    prompt ids extend the previous prompt + sampled ids. Under exact
    history the backend continues the served ids by tokens and raises on
    an undeclared rewrite, and the collector refuses a non-nesting step
    within an epoch -- so the assertion here is the measured fact on top
    of the enforced one."""
    import taaf.trace

    assert run.state in {"won", "gave_up"}, f"expected a clean game, got state={run.state!r} ({run.solver_note})"
    folders = [p for p in (job_dir / "traces").iterdir() if (p / "_SUCCESS").exists()]
    assert len(folders) == 1
    trace = taaf.trace.load_trace(folders[0])
    assert trace.n_steps >= 2, "a one-step game cannot exercise nesting"
    breaks: list[tuple[int, int, int, int]] = []
    for step in trace.steps:
        if step.extra.get("nests_previous") is not False:
            continue
        previous = trace.steps[step.step_index - 1]
        served = [*previous.input_token_ids, *previous.output_token_ids]
        divergence = next(
            (i for i, (a, b) in enumerate(zip(served, step.input_token_ids)) if a != b),
            min(len(served), len(step.input_token_ids)),
        )
        breaks.append((step.step_index, len(previous.input_token_ids), len(served), divergence))
    # A fresh render after the first step is legitimate only on a declared
    # boundary (a bumped epoch, i.e. a compaction); the plain variants never
    # compact, so for them this is "zero breaks".
    continued = [bool(step.extra.get("continued")) for step in trace.steps]
    epochs = [int(step.extra.get("history_epoch") or 0) for step in trace.steps]
    boundaries = [i for i, c in enumerate(continued) if not c and i > 0]
    assert all(epochs[i] > epochs[i - 1] for i in boundaries), (continued, epochs)
    assert trace.extra["nesting_breaks"] == len(boundaries), (
        f"{len(breaks)} of {trace.n_steps} steps do not extend the previous prompt + sampled ids "
        f"(step_index, previous_prompt_len, previous_prompt+output_len, first_divergence): {breaks}"
    )
    # The compacting variant is the one that narrows the window, not merely
    # the one that overrides something: a variant may carry overrides for an
    # unrelated reason and still never compact.
    if "context_window_tokens" in overrides:
        if not boundaries and trace.n_steps < _STEPS_TO_COMPACT:
            pytest.skip(f"the game ended in {trace.n_steps} steps, before the compacting window could fire")
        assert boundaries, f"the compacting window must have compacted at least once in {trace.n_steps} steps"
    else:
        assert not boundaries, continued
    if multimodal_context:
        assert all(step.media for step in trace.steps), "the grid image must ride every step"


@pytest.mark.gpu
@pytest.mark.asyncio
async def test_duck_direct_boot_plays_and_nests(request: pytest.FixtureRequest) -> None:
    """One boot of the default engine: the bare duck beats taps level 1
    through the solver's own start, then the three nesting variants play
    against the same live engine, handed on through the holder."""
    _require_b200(request)
    jobs = _kept_jobs(request, "duck_direct_boot")
    import platypus.llm_solver
    from duck import DuckAgent

    kept = platypus.llm_solver.KeptBackend()
    first = DuckAgent.build(engine="qwen3.8-27b")
    # The backend matrix against the engine this boot serves, before its first
    # game; the variants below play on the same engine and do not repeat it.
    first.llm_config.verify_backend = True
    run = await _play_taps(first, None, keep_backend=kept)
    try:
        assert run.state == "won", f"expected a level-1 win on taps, got state={run.state!r}"
        assert run.levels_completed >= 1
        assert kept.backend is not None and kept.factory is not None
        for i, (multimodal_context, overrides) in enumerate(_NESTING_VARIANTS):
            # The variant's own cap where it states one, otherwise a reply
            # bounded for a level-1 smoke: the shipped cap is for a scored
            # game, and four games share this engine's time.
            extra = dict[str, Any](overrides)
            reply_cap = extra.pop("reply_cap_tokens", 2048)
            agent = DuckAgent.build(
                engine="qwen3.8-27b", multimodal_context=multimodal_context, reply_cap_tokens=reply_cap, **extra
            )
            agent.llm_config.traces = True
            assert agent.llm_config.exact_history, "the local presets promise exact history"
            job_dir = jobs / f"job{i}"
            # The filled holder: this run serves from the kept engine and closes nothing.
            nested = await _play_taps(agent, job_dir, keep_backend=kept)
            _assert_steps_nest(job_dir, nested, multimodal_context=multimodal_context, overrides=overrides)
    finally:
        await kept.close()
