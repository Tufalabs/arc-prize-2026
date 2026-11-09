"""The LLMAgent/LLMAgentSolver seam end to end (U1 minimal, U6, U8, I1
token reporting).

The solver's topology is under test everywhere: the host backend sits
behind the always-on Unix-socket wrapper and every game talks to its own
``ProxyBackend`` client — including the free-tier stub columns, so plain
``make test`` exercises the real wire path. The api-tier test runs the
same agent against a cheap real model."""

from __future__ import annotations

import copy
import os
import pickle
from dataclasses import dataclass
from typing import Callable

import pytest
import taaf.benchmark as taaf_benchmark
import ullm.backend
import ullm.backends.proxy
import ullm.backends.stub as ullm_stub
import ullm.conv as ullm_conv
import ullm.factory
from taaf.game import GameRun
from taaf.game_examples import ExampleGame

import platypus.llm_solver
from platypus import (
    Action,
    GiveUp,
    LLMAgent,
    LLMAgentSolver,
    LLMConfig,
    LLMMoveContext,
    Observation,
    RunConfig,
)
from reference_agents import PromptedRandomAgent


def _stub_llm_config() -> LLMConfig:
    """A '8'-forever stub behind the trivial factory; the solver still
    serves it over the socket, so children need no stub-side state."""
    backend = ullm_stub.StubBackend.from_script("8", on_exhausted="loop")
    return LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
    )


def _agent(
    mode: str, *, max_moves: int = 4, max_llm_calls: int | None = None, profile: str = "play", seed: int = 7
) -> PromptedRandomAgent:
    return PromptedRandomAgent(
        max_moves=max_moves,
        max_llm_calls=max_llm_calls,
        profile=profile,
        randomize_sum=False,  # the scripted stub can only ever answer "8"
        llm_config=_stub_llm_config(),
        run_config=RunConfig(process_mode=mode, rng_seed=seed),  # type: ignore[arg-type]
    )


def _token_trace(run: GameRun) -> tuple[tuple[int, int], ...]:
    per_action = tuple((rec.generated_tokens, rec.uncached_input_tokens) for rec in run.history)
    return (*per_action, (run.final_generated_tokens, run.final_uncached_input_tokens))


async def _run(agent: PromptedRandomAgent, n_games: int = 1) -> list[GameRun]:
    benchmark = taaf_benchmark.Benchmark(
        games=[ExampleGame(label=f"game{i}") for i in range(n_games)],
        solver=LLMAgentSolver(agent=agent),
    )
    await benchmark.run()
    return benchmark.game_runs


@pytest.mark.parametrize("timeout", [None, 17.0])
async def test_solver_passes_configured_backend_watchdog(
    monkeypatch: pytest.MonkeyPatch, timeout: float | None
) -> None:
    original_start = ullm.backends.proxy.start_backend_server
    observed: list[float | None] = []

    async def start(
        backend: ullm.backend.Backend,
        socket_path: str,
        *,
        default_inactivity_timeout_s: float | None,
    ) -> ullm.backends.proxy.BackendHostHandle:
        handle = await original_start(backend, socket_path, default_inactivity_timeout_s=default_inactivity_timeout_s)
        observed.append(handle.state.default_inactivity_timeout_s)
        return handle

    monkeypatch.setattr(ullm.backends.proxy, "start_backend_server", start)
    # Process environment must not silently override deployment configuration.
    monkeypatch.setenv("ULLM_PROXY_INACTIVITY_S", "0.00001")
    agent = _agent("inprocess", max_moves=1)
    agent.run_config.backend_inactivity_timeout_s = timeout
    runs = await _run(agent)
    assert observed == [timeout]
    assert runs[0].state == "gave_up"
    assert runs[0].history[0].generated_tokens > 0


async def test_keep_backend_leaves_the_engine_open_for_a_second_run() -> None:
    """A solver asked to keep its backend hands the started factory and backend
    to the caller with nothing but the socket closed, and a second solver plays
    against that same backend; the caller closes it."""
    kept = platypus.llm_solver.KeptBackend()
    benchmark = taaf_benchmark.Benchmark(
        games=[ExampleGame(label="game0")], solver=LLMAgentSolver(agent=_agent("subprocess"), keep_backend=kept)
    )
    await benchmark.run()
    assert kept.backend is not None and kept.factory is not None
    assert isinstance(benchmark.solver, LLMAgentSolver) and benchmark.solver.keep_backend is kept, (
        "one holder across the copies"
    )
    assert pickle.loads(pickle.dumps(kept)).backend is None, "a pickle carries an empty holder"
    # A second run over the same engine: its own factory spec is never started.
    again = taaf_benchmark.Benchmark(
        games=[ExampleGame(label="game1")], solver=LLMAgentSolver(agent=_agent("subprocess"), keep_backend=kept)
    )
    await again.run()
    (run,) = again.game_runs
    assert run.history and run.history[0].generated_tokens > 0
    assert kept.backend is not None, "the reusing run closed nothing"
    await kept.close()


_TAKEN_OVER: list[ullm.backend.Backend] = []


@dataclass
class _HostTakeover(PromptedRandomAgent):
    """Takes the run over on the host instead of playing; the benchmark runs
    a deep copy of it, so what it saw goes to the module list."""

    async def run_on_host(self, backend: ullm.backend.Backend, job_dir: object) -> bool:
        """Note the started backend and keep every game unstarted."""
        del job_dir
        _TAKEN_OVER.append(backend)
        return True


async def test_an_agent_can_take_the_run_over_on_the_host() -> None:
    """The traffic replay's seam: the solver starts and gates the backend as
    for a game run, then hands it to the agent. No game plays, and each ends
    given up with the reason on it rather than crashed at teardown."""
    _TAKEN_OVER.clear()
    agent = _HostTakeover(
        max_moves=1,
        profile="play",
        randomize_sum=False,
        llm_config=_stub_llm_config(),
        run_config=RunConfig(process_mode="inprocess", rng_seed=7),  # type: ignore[arg-type]
    )
    runs = await _run(agent, n_games=2)
    (backend,) = _TAKEN_OVER
    assert isinstance(backend, ullm_stub.StubBackend)
    assert [run.state for run in runs] == ["gave_up", "gave_up"]
    assert all(not run.history and "took the run over" in (run.solver_note or "") for run in runs)


async def test_prompted_random_agent_through_subprocess() -> None:
    # The full stack: stub backend behind the host socket, per-game
    # ProxyBackend clients inside spawned children, one LLM call per
    # decision move, per-action token attribution (I1), and the solver's
    # host-vs-reported token cross-check (it raises on mismatch).
    runs = await _run(_agent("subprocess"), n_games=2)
    for run in runs:
        assert run.state == "gave_up"  # max_moves=4 random moves don't win ExampleGame
        assert len(run.history) == 4
        for rec in run.history:
            assert rec.generated_tokens > 0  # each decision move made one LLM call
            assert rec.uncached_input_tokens > 0
        # The give-up decision made no LLM call, so no trailing tokens.
        assert run.final_generated_tokens == 0


async def test_cross_mode_equivalence_including_tokens() -> None:
    # I9: identical outcomes AND identical per-action token counts across
    # process modes under fixed seeds.
    traces = {}
    for mode in ("inprocess", "subprocess"):
        runs = await _run(_agent(mode), n_games=2)
        traces[mode] = [
            (run.state, run.levels_completed, tuple(r.action.id.name for r in run.history), _token_trace(run))
            for run in runs
        ]
    assert traces["inprocess"] == traces["subprocess"]


async def test_max_llm_calls_caps_seam_checks() -> None:
    # The first max_llm_calls decision moves carry LLM tokens; later moves
    # are pure random with zero-token ActionRecords.
    runs = await _run(_agent("inprocess", max_llm_calls=2))
    run = runs[0]
    assert len(run.history) == 4
    assert all(rec.generated_tokens > 0 for rec in run.history[:2])
    assert all(rec.generated_tokens == 0 for rec in run.history[2:])


async def test_give_up_tokens_land_on_finish_game() -> None:
    # R11.03: tokens spent on a move that produced no action (the give-up
    # decision) are reported through finish_game, not an ActionRecord.
    @dataclass
    class CheckThenQuitAgent(LLMAgent):
        async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp:
            turn_manager = self.new_turn_manager(ctx, profile="play", system_prompt="calc")
            turn_manager.conversation.elements.append(
                ullm_conv.UserPrompt(content=ullm_conv.StringContent(text="3+5?"))
            )
            await turn_manager.run()
            return GiveUp("done thinking")

    agent = CheckThenQuitAgent(
        llm_config=_stub_llm_config(),
        run_config=RunConfig(process_mode="inprocess", rng_seed=0),
    )
    benchmark = taaf_benchmark.Benchmark(games=[ExampleGame()], solver=LLMAgentSolver(agent=agent))
    await benchmark.run()
    run = benchmark.game_runs[0]
    assert run.state == "gave_up"
    assert len(run.history) == 0
    assert run.final_generated_tokens > 0
    assert run.final_uncached_input_tokens > 0


async def test_unknown_sampling_profile_is_loud() -> None:
    runs = await _run(_agent("inprocess", profile="nope"))
    run = runs[0]
    assert run.state == "crashed"
    assert run.solver_note is not None and "unknown sampling profile 'nope'" in run.solver_note


def test_llm_agent_template_pickles_and_deepcopies() -> None:
    # A1/A3: the template (unstarted factory spec included) must pickle
    # and deepcopy; the solver only ever starts a copy of the factory.
    agent = _agent("subprocess")
    for clone in (pickle.loads(pickle.dumps(agent)), copy.deepcopy(agent)):
        assert isinstance(clone, PromptedRandomAgent)
        assert clone.llm_config.backend_factory is not None
        assert sorted(clone.llm_config.sampling_profiles) == ["play"]


def _api_key(env_key: str) -> str:
    if os.environ.get("ULLM_API_TESTS") != "1":
        pytest.skip("api tests require explicit opt-in: ULLM_API_TESTS=1")
    key = os.environ.get(env_key)
    if not key:
        pytest.fail(f"ULLM_API_TESTS=1 but {env_key} is not set")
    return key


def _openrouter_backend() -> "ullm.backend.Backend":
    import ullm.backends.oai as _oai

    return _oai.OpenAIServer(
        base_url="https://openrouter.ai/api/v1",
        api_key=_api_key("OPENROUTER_API_KEY"),
        model="qwen/qwen3-30b-a3b-instruct-2507",
        send_back_reasoning="none",
    )


def _anthropic_backend() -> "ullm.backend.Backend":
    import ullm.backends.anthropic as _anthropic

    # haiku rejects adaptive thinking; run the column thinking-free (same
    # choice as the ullm matrix).
    return _anthropic.Anthropic(
        api_key=_api_key("ANTHROPIC_API_KEY"), model="claude-haiku-4-5", set_adaptive_thinking=False
    )


@pytest.mark.gpu
async def test_prompted_random_agent_on_gpu_solver_engine(request: pytest.FixtureRequest) -> None:
    # The full stack against a real LOCAL engine: the solver starts its own
    # engine from the (unstarted, picklable) factory, serves it over the
    # socket, and subprocess children make the calls through ProxyBackend
    # clients. Its own srun job (``gpu_solver_engine`` in the Makefile's
    # GPU_MATRIX_COLUMNS): beside the matrix's direct column its engine does
    # not fit the card, which the matrix engine takes most of.
    # Mirrors the ullm tier conftest's guard (packages/ullm/tests): skip
    # quietly when gpu wasn't selected, fail loudly when it was but no GPU
    # is visible. Inlined — cross-package conftest imports are ambiguous.
    pytest.importorskip("vllm")
    import torch

    if not torch.cuda.is_available():
        if "gpu" in (request.config.option.markexpr or ""):
            pytest.fail("gpu tests selected but no GPU is visible; run under `make test-gpu-srun`")
        pytest.skip("no GPU available")
    import ullm.backends.vllm_factory as _vllm_factory
    import ullm.backends.vllmx as _vllmx
    import vllm

    GPU_TEST_MODEL = "Qwen/Qwen3-VL-2B-Instruct"  # the small instruct VL model; the ullm matrix runs Qwen3.5-9B

    factory = _vllm_factory.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(
            model=GPU_TEST_MODEL,
            max_model_len=8192,
            gpu_memory_utilization=0.30,
            enforce_eager=True,
        ),
        backend_cls=_vllmx.VLLM,
        backend_kwargs={"tool_call_parser": "hermes"},
        mode="direct",  # the LLMAgentSolver socket is the only process boundary
    )
    agent = PromptedRandomAgent(
        max_moves=3,
        # The frame rides along as an image on every seam check: the test
        # model is a VL model, so this exercises the multimodal path
        # through the solver socket + subprocess children for free.
        attach_frame_image=True,
        image_scale=4,
        llm_config=LLMConfig(
            backend_factory=factory,
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        ),
        run_config=RunConfig(process_mode="subprocess", rng_seed=1),
    )
    runs = await _run(agent)
    run = runs[0]
    assert run.state == "gave_up"  # every seam check answered '8'; crash would surface here
    assert len(run.history) == 3
    assert all(rec.generated_tokens > 0 for rec in run.history)
    assert all(rec.uncached_input_tokens > 0 for rec in run.history)


@pytest.mark.api
@pytest.mark.parametrize("make_backend", [_openrouter_backend, _anthropic_backend], ids=["openrouter", "anthropic"])
async def test_prompted_random_agent_against_cheap_api_model(
    make_backend: Callable[[], ullm.backend.Backend],
) -> None:
    # The same stack against real (cheap) models — U6's two provider
    # families through the full platypus seam (socket, profiles, token
    # cross-check). The 3+5 seam check is the assertion: a wrong answer
    # raises inside the agent and the game ends "crashed" not "gave_up".
    backend = make_backend()
    agent = PromptedRandomAgent(
        max_moves=2,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
        ),
        run_config=RunConfig(process_mode="subprocess", rng_seed=1),
    )
    runs = await _run(agent)
    run = runs[0]
    assert run.state == "gave_up"
    assert all(rec.generated_tokens > 0 for rec in run.history)
    assert all(rec.uncached_input_tokens > 0 for rec in run.history)
