"""A kernel-using agent inside the capture/replay harness (U9 + I13).

The toy agent runs real kernel code every move and derives its ACTION from
the kernel's output, so replay only reproduces bit-exact if the kernel
environment behaves identically — the accepted third nondeterminism seam
(contract §9) exercised end to end: LLM traffic is replayed from the
recorded stream, kernel code re-executes live.

In-process on purpose: kernel behavior is per-game (the process boundary
cannot influence it), and the cross-process pool-cap machinery is covered
at the ullm level. The real subprocess kernel topology is covered by the
duck's own tests.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
import ullm.tools.python
from taaf.game_examples import ExampleGame

from platypus import (
    Action,
    GameResult,
    GiveUp,
    LLMAgent,
    LLMAgentSolver,
    LLMConfig,
    LLMMoveContext,
    Observation,
    RunConfig,
)
from platypus.reference import generate_references, load_references, replay_references

N_GAMES = 2


@dataclass
class KernelArithmeticBackend(ullm.backends.stub.StubBackend):
    """Scripted 'model' that drives the python tool: first call emits a
    tool call computing the sum found in the prompt, second call reports
    the tool result verbatim. Request-dependent and stateless across
    games, so concurrent games can share one instance."""

    def _prepare_input(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        turn: ullm.conv.ResponseTurn,
        conversation: ullm.conv.Conversation,
        send_back_reasoning: str,
    ) -> Any:
        prompts = [e for e in conversation.elements if isinstance(e, ullm.conv.UserPrompt)]
        match = re.search(r"(\d+)\s*\+\s*(\d+)", prompts[-1].content.to_string())
        assert match, "prompt carries no sum"
        results = [e for e in turn.elements if isinstance(e, ullm.conv.ToolResult)]
        if not results:
            code = f"print({match.group(1)} + {match.group(2)})"
            return ullm.conv.LLMResponse(
                final_content=ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})
            )
        return ullm.conv.LLMResponse(
            final_content=ullm.conv.StringContent(text=results[-1].content.to_string().strip())
        )


@dataclass(kw_only=True)
class KernelSumAgent(LLMAgent):
    """Toy agent: each move asks the 'model' to compute a random sum with
    the python tool (through a per-move bound view), then picks its action
    FROM the kernel's answer — kernel output flows into the action stream
    the reference pins."""

    max_moves: int = 3
    repl: ullm.tools.python.PythonTool = field(
        default_factory=lambda: ullm.tools.python.PythonTool(timeout=60.0)  # module-default pool
    )
    moves_made: int = 0
    host_notes: list[tuple[int, str]] = field(default_factory=lambda: list[tuple[int, str]]())

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp:
        if obs.outcome == "level_won":
            return Action("RESET")
        if self.moves_made >= self.max_moves:
            return GiveUp("done")
        self.moves_made += 1
        if obs.outcome == "lost":
            return Action("RESET")
        a, b = ctx.rng.randint(1, 9), ctx.rng.randint(1, 9)
        move_index = self.moves_made

        def _note(value: object) -> str:
            self.host_notes.append((move_index, str(value)))
            return "ok"

        view = self.repl.bound(host_functions={"note": _note})
        turn_manager = self.new_turn_manager(ctx, profile="play", system_prompt="calc", tools=[view])
        turn_manager.conversation.elements.append(
            ullm.conv.UserPrompt(content=ullm.conv.StringContent(text=f"compute {a}+{b} with python"))
        )
        await turn_manager.run()
        answer = turn_manager.conversation.last_turn.last_response.final_content.to_string()
        assert str(a + b) in answer, f"kernel/LLM loop failed: expected {a + b}, got {answer!r}"
        choices = [name for name in obs.available_actions if name not in ("RESET", "ACTION6")]
        return Action(choices[int(answer) % len(choices)])

    async def _on_finish(  # pyright: ignore[reportIncompatibleMethodOverride] — covariant ctx narrowing, as LLMAgent._do_move
        self, ctx: LLMMoveContext, result: GameResult
    ) -> None:
        await self.repl.clear_state()


def _llm_config(*, capture: bool = False, replay_dir: str | None = None) -> LLMConfig:
    profiles = {"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=100)}
    if replay_dir is not None:
        return LLMConfig(sampling_profiles=profiles, replay_dir=replay_dir)
    backend = KernelArithmeticBackend.from_script("unused; _prepare_input scripts dynamically")
    return LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
        sampling_profiles=profiles,
        capture=capture,
    )


def _benchmark(llm_config: LLMConfig, job_dir: Path | None) -> tuple[taaf.benchmark.Benchmark, KernelSumAgent]:
    agent = KernelSumAgent(llm_config=llm_config, run_config=RunConfig(rng_seed=5, process_mode="inprocess"))
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame(label=f"kernelgame{i}") for i in range(N_GAMES)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
    )
    return benchmark, agent


async def test_kernel_agent_capture_replay_roundtrip(tmp_path: Path) -> None:
    capture, template = _benchmark(_llm_config(capture=True), tmp_path / "job")
    try:
        archives = await generate_references(capture, tmp_path / "refs", "kernel", minimal_diagnostics=True)
    finally:
        await template.repl.clear_state()  # per-game copies clean up in _on_finish; this is the template's
    assert len(archives) == N_GAMES
    expected = load_references(archives, tmp_path / "unpacked")
    # Tool-call rounds are part of the recorded traffic: each move is two
    # LLM calls (tool call, then answer), and actions depend on the answer.
    assert all(len(fields["actions"]) > 0 for fields in expected.values())

    replay, template = _benchmark(_llm_config(replay_dir=str(tmp_path / "unpacked")), None)
    try:
        await replay_references(replay, expected)  # bit-exact, kernel re-executing live
    finally:
        await template.repl.clear_state()
