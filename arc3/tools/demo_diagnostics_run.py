"""Generate a clickable diagnostics demo run, then (optionally) serve it.

Multiple games x multiple passes of a demo agent whose moves carry a
deliberately rich span tree — nested free-form spans, kv (including an
auto-blocked long string), png-grid artifacts, a pickle repro artifact,
a caught-error span, and multimodal llm/turn spans — all against the
offline ArithmeticStubBackend, so it runs anywhere in seconds.

    uv run python tools/demo_diagnostics_run.py            # generate
    uv run python tools/demo_diagnostics_run.py --serve    # generate + serve

Then browse http://127.0.0.1:<port>/ — the listing links each run dir's
viewer; TAAF's own diagnostics.html sits next to it with its
"Detailed analysis by solver" links resolving into the same viewer.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path

import taaf.benchmark
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
from taaf.game_examples import ExampleGame

from platypus import Action, GiveUp, LLMAgentSolver, LLMConfig, LLMMoveContext, Observation, RunConfig
from platypus.viewer import serve
from reference_agents import PromptedRandomAgent


@dataclass
class ReasoningArithmeticStub(ullm.backends.stub.ArithmeticStubBackend):
    """Demo variant: attaches synthetic reasoning to every response so the
    viewer's collapsed-reasoning rendering has something to show."""

    def _process_output(self, sampling_params, llm_output, conversation, turn, element) -> None:  # type: ignore[no-untyped-def]
        """Attach canned reasoning to the stub's reply, so the demo run shows
        the reasoning-bearing shape of a real llm/turn span."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        reasoning = (
            f"The user asks for a small sum. Reading the prompt again to be sure. "
            f"Adding the two operands carefully, digit by digit, gives {llm_output}. "
            f"Double-checking by computing in the other order: same result. Answering just the number."
        )
        element.reasoning_content = ullm.conv.StringContent(text=reasoning)
        assert element.token_usage is not None
        element.token_usage.reasoning_tokens = max(1, len(reasoning) // 4)
        element.token_usage.completion_tokens += element.token_usage.reasoning_tokens


@dataclass
class DemoAgent(PromptedRandomAgent):
    """PromptedRandomAgent plus demo-only span depth: every decision move
    opens observe/plan phases with nested hypothesize/verify children, so
    the viewer has real hierarchy to fold.

    ``crash_game_id`` crashes that game with an UNHANDLED error raised
    several spans deep on move ``crash_on_move`` — the red error trail
    then runs game → move → plan → commit → apply, and the game ends
    "crashed" (on_agent_error="crash_game": other games continue)."""

    crash_game_id: str | None = "demogame2"
    crash_on_move: int = 3

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp:
        """A move with a deliberately rich span tree, so the demo shows what
        good instrumentation renders as."""
        if obs.outcome != "playing" or (self.max_moves is not None and self.moves_made >= self.max_moves):
            return await super()._do_move(ctx, obs)
        with ctx.diag.span("agent/observe") as observe:
            observe.kv(
                level=obs.level,
                available=",".join(obs.available_actions),
                notes="observation " * 60,  # long string: auto-blocked, renders collapsed
            )
            observe.artifact("frame", obs.frame.data, codec="png-grid")
        with ctx.diag.span("agent/plan") as plan:
            for attempt in range(2):
                with plan.span("agent/hypothesize", f"hypothesis {attempt}") as hypothesis:
                    hypothesis.kv(rule=f"pressing keys changes pixels (variant {attempt})")
                    with hypothesis.span("agent/verify") as verify:
                        verify.kv(pixels_checked=64 * 64, matched=attempt == 1)
                        if attempt == 0:
                            # A caught failure: the span closes with outcome
                            # error (red in the viewer); the move continues.
                            try:
                                with verify.span("agent/simulate"):
                                    raise ValueError("simulation diverged on frame 3")
                            except ValueError:
                                pass
            if len(ctx.history) == 1:  # once per game: a repro-style pickle artifact
                plan.artifact(
                    "repro",
                    {"frames": [obs.frame.data], "hypothesis": "variant 1"},
                    codec="pickle",
                )
            if ctx.game.game_id == self.crash_game_id and len(ctx.history) == self.crash_on_move:
                with plan.span("agent/commit") as commit:
                    commit.kv(applying="variant 1")
                    with commit.span("agent/apply"):
                        raise RuntimeError("demo crash: apply step hit an unhandled condition")
            plan.set_outcome("chose variant 1")
            return await super()._do_move(ctx, obs)  # seam-check span + llm/turn nest in here


async def generate_duck(job_dir: Path) -> None:
    """A duck run (scripted corners scenario borrowed from the test
    suite — real kernels, zero API spend): duck/analyze spans, kernel
    events, and conversations that carry PRE-HISTORY, growing move over
    move."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    import test_duck_scripted_reference as duck_scripted
    from reference_support import ReferenceRun

    if job_dir.exists():
        shutil.rmtree(job_dir)
    case = duck_scripted.CORNERS
    backend = ullm.backends.stub.StubBackend(responses=duck_scripted._corners_script(), on_exhausted="raise")
    benchmark = case.build(
        ReferenceRun(
            llm_config=LLMConfig(
                backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
                sampling_profiles=case.profiles,
            ),
            job_dir=job_dir,
            seeds={},
        )
    )
    await benchmark.run()
    print(f"duck demo at {job_dir} — open its later moves for conversations with pre-history")


async def generate(job_dir: Path, *, n_games: int, n_passes: int) -> None:
    """Run the demo benchmark into a fresh ``job_dir``."""
    if job_dir.exists():
        shutil.rmtree(job_dir)
    config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ReasoningArithmeticStub()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
    )
    agent = DemoAgent(
        max_moves=8,
        attach_frame_image=True,
        image_scale=4,
        llm_config=config,
        run_config=RunConfig(rng_seed=0, process_mode="inprocess", label="p8-demo"),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[ExampleGame(label=f"demogame{i}") for i in range(n_games)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=n_passes,
        job_dir=job_dir,
    )
    await benchmark.run()
    print(f"\ndemo run at {job_dir}")
    print(f"  serve:  uv run python tools/demo_diagnostics_run.py --serve --out {job_dir.parent}")
    print(f"  viewer: http://127.0.0.1:8765/{job_dir.name}/viewer/index.html")


def main() -> None:
    """Generate the demo run, and serve it when asked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("demo_runs"), help="directory holding demo run dirs")
    parser.add_argument("--name", default="p8_demo", help="run dir name under --out")
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--serve", action="store_true", help="serve --out after generating")
    parser.add_argument("--serve-only", action="store_true", help="skip generation, just serve --out")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.serve_only:
        asyncio.run(generate(args.out / args.name, n_games=args.games, n_passes=args.passes))
        asyncio.run(generate_duck(args.out / "duck_demo"))
    if args.serve or args.serve_only:
        serve(args.out, port=args.port)


if __name__ == "__main__":
    main()
