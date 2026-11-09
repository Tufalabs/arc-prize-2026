"""Generate a clickable diagnostics demo run, then (optionally) serve it.

Several games x several passes of a demo agent whose single call carries a
deliberately rich span tree — nested free-form spans, kv (including an
auto-blocked long string), png-grid artifacts, a pickle repro artifact, a
caught-error span, an unhandled crash several spans deep in one game, and
multimodal llm/turn spans — all against the offline ArithmeticStubBackend,
so it runs anywhere in seconds.

    uv run python tools/demo_diagnostics_run.py            # generate
    uv run python tools/demo_diagnostics_run.py --serve    # generate + serve

Then browse http://127.0.0.1:<port>/ — the listing links each run dir's
viewer; TAAF's own diagnostics.html sits next to it with its per-game
"Detailed analysis by solver" links resolving into the same viewer.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory

import taaf.benchmark
import taaf.game
from platypus import GameView, GiveUp, LLMAgentSolver, LLMConfig, LLMMoveContext, RunConfig
from platypus.reference_agents import ANSWERS, PromptedFlipAgent
from platypus.viewer import serve
from taaf.game_examples import constant_game, flip_game


@dataclass
class ReasoningArithmeticStub(ullm.backends.stub.ArithmeticStubBackend):
    """Demo variant: attaches synthetic reasoning to every response so the
    viewer's collapsed-reasoning rendering has something to show."""

    def _process_output(
        self, sampling_params: Any, llm_output: Any, conversation: Any, turn: Any, element: Any
    ) -> None:
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
class DemoAgent(PromptedFlipAgent):
    """``PromptedFlipAgent`` plus demo-only span depth: one observe phase,
    then a plan phase per answer with nested hypothesize/verify children
    and the answer submitted from inside it — so the viewer has real
    hierarchy to fold, and shows an answer landing where it was made.

    ``crash_game_id`` crashes that game with an UNHANDLED error raised
    several spans deep while making answer ``crash_on_answer`` — the red
    error trail then runs game → call → plan → commit → apply, and the
    game ends "crashed" (``on_agent_error="crash_game"``: others continue)
    with the answers it had already made still scored."""

    crash_game_id: str | None = "demogame2"
    crash_on_answer: int = 2

    async def _do_move(self, ctx: LLMMoveContext, view: GameView) -> GiveUp | None:
        """One call with a deliberately rich span tree, so the demo shows
        what good instrumentation renders as."""
        with ctx.diag.span("agent/observe") as observe:
            observe.kv(
                n_given=len(view.given),
                n_to_answer=len(view.questions),
                notes="observation " * 60,  # long string: auto-blocked, renders collapsed
            )
            observe.artifact("question", view.questions[0].data, codec="png-grid")
        for index, name in enumerate(self.answers):
            with ctx.diag.span("agent/plan", f"answer {index + 1}") as plan:
                for attempt in range(2):
                    with plan.span("agent/hypothesize", f"hypothesis {attempt}") as hypothesis:
                        hypothesis.kv(rule=f"the answer mirrors the question (variant {attempt})")
                        with hypothesis.span("agent/verify") as verify:
                            verify.kv(pairs_checked=len(view.given), matched=attempt == 1)
                            if attempt == 0:
                                # A caught failure: the span closes with outcome
                                # error (red in the viewer); the call continues.
                                try:
                                    with verify.span("agent/simulate"):
                                        raise ValueError("simulation diverged on pair 2")
                                except ValueError:
                                    pass
                if index == 0:  # once per game: a repro-style pickle artifact
                    plan.artifact(
                        "repro",
                        {"question": view.questions[0].data, "hypothesis": "variant 1"},
                        codec="pickle",
                    )
                if ctx.game_id == self.crash_game_id and index == self.crash_on_answer:
                    with plan.span("agent/commit") as commit:
                        commit.kv(applying="variant 1")
                        with commit.span("agent/apply"):
                            raise RuntimeError("demo crash: apply step hit an unhandled condition")
                plan.set_outcome("chose variant 1")
                await self._check_llm_seam(ctx, view)  # seam-check span + llm/turn nest in here
                ctx.submit(ANSWERS[name](view))  # the infra/submit span lands under this plan
        return None


async def generate(job_dir: Path, *, n_passes: int) -> None:
    """Run the demo benchmark into a fresh ``job_dir``."""
    if job_dir.exists():
        shutil.rmtree(job_dir)
    config = LLMConfig(
        backend_factory=ullm.factory.PrebuiltBackend(backend=ReasoningArithmeticStub()),
        sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=50)},
    )
    agent = DemoAgent(
        attach_grid_image=True,
        image_scale=4,
        llm_config=config,
        run_config=RunConfig(rng_seed=0, process_mode="inprocess", label="diagnostics-demo"),
    )
    # A task the mirror solves, one it cannot, and the one that crashes.
    games: list[taaf.game.Game] = [flip_game("demogame0"), constant_game("demogame1"), flip_game("demogame2")]
    benchmark = taaf.benchmark.Benchmark(
        games=games, solver=LLMAgentSolver(agent=agent), n_passes=n_passes, job_dir=job_dir
    )
    await benchmark.run()
    print(f"\ndemo run at {job_dir}")
    print(f"  serve:  uv run python tools/demo_diagnostics_run.py --serve-only --out {job_dir.parent}")
    print(f"  viewer: http://127.0.0.1:8765/{job_dir.name}/viewer/index.html")


def main() -> None:
    """Generate the demo run, and serve it when asked."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("demo_runs"), help="directory holding demo run dirs")
    parser.add_argument("--name", default="diagnostics_demo", help="run dir name under --out")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--serve", action="store_true", help="serve --out after generating")
    parser.add_argument("--serve-only", action="store_true", help="skip generation, just serve --out")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not args.serve_only:
        asyncio.run(generate(args.out / args.name, n_passes=args.passes))
    if args.serve or args.serve_only:
        serve(args.out, port=args.port)


if __name__ == "__main__":
    main()
