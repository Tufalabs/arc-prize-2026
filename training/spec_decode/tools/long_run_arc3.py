"""Deploy a B200 long run: the slurm-b200 Flash-Next cell as it stands over a
game set, several passes, on one B200, with the target and drafter a run names.

    uv run python tools/long_run.py --label e307 [--games official_25] [--passes 10]
    uv run python tools/long_run.py --label reap307 --target DIR --drafter DIR
    uv run python tools/long_run.py --label reap307_undrafted --target DIR --no-drafter

It is main's comparison run (the long runs of 2026-09-28: the official 25 x 10
at the cell's 100 slots, ids-only traces, 24 h job, the cell's 9 h per game)
with two knobs: ``--target`` serves another Flash-Next checkpoint of the same
architecture (an expert prune), and ``--drafter`` / ``--no-drafter`` give it
its own MTP head or none. A drafter is only valid for the target it was trained
against, so a new ``--target`` must say which it drafts with. The traces are a
drafter's training data (``drafter.chains build``). Fire and forget: the job
dir and id are printed.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import getpass
import pathlib

import taaf.benchmark
import taaf.deploy_slurm
import taaf.game_api
import taaf.standard_benchmarks

import platypus.deployment
from duck.agent import DuckAgent
from platypus import LLMAgentSolver

RESULTS = pathlib.Path("/data/shared/research/arc_3_results") / getpass.getuser() / "long_runs"

#: Sweep 13's hard arc-like re-arc-3 games (arc3-experiments,
#: ``sweep13/duck_sweep.py`` ``ARC_LIKE_HARD``).
ARC_LIKE_HARD = (
    "anchor_drift-0001", "ball_bounce-0001", "en18-0001", "escape_velocity-0001",
    "glass_orchard-0001", "jc26-0001", "js26-0001", "lantern_moths-0001",
    "marble_loom-0001", "paint_snakes-0001", "talos_beams-0001",
)  # fmt: skip


def _benchmark(games: str, solver: LLMAgentSolver) -> taaf.benchmark.Benchmark:
    if games == "official_25":
        return taaf.standard_benchmarks.make_benchmark_official_25(solver=solver)
    if games == "arc_like_hard":
        return taaf.benchmark.Benchmark(
            label="arc_like_hard",
            games=[taaf.game_api.GameAPI(env_name=game_id) for game_id in ARC_LIKE_HARD],
            solver=solver,
            n_passes=1,
        )
    raise SystemExit(f"unknown game set {games!r}")


def build_agent(target: str | None, drafter: str | None, no_drafter: bool) -> DuckAgent:
    """The cell as it stands, serving ``target`` and drafting with ``drafter``."""
    if target and not (drafter or no_drafter):
        raise SystemExit("a drafter is only valid for its target: with --target, pass --drafter DIR or --no-drafter")
    if drafter and no_drafter:
        raise SystemExit("--drafter and --no-drafter exclude each other")
    agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    # Ids-only traces, as main's long runs keep them: the calibration replay and
    # the drafter read every step's token ids (job_dir/traces).
    agent.llm_config.traces = True
    agent.llm_config.trace_content = "ids"
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    if target:
        engine_args.model = target
    if no_drafter:
        engine_args.speculative_config = None
    elif drafter:
        if not engine_args.speculative_config:
            raise SystemExit("the cell does not draft")
        engine_args.speculative_config = {**engine_args.speculative_config, "model": drafter}
    agent.prepare_for_backend()
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", required=True)
    parser.add_argument("--games", choices=("official_25", "arc_like_hard"), default="official_25")
    parser.add_argument("--passes", type=int, default=10)
    parser.add_argument("--target", default=None, help="a Flash-Next checkpoint dir (default: the cell's e307)")
    parser.add_argument("--drafter", default=None, help="its MTP drafter dir (default: the cell's)")
    parser.add_argument("--no-drafter", action="store_true")
    parser.add_argument("--node", default=None)
    args = parser.parse_args()

    agent = build_agent(args.target, args.drafter, args.no_drafter)
    solver = LLMAgentSolver(agent=agent)
    benchmark = _benchmark(args.games, solver)
    benchmark.n_passes = args.passes
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"arc3_{args.games}_{args.passes}x_{args.label}_{stamp}"
    benchmark.label = f"long_runs_{name}"
    benchmark.job_dir = RESULTS / name
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    speculative = engine_args.speculative_config or {}
    print(
        f"{len(benchmark.games)} games x {benchmark.n_passes} passes, max_concurrent={agent.run_config.max_concurrent}\n"
        f"target={engine_args.model}\ndrafter={speculative.get('model')} k={speculative.get('num_speculative_tokens')}\n"
        f"job_dir={benchmark.job_dir}",
        flush=True,
    )
    target = taaf.deploy_slurm.TufaSlurmTarget(
        # The dev partition's 12 h (the long partition is not open to every
        # account): main's ARC-3 long run took 6 h 52 m, ARC-2's 3 h 39 m.
        time="12:00:00",
        max_runtime_s=41400,
        main_project=platypus.deployment.detect_main_project(),
        nodelist=args.node,
        gpu_count=1,
        cpus_per_gpu=24,
        mem_per_gpu_bytes=280 * 1024**3,
        partition="dev",
        account="research",
    )
    handle = asyncio.run(benchmark.deploy(target))
    print("submitted:", getattr(handle, "job_id", handle), flush=True)


if __name__ == "__main__":
    main()
