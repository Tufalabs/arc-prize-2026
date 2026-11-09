"""Games of the seal played by an API teacher (``teachers.API_TEACHERS``),
in the shape of Flash-Next's B200 column.

The seal is the column's -- ``SealAgent.build(engine="qwen3.8-flash-next",
environment="slurm-b200", kimi_student=True)`` -- with the backend replaced
by the teacher's API, so the column's run settings hold (games 12 s apart,
the column's budgets and windows) and the model-visible ones are those a
student trained on the traces plays with (``seal.agent.KIMI_STUDENT``). On
top, what an API needs:

- the teacher's own slots and reasoning parameter (the seal's effort None);
- no token-native counting or exact history (an API returns no token ids and
  has no /tokenize): the seal's estimate-mode accounting, calibrated on
  Flash-Next's tokenizer (``ApiTeacher.bound_chars_per_token``,
  ``image_tokens``), so the prompt side's windows hold in Flash-Next tokens;
  the bills, in the teacher's tokens, are held to the API's own window;
- the reply cap and the level budget, which count the teacher's billed
  tokens, at the column's values over ``ApiTeacher.flash_tokens_per_token``,
  so they too come out at the column's in Flash-Next tokens
  (2026-10-27);
- no token banned in a summary (the API takes no logit_bias);
- the API's sampling;
- no host watchdog over a reply, so a run waits out a provider outage for the
  backend's whole retry window and plays on after it, each game's wall cap
  (``--game-hours``) allowing for it.

The games (``--set``, ``GAME_SETS``): by default pathfinder and taps first,
the canaries, then TAAF's official 25 and duck sweep 13's eleven arc-like
hard; ``main`` is those 36 without the canaries, ``re-arc-rest`` every other
re-arc-3 game in re-arc's lists, ``re-arc-unlisted`` the ones outside them.
The re-arc-3 games' baselines are times 3, as the seal sweep scores them;
``--games`` plays a subset.

It runs as a CPU job through TAAF's SLURM deploy, from the workspace lock,
with the key in the launching shell's environment (TAAF's job inherits it):

    uv run python tools/teachers/seal_smoke.py --api kimi-k3.1-pro [--set NAME] [--games ID,...] [--traces]
        [--slots N] [--seed N] [--after JOBID] [--game-hours H] [--mem-gib N] [--cpus N]
        [--partition P] [--qos Q] [--exclude NODES] [--root DIR] [--inline] [--dry-run]

``--after`` holds the job until that job has ended (``afterany``), so runs
can be chained. ``--inline`` plays in this process instead, for a partition
with no room.

``--traces`` writes one TAAF trace per game (``llm_config.traces``): the
hosted-API grade, messages, output text and message, media, and the
teacher's per-token alternatives, which is what a re-templating SFT consumer
reads. They land on node-local scratch and are bundled into the job dir at
teardown (``Benchmark.retain_traces``); the backend then runs unstreamed, as
tracing needs the whole completion. ``README.md`` beside this file has the
measurements behind the teacher entry and what to expect of the API.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import math
import time
from pathlib import Path

import taaf.benchmark
import taaf.deploy_inline
import taaf.deploy_slurm
import taaf.game
import taaf.game_api
import taaf.standard_benchmarks
import teachers
import ullm.backend
import ullm.factory

import platypus.deployment
from platypus import LLMAgentSolver
from seal import SealAgent

#: Every run under the shared results tree, the launching user's own.
ROOT = Path(f"/data/shared/research/arc_3_results/{getpass.getuser()}/distill")
#: An inline run's traces: no node scratch there, so a local folder.
INLINE_TRACE_ROOT = Path(f"/scratch/{getpass.getuser()}/tmp")
#: Traces on the node's own scratch, bundled into the job dir at teardown.
TRACE_ROOT = Path("${TAAF_LOCAL_JOB_ROOT}/traces")

#: Duck sweep 13's arc-like hard set, verbatim from the seal sweep
#: (``seal_sweep1.ARC_LIKE_HARD``), at the pinned re-arc-3.
ARC_LIKE_HARD: tuple[str, ...] = (
    "anchor_drift-0001", "ball_bounce-0001", "en18-0001", "escape_velocity-0001",
    "glass_orchard-0001", "jc26-0001", "js26-0001", "lantern_moths-0001",
    "marble_loom-0001", "paint_snakes-0001", "talos_beams-0001",
)  # fmt: skip
#: Played first: the canaries.
CANARIES: tuple[str, ...] = ("pathfinder-0001", "taps-0001")
#: The re-arc-3 baselines are optimal solves, the official 25's human play.
BASELINE_MULTIPLIER = 3.0
RUN_SEED = 20261002


def build_api_agent(teacher: teachers.ApiTeacher, *, traces: bool = False) -> SealAgent:
    """The B200 column's seal on a provider's API, at the teacher's slots,
    with its token figures in Flash-Next's tokens and the student's
    model-visible settings."""
    agent = SealAgent.build(
        engine="qwen3.8-flash-next",
        environment="slurm-b200",
        backend_factory=ullm.factory.PrebuiltBackend(backend=teacher.backend(streaming=not traces), close_backend=True),
        kimi_student=True,
        reasoning_effort=None,
        zero_recurrent=False,
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,  # an engine argument: the API sizes its own images
        engine_ceiling_tokens=teacher.max_context_tokens,
    )
    # After the preset, which sets the column's own.
    if not teacher.images:
        agent.multimodal_context = ""
    agent.token_native_context = False
    agent.llm_config.exact_history = False
    agent.bound_chars_per_token = teacher.bound_chars_per_token
    agent.image_tokens_override = teacher.image_tokens
    # The bills count in the teacher's tokenizer, the window in Flash-Next's:
    # a bill past the window is the estimate missing, which the next anchor
    # heals. Only the API's own window is a hard limit.
    agent.usage_window_tokens = teacher.max_context_tokens
    # The column's reply cap and level budget hold in Flash-Next tokens: the
    # teacher's billed tokens run ``flash_tokens_per_token`` Flash-Next tokens
    # each.
    ratio = teacher.flash_tokens_per_token
    if agent.reply_cap_tokens is not None:
        agent.reply_cap_tokens = round(agent.reply_cap_tokens / ratio)
        # The window holds back the reply cap for every reply, the safety
        # margin beside it, and counts Flash-Next tokens while the cap now
        # counts the teacher's. The margin takes what that leaves out, so a
        # full-cap reply fits the student's window; under a smaller cap (the
        # end of a budget) it holds back more than needed. The cap itself
        # stays the API's: it is also what a request sends as its max_tokens.
        agent.safety_margin_tokens += max(0, math.ceil(agent.reply_cap_tokens * ratio) - agent.reply_cap_tokens)
    if agent.max_tokens_per_level is not None:
        agent.max_tokens_per_level = tuple(round(t / ratio) for t in agent.max_tokens_per_level)
    if agent.max_tokens_per_game is not None:
        agent.max_tokens_per_game = round(agent.max_tokens_per_game / ratio)
    agent.compaction_forbidden_token_ids = ()
    agent.llm_config.sampling_profiles = {
        "play": ullm.backend.GenericSamplingParameters(**teacher.play_sampling, max_tokens=None)
    }
    agent.llm_config.traces = traces
    agent.run_config.rng_seed = RUN_SEED
    # Under queue_order a playing game outranks every later one, so its slot
    # offers come straight back.
    agent.run_config.max_concurrent = teacher.play_slots
    # The host's per-request watchdog off: a reply may wait out a provider
    # overload for the backend's whole retry window (twelve hours for Kimi),
    # each attempt bounded by the client's own timeout, so the run hibernates
    # through an outage rather than crash its games (2026-10-29).
    agent.run_config.backend_inactivity_timeout_s = None
    return agent


#: The game sets ``--set`` names, with how many games each holds.
GAME_SETS = {
    # The canaries, then the main 36.
    "smoke": 38,
    # The official 25 at their own baselines, then the arc-like hard.
    "main": 36,
    # Every other re-arc-3 game in re-arc's train and eval lists (the
    # canaries among them), for the much easier games' volume
    # (2026-10-28): the lists' 164 that are not official, less the five
    # arc-like hard among them. The other six arc-like hard are in neither.
    "re-arc-rest": 159,
    # The re-arc-3 games outside those lists (2026-10-28): a newer,
    # mostly codex-reviewed batch, less the six arc-like hard among them and
    # its one debugging game.
    "re-arc-unlisted": 74,
}


def build_games(game_set: str = "smoke") -> list[taaf.game.Game]:
    """One of ``GAME_SETS``, the re-arc-3 games' baselines times
    ``BASELINE_MULTIPLIER``."""

    def re_arc_3(game_ids: tuple[str, ...]) -> list[taaf.game.Game]:
        return [
            taaf.game_api.GameAPI(env_name=game_id, baseline_multiplier=BASELINE_MULTIPLIER) for game_id in game_ids
        ]

    if game_set == "re-arc-rest":
        import re_arc

        lists = ["train", "eval"]
        official = set(re_arc.list_game_ids(datasets=lists, include_tags="official"))
        ids = tuple(g for g in re_arc.list_game_ids(datasets=lists) if g not in official and g not in ARC_LIKE_HARD)
        games = re_arc_3(ids)
    elif game_set == "re-arc-unlisted":
        import re_arc

        listed = set(re_arc.list_game_ids(datasets=["train", "eval"]))
        ids = tuple(
            g for g in re_arc.list_game_ids(exclude_tags="debugging") if g not in listed and g not in ARC_LIKE_HARD
        )
        games = re_arc_3(ids)
    else:
        games = re_arc_3(CANARIES) if game_set == "smoke" else []
        games += taaf.standard_benchmarks.make_benchmark_official_25().games
        games += re_arc_3(ARC_LIKE_HARD)
    if len(games) != GAME_SETS[game_set]:
        raise RuntimeError(f"expected {GAME_SETS[game_set]} games in {game_set!r}, built {len(games)}")
    return games


async def launch(
    api: str,
    dry_run: bool,
    partition: str,
    hours: int,
    only: list[str] | None,
    live_diagnostics: bool,
    traces: bool,
    game_set: str = "smoke",
    slots: int | None = None,
    seed: int | None = None,
    after: str | None = None,
    game_hours: float | None = None,
    mem_gib: int = 64,
    inline: bool = False,
    qos: str | None = None,
    exclude: str | None = None,
    root: Path = ROOT,
    cpus: int = 16,
) -> None:
    """Build the teacher's seal and the games, print the settings that
    matter, and submit the run as a CPU job (nothing with ``dry_run``);
    ``after`` holds the job until that one has ended, however it ended."""
    teacher = teachers.API_TEACHERS[api]
    agent = build_api_agent(teacher, traces=traces)
    if slots is not None:
        agent.run_config.max_concurrent = slots
    if seed is not None:
        agent.run_config.rng_seed = seed
    if game_hours is not None:
        agent.run_config.max_runtime_s_per_game = game_hours * 3600
    if live_diagnostics or inline:
        # Straight into the job dir, watchable as the games play, instead of
        # node-local scratch archived every 30 min: for a few games only, as
        # every event is a line appended on the network filesystem.
        agent.run_config.diagnostics_scratch = ""
    games = build_games(game_set)
    if only:
        unknown = set(only) - {getattr(g, "env_name", None) for g in games}
        if unknown:
            raise SystemExit(f"not among the {game_set!r} games: {sorted(unknown)}")
        games = [g for g in games if getattr(g, "env_name", None) in only]
    stamp = time.strftime("%Y%m%d-%H%M%S")
    benchmark = taaf.benchmark.Benchmark(
        label=f"{teacher.name}_seal_{game_set}", games=games, solver=LLMAgentSolver(agent=agent), n_passes=1
    )
    benchmark.job_dir = root / teacher.name / f"seal-{game_set}-s{agent.run_config.rng_seed}-{stamp}"
    if traces:
        # Inline there is no node scratch: a local folder instead, bundled the same way.
        benchmark.trace_root = INLINE_TRACE_ROOT / f"{teacher.name}-traces-{stamp}" if inline else TRACE_ROOT
        benchmark.retain_traces = True
    print(f"{len(games)} games ({game_set}): {[getattr(g, 'env_name', g) for g in games]}", flush=True)
    print(f"job dir: {benchmark.job_dir}", flush=True)
    print(
        f"slots: {agent.run_config.max_concurrent}, seed {agent.run_config.rng_seed}, "
        f"game wall cap {agent.run_config.max_runtime_s_per_game} s, job memory {mem_gib} GiB, {cpus} CPUs",
        flush=True,
    )
    print(
        f"accounting: {agent.bound_chars_per_token} chars a token, {agent.image_tokens_override} an image, "
        f"reply cap {agent.reply_cap_tokens}, margin {agent.safety_margin_tokens}, level budget {agent.max_tokens_per_level}, "
        f"window {agent.context_window_tokens}, compaction to {agent.compact_to_tokens}, "
        f"bills held to {agent.usage_window_tokens}",
        flush=True,
    )
    print(f"diagnostics: {'live in the job dir' if live_diagnostics else 'scratch, archived'}", flush=True)
    print(f"traces: {'on, bundled into the job dir' if traces else 'off'}", flush=True)
    print(f"after: {after or '-'}", flush=True)
    if dry_run:
        return
    if inline:
        # In this process, on this machine (a login node when the partition is full).
        await benchmark.deploy(taaf.deploy_inline.InlineTarget(max_runtime_s=hours * 3600.0))
        print(f"ran inline -> {benchmark.job_dir}", flush=True)
        return
    target = taaf.deploy_slurm.TufaSlurmTarget(
        gpu="CPU",
        gpu_count=0,
        partition=partition,
        # The long and arc partitions take only their own accounts.
        account=partition if partition in ("long", "arc") else "research",
        time=f"{hours:02d}:00:00",
        total_cpus=cpus,
        total_mem_bytes=mem_gib * 1024**3,
        main_project=platypus.deployment.detect_main_project(),
        extra_sbatch_flags=[f"--dependency=afterany:{after}"] * bool(after)
        + [f"--qos={qos}"] * bool(qos)
        + [f"--exclude={exclude}"] * bool(exclude),
    )
    handle = await benchmark.deploy(target)
    print(f"submitted {getattr(handle, 'job_id', handle)} -> {benchmark.job_dir}", flush=True)


def main() -> None:
    """The command line: the module docstring's usage."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--api", required=True, choices=sorted(teachers.API_TEACHERS), help="the API teacher")
    parser.add_argument("--dry-run", action="store_true", help="build the agent and games, submit nothing")
    parser.add_argument("--partition", default="arc", help="the SLURM partition (default arc)")
    parser.add_argument("--hours", type=int, default=48, help="job walltime (default 48)")
    parser.add_argument("--games", default=None, help="comma-separated game ids: play only these, in the smoke's order")
    parser.add_argument(
        "--live-diagnostics", action="store_true", help="write diagnostics into the job dir as they happen (few games)"
    )
    parser.add_argument("--traces", action="store_true", help="write a TAAF trace per game, bundled into the job dir")
    parser.add_argument("--set", default="smoke", choices=sorted(GAME_SETS), help="the game set (default smoke)")
    parser.add_argument("--slots", type=int, default=None, help="games at once (default the teacher's play_slots)")
    parser.add_argument("--seed", type=int, default=None, help=f"the run seed (default {RUN_SEED})")
    parser.add_argument("--after", default=None, help="a SLURM job id: start once it has ended, however it ended")
    # Twelve hours of play, as the runs of 2026-10-28 capped it, plus as long
    # again for a provider outage to be waited out (2026-10-29).
    parser.add_argument("--game-hours", type=float, default=24.0, help="each game's wall cap (default 24)")
    # 38 games at once, traced with logprobs held packed, peaked at 11 GiB
    # that reclaim could not free (2026-10-29).
    parser.add_argument("--mem-gib", type=int, default=64, help="the job's memory (default 64 GiB)")
    parser.add_argument("--inline", action="store_true", help="run in this process, not as a SLURM job")
    parser.add_argument("--qos", default=None, help="the SLURM QOS (e.g. dev, for the dev partition)")
    parser.add_argument("--exclude", default=None, help="SLURM nodes to keep off")
    parser.add_argument("--root", type=Path, default=ROOT, help=f"where the job dir goes (default {ROOT})")
    parser.add_argument(
        "--cpus",
        type=int,
        default=16,
        help="the job's CPUs (default 16; 159 games averaged 1.4 cores, waiting on the API)",
    )
    args = parser.parse_args()
    only = args.games.split(",") if args.games else None
    asyncio.run(
        launch(
            args.api,
            args.dry_run,
            args.partition,
            args.hours,
            only,
            args.live_diagnostics,
            args.traces,
            args.set,
            args.slots,
            args.seed,
            args.after,
            args.game_hours,
            args.mem_gib,
            args.inline,
            args.qos,
            args.exclude,
            args.root,
            args.cpus,
        )
    )


if __name__ == "__main__":
    main()
