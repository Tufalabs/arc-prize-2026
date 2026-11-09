"""ARC2 tasks played by an API teacher (``teachers.API_TEACHERS``) through the
duck, in the shape of Flash-Next's B200 column.

The duck is the column's -- ``DuckAgent.build(engine="qwen3.8-flash-next",
environment="slurm-b200", kimi_student=True)`` -- with the backend replaced
by the teacher's API, so the column's run settings hold (its budgets and
windows) and the model-visible ones are those a student trained on the
traces plays with (``duck.agent.KIMI_STUDENT``). On top, what an API needs:

- the teacher's own slots and reasoning parameter (the duck's effort None);
- no token-native counting or exact history (an API returns no token ids and
  has no /tokenize): the duck's estimate-mode accounting, calibrated on
  Flash-Next's tokenizer (``ApiTeacher.bound_chars_per_token``,
  ``image_tokens``), so the prompt side's windows hold in Flash-Next tokens;
  the bills, in the teacher's tokens, are held to the API's own window;
- the reply cap and the game budget, which count the teacher's billed
  tokens, at the column's values over ``ApiTeacher.flash_tokens_per_token``,
  so they too come out at the column's in Flash-Next tokens
  (2026-10-27);
- no token banned in a summary (the API takes no logit_bias);
- the API's sampling;
- no host watchdog over a reply, and each task's wall cap raised by the
  backend's retry window, so a run waits out a provider outage and plays on
  after it.

The tasks: TAAF's named sets, merged in order, the easy ten
(``arc2-train-easy``) by default; ``--ids`` plays a subset.

It runs as a CPU job through TAAF's SLURM deploy, from the workspace lock,
with the key in the launching shell's environment (TAAF's job inherits it):

    uv run python tools/teachers/duck_smoke.py --api kimi-k3.1-pro [--set NAME,...] [--ids ID,...] [--traces] [--dry-run]

``--traces`` writes one TAAF trace per task (``llm_config.traces``): the
hosted-API grade, messages, output text and message, and media, which is
what a re-templating SFT consumer reads. They land on node-local scratch and
are bundled into the job dir at teardown (``Benchmark.retain_traces``); the
backend then runs unstreamed, as tracing needs the whole completion.
``README.md`` beside this file has the student, the traces and the API.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import math
import time
from pathlib import Path

import teachers
import ullm.backend
import ullm.factory
from duck import DuckAgent

import platypus.deployment
import taaf.deploy_slurm
import taaf.game
import taaf.standard_benchmarks
from platypus import LLMAgentSolver

#: ARC2 runs under the shared ARC2 results tree, the launching user's own.
ROOT = Path(f"/data/shared/research/arc_2_results/{getpass.getuser()}/distill")
#: Traces on the node's own scratch, bundled into the job dir at teardown.
TRACE_ROOT = Path("${TAAF_LOCAL_JOB_ROOT}/traces")
#: The seal smoke's run seed.
RUN_SEED = 20261002


def build_api_agent(teacher: teachers.ApiTeacher, *, traces: bool = False) -> DuckAgent:
    """The B200 column's duck on a provider's API, at the teacher's slots,
    with its token figures in Flash-Next's tokens and the student's
    model-visible settings."""
    backend = teacher.backend(streaming=not traces)
    agent = DuckAgent.build(
        engine="qwen3.8-flash-next",
        environment="slurm-b200",
        backend_factory=ullm.factory.PrebuiltBackend(backend=backend, close_backend=True),
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
        agent.images = "off"
    agent.token_native_context = False
    agent.llm_config.exact_history = False
    agent.bound_chars_per_token = teacher.bound_chars_per_token
    agent.image_tokens_override = teacher.image_tokens
    # The bills count in the teacher's tokenizer, the window in Flash-Next's:
    # a bill past the window is the estimate missing, which the next anchor
    # heals. Only the API's own window is a hard limit.
    agent.usage_window_tokens = teacher.max_context_tokens
    # The column's reply cap and game budget hold in Flash-Next tokens: the
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
        # stays the API's: it is also what a request sends as its max_tokens,
        # an intermediate summary's included.
        agent.safety_margin_tokens += max(0, math.ceil(agent.reply_cap_tokens * ratio) - agent.reply_cap_tokens)
    if agent.max_tokens_per_game is not None:
        agent.max_tokens_per_game = round(agent.max_tokens_per_game / ratio)
    agent.compaction_forbidden_token_ids = ()
    agent.llm_config.sampling_profiles = {
        "play": ullm.backend.GenericSamplingParameters(**teacher.play_sampling, max_tokens=None)
    }
    agent.llm_config.traces = traces
    agent.run_config.rng_seed = RUN_SEED
    agent.run_config.max_concurrent = teacher.play_slots
    # The host's per-request watchdog off, and each task's wall cap raised by
    # the backend's retry window: a reply may wait out a provider's refusals
    # for that long, each attempt bounded by the client's own timeout, so the
    # run hibernates through them rather than crash its tasks
    # (2026-10-29).
    agent.run_config.backend_inactivity_timeout_s = None
    window = getattr(backend, "transient_retry_window_s", None)
    if window is not None and agent.run_config.max_runtime_s_per_game is not None:
        agent.run_config.max_runtime_s_per_game += window
    return agent


async def launch(
    api: str,
    datasets: list[str],
    ids: list[str] | None,
    passes: int,
    dry_run: bool,
    partition: str,
    hours: int,
    live_diagnostics: bool,
    traces: bool,
) -> None:
    """Build the teacher's duck and the task set, print the settings that
    matter, and submit the run as a CPU job (nothing with ``dry_run``)."""
    teacher = teachers.API_TEACHERS[api]
    agent = build_api_agent(teacher, traces=traces)
    if live_diagnostics:
        # Straight into the job dir, watchable as the tasks play, instead of
        # node-local scratch archived every 30 min: for a few tasks only, as
        # every event is a line appended on the network filesystem.
        agent.run_config.diagnostics_scratch = ""
    benchmark = taaf.standard_benchmarks.make_benchmark_named(
        datasets, LLMAgentSolver(agent=agent), label=f"{teacher.name}_{'+'.join(datasets)}", ids=ids, n_passes=passes
    )
    stamp = time.strftime("%Y%m%d-%H%M%S")
    benchmark.job_dir = ROOT / teacher.name / f"{'+'.join(datasets)}-{stamp}"
    if traces:
        benchmark.trace_root = TRACE_ROOT
        benchmark.retain_traces = True
    task_ids = [g.spec_data.game_id for g in benchmark.games if isinstance(g, taaf.game.ArcGame)]
    print(f"{len(benchmark.games)} tasks x {passes} passes: {task_ids}", flush=True)
    print(f"job dir: {benchmark.job_dir}", flush=True)
    print(
        f"slots: {agent.run_config.max_concurrent}, task wall cap {agent.run_config.max_runtime_s_per_game} s",
        flush=True,
    )
    print(
        f"accounting: {agent.bound_chars_per_token} chars a token, {agent.image_tokens_override} an image, "
        f"reply cap {agent.reply_cap_tokens}, margin {agent.safety_margin_tokens}, game budget {agent.max_tokens_per_game}, "
        f"window {agent.context_window_tokens}, compaction to {agent.compact_to_tokens}, "
        f"bills held to {agent.usage_window_tokens}",
        flush=True,
    )
    print(
        f"summary_thinking {agent.summary_thinking}, images {agent.images}, reasoning_effort {agent.reasoning_effort}",
        flush=True,
    )
    print(f"diagnostics: {'live in the job dir' if live_diagnostics else 'scratch, archived'}", flush=True)
    print(f"traces: {'on, bundled into the job dir' if traces else 'off'}", flush=True)
    if dry_run:
        return
    target = taaf.deploy_slurm.TufaSlurmTarget(
        gpu="CPU",
        gpu_count=0,
        partition=partition,
        # The long and arc partitions take only their own accounts.
        account=partition if partition in ("long", "arc") else "research",
        time=f"{hours:02d}:00:00",
        total_cpus=64,
        total_mem_bytes=256 * 1024**3,
        main_project=platypus.deployment.detect_main_project(),
    )
    handle = await benchmark.deploy(target)
    print(f"submitted {getattr(handle, 'job_id', handle)} -> {benchmark.job_dir}", flush=True)


def main() -> None:
    """The command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--api", required=True, choices=sorted(teachers.API_TEACHERS), help="the API teacher")
    parser.add_argument(
        "--set", default="arc2-train-easy", help="TAAF's named task sets, comma-separated (default arc2-train-easy)"
    )
    parser.add_argument("--ids", default=None, help="comma-separated task ids: play only these")
    parser.add_argument("--passes", type=int, default=1, help="passes over the set (default 1)")
    parser.add_argument("--dry-run", action="store_true", help="build the agent and tasks, submit nothing")
    parser.add_argument("--partition", default="arc", help="the SLURM partition (default arc)")
    parser.add_argument("--hours", type=int, default=48, help="job walltime (default 48)")
    parser.add_argument(
        "--live-diagnostics", action="store_true", help="write diagnostics into the job dir as they happen (few tasks)"
    )
    parser.add_argument("--traces", action="store_true", help="write a TAAF trace per task, bundled into the job dir")
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else None
    asyncio.run(
        launch(
            args.api,
            args.set.split(","),
            ids,
            args.passes,
            args.dry_run,
            args.partition,
            args.hours,
            args.live_diagnostics,
            args.traces,
        )
    )


if __name__ == "__main__":
    main()
