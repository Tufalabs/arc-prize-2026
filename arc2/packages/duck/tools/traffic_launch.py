"""Launch the traffic benchmark's jobs: a B200 recording, a B200 replay, or a
Kaggle replay.

    uv run python packages/duck/tools/traffic_launch.py record --lanes 100 --passes 3
    uv run python packages/duck/tools/traffic_launch.py replay --workload FILE [--lanes N]
    uv run python packages/duck/tools/traffic_launch.py kaggle --workload FILE [--arm NAME]
    uv run python packages/duck/tools/traffic_launch.py smoke --arm NAME

``record`` plays arc2-eval as the Kaggle column's duck would -- its scheduling
and budgets, on the B200's engine -- with ids-only traces on, so the run's
traces are a workload's source (``traffic_workload.py`` cuts one). ``replay``
deploys a ``TrafficDuck`` of the B200 column, at the lane count asked for, over
a workload file. ``kaggle`` deploys the Kaggle column's ``TrafficDuck`` as it
stands, its concurrency included, to a four-L4 notebook set up as
``notebooks/demo.ipynb`` sets one up, the workload riding in the bundle.
``smoke`` plays arc2-eval for a short while on four B200s set up as the four
L4s' engine as far as a B200 runs it, to see an arm serve before Kaggle quota
is spent on it. ``--arm`` names the knobs a run changes (``ARMS``). Replays
and Kaggle runs sample with temperature alone and open torch-profiler windows
in a tail after the reading, by default (``--sampling preset``,
``--no-profile``). All are fire and forget: the job dir is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import getpass
import os
import pathlib
import tempfile
from collections.abc import Callable

import ullm.backend
import ullm.backends.deployment
from duck.agent import DuckAgent
from duck.traffic import TrafficDuck
from ullm.backends.vllm_factory import EngineProfile

import platypus.deployment
import taaf.deploy_kaggle
import taaf.deploy_slurm
import taaf.standard_benchmarks
from platypus import LLMAgentSolver

RESULTS = pathlib.Path("/data/shared/research/arc_2_results") / getpass.getuser() / "traffic"


def _kaggle_like(agent: DuckAgent, lanes: int) -> None:
    """The Kaggle Flash column's scheduling and budgets on the B200 column:
    the fields where the two differ in what the games do, not in the engine."""
    agent.max_tokens_per_game = None
    agent.run_config.max_runtime_s_per_game = None
    agent.run_config.shuffle_games = True
    agent.run_config.slot_policy = "least_tokens"
    agent.run_config.max_concurrent = lanes


def _l4_like(agent: DuckAgent) -> None:
    """The four-L4 cell's engine on B200s, as far as a B200 runs it: two
    pipeline stages of two tensor ranks, the cell's step, no dense graph
    capture, the 60 GiB tier, and the Triton GDN prefill the L4 falls back to.
    Not its memory: the utilization stays the B200's."""
    factory = agent.llm_config.backend_factory
    engine_args = factory.engine_args  # type: ignore[union-attr]
    engine_args.tensor_parallel_size = 2
    engine_args.pipeline_parallel_size = 2
    engine_args.max_num_batched_tokens = 3300
    engine_args.additional_config = {**(engine_args.additional_config or {}), "gdn_prefill_backend": "triton"}
    factory.env_overrides["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "64"  # type: ignore[union-attr]
    # The L4's all-reduce between tensor ranks: vLLM's custom one, then NCCL.
    # A B200 dispatches to FlashInfer's first (and has symmetric memory),
    # which hides what breaks the custom one on the L4.
    factory.env_overrides["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"  # type: ignore[union-attr]
    factory.env_overrides["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"  # type: ignore[union-attr]
    factory.dense_cuda_capture = False  # type: ignore[union-attr]
    factory.cpu_kv_offload_gib = 60.0  # type: ignore[union-attr]
    agent.max_tokens_per_game = None


def _expandable_segments(agent: DuckAgent) -> None:
    """PyTorch's expandable segments. vLLM refuses them beside any KV
    connector -- the host tier is one -- unless its cumem allocator holds the
    KV cache, which keeps expandable segments off for that pool alone. And
    vLLM's custom all-reduce shares its graph buffers between the tensor
    ranks over CUDA IPC, which expandable-segment memory does not support: on
    the L4s it failed capturing graphs (Kaggle v3), so the ranks all-reduce
    over NCCL instead."""
    factory = agent.llm_config.backend_factory
    factory.env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # type: ignore[union-attr]
    factory.engine_args.enable_cumem_allocator = True  # type: ignore[union-attr]
    factory.engine_args.disable_custom_all_reduce = True  # type: ignore[union-attr]


def _expandable(agent: DuckAgent) -> None:
    """Expandable segments, with the indexer's bound at the engine default."""
    _expandable_segments(agent)
    agent.llm_config.backend_factory.env_overrides.pop("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", None)  # type: ignore[union-attr]


def _indexer_default(agent: DuckAgent) -> None:
    """The indexer's prefill scoring buffer at the engine's own bound (512 MB)
    in place of the four-L4 cell's 64."""
    agent.llm_config.backend_factory.env_overrides.pop("VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", None)  # type: ignore[union-attr]


def _custom_ar_off(agent: DuckAgent) -> None:
    """NCCL between the tensor ranks in place of vLLM's custom all-reduce:
    the price expandable segments pays on the L4s."""
    agent.llm_config.backend_factory.engine_args.disable_custom_all_reduce = True  # type: ignore[union-attr]


def _gdn_decode_triton(agent: DuckAgent) -> None:
    """The Triton GDN decode kernel, where the cell runs vLLM's fused CUDA one."""
    agent.llm_config.backend_factory.env_overrides["VLLM_GDN_DECODE_KERNEL"] = "triton"  # type: ignore[union-attr]


def _bf16_state(agent: DuckAgent) -> None:
    """The recurrent state in bf16, where the checkpoint's float32 is the default."""
    agent.llm_config.backend_factory.engine_args.mamba_ssm_cache_dtype = "bfloat16"  # type: ignore[union-attr]


def _bf16_state_two_blocks(agent: DuckAgent) -> None:
    """The bf16 state with the step budget at two of its 1,600-token blocks
    and a hundred for the decodes: the prefill per step the cell's 3,236 buys
    with one 3,136-token block, which two bf16 blocks and the decodes
    outgrow."""
    _bf16_state(agent)
    agent.llm_config.backend_factory.engine_args.max_num_batched_tokens = 3300  # type: ignore[union-attr]


def _unit64(agent: DuckAgent) -> None:
    """Cache hits inside a block, every 64 tokens: divides the 3,136-token
    block and the bf16 state's 1,600 on the L4."""
    agent.llm_config.backend_factory.engine_args.prefix_match_unit = 64  # type: ignore[union-attr]


def _utilization_096(agent: DuckAgent) -> None:
    """The pool sized at 0.96 of each card, where the four-L4 cell runs 0.91."""
    agent.llm_config.backend_factory.engine_args.gpu_memory_utilization = 0.96  # type: ignore[union-attr]


def _all_l4(agent: DuckAgent) -> None:
    """Every four-L4 candidate a B200 can smoke-test, at once: expandable
    segments (the indexer's bound kept, as the cell goes forward), the Triton
    GDN decode, the bf16 state with the step budget at one of its blocks and
    the decodes, and cache hits inside a block. The bf16 state halves the
    3,136-token block, and the attention kernel's alignment rounds the half
    up: 1,600 tokens on 2x2 B200s, 1,568 if a card's kernel aligns finer. So
    the unit is 32, which divides both, and the budget 1,700 holds one block
    and the decodes either way."""
    _expandable_segments(agent)
    _gdn_decode_triton(agent)
    _bf16_state(agent)
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.max_num_batched_tokens = 1700
    engine_args.prefix_match_unit = 32


def _eager(agent: DuckAgent) -> None:
    """No CUDA graphs. Not shippable -- every step pays its launches -- but it
    settles whether a fault is capture alone, and lets two kernels be
    compared eager against eager."""
    agent.llm_config.backend_factory.engine_args.enforce_eager = True  # type: ignore[union-attr]


def _no_mtp(agent: DuckAgent) -> None:
    """No drafter, where the B200's Flash-Next column drafts by default
    (ullm's engine, since the spec-decode branch)."""
    agent.llm_config.backend_factory.engine_args.speculative_config = None  # type: ignore[union-attr]


def _mtp(k: int) -> Callable[[DuckAgent], None]:
    """An arm drafting ``k`` tokens a step with the checkpoint's MTP head."""

    def arm(agent: DuckAgent) -> None:
        """Flash-Next's MTP head drafting, loaded from its own directory
        (arc3-platypus ``tools/checkpoints/extract_mtp_drafter.py``) so it does
        not re-read the target's shards. Its bf16 experts need their own MoE
        backend, since Marlin serves no bf16; its KV takes the target's cache
        dtype. Draft tokens are sampled and verified against their
        probabilities: vLLM's argmax default is verified as one-hot, which caps
        acceptance at the target's top-1 probability."""
        engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
        engine_args.speculative_config = {
            "method": "mtp",
            "num_speculative_tokens": k,
            "model": engine_args.model + "-drafter",
            "quantization": engine_args.quantization,
            "draft_sample_method": "probabilistic",
            "moe_backend": "triton",
        }

    return arm


def _gdn_recoverssm(agent: DuckAgent) -> None:
    """GDN RecoverSSM on the drafted arm it follows (fork
    ``spec-decode-recoverssm``): the verify runs from the checkpoint
    and records per position what the accepted state needs, so no Mamba group
    keeps per-draft-token state blocks. Its verify is Triton where the plain
    drafted arm runs vLLM's fused CUDA kernel -- ``gdn_decode_triton`` on a
    drafted arm isolates that swap."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("gdn_recoverssm needs an mtp arm before it")
    speculative["gdn_recoverssm"] = True


def _prefix_match_unit_16(agent: DuckAgent) -> None:
    """Prefix-cache keys every 16 tokens: a unit that divides the block with
    the MTP drafter (1,616 = 16 x 101 on ARC3's RTX at k=3, 2,000 on the B200
    at k=9) as well as without it (1,600), so drafted and plain arms can
    share one unit."""
    agent.llm_config.backend_factory.engine_args.prefix_match_unit = 16  # type: ignore[union-attr]


def _indexer_fp8(agent: DuckAgent) -> None:
    """The QSA indexer's compressed-key cache in plain e4m3 where the column
    keeps bf16 (upstream #54890, in the fork): half the bytes a cached key,
    the drafter's QSA layer included. A score risk: unscaled fp8 keys change
    which tokens the indexer selects."""
    agent.llm_config.backend_factory.engine_args.attention_config.indexer_kv_dtype = "fp8"  # type: ignore[union-attr]


# The target's bf16 GEMMs the fp8_dense arm quantizes (tier 1): the GDN q/k/v/z
# and output projections, QSA's q/k/v(/gate) and output projections, the
# shared expert. The target only (the drafter's layers sit under mtp.).
FP8_DENSE_TIER1 = (
    "language_model.model.layers.*.linear_attn.in_proj_qkvz",
    "language_model.model.layers.*.linear_attn.out_proj",
    "language_model.model.layers.*.self_attn.qkv_proj",
    "language_model.model.layers.*.self_attn.o_proj",
    "language_model.model.layers.*.mlp.shared_expert.gate_up_proj",
    "language_model.model.layers.*.mlp.shared_expert.down_proj",
)


def _fp8_dense(agent: DuckAgent) -> None:
    """FP8 for the target's dense GEMMs the checkpoint leaves bf16, tier 1
    (``FP8_DENSE_TIER1``), quantized at load (fork: ModelOpt's excluded
    layers matching ``additional_config["online_fp8_linear"]``): per output
    channel weights, per token activations. Kept bf16: the GDN gate inputs
    and conv, the QSA indexer projection, the router and shared-expert gate,
    the LM head, the embedding, the hyper-connections. A score risk: GDN
    projection error accumulates in the recurrent state."""
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.additional_config = {
        **(engine_args.additional_config or {}),
        "online_fp8_linear": list(FP8_DENSE_TIER1),
    }


def _draft_top_k_top_p(agent: DuckAgent) -> None:
    """The drafter's distribution cut to the request's own top_k and top_p
    before a draft is drawn (fork: ``speculative_config.draft_top_k_top_p``),
    where vLLM draws drafts at temperature alone: the output distribution is
    unchanged, acceptance rises for requests sampling with top_k or top_p."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("draft_top_k_top_p needs a drafting cell or an mtp arm before it")
    speculative["draft_top_k_top_p"] = True


def _relaxed_accept(agent: DuckAgent) -> None:
    """A public notebook's relaxed draft acceptance at 0.3 (the fork's
    ``speculative_config.accept_threshold_single/_acc``, SGLang's
    ``--speculative-accept-threshold-*``; arc3-platypus competitor-tricks T8):
    a draft token the target gives at least 0.3 is accepted outright, any
    other with probability min(1, p / (0.3 q)). A score risk: sampling
    sharpens toward the draft."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("relaxed_accept needs a drafting cell or an mtp arm before it")
    speculative["accept_threshold_single"] = 0.3
    speculative["accept_threshold_acc"] = 0.3


def _autoround(agent: DuckAgent) -> None:
    """The target from Intel's AutoRound W4A16 experts, cut to e307's, with
    e307's FP8 PLE table (``FLASHNEXT_E307_AUTOROUND_CHECKPOINT``;
    arc3-platypus competitor-tricks T1): INT4 experts on GPTQ Marlin through
    vLLM's ``inc`` quantization, everything else the base model's bf16. The
    drafter stays e307's, fine-tuned on the NVFP4 target's hidden states. A
    score risk: other expert weights."""
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.model = ullm.backends.deployment.FLASHNEXT_E307_AUTOROUND_CHECKPOINT
    engine_args.quantization = "inc"


def _qad(agent: DuckAgent) -> None:
    """The target from local-inference-lab's quantization-aware distillation,
    cut to e307's experts, with e307's FP8 PLE table
    (``FLASHNEXT_E307_QAD_CHECKPOINT``; arc3-platypus competitor-tricks T7):
    NVFP4 experts on the cell's Marlin, MXFP8 shared experts and attention,
    ModelOpt's mixed precision. The drafter stays e307's, fine-tuned on the
    PTQ target's hidden states. A score risk either way: other weights
    throughout."""
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.model = ullm.backends.deployment.FLASHNEXT_E307_QAD_CHECKPOINT
    engine_args.quantization = "modelopt_mixed"


def _baseline(agent: DuckAgent) -> None:
    """The column as it stands."""


# The knob arms a run can take, each a change on top of the column.
def _v7_like(agent: DuckAgent) -> None:
    """Kaggle v7 of 2026-09-19 (flash_optimization_arc2.md, 552.4 generated
    tok/s) and main's round 3 (556.4): its agent's window and prune, so the
    engine derives v7's 65,536-token max_model_len; its images' gridlines,
    legend and text grids, so the processor derives v7's max_pixels; its
    presence penalty of 1.5; the host tier's engine log on. With v7's
    workload and 70 lanes (the command line), and --decode-gate where round 3
    prefilled each lane's history: the run that sets a revision against
    main's round 3."""
    agent.compaction_strategy = "prune"
    agent.context_window_tokens = 65536
    agent.reply_cap_tokens = 16384
    agent.image_gridlines = True
    agent.image_legend = True
    agent.text_grids = True
    profile = agent.llm_config.sampling_profiles[agent.profile]
    profile.extra_params = {**(profile.extra_params or {}), "presence_penalty": 1.5}  # type: ignore[union-attr]
    agent.llm_config.backend_factory.cpu_kv_offload_extra = {"engine_log": True}  # type: ignore[union-attr]
    # v7's pacing: 5 s between a lane's calls (the replay's and the agent's),
    # and the runner's 0.5 s between starts.
    if isinstance(agent, TrafficDuck):
        agent.request_gap_s = 5.0
    agent.run_config.min_request_gap_s = 5.0
    agent.run_config.start_gap_s = 0.5


ARMS = {
    "baseline": _baseline,
    "expandable": _expandable,
    "indexer_default": _indexer_default,
    "custom_ar_off": _custom_ar_off,
    "gdn_decode_triton": _gdn_decode_triton,
    "bf16_state": _bf16_state,
    "bf16_state_two_blocks": _bf16_state_two_blocks,
    "unit64": _unit64,
    "utilization_096": _utilization_096,
    "all_l4": _all_l4,
    "eager": _eager,
    "no_mtp": _no_mtp,
    "mtp1": _mtp(1),
    "mtp2": _mtp(2),
    "mtp3": _mtp(3),
    "mtp4": _mtp(4),
    "gdn_recoverssm": _gdn_recoverssm,
    "prefix_match_unit_16": _prefix_match_unit_16,
    "indexer_fp8": _indexer_fp8,
    "fp8_dense": _fp8_dense,
    "draft_top_k_top_p": _draft_top_k_top_p,
    "v7_like": _v7_like,
    "relaxed_accept": _relaxed_accept,
    "autoround": _autoround,
    "qad": _qad,
}


def _replayed_profile(agent: TrafficDuck) -> ullm.backend.GenericSamplingParameters:
    """The sampling profile the replay copies; the helpers below replace it in
    a new mapping, as the presets set the profiles."""
    profile = agent.llm_config.sampling_profiles[agent.profile]
    if not isinstance(profile, ullm.backend.GenericSamplingParameters):
        raise SystemExit(f"sampling profile {agent.profile!r} is {type(profile).__name__}")
    return profile


def _plain_sampling(agent: TrafficDuck) -> None:
    """Temperature alone: top_p 1 and top_k, min_p and the presence penalty
    off, on the play profile the replay copies -- every ARC2/ARC3 test run
    samples so (2026-09-26). The preset's play profile carries top_k
    194; its temperature (0.8) stays."""
    profile = _replayed_profile(agent)
    agent.llm_config.sampling_profiles = {
        **agent.llm_config.sampling_profiles,
        agent.profile: dataclasses.replace(
            profile, top_p=1.0, extra_params={"top_k": 0, "min_p": 0.0, "presence_penalty": 0.0}
        ),
    }


def _top_p(agent: TrafficDuck, top_p: float) -> None:
    """The replayed profile's nucleus at ``top_p``, on top of the sampling
    chosen: ``--sampling preset --top-p 0.8899`` is the preset's profile with
    upstream's sweep-3 nucleus (top_k 194, presence 0), the case a drafter
    sampling at temperature alone loses acceptance on."""
    profile = _replayed_profile(agent)
    agent.llm_config.sampling_profiles = {
        **agent.llm_config.sampling_profiles,
        agent.profile: dataclasses.replace(profile, top_p=top_p),
    }


def _ban_tokens(agent: TrafficDuck, token_ids: list[int]) -> None:
    """Every replayed call bans these token ids (logit bias -100, as the
    summary bans its tool-call token): a test of the ban under drafting,
    which ullm's vLLM backend fails the call on if a banned id comes out."""
    profile = _replayed_profile(agent)
    agent.llm_config.sampling_profiles = {
        **agent.llm_config.sampling_profiles,
        agent.profile: dataclasses.replace(
            profile,
            extra_params={
                **profile.extra_params,
                "logit_bias": {**profile.extra_params.get("logit_bias", {}), **{t: -100.0 for t in token_ids}},
            },
        ),
    }


def _profile_after_window(
    agent: TrafficDuck, window_s: float, with_stack: bool, iterations: int = 100, tail_s: float = 240.0
) -> None:
    """Torch-profiler windows after the reading: the lanes play on for a tail
    the reading leaves out, and two windows open in it, by time rather than by
    pool. A window's close stalls the engine for up to a minute (68 s seen on
    ARC3's RTX, a minute under a B200 drafter), so they sit 90 s apart. The
    replay starts as the engine becomes ready, which is where not_before_s
    counts from. Under pipeline stages one window: the profiler's step cap
    stops the first stage's profiler inside a step and the last stage's only
    on the stop call, and the next window's start then segfaults the first
    stage's workers (2026-09-28). A longer window (``iterations``
    steps) needs a longer tail for its trace to be written."""
    staged = agent.llm_config.backend_factory.engine_args.pipeline_parallel_size > 1  # type: ignore[union-attr]
    agent.llm_config.backend_factory.profile = EngineProfile(  # type: ignore[union-attr]
        kv_cache_above=0.0,
        not_before_s=window_s + 30.0,
        windows=1 if with_stack or staged else 2,
        spacing_s=90.0,
        iterations=iterations,
        with_stack=with_stack,
    )
    agent.tail_s = tail_s


def _drafter_moe_backend(agent: DuckAgent, backend: str | None) -> None:
    """The mtp arm's head served on ``backend`` in place of the arm's own."""
    if backend is None:
        return
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("--drafter-moe-backend needs an mtp arm")
    speculative["moe_backend"] = backend


def _pp_layer_partition(agent: DuckAgent, partition: str | None) -> None:
    """The decoder layers each pipeline stage builds, first stage first."""
    if partition is None:
        return
    agent.llm_config.backend_factory.env_overrides["VLLM_PP_LAYER_PARTITION"] = partition  # type: ignore[union-attr]


def _target(
    hours: float,
    *,
    gpus: int = 1,
    gpu: taaf.deploy_slurm.GPU = "B200",
    walltime: str | None = None,
    node: str | None = None,
) -> taaf.deploy_slurm.TufaSlurmTarget:
    """One B200, sized like the column's own launches, for ``hours`` of run;
    on ``node`` when named, so runs to be compared share a machine. The
    caches are deploy_slurm's: its job script clears the submitter's uv, temp
    and XDG locations and puts them on node-local storage (node_local_storage,
    on by default), and a deploy runs no uv on this machine."""
    budget_s = int(hours * 3600)
    wall_s = budget_s + 30 * 60
    return taaf.deploy_slurm.TufaSlurmTarget(
        time=walltime or f"{wall_s // 3600:02d}:{wall_s % 3600 // 60:02d}:00",
        max_runtime_s=budget_s,
        main_project=platypus.deployment.detect_main_project(),
        nodelist=node,
        gpu=gpu,
        # The same 24 CPUs and 260 GiB in all, over however many cards.
        gpu_count=gpus,
        cpus_per_gpu=24 // gpus,
        mem_per_gpu_bytes=260 * 1024**3 // gpus,
    )


# The machine a Kaggle run lands on (--machine-log): the card topology and a
# full nvidia-smi query at the start and the end, a peer/host copy probe
# before the engine starts (machine_probe.py, carried as a payload file), and
# every 15 s each card's PCIe link, clocks, power and throttle reasons, and
# the PCIe traffic, for the whole run; and the environment the engine's
# processes started with, once its four workers are up (engine_env_probe.sh:
# whether a variable the run sets reached the processes that use it). All
# into /kaggle/working, so the bundle carries them; none of it fatal.
MACHINE_PROBE = pathlib.Path(__file__).with_name("machine_probe.py")
ENGINE_ENV_PROBE = pathlib.Path(__file__).with_name("engine_env_probe.sh")
MACHINE_SETUP = (
    "( nvidia-smi topo -m > /kaggle/working/machine_topo.txt 2>&1 ;"
    " nvidia-smi -q > /kaggle/working/machine_smi_start.txt 2>&1 ) || true",
    '( "$PYTHON" "$TAAF_KAGGLE_BUNDLE_DIR/payload/machine_probe.py" > /kaggle/working/machine_p2p.json'
    ' 2> /kaggle/working/machine_p2p.err ) || echo "machine probe failed"',
    "( setsid nohup nvidia-smi --query-gpu=timestamp,index,pcie.link.gen.current,pcie.link.width.current,"
    "clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active,"
    # Last, so the columns before keep their places: the cards' memory over
    # the run, whose high-water mark is the headroom the utilization search
    # reads (an out-of-memory at the edge comes in serving, not start-up).
    "memory.used,memory.free"
    " --format=csv -l 15 > /kaggle/working/machine_gpu_samples.csv 2>&1 < /dev/null & ) || true",
    "( setsid nohup nvidia-smi dmon -s pct -d 15 -o T > /kaggle/working/machine_dmon.log 2>&1 < /dev/null & ) || true",
    '( setsid nohup bash "$TAAF_KAGGLE_BUNDLE_DIR/payload/engine_env_probe.sh" > /kaggle/working/engine_env.txt'
    " 2>&1 < /dev/null & ) || true",
)
MACHINE_TEARDOWN = "nvidia-smi -q > /kaggle/working/machine_smi_end.txt 2>&1 || true"
# Everything the run left in the working directory, packed into one file at
# its end, crash or not (TAAF runs teardown commands in a finally): one file to
# carry off an interactive session by hand. engine_profile/ is left out, as
# engine_profile.tar already holds it.
BUNDLE_COMMAND = (
    "tar --warning=no-file-changed --exclude=./run_bundle.tar.gz --exclude=./engine_profile"
    " -czf run_bundle.tar.gz . ; ls -l run_bundle.tar.gz"
)


def _kaggle_target(
    hours: float, slug: str = "tufa-arc2-traffic", machine_log: bool = False, run_id: str | None = None
) -> taaf.deploy_kaggle.KaggleTarget:
    """The competition's four L4s, as ``notebooks/demo.ipynb`` deploys the duck
    there, for ``hours`` of run. The run ends when the replay does; the budget
    only bounds a replay that hangs. A kernel of its own (``slug``, with its
    source dataset ``<slug>-src``) runs beside the default one instead of
    replacing its version. With ``run_id`` the source dataset is the run's own
    (``<slug>-src-<run_id>``): a kernel version attaches a dataset's current
    version, which can still be the previous run's while the new one goes
    live (TAAF's wait for it only warns), and a dataset of the run's own has
    no previous version to attach."""
    owner = platypus.deployment.kaggle_owner()
    return taaf.deploy_kaggle.KaggleTarget(
        username=owner,
        kernel_slug=slug,
        kernel_title=slug.replace("-", " "),
        dataset_ref=f"{owner}/{slug}-src" + (f"-{run_id}" if run_id else ""),
        accelerator="NvidiaL4",
        run_as_submission=False,
        max_runtime_s=hours * 3600,
        submission_stop_runtime_s=12 * 3600.0 - taaf.deploy_kaggle.SUBMISSION_END_MARGIN_S,
        teardown_commands=[*([MACHINE_TEARDOWN] if machine_log else []), BUNDLE_COMMAND],
    )


def main() -> None:
    """Parse the job to launch and deploy it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    record = sub.add_parser("record")
    record.add_argument("--lanes", type=int, default=100)
    record.add_argument("--passes", type=int, default=3)
    record.add_argument("--hours", type=float, default=1.5)
    record.add_argument("--tier-gib", type=float, default=48.0)
    # Below the cell's 0.97 when many lanes' CUDA graphs outgrow the headroom
    # the capture was profiled with (300 lanes at k=4 ran out of memory).
    record.add_argument("--utilization", type=float, default=None)
    replay = sub.add_parser("replay")
    replay.add_argument("--workload", required=True)
    # The B200 column with the Kaggle column's scheduling (_kaggle_like).
    # The column's own concurrency unless named: the column's max_concurrent as
    # built, so the replay tracks gameplay rather than a copied figure. Lanes
    # past the workload's replay its lanes again, nonced.
    replay.add_argument("--lanes", type=int, default=None)
    # The cards' utilization in place of the column's.
    replay.add_argument("--utilization", type=float, default=None)
    replay.add_argument("--hours", type=float, default=1.5)
    replay.add_argument("--tier-gib", type=float, default=None)
    kaggle = sub.add_parser("kaggle")
    kaggle.add_argument("--workload", required=True)
    kaggle.add_argument("--hours", type=float, default=9.0)
    # A knob that frees pool needs more lanes than the column's, or the pool
    # stops binding; the engine's sequence cap follows (derive_max_num_seqs).
    kaggle.add_argument("--max-concurrent", type=int, default=None)
    # A memory knob is a stability question, not a throughput one: the cell
    # ships 0.91, 0.92 ran out of memory in serving.
    kaggle.add_argument("--utilization", type=float, default=None)
    for p in (replay, kaggle):
        # The CUDA graph capture ladder (comma-separated token counts) in place of
        # the cell's (the four-L4 Flash cell names its own, the others take the
        # engine's): FULL decode graphs take its rungs rounded to 1 + k
        # tokens, PIECEWISE ones its rungs as given, the drafter's later
        # positions its rungs up to max_num_seqs as request counts; above the
        # top a batch runs eagerly (upstream's flash_optimization_arc2.md, "Drafting
        # on the four L4s").
        p.add_argument("--capture-sizes", default=None)
    for p in (replay, kaggle):
        p.add_argument("--window-s", type=float, default=1500.0)  # ten minutes of warm-up, fifteen read
        p.add_argument("--settle-s", type=float, default=600.0)
    # The engine without its host tier.
    replay.add_argument("--no-tier", action="store_true")
    for p in (replay, kaggle):
        # Decoder layers per pipeline stage (vLLM's VLLM_PP_LAYER_PARTITION,
        # e.g. "26,22"), in place of the four-L4 Flash cell's 27/21: the
        # drafter and its KV sit on the last stage alone.
        p.add_argument("--pp-layer-partition", default=None)
    for p in (replay, kaggle):
        # Temperature alone (_plain_sampling), or the preset's play profile.
        p.add_argument("--sampling", choices=("plain", "preset"), default="plain")
        # The nucleus on top of it (_top_p).
        p.add_argument("--top-p", type=float, default=None)
        # Torch-profiler windows after the reading (_profile_after_window).
        p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
        # One profiler window with Python stacks, to name the host calls
        # behind a stall; the traces are several times larger, so it is a
        # single window.
        p.add_argument("--profile-stack", action="store_true")
        # Engine steps a window records, and the tail the lanes play on for
        # after the reading, which must hold the window and its trace's write.
        p.add_argument("--profile-iterations", type=int, default=100)
        p.add_argument("--tail-s", type=float, default=240.0)
        # The engine core's batch queue, a line per step (the fork's
        # VLLM_BATCH_QUEUE_TRACE), and each worker's RPCs, a line per call
        # (VLLM_WORKER_RPC_TRACE), on one clock: when each batch was sent,
        # picked up by each stage and answered.
        p.add_argument("--batch-queue-trace", action="store_true")
        # Who holds the pool, about once a second (the fork's VLLM_KV_BLOCKS_TRACE):
        # per request its blocks in each KV cache group.
        p.add_argument("--kv-blocks-trace", action="store_true")
        # The decode gate (TrafficDuck.decode_gate): every lane's first prompt
        # prefilled on its thread before the window, which opens warm.
        p.add_argument("--decode-gate", action="store_true")
        # The engine's environment, KEY=VALUE, repeatable (e.g. an NCCL knob).
        p.add_argument("--engine-env", action="append", default=[])
    # The engine's processes' output also in engine_output.log in the job dir
    # (ullm's engine_output_log); a Kaggle run always has it, as the console
    # is its only other record.
    replay.add_argument("--engine-output-log", action="store_true")
    for p in (replay, kaggle):
        # Keep the prefix-cache block a drafter's lookup would drop (upstream
        # #53388): a continuation shares the token the drafter's cached row
        # depends on, so the drop only recomputes. The drafting cell sets it.
        p.add_argument("--no-eagle-block-drop", action="store_true")
    # The mtp arms' head on Kaggle, as its own dataset (OWNER/SLUG) mounted
    # beside the checkpoint's: the <model>-drafter rule has no directory to
    # find on a dataset mount, and the checkpoint's dataset is not rewritten.
    kaggle.add_argument("--drafter-dataset", default=None)
    # The kernel's slug: a new one runs beside the default kernel's current
    # version instead of pushing over it.
    kaggle.add_argument("--slug", default="arc2-traffic-duck")
    kaggle.add_argument("--machine-log", action="store_true")
    # The source dataset the run's own, <slug>-src-<run id> (_kaggle_target).
    kaggle.add_argument("--run-id", default=None)
    # The KV pool capped at this many GiB instead of sized from the card's
    # utilization: squeezes the pool so a replay's lanes overflow into the
    # host tier and exercise its loads, at a throughput that is not the cell's.
    replay.add_argument("--kv-cache-gib", type=float, default=None)
    # The drafter from this directory instead of the cell's or the arm's: a
    # trained head (arc3-platypus branch drafter-sft).
    replay.add_argument("--drafter", default=None)
    # Token ids every replayed call bans (logit bias -100), repeatable.
    replay.add_argument("--ban-token", type=int, action="append", default=None)
    # Each call's reply ids kept in traffic_calls.jsonl, to read the text.
    replay.add_argument("--record-outputs", action="store_true")
    # The head's MoE backend instead of the one its cell or arm serves it on.
    for p in (replay, kaggle):
        p.add_argument("--drafter-moe-backend", default=None)
    smoke = sub.add_parser("smoke")
    for p in (record, replay, kaggle, smoke):
        # Cache hits every this many tokens; 0 leaves the cell's own (80 on
        # the Flash-Next cells, the four L4s included, which divides the block
        # undrafted and at k=4; at k=3 the B200's block is 1,792 and needs 64).
        p.add_argument("--prefix-match-unit", type=int, default=0)
        # The drafting cell's own k (its drafter as it ships, only the count
        # changed); the prefix-match unit must divide the block that k gives.
        p.add_argument("--num-speculative-tokens", type=int, default=None)
    for p in (replay, kaggle, smoke):
        # Repeatable: the arms apply in order, each a change on top of the last.
        p.add_argument("--arm", choices=sorted(ARMS), action="append", default=None)
    for p in (replay, kaggle):
        # A lane's pause between calls, where the TrafficDuck's (none) would
        # not compare: 5 for a replay of the 09-19 workload.
        p.add_argument("--request-gap-s", type=float, default=None)
    for p in (record, replay, kaggle, smoke):
        p.add_argument("--label", default="")
        # Build the agent and write its solver's repr to this file, deploying nothing.
        p.add_argument("--dry-run", default=None)
    for p in (record, replay):
        p.add_argument("--node", default=None)
    args = parser.parse_args()
    arms = getattr(args, "arm", None) or ["baseline"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.job == "record":
        agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
        _kaggle_like(agent, args.lanes)
        # The tier only moves the recording's timing, and every game started
        # keeps its process while parked: the host memory goes to those.
        agent.llm_config.backend_factory.cpu_kv_offload_gib = args.tier_gib  # type: ignore[union-attr]
        if args.utilization is not None:
            agent.llm_config.backend_factory.engine_args.gpu_memory_utilization = args.utilization  # type: ignore[union-attr]
        agent.llm_config.traces = True
        agent.llm_config.trace_content = "ids"
        passes = args.passes
        name = f"record_{args.lanes}lanes_{passes}x_{stamp}"
    elif args.job == "kaggle":
        agent = TrafficDuck.build(
            engine="qwen3.8-flash-next",
            environment="kaggle-4xl4",
            workload=args.workload,
            window_s=args.window_s,
            settle_s=args.settle_s,
        )
        if args.sampling == "plain":
            _plain_sampling(agent)
        if args.top_p is not None:
            _top_p(agent, args.top_p)
        if args.profile:
            _profile_after_window(agent, args.window_s, args.profile_stack, args.profile_iterations, args.tail_s)
        # The cell drafts from its own dataset; an arm that replaces its head
        # names a path no kernel mounts, so the head comes as a dataset too.
        cell_drafter = (agent.llm_config.backend_factory.engine_args.speculative_config or {}).get("model")  # type: ignore[union-attr]
        for arm in arms:
            ARMS[arm](agent)
        speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
        if speculative and speculative.get("model") != cell_drafter and not args.drafter_dataset:
            raise SystemExit("an arm replaced the cell's drafter: name its head on Kaggle with --drafter-dataset")
        if args.drafter_dataset:
            if not speculative:
                raise SystemExit("--drafter-dataset needs a drafting engine")
            agent.run_config.kaggle_dataset_sources = (
                *agent.run_config.kaggle_dataset_sources,
                args.drafter_dataset,
            )
            speculative["model"] = f"/kaggle/input/datasets/{args.drafter_dataset}"
        if args.no_eagle_block_drop:
            if not speculative:
                raise SystemExit("--no-eagle-block-drop needs an mtp arm")
            speculative["disable_eagle_block_drop"] = True
        _drafter_moe_backend(agent, args.drafter_moe_backend)
        _pp_layer_partition(agent, args.pp_layer_partition)
        if args.prefix_match_unit:
            agent.llm_config.backend_factory.engine_args.prefix_match_unit = args.prefix_match_unit  # type: ignore[union-attr]
        if args.max_concurrent is not None:
            agent.run_config.max_concurrent = args.max_concurrent
        if args.utilization is not None:
            agent.llm_config.backend_factory.engine_args.gpu_memory_utilization = args.utilization  # type: ignore[union-attr]
        util = agent.llm_config.backend_factory.engine_args.gpu_memory_utilization  # type: ignore[union-attr]
        passes = 1
        name = (
            f"kaggle_{agent.run_config.max_concurrent}lanes_util{util}_{'profile_' if args.profile else ''}"
            f"{'plain_' if args.sampling == 'plain' else ''}{f'topp{args.top_p:g}_' if args.top_p is not None else ''}"
            f"{'nodrop_' if args.no_eagle_block_drop else ''}"
            f"{'+'.join(arms)}{'_' + args.drafter_dataset.split('/')[-1] if args.drafter_dataset else ''}"
            f"{'_' + args.drafter_moe_backend if args.drafter_moe_backend else ''}_{stamp}"
        )
    elif args.job == "smoke":
        agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
        _kaggle_like(agent, 50)
        _l4_like(agent)
        for arm in arms:
            ARMS[arm](agent)
        passes = 1
        name = f"smoke_{'+'.join(arms)}_{stamp}"
    else:
        agent = TrafficDuck.build(
            engine="qwen3.8-flash-next",
            environment="slurm-b200",
            workload=args.workload,
            window_s=args.window_s,
            settle_s=args.settle_s,
        )
        lanes = args.lanes if args.lanes is not None else agent.run_config.max_concurrent
        _kaggle_like(agent, lanes)
        # To replay on a recording's own engine, give it the recording's tier.
        if args.tier_gib is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = args.tier_gib  # type: ignore[union-attr]
        if args.no_tier:
            agent.llm_config.backend_factory.cpu_kv_offload = False  # type: ignore[union-attr]
        if args.kv_cache_gib is not None:
            agent.llm_config.backend_factory.engine_args.kv_cache_memory_bytes = int(args.kv_cache_gib * 2**30)  # type: ignore[union-attr]
        if args.prefix_match_unit:
            agent.llm_config.backend_factory.engine_args.prefix_match_unit = args.prefix_match_unit  # type: ignore[union-attr]
        if args.sampling == "plain":
            _plain_sampling(agent)
        if args.top_p is not None:
            _top_p(agent, args.top_p)
        if args.ban_token:
            _ban_tokens(agent, args.ban_token)
        agent.record_outputs = args.record_outputs
        if args.profile:
            _profile_after_window(agent, args.window_s, args.profile_stack, args.profile_iterations, args.tail_s)
        for arm in arms:
            ARMS[arm](agent)
        if args.no_eagle_block_drop:
            speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
            if not speculative:
                raise SystemExit("--no-eagle-block-drop needs an mtp arm")
            speculative["disable_eagle_block_drop"] = True
        if args.drafter:
            speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
            if not speculative:
                raise SystemExit("--drafter needs an mtp arm")
            speculative["model"] = args.drafter
        _drafter_moe_backend(agent, args.drafter_moe_backend)
        _pp_layer_partition(agent, args.pp_layer_partition)
        passes = 1
        name = (
            f"replay_{lanes}lanes_{'notier_' if args.no_tier else ''}{'profile_' if args.profile else ''}"
            f"{'plain_' if args.sampling == 'plain' else ''}"
            f"{f'topp{args.top_p:g}_' if args.top_p is not None else ''}"
            f"{f'unit{args.prefix_match_unit}_' if args.prefix_match_unit else ''}"
            f"{'nodrop_' if args.no_eagle_block_drop else ''}"
            f"{'pp' + args.pp_layer_partition.replace(',', '-') + '_' if args.pp_layer_partition else ''}"
            f"{f'kv{args.kv_cache_gib:g}g_' if args.kv_cache_gib is not None else ''}"
            f"{'ban' + '-'.join(map(str, args.ban_token)) + '_' if args.ban_token else ''}"
            f"{'+'.join(arms)}{'_' + pathlib.Path(args.drafter).name if args.drafter else ''}"
            f"{'_' + args.drafter_moe_backend if args.drafter_moe_backend else ''}_{stamp}"
        )
    if getattr(args, "capture_sizes", None):
        sizes = sorted({int(size) for size in args.capture_sizes.split(",")})
        engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
        compilation = engine_args.compilation_config  # type: ignore[union-attr]
        if compilation is None:
            engine_args.compilation_config = {"cudagraph_capture_sizes": sizes}
        elif isinstance(compilation, dict):
            compilation["cudagraph_capture_sizes"] = sizes  # type: ignore[index]
        else:
            compilation.cudagraph_capture_sizes = sizes  # type: ignore[union-attr]
        name = f"{name}_cg{len(sizes)}-{sizes[-1]}"
    # A replay sets its unit in its own branch, ahead of its arms.
    if args.job != "replay" and args.prefix_match_unit:
        agent.llm_config.backend_factory.engine_args.prefix_match_unit = args.prefix_match_unit  # type: ignore[union-attr]
        name = f"{name}_unit{args.prefix_match_unit}"
    if getattr(args, "request_gap_s", None) is not None:
        agent.request_gap_s = args.request_gap_s  # type: ignore[attr-defined]
        name = f"{name}_gap{args.request_gap_s:g}"
    if getattr(args, "num_speculative_tokens", None) is not None:
        speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
        if not speculative:
            raise SystemExit("--num-speculative-tokens needs a drafting cell")
        speculative["num_speculative_tokens"] = args.num_speculative_tokens
        name = f"{name}_k{args.num_speculative_tokens}"
    if args.label:
        name = f"{name}_{args.label}"
    if args.job == "kaggle" or getattr(args, "engine_output_log", False):
        agent.llm_config.backend_factory.engine_output_log = True  # type: ignore[union-attr]
    for item in getattr(args, "engine_env", []):
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit(f"--engine-env takes KEY=VALUE, got {item!r}")
        agent.llm_config.backend_factory.env_overrides[key] = value  # type: ignore[union-attr]
    if getattr(args, "decode_gate", False):
        agent.decode_gate = True  # type: ignore[attr-defined]
    if getattr(args, "machine_log", False):
        agent.run_config.kaggle_bundle_files = (
            *agent.run_config.kaggle_bundle_files,
            str(MACHINE_PROBE),
            str(ENGINE_ENV_PROBE),
        )
        agent.run_config.kaggle_setup_commands = (*agent.run_config.kaggle_setup_commands, *MACHINE_SETUP)
    agent.prepare_for_backend()
    solver = LLMAgentSolver(agent=agent)
    # A replay plays no game, so one task stands in for the benchmark TAAF runs.
    limit = 1 if args.job in ("replay", "kaggle") else None
    benchmark = taaf.standard_benchmarks.make_benchmark_named("arc2-eval", solver, n_passes=passes, limit=limit)
    # A Kaggle deploy only stages its bundle here, so it goes to local temp
    # (as in the demo); the run's output is pulled from the kernel.
    job_dir = (pathlib.Path(tempfile.gettempdir()) / "traffic" if args.job == "kaggle" else RESULTS) / name
    benchmark.label = f"traffic_{name}"
    benchmark.job_dir = job_dir
    if getattr(args, "batch_queue_trace", False):
        traces = pathlib.Path("/kaggle/working") if args.job == "kaggle" else job_dir
        env = agent.llm_config.backend_factory.env_overrides  # type: ignore[union-attr]
        env["VLLM_BATCH_QUEUE_TRACE"] = str(traces / "batch_queue_trace.jsonl")
        env["VLLM_WORKER_RPC_TRACE"] = str(traces / "worker_rpc_trace")
    if getattr(args, "kv_blocks_trace", False):
        traces = pathlib.Path("/kaggle/working") if args.job == "kaggle" else job_dir
        env = agent.llm_config.backend_factory.env_overrides  # type: ignore[union-attr]
        env["VLLM_KV_BLOCKS_TRACE"] = str(traces / "kv_blocks_trace.jsonl")
    if getattr(args, "dry_run", None):
        pathlib.Path(args.dry_run).write_text(f"benchmark.solver: {solver!r}\n")
        print(f"dry run: the solver's repr is in {args.dry_run}", flush=True)
        return
    benchmark.submission_path = (
        pathlib.Path("/kaggle/working/submission.json") if args.job == "kaggle" else job_dir / "submission.json"
    )
    print(
        f"{args.job}: {len(benchmark.games)} games x {benchmark.n_passes} pass(es), "
        f"max_concurrent={agent.run_config.max_concurrent}\njob_dir={job_dir}",
        flush=True,
    )
    if args.job == "kaggle":
        # The fork's overlay is built here from upstream's base wheel, kept
        # between deploys under ~/.cache unless ULLM_VLLM_WHEEL_CACHE names a
        # place; on a machine whose XDG cache lives elsewhere, there.
        xdg = os.environ.get("XDG_CACHE_HOME")
        if xdg and not os.environ.get("ULLM_VLLM_WHEEL_CACHE"):
            os.environ["ULLM_VLLM_WHEEL_CACHE"] = str(pathlib.Path(xdg) / "ullm" / "vllm-base-wheels")
        target = _kaggle_target(args.hours, args.slug, machine_log=args.machine_log, run_id=args.run_id)
    elif args.job == "smoke":
        # Thirty minutes of walltime, of which TAAF's soft end keeps ten for
        # the teardown: a boot and some ten minutes of play.
        target = _target(0.5, gpus=4, walltime="00:30:00")
    else:
        target = _target(args.hours, node=args.node)
    handle = asyncio.run(benchmark.deploy(target))
    print("submitted:", getattr(handle, "job_id", None) or getattr(handle, "kernel_id", handle), flush=True)


if __name__ == "__main__":
    main()
