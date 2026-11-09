"""Launch the traffic benchmark's jobs: a B200 recording, a B200 replay, or a
Kaggle replay on the RTX.

    uv run python tools/traffic_launch.py record --lanes 120
    uv run python tools/traffic_launch.py replay --workload FILE [--lanes N]
    uv run python tools/traffic_launch.py kaggle --workload FILE [--arm NAME]

``record`` plays the Kaggle set as the RTX column's duck would -- its
scheduling, on the B200's engine -- with ids-only traces on, so the run's
traces are a workload's source (``traffic_workload.py`` cuts one). ``replay``
deploys a ``TrafficDuck`` of the B200 column over a workload file. ``kaggle``
deploys the RTX column's ``TrafficDuck`` as it stands, its concurrency
included, the workload riding in the deploy's bundle. ``--arm`` names the knobs
a run changes (``ARMS``). All are fire and forget: the job dir is printed.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime
import getpass
import pathlib
import tempfile
from typing import Any, Callable, cast

import taaf.deploy_kaggle
import taaf.deploy_slurm
import taaf.standard_benchmarks
import ullm.backend
import ullm.backends.deployment
from ullm.backends.vllm_factory import EngineProfile

import platypus.deployment
from duck.agent import DuckAgent
from duck.traffic import TrafficDuck
from platypus import LLMAgentSolver
from seal.agent import SealAgent

RESULTS = pathlib.Path("/data/shared/research/arc_3_results") / getpass.getuser() / "traffic"


def _rtx_like(agent: DuckAgent, lanes: int) -> None:
    """The RTX column's scheduling and budgets on the B200 column: the fields
    where the two differ in what the games do, not in the engine."""
    agent.max_tokens_per_level = None
    agent.run_config.max_runtime_s_per_game = None
    agent.run_config.shuffle_games = True
    agent.run_config.slot_policy = "least_level_tokens"
    agent.run_config.max_concurrent = lanes


def _baseline(agent: DuckAgent) -> None:
    """The column as it stands."""


def _kernel_stack(agent: DuckAgent) -> None:
    """Enable the guarded Flash-Next inference stack without changing the workload."""
    factory = agent.llm_config.backend_factory
    factory.env_overrides.update(ullm.backends.deployment.FLASHNEXT_THROUGHPUT_ENV)  # type: ignore[union-attr]


def _no_fused_commit(agent: DuckAgent) -> None:
    """The throughput stack without its deferred GDN state commit, the one
    switch whose failure would leave a request's recurrent state short of
    tokens it accepted. XKB_XGDN goes with it: it is the CUDA build of the
    fused verify and runs nowhere else."""
    factory = agent.llm_config.backend_factory
    factory.env_overrides.update({"GDN_FUSED_COMMIT": "0", "XKB_XGDN": "0"})  # type: ignore[union-attr]


def _bf16_state(agent: DuckAgent) -> None:
    """The recurrent state in bf16, where the checkpoint's float32 is the
    default: ARC2's four-L4 column reads +23% generated with it."""
    agent.llm_config.backend_factory.engine_args.mamba_ssm_cache_dtype = "bfloat16"  # type: ignore[union-attr]


def _expandable(agent: DuckAgent) -> None:
    """PyTorch's expandable segments, against the fragmentation that ends a
    run: every allocator failure on record asked for less than the card had
    reserved but unusable.

    vLLM refuses them beside any KV connector -- the host tier is one --
    unless its cumem allocator holds the KV cache, which keeps expandable
    segments off for that pool alone. The L4s' blocker does not exist here:
    custom all-reduce shares its graph buffers between tensor ranks over CUDA
    IPC, which expandable-segment memory cannot do, and this card is one rank.
    """
    factory = agent.llm_config.backend_factory
    factory.env_overrides["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # type: ignore[union-attr]
    factory.engine_args.enable_cumem_allocator = True  # type: ignore[union-attr]


def _indexer_64(agent: DuckAgent) -> None:
    """The indexer's prefill scoring buffer bounded at 64 MB, where this column
    runs the engine's own 512. The four-L4 column ships the bound and reads no
    throughput cost for it; there it is what ended a run's allocator failures
    (52 in twenty minutes without it, 2 with)."""
    factory = agent.llm_config.backend_factory
    factory.env_overrides["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "64"  # type: ignore[union-attr]


def _b12x_a16(agent: DuckAgent) -> None:
    """The b12x MoE kernels with their FP4 activations forced to bf16: b12x's
    speed question without its score question, since that is Marlin's numerics.

    This is vLLM's own b12x backend (#52018, inside our pin), not FlashInfer's
    `flashinfer_b12x`, which faults on this card and has no fix upstream. The
    cell names the backend because `auto` never reaches it: the change that
    would put b12x before Marlin is unmerged and postdates the pin.
    """
    factory = agent.llm_config.backend_factory
    factory.engine_args.kernel_config.moe_backend = "b12x"  # type: ignore[union-attr]
    factory.env_overrides["VLLM_B12X_MOE_FP4_FORCE_A16"] = "1"  # type: ignore[union-attr]


def _b12x_fp4(agent: DuckAgent) -> None:
    """b12x on the checkpoint's own FP4 activations (W4A4), which is what the
    kernel is for and what carries the score risk: every FlashInfer W4A4
    backend quantizes expert inputs against one scale shared across experts
    (TRT-LLM's shape scored 28.19 against Marlin's 30.79), where this one takes
    the checkpoint's per-expert scales. Throughput here, score on its own run."""
    factory = agent.llm_config.backend_factory
    factory.engine_args.kernel_config.moe_backend = "b12x"  # type: ignore[union-attr]
    factory.env_overrides.pop("VLLM_B12X_MOE_FP4_FORCE_A16", None)  # type: ignore[union-attr]


def _marlin(agent: DuckAgent) -> None:
    """Marlin on the B200: the cell's W4A16 arithmetic in the older warp-level
    tensor-core instructions, and the baseline that mode was read against. The
    drafter's experts follow the target's backend."""
    factory = agent.llm_config.backend_factory
    factory.engine_args.kernel_config.moe_backend = "marlin"  # type: ignore[union-attr]
    factory.env_overrides.pop(ullm.backends.deployment.CUTEDSL_W4A16_ENV, None)  # type: ignore[union-attr]


def _eager(agent: DuckAgent) -> None:
    """No CUDA graphs. Not shippable -- every step pays its launches -- but it
    is what settles whether b12x's fault is capture alone, and it lets b12x and
    Marlin be compared eager against eager."""
    agent.llm_config.backend_factory.engine_args.enforce_eager = True  # type: ignore[union-attr]


def _gdn_prefill_flashinfer(agent: DuckAgent) -> None:
    """FlashInfer's GDN prefill kernel in place of Triton's.

    Upstream admits it on 9.0 and 10.x only, so a 12.0 card falls to Triton
    for the phase mixed steps spend the most in; our fork admits SM120 under
    the same constraints (head_k_dim 128, CUDA 13). It is in no prebuilt
    jit-cache, so it compiles on first use -- minutes, and on Kaggle with the
    nvcc the bundle ships.
    """
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.additional_config = {
        **(engine_args.additional_config or {}),
        "gdn_prefill_backend": "flashinfer",
    }


def _gdn_decode_triton(agent: DuckAgent) -> None:
    """The Triton GDN decode kernel where this column runs vLLM's fused CUDA
    one. GDN is ~4.4 ms of a ~34 ms decode step here, so either kernel is
    capped at a few percent; ARC2's four L4s read +1.5% for Triton, under the
    3% bar that would keep it."""
    factory = agent.llm_config.backend_factory
    factory.env_overrides["VLLM_GDN_DECODE_KERNEL"] = "triton"  # type: ignore[union-attr]


def _siblings_all(agent: DuckAgent) -> None:
    """Park mode's sibling refresh as it was before 2026-09-29: a use of a
    boundary moves every stored sibling young, dead checkpoints included. The
    baseline against ``victim_offload.DEFAULT_SIBLING_REFRESH``."""
    factory = agent.llm_config.backend_factory
    factory.cpu_kv_offload_extra = {**(factory.cpu_kv_offload_extra or {}), "sibling_refresh": "all"}  # type: ignore[union-attr]


def _prefix_match_unit_64(agent: DuckAgent) -> None:
    """Hash prefix-cache keys every 64 tokens instead of once a block, so a hit
    can end inside a block wherever align mode kept a recurrent state.

    It must divide every cacheable group's block size, which bf16 makes 1,600.
    The value matters little -- the expected recompute it leaves is half the
    unit, so anything well under the block captures most of it -- and what
    sank it on the four L4s (-3.9%) was pool: the prompt-end state copies a
    partial hit resumes from, ~2.5 blocks a running request, which a coarser
    unit would not reclaim. Here recompute is ~20% of uncached input and the
    tier answers most pool misses, so the trade may land differently.
    """
    agent.llm_config.backend_factory.engine_args.prefix_match_unit = 64  # type: ignore[union-attr]


def _no_mtp(agent: DuckAgent) -> None:
    """No drafter, where the Flash-Next column drafts by default (ullm's
    engine, since the spec-decode branch)."""
    agent.llm_config.backend_factory.engine_args.speculative_config = None  # type: ignore[union-attr]


def _mtp(k: int) -> Callable[[DuckAgent], None]:
    """An arm drafting ``k`` tokens a step with the checkpoint's MTP head."""

    def arm(agent: DuckAgent) -> None:
        """Flash-Next's MTP head drafting, loaded from its own directory
        (``tools/checkpoints/extract_mtp_drafter.py``) so it does not re-read
        the target's shards. Its bf16 experts need their own MoE backend,
        since Marlin serves no bf16; its KV takes the target's cache dtype.
        Draft tokens are sampled and verified against their probabilities:
        vLLM's argmax default is verified as one-hot, which caps acceptance at
        the target's top-1 probability (``settings.md``)."""
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
    """Prefix-cache keys every 16 tokens: the coarsest unit that divides the
    block under the MTP drafter (1,616 = 16 x 101) as well as without it
    (1,600), so drafted and plain arms can share one unit."""
    agent.llm_config.backend_factory.engine_args.prefix_match_unit = 16  # type: ignore[union-attr]


def _indexer_fp8(agent: DuckAgent) -> None:
    """The QSA indexer's compressed-key cache in plain e4m3 for a column that
    keeps bf16 (upstream #54890, in the fork; the Flash-Next cells ship it,
    ``duck.presets._flashnext_kv_caches``): half the bytes a cached key, the
    drafter's QSA layer included. A score risk: unscaled fp8 keys change which
    tokens the indexer selects. Under INT4 the block follows
    (``_quantized_cache_block``)."""
    agent.llm_config.backend_factory.engine_args.attention_config.indexer_kv_dtype = "fp8"  # type: ignore[union-attr]
    _quantized_cache_block(agent)


def _batched_512(agent: DuckAgent) -> None:
    """At most 512 tokens a step (max_num_batched_tokens), the cell's largest
    captured CUDA graph (max_cudagraph_capture_size), so no step runs eagerly;
    the RTX Flash-Next cell ships it (``packages/ullm/flash_optimization.md``,
    "512 tokens a step")."""
    agent.llm_config.backend_factory.engine_args.max_num_batched_tokens = 512  # type: ignore[union-attr]


def _transient_kv_pool(agent: DuckAgent) -> None:
    """The QSA compressor rings in a block pool of their own (the fork's
    ``CacheConfig.transient_kv_pool``): a ring holds a few kilobytes but took
    a whole shared block per running request, one of the seven fixed blocks
    a request holds on the RTX."""
    agent.llm_config.backend_factory.engine_args.transient_kv_pool = True  # type: ignore[union-attr]


# A GDN layer's recurrent page on one rank, bytes, by drafted k, as the hybrid
# alignment sizes the block from it: the bf16 state, the conv window over the
# drafts and, with a drafter, RecoverSSM's float32 records per verified
# position -- the fork's state calculators at the checkpoint's GDN dimensions.
# A drafted arm without RecoverSSM holds less, so its block comes out a little
# larger than it needs and still holds the page. FP8 rows take
# FP8_STATE_SAVING less.
GDN_PAGE_BYTES = {0: 1_634_304, 1: 1_720_704, 2: 1_774_144, 3: 1_827_584, 4: 1_881_024}
FP8_STATE_SAVING = 761_856
# A token's bytes in the attention group's block, as the fork's alignment counts
# them: an FP8 cache its own count (two KV heads, left as is); an INT4 one what
# the group stores (fork f3302381), the cell padded to the compressed-key
# cache's multiple plus that cache's row share, by whether the indexer cache is
# FP8.
FP8_KV_TOKEN_BYTES = 1_024
INT4_KV_TOKEN_BYTES = {True: 608, False: 640}


def _quantized_cache_block(agent: DuckAgent) -> None:
    """The block the options the engine args carry take: the smallest multiple
    of the cell's cache-hit unit, which the engine refuses a block it does not
    divide, that holds the recurrent page at a token's bytes; nothing when
    neither INT4 nor the FP8 state is on. The arms call it, and the launcher
    again once every knob (the drafted k and the unit among them) is set."""
    engine_args: Any = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    int4 = cast(str, engine_args.kv_cache_dtype) == "int4_per_token_head"
    fp8_state = cast(str, engine_args.mamba_ssm_cache_dtype) == "fp8_e4m3"
    if not (int4 or fp8_state):
        return
    fp8_indexer = cast(str, engine_args.attention_config.indexer_kv_dtype) == "fp8"
    speculative = cast("dict[str, Any] | None", engine_args.speculative_config)
    k: int = speculative["num_speculative_tokens"] if speculative else 0
    unit = cast("int | None", engine_args.prefix_match_unit)
    if k not in GDN_PAGE_BYTES or not unit:
        raise SystemExit(f"no block known for k={k} at cache-hit unit {unit}")
    page = GDN_PAGE_BYTES[k] - (FP8_STATE_SAVING if fp8_state else 0)
    token_bytes = INT4_KV_TOKEN_BYTES[fp8_indexer] if int4 else FP8_KV_TOKEN_BYTES
    tokens = -(-page // token_bytes)
    engine_args.block_size = -(-tokens // unit) * unit


def _int4_kv(agent: DuckAgent) -> None:
    """The QSA layers' main KV cache in INT4 for a column that keeps
    ``fp8_per_token_head`` (the fork's ``TUFA.md``; the Flash-Next cells ship
    it, ``duck.presets._flashnext_kv_caches``): Hadamard-rotated, a scale and
    zero point per token and head, two values a byte (what it buys:
    ``packages/ullm/flash_optimization.md``, "INT4 KV cache"). The drafter's
    QSA layer stores INT4 too
    (``drafter_fp8_kv`` keeps it FP8). The block follows
    (``_quantized_cache_block``)."""
    agent.llm_config.backend_factory.engine_args.kv_cache_dtype = "int4_per_token_head"  # type: ignore[union-attr]
    _quantized_cache_block(agent)


def _fp8_state(agent: DuckAgent) -> None:
    """The GDN recurrent state in FP8 rows where the column keeps bf16 (the
    fork's ``TUFA.md``): each row's e4m3 values and its float32
    scale, rounded stochastically; a request's recurrent blocks about 40%
    smaller. The block follows (``_quantized_cache_block``)."""
    agent.llm_config.backend_factory.engine_args.mamba_ssm_cache_dtype = "fp8_e4m3"  # type: ignore[union-attr]
    _quantized_cache_block(agent)


def _drafter_fp8_kv(agent: DuckAgent) -> None:
    """The MTP drafter's QSA cache in FP8 per token and head whatever the
    target's (the draft loader builds the drafter with it): beside
    ``int4_kv``, the drafter's acceptance back at FP8's level for about 6% of
    INT4's pool on the B200 (``packages/ullm/flash_optimization.md``, "INT4 KV
    cache")."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("drafter_fp8_kv needs a drafting cell or an mtp arm before it")
    speculative["kv_cache_dtype"] = "fp8_per_token_head"


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


def _index_share(agent: DuckAgent) -> None:
    """The drafter's MTP steps after the first reuse its QSA top-k indices
    (``speculative_config.index_share_for_mtp_iteration``, off in both
    checkpoints' configs; competitor-tricks T5): fewer indexer passes a
    draft, at whatever it costs acceptance."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("index_share needs a drafting cell")
    speculative["index_share_for_mtp_iteration"] = True


def _relaxed_accept(agent: DuckAgent) -> None:
    """A public notebook's relaxed draft acceptance at 0.3 (the fork's
    ``speculative_config.accept_threshold_single/_acc``, SGLang's
    ``--speculative-accept-threshold-*``; competitor-tricks T8): a draft token
    the target gives at least 0.3 is accepted outright, any other with
    probability min(1, p / (0.3 q)). A score risk: sampling sharpens toward
    the draft."""
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("relaxed_accept needs a drafting cell")
    speculative["accept_threshold_single"] = 0.3
    speculative["accept_threshold_acc"] = 0.3


def _autoround(agent: DuckAgent) -> None:
    """The target from Intel's AutoRound W4A16 experts, cut to e307's, with
    e307's FP8 PLE table (``FLASHNEXT_E307_AUTOROUND_CHECKPOINT``;
    competitor-tricks T1): INT4 experts on GPTQ Marlin through vLLM's ``inc``
    quantization, everything else the base model's bf16. The drafter stays
    e307's, fine-tuned on the NVFP4 target's hidden states. A score risk:
    other expert weights."""
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.model = ullm.backends.deployment.FLASHNEXT_E307_AUTOROUND_CHECKPOINT
    engine_args.quantization = "inc"


def _qad(agent: DuckAgent) -> None:
    """The target from local-inference-lab's quantization-aware distillation,
    cut to e307's experts, with e307's FP8 PLE table
    (``FLASHNEXT_E307_QAD_CHECKPOINT``; competitor-tricks T7): NVFP4 experts
    on the cell's Marlin, MXFP8 shared experts and attention, ModelOpt's
    mixed precision. The drafter stays e307's, fine-tuned on the PTQ target's
    hidden states. A score risk either way: other weights throughout."""
    engine_args = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    engine_args.model = ullm.backends.deployment.FLASHNEXT_E307_QAD_CHECKPOINT
    engine_args.quantization = "modelopt_mixed"


# The knob arms a run can take, each a change on top of the column. They
# compose: --arm is repeatable, and they are applied in the order given.
ARMS = {
    "kernel_stack": _kernel_stack,
    "no_fused_commit": _no_fused_commit,
    "baseline": _baseline,
    "bf16_state": _bf16_state,
    "expandable": _expandable,
    "indexer_64": _indexer_64,
    "b12x_a16": _b12x_a16,
    "b12x_fp4": _b12x_fp4,
    "marlin": _marlin,
    "eager": _eager,
    "gdn_prefill_flashinfer": _gdn_prefill_flashinfer,
    "gdn_decode_triton": _gdn_decode_triton,
    "siblings_all": _siblings_all,
    "prefix_match_unit_64": _prefix_match_unit_64,
    "prefix_match_unit_16": _prefix_match_unit_16,
    "no_mtp": _no_mtp,
    "mtp1": _mtp(1),
    "mtp2": _mtp(2),
    "mtp3": _mtp(3),
    "gdn_recoverssm": _gdn_recoverssm,
    "indexer_fp8": _indexer_fp8,
    "transient_kv_pool": _transient_kv_pool,
    "batched_512": _batched_512,
    "int4_kv": _int4_kv,
    "fp8_state": _fp8_state,
    "drafter_fp8_kv": _drafter_fp8_kv,
    "fp8_dense": _fp8_dense,
    "index_share": _index_share,
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
    samples so (2026-09-26). The preset's play profile carries
    top_k 194 (top_p and the presence penalty are off there too); its
    temperature (0.6) stays."""
    profile = _replayed_profile(agent)
    agent.llm_config.sampling_profiles = {
        **agent.llm_config.sampling_profiles,
        agent.profile: dataclasses.replace(
            profile, top_p=1.0, extra_params={"top_k": 0, "min_p": 0.0, "presence_penalty": 0.0}
        ),
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


def _profile_after_window(agent: TrafficDuck, window_s: float, with_stack: bool, record_shapes: bool) -> None:
    """Torch-profiler windows after the reading: the lanes play on for a tail
    the reading leaves out, and two windows open in it, by time rather than by
    pool. A window's close stalls the engine for up to a minute (68 s seen on
    the RTX, a minute under a B200 drafter), so they sit 90 s apart. The
    replay starts as the engine becomes ready, which is where not_before_s
    counts from. ``record_shapes`` gives every eagerly launched op its input
    shapes; a kernel replayed from a CUDA graph has none to record."""
    agent.llm_config.backend_factory.profile = EngineProfile(  # type: ignore[union-attr]
        kv_cache_above=0.0,
        not_before_s=window_s + 30.0,
        windows=1 if with_stack else 2,
        spacing_s=90.0,
        with_stack=with_stack,
        record_shapes=record_shapes,
    )
    agent.tail_s = 240.0


def _drafter_moe_backend(agent: DuckAgent, backend: str | None) -> None:
    """The mtp arm's head served on ``backend`` in place of the arm's own."""
    if backend is None:
        return
    speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
    if not speculative:
        raise SystemExit("--drafter-moe-backend needs an mtp arm")
    speculative["moe_backend"] = backend


def _serve_checkpoint(agent: DuckAgent, model: str) -> None:
    """Another checkpoint of the cell's architecture (an expert prune) on the
    B200 column, from its directory on shared storage. A drafter is only valid
    for its target, so a drafting cell names its head too (``--drafter``); the
    column's own head is the column's checkpoint's."""
    engine_args: Any = agent.llm_config.backend_factory.engine_args  # type: ignore[union-attr]
    speculative = cast("dict[str, Any] | None", engine_args.speculative_config)
    if speculative and str(speculative["model"]).startswith(cast(str, engine_args.model)):
        raise SystemExit("--model on a drafting cell needs --drafter, the head of that checkpoint")
    engine_args.model = model


def _seal_engine(agent: TrafficDuck) -> None:
    """The seal's B200 column in place of the duck's: the engine factory, the
    sampling and the slots ``seal.presets.configure`` builds, taken from a
    built seal rather than restated, so the replay serves what a seal run
    serves. The two columns differ in the host tier, the image processor's
    cache and the slots; every knob a replay sets afterwards lands on the
    seal's factory."""
    seal = SealAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    agent.llm_config.backend_factory = seal.llm_config.backend_factory
    agent.llm_config.sampling_profiles = seal.llm_config.sampling_profiles
    agent.run_config.max_concurrent = seal.run_config.max_concurrent


def _target(hours: float, node: str | None = None, mem_gib: float = 224.0) -> taaf.deploy_slurm.TufaSlurmTarget:
    """One B200 for ``hours`` of run; on ``node`` when named, so runs to be
    compared share a machine. ``mem_gib`` of host memory: the engine holds
    ~182 GiB it cannot give back (2026-10-11, 50-lane replays: the 96 GiB
    tier, the 64 GiB pinned PLE table, ~6 GiB of other shared memory and
    ~13 GiB of process memory); past that the checkpoint's page cache, which
    the kernel reclaims under the limit."""
    budget_s = int(hours * 3600)
    wall_s = budget_s + 30 * 60
    return taaf.deploy_slurm.TufaSlurmTarget(
        time=f"{wall_s // 3600:02d}:{wall_s % 3600 // 60:02d}:00",
        max_runtime_s=budget_s,
        main_project=platypus.deployment.detect_main_project(),
        nodelist=node,
        gpu_count=1,
        cpus_per_gpu=24,
        mem_per_gpu_bytes=int(mem_gib * 1024**3),
    )


def _kaggle_target(
    hours: float, slug: str = "tufa-arc3-traffic", run_id: str | None = None
) -> taaf.deploy_kaggle.KaggleTarget:
    """The RTX box, as ``notebooks/demo.ipynb`` deploys an agent there. The run
    ends when the replay does; the budget only bounds a replay that hangs.
    A kernel of its own (``slug``, with its source dataset ``<slug>-ds``) runs
    beside the default one instead of replacing its version. With ``run_id``
    the source dataset is the run's own (``<slug>-ds-<run_id>``): a kernel
    version attaches a dataset's current version, which can still be the
    previous run's while the new one goes live (TAAF's wait for it only
    warns), and a dataset of the run's own has no previous version to
    attach."""
    owner = platypus.deployment.kaggle_owner()
    return taaf.deploy_kaggle.KaggleTarget(
        username=owner,
        kernel_slug=slug,
        dataset_ref=f"{owner}/{slug}-ds" + (f"-{run_id}" if run_id else ""),
        run_as_submission=False,
        max_runtime_s=hours * 3600,
    )


def main() -> None:
    """Parse the job to launch and deploy it."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="job", required=True)
    record = sub.add_parser("record")
    record.add_argument("--lanes", type=int, default=120)  # twice the RTX column's slots
    record.add_argument("--passes", type=int, default=2)
    record.add_argument("--hours", type=float, default=2.0)
    replay = sub.add_parser("replay")
    replay.add_argument("--workload", required=True)
    # The column's own concurrency unless named: the B200 column's
    # max_concurrent as built, so the replay tracks gameplay rather than a
    # copied figure. Lanes past the workload's replay its lanes again, nonced.
    replay.add_argument("--lanes", type=int, default=None)
    replay.add_argument("--hours", type=float, default=1.5)
    # The column whose engine the replay serves: the duck's, or the seal's
    # (_seal_engine), whose slots are then the lanes unless named.
    replay.add_argument("--column", choices=("duck", "seal"), default="duck")
    kaggle = sub.add_parser("kaggle")
    kaggle.add_argument("--workload", required=True)
    kaggle.add_argument("--hours", type=float, default=9.0)
    # A knob that frees pool needs more lanes than the column's, or the pool
    # stops binding; the engine's sequence cap follows (derive_max_num_seqs).
    kaggle.add_argument("--max-concurrent", type=int, default=None)
    for p in (replay, kaggle):
        # A memory knob is a stability question, not a throughput one: the
        # RTX dies in play just past its cell's budget (flash_optimization.md,
        # "The utilization ceiling"); on the B200 a kernel whose CUDA graphs
        # take more than the cell's leaves its warm-up short of memory at a
        # high lane count.
        p.add_argument("--utilization", type=float, default=None)
        p.add_argument("--window-s", type=float, default=1500.0)  # ten minutes of warm-up, fifteen read
        p.add_argument("--settle-s", type=float, default=600.0)
    # The engine without its host tier, as a drafter needs it today.
    replay.add_argument("--no-tier", action="store_true")
    # The host tier at this size instead of the cell's (at k=3 on the B200 a
    # slot is 26,836,992 bytes: 30 GiB holds 1,200).
    replay.add_argument("--tier-gib", type=float, default=None)
    for p in (replay, kaggle):
        # Temperature alone (_plain_sampling), or the preset's play profile.
        p.add_argument("--sampling", choices=("plain", "preset"), default="plain")
        # Torch-profiler windows after the reading (_profile_after_window).
        p.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
        # One profiler window with Python stacks, to name the host calls
        # behind a stall; the traces are several times larger, so it is a
        # single window.
        p.add_argument("--profile-stack", action="store_true")
        # The profiler windows record each eager op's input shapes too.
        p.add_argument("--profile-shapes", action="store_true")
        # The fork's memory log every this many seconds, kept on through
        # serving (VLLM_MEMORY_LOG_SERVING): the card's free memory and
        # torch's peaks, in the engine's output.
        p.add_argument("--memory-log-s", type=float, default=None)
        # Every lane's first prompt prefilled before the window (TrafficDuck's
        # decode_gate): the window opens on parked conversations, so a short
        # settle does.
        p.add_argument("--decode-gate", action="store_true")
        # The decode gate, a minute of play and the profiler windows after it,
        # and no reading: a step's compute against another arm's, at the same
        # lanes.
        p.add_argument("--profile-only", action="store_true")
    for p in (record, replay, kaggle):
        # Cache hits every this many tokens; 0 leaves the cell's own
        # (``duck.presets``); a quantized cache's block follows it
        # (``_quantized_cache_block``).
        p.add_argument("--prefix-match-unit", type=int, default=0)
        # The drafting cell's own k (its drafter as it ships, only the count
        # changed); the prefix-match unit must divide the block that k gives.
        p.add_argument("--num-speculative-tokens", type=int, default=None)
    for p in (replay, kaggle):
        # Keep the prefix-cache block a drafter's lookup would drop (upstream
        # #53388): a continuation shares the token the drafter's cached row
        # depends on, so the drop only recomputes.
        p.add_argument("--no-eagle-block-drop", action="store_true")
    # The mtp arms' head on Kaggle, as its own dataset (OWNER/SLUG) mounted
    # beside the checkpoint's: the <model>-drafter rule has no directory to
    # find on a dataset mount, and the checkpoint's dataset is not rewritten.
    # A trained head is jeroencottaar/qwen3-8-flash-next-e307-mtp-v1.
    kaggle.add_argument("--drafter-dataset", default=None)
    # The kernel's slug: a new one runs beside the default kernel's current
    # version instead of pushing over it.
    kaggle.add_argument("--slug", default="arc3-traffic-duck")
    # The source dataset the run's own, <slug>-ds-<run id> (_kaggle_target).
    kaggle.add_argument("--run-id", default=None)
    # The KV pool capped at this many GiB instead of sized from the card's
    # utilization: squeezes the pool so a replay's lanes overflow into the
    # host tier and exercise its loads, at a throughput that is not the cell's.
    replay.add_argument("--kv-cache-gib", type=float, default=None)
    # The mtp arms' head from this directory instead of the checkpoint's own:
    # a trained head, a copy of the checkpoint's drafter directory with its
    # mtp.* tensors replaced (branch drafter-sft; its result is
    # arc_3_results/shared/drafter/heads/v1-final).
    replay.add_argument("--drafter", default=None)
    # Another checkpoint of the cell's architecture (an expert prune), as the
    # directory that holds it; with a drafting cell, --drafter names its head.
    replay.add_argument("--model", default=None)
    # Token ids every replayed call bans (logit bias -100), repeatable.
    replay.add_argument("--ban-token", type=int, action="append", default=None)
    # Each call's reply ids kept in traffic_calls.jsonl, to read the text.
    replay.add_argument("--record-outputs", action="store_true")
    # The head's MoE backend instead of the mtp arms' triton, which serves its
    # bf16 experts: marlin for NVFP4 ones (drafter-sft's drafter.nvfp4, as
    # heads/v1-nvfp4), as the target's are served.
    for p in (replay, kaggle):
        p.add_argument("--drafter-moe-backend", default=None)
    for p in (replay, kaggle):
        p.add_argument("--arm", choices=sorted(ARMS), action="append", default=None)
        # The host tier's park log (kv_park.log in the job directory), off by
        # default: every transfer with its cause, and the tier's contexts.
        p.add_argument("--park-log", action="store_true")
        # A lane's pause between calls, where the TrafficDuck's (none) would
        # not compare: 5 for a replay of the 09-19 or 09-20 workloads.
        p.add_argument("--request-gap-s", type=float, default=None)
    for p in (record, replay, kaggle):
        p.add_argument("--label", default="")
    for p in (record, replay):
        p.add_argument("--node", default=None)
        # Host memory for the job (_target says what the engine holds).
        p.add_argument("--mem-gib", type=float, default=224.0)
    # The fork's VLLM_KV_BLOCKS_TRACE, into kv_blocks_trace.jsonl in the job
    # directory: about once a second the pool's free blocks and each request's
    # real and null blocks in every KV cache group (who holds the pool).
    replay.add_argument("--kv-blocks-trace", action="store_true")
    # The fork's VLLM_BATCH_QUEUE_TRACE, into batch_queue_trace.jsonl in the
    # job directory: per engine step, the batch it scheduled and sent and the
    # one it waited on, and how long -- how far the host runs ahead of the GPU.
    replay.add_argument("--batch-queue-trace", action="store_true")
    for p in (replay, kaggle):
        # The backend matrix against the started engine before the replay
        # (llm_config.verify_backend): a run that checks a code change
        # first shows the engine still answers sensibly.
        p.add_argument("--verify-backend", action="store_true")
    args = parser.parse_args()
    if getattr(args, "profile_only", False):
        args.decode_gate = True
        args.profile = True
        args.window_s = 60.0
        args.settle_s = 0.0
    arms = getattr(args, "arm", None) or ["baseline"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.job == "record":
        agent = DuckAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
        _rtx_like(agent, args.lanes)
        agent.llm_config.traces = True
        agent.llm_config.trace_content = "ids"
        name = f"record_{args.lanes}lanes_{args.passes}x_{stamp}"
    elif args.job == "replay":
        agent = TrafficDuck.build(
            engine="qwen3.8-flash-next",
            environment="slurm-b200",
            workload=args.workload,
            window_s=args.window_s,
            settle_s=args.settle_s,
        )
        if args.column == "seal":
            _seal_engine(agent)
        lanes = args.lanes if args.lanes is not None else agent.run_config.max_concurrent
        _rtx_like(agent, lanes)
        if args.no_tier:
            agent.llm_config.backend_factory.cpu_kv_offload = False  # type: ignore[union-attr]
        if args.tier_gib is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = args.tier_gib  # type: ignore[union-attr]
        if args.kv_cache_gib is not None:
            agent.llm_config.backend_factory.engine_args.kv_cache_memory_bytes = int(args.kv_cache_gib * 2**30)  # type: ignore[union-attr]
        if args.utilization is not None:
            agent.llm_config.backend_factory.engine_args.gpu_memory_utilization = args.utilization  # type: ignore[union-attr]
        if args.prefix_match_unit:
            agent.llm_config.backend_factory.engine_args.prefix_match_unit = args.prefix_match_unit  # type: ignore[union-attr]
        if args.sampling == "plain":
            _plain_sampling(agent)
        if args.ban_token:
            _ban_tokens(agent, args.ban_token)
        agent.record_outputs = args.record_outputs
        if args.profile:
            _profile_after_window(agent, args.window_s, args.profile_stack, args.profile_shapes)
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
        if args.model:
            _serve_checkpoint(agent, args.model)
        _drafter_moe_backend(agent, args.drafter_moe_backend)
        name = (
            f"replay_{lanes}lanes_{'seal_' if args.column == 'seal' else ''}{'notier_' if args.no_tier else ''}"
            f"{f'tier{args.tier_gib:g}g_' if args.tier_gib is not None else ''}{'profile_' if args.profile else ''}"
            f"{'plain_' if args.sampling == 'plain' else ''}"
            f"{f'unit{args.prefix_match_unit}_' if args.prefix_match_unit else ''}"
            f"{'nodrop_' if args.no_eagle_block_drop else ''}"
            f"{f'kv{args.kv_cache_gib:g}g_' if args.kv_cache_gib is not None else ''}"
            f"{f'util{args.utilization:g}_' if args.utilization is not None else ''}"
            f"{'ban' + '-'.join(map(str, args.ban_token)) + '_' if args.ban_token else ''}"
            f"{'+'.join(arms)}{'_' + pathlib.Path(args.model).name if args.model else ''}"
            f"{'_' + pathlib.Path(args.drafter).name if args.drafter else ''}"
            f"{'_' + args.drafter_moe_backend if args.drafter_moe_backend else ''}_{stamp}"
        )
    else:
        agent = TrafficDuck.build(
            engine="qwen3.8-flash-next",
            environment="kaggle-rtx",
            workload=args.workload,
            window_s=args.window_s,
            settle_s=args.settle_s,
        )
        if args.sampling == "plain":
            _plain_sampling(agent)
        if args.profile:
            _profile_after_window(agent, args.window_s, args.profile_stack, args.profile_shapes)
        for arm in arms:
            ARMS[arm](agent)
        speculative = agent.llm_config.backend_factory.engine_args.speculative_config  # type: ignore[union-attr]
        # A drafting cell mounts its own head; an mtp arm's is a directory here.
        head = cast("dict[str, object]", speculative or {}).get("model")
        mounted = isinstance(head, str) and head.startswith("/kaggle/input/")
        if speculative and not args.drafter_dataset and not mounted:
            raise SystemExit("an mtp arm on Kaggle needs --drafter-dataset")
        if args.drafter_dataset:
            if not speculative:
                raise SystemExit("--drafter-dataset needs an mtp arm")
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
        if args.max_concurrent is not None:
            agent.run_config.max_concurrent = args.max_concurrent
        if args.utilization is not None:
            agent.llm_config.backend_factory.engine_args.gpu_memory_utilization = args.utilization  # type: ignore[union-attr]
        util = agent.llm_config.backend_factory.engine_args.gpu_memory_utilization  # type: ignore[union-attr]
        name = (
            f"kaggle_{agent.run_config.max_concurrent}lanes_util{util}_{'profile_' if args.profile else ''}"
            f"{'plain_' if args.sampling == 'plain' else ''}{'nodrop_' if args.no_eagle_block_drop else ''}"
            f"{'+'.join(arms)}{'_' + args.drafter_dataset.split('/')[-1] if args.drafter_dataset else ''}"
            f"{'_' + args.drafter_moe_backend if args.drafter_moe_backend else ''}_{stamp}"
        )
    if getattr(args, "park_log", False):
        factory = agent.llm_config.backend_factory
        factory.cpu_kv_offload_extra = {**(factory.cpu_kv_offload_extra or {}), "park_log": True}  # type: ignore[union-attr]
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
    # The quantized caches' block, for the k now set.
    _quantized_cache_block(agent)
    if getattr(args, "decode_gate", False):
        agent.decode_gate = True  # type: ignore[attr-defined]
        name = f"{name}_{'profileonly' if args.profile_only else 'gate'}"
    if getattr(args, "verify_backend", False):
        agent.llm_config.verify_backend = True
        name = f"{name}_verified"
    if args.label:
        name = f"{name}_{args.label}"
    if getattr(args, "kv_blocks_trace", False):
        factory = agent.llm_config.backend_factory
        factory.env_overrides["VLLM_KV_BLOCKS_TRACE"] = str(RESULTS / name / "kv_blocks_trace.jsonl")  # type: ignore[union-attr]
    if getattr(args, "batch_queue_trace", False):
        factory = agent.llm_config.backend_factory
        factory.env_overrides["VLLM_BATCH_QUEUE_TRACE"] = str(RESULTS / name / "batch_queue_trace.jsonl")  # type: ignore[union-attr]
    if args.job != "record" and args.memory_log_s is not None:
        factory = agent.llm_config.backend_factory
        factory.memory_log_s = args.memory_log_s  # type: ignore[union-attr]
        factory.env_overrides["VLLM_MEMORY_LOG_SERVING"] = "1"  # type: ignore[union-attr]
    agent.prepare_for_backend()
    solver = LLMAgentSolver(agent=agent)
    benchmark = taaf.standard_benchmarks.make_benchmark_kaggle_official_110(solver=solver)
    if args.job == "record":
        benchmark.n_passes = args.passes
    else:
        # A replay plays no game, so one stands in for the benchmark TAAF runs.
        benchmark.games = benchmark.games[:1]
        benchmark.n_passes = 1
    # A Kaggle deploy only stages its bundle here, so it goes to local temp
    # (as in the demo); the run's output is pulled from the kernel.
    job_dir = (pathlib.Path(tempfile.gettempdir()) / "traffic" if args.job == "kaggle" else RESULTS) / name
    benchmark.label = f"traffic_{name}"
    benchmark.job_dir = job_dir
    print(
        f"{args.job}: {len(benchmark.games)} games x {benchmark.n_passes} pass(es), "
        f"max_concurrent={agent.run_config.max_concurrent}\njob_dir={job_dir}",
        flush=True,
    )
    target = (
        _kaggle_target(args.hours, args.slug, args.run_id)
        if args.job == "kaggle"
        else _target(args.hours, args.node, args.mem_gib)
    )
    handle = asyncio.run(benchmark.deploy(target))
    print("submitted:", getattr(handle, "job_id", None) or getattr(handle, "kernel_id", handle), flush=True)


if __name__ == "__main__":
    main()
