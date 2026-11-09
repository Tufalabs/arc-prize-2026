"""Serving one engine in one environment, as two coordinates.

A served model's configuration splits three ways, and keeping the three apart
is what lets one of them change without disturbing the others:

- **engine** — facts about a checkpoint: its numerics, its parsers, its
  template. True wherever it runs.
- **environment** — facts about the box and the deploy target: how many GPUs,
  where weights come from, whether the hub is reachable.
- the **pair** — what only the two together decide, memory above all.

:func:`build` walks those in that order, over a small layer of corrections to
vLLM's own defaults, and hands back everything it takes to serve the model
here. The pair's block is deliberately one branch per cell:
``gpu_memory_utilization`` is measured, not derived, so a cell nobody has
served raises rather than borrowing a neighbour's number.

Nothing here is task-specific -- no window, no concurrency, no sampling. Those
belong to whatever is playing the games, which is why a consumer that is not
the ARC3 harness can use this module on its own.

The full placement rule, and every parameter placed under it, is
``settings.md`` at the repo root of arc3-platypus.
"""

import dataclasses
from typing import Any, Literal

import ullm.backend
import ullm.backends.config_local
import ullm.backends.oai
import ullm.backends.vllm_factory
import ullm.backends.vllmx
import ullm.factory
import ullm.kaggle
import ullm.support as uls
import vllm
import vllm.config
import vllm.envs

#: Every checkpoint this module can serve.
ENGINES = ("qwen3.8-27b", "qwen3.6-27b", "qwen3.8-flash-next")

#: Every box it can serve them on. Which of these a given task actually runs
#: on is that task's own business to say.
ENVIRONMENTS = ("slurm-b200", "kaggle-rtx", "kaggle-4xl4")

# The share of the KV pool the engine keeps free when admitting a waiting
# request. vLLM's own value is zero: admission fills the pool to the last
# block, every later decode step that needs one preempts, in storms, and a
# live set larger than the pool makes that the steady state. With a reserve
# the engine holds admissions near the boundary, running requests grow into
# the reserve, and the rest queue with their contexts in the host tier --
# which is the regime the tier is for (kv_offload.md, "Preemption is the zero
# watermark"). It binds only while the pool is at the line; under the tier's
# park mode a smaller reserve preempts in bursts (kv_offload.md, "Park mode").
ENGINE_ADMISSION_WATERMARK = 0.10

# vLLM reads this only at engine init, so it has to be in the environment
# before the worker spawns rather than set on the args. The V2 runner is
# vLLM's recommended path and the only one that implements the
# recurrent-state blank, so nothing here runs without it.
UNIVERSAL_ENGINE_ENV: dict[str, str] = {"VLLM_USE_V2_MODEL_RUNNER": "1"}


# The Flash-Next throughput stack with KN (dense GDN qkvz in [K, N] layout),
# in three parts by where each can run. Whole, it is served by the Kaggle RTX
# Flash-Next cell and the traffic arm `kernel_stack`.
#
# Triton, tensor-view and cuBLAS changes that take each tensor rank's shard
# and run the sampling and drafting on the last pipeline stage, where the
# engine puts them: any card, any topology. The four-L4 cell serves this part.
FLASHNEXT_THROUGHPUT_PORTABLE_ENV: dict[str, str] = {
    "XKB_SAMPLER_FOLD": "2",
    "XKB_HC_FUSED": "1",
    "VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD": "1024",
    "XKB_GDN_NOCOPY": "1",
    "XKB_SPARSE_SAMPLER": "1",
    "XKB_DRAFT_PRUNE": "2",
    "XKB_DENSE_KN": "1",
    "XKB_GDN_MIX": "2",
}
# RecoverSSM's state commit folded into the next step's verify: the engine
# refuses it under pipeline stages.
FLASHNEXT_THROUGHPUT_FUSED_COMMIT_ENV: dict[str, str] = {"GDN_FUSED_COMMIT": "1"}
# CUDA images built for capability 12.0 alone (the MoE, the router, the
# hyper-connections, the L2 prefetch of the expert weights, and XKB_XGDN, the
# fused commit's verify): any other card keeps the engine's kernels.
FLASHNEXT_THROUGHPUT_SM120_ENV: dict[str, str] = {
    "XKB_XROUTER": "1",
    "XKB_XMOE": "1",
    "XKB_XHC": "1",
    "XKB_L2PF": "1",
    "XKB_XGDN": "1",
    "XKB_XHC_ROWS": "1-32,257-288",
    "XKB_XHC_S": "32:24,288:16",
    "XKB_XROUTER_M": "1-1024",
    "XKB_XROUTER_S": "128:8,384:5,1024:4",
    "XKB_XHC_UP": "1-32,193-288",
    "XKB_L2PF_MB": "32",
    "XKB_L2PF_CTAS": "16",
    "XKB_L2PF_NS": "25",
    "XKB_L2PF_WARPS": "4",
    "XKB_L2PF_MINROWS": "64",
    "XKB_L2PF_MAXROWS": "512",
}
FLASHNEXT_THROUGHPUT_ENV: dict[str, str] = {
    **FLASHNEXT_THROUGHPUT_PORTABLE_ENV,
    **FLASHNEXT_THROUGHPUT_FUSED_COMMIT_ENV,
    **FLASHNEXT_THROUGHPUT_SM120_ENV,
}


@dataclasses.dataclass
class Deployment(uls.BaseClass):
    """What it takes to serve one engine in one environment.

    Three members going three places: the factory this process starts, and the
    two deploy assets a submission's kernel needs, which belong to whatever
    owns the run rather than to the engine.
    """

    factory: ullm.factory.BackendFactory  # how this process reaches the model
    kaggle_engine: ullm.kaggle.KaggleEngine | None = None  # the wheelhouse a submission installs
    kaggle_dataset_sources: tuple[str, ...] = ()  # datasets the kernel mounts to have the weights
    kaggle_model_sources: tuple[str, ...] = ()  # models the kernel mounts to have the weights


def _qwen_backend_kwargs(model_family: str) -> dict[str, Any]:
    """Return Qwen backend defaults without configuring or constructing an engine."""
    if model_family not in {"qwen36", "qwen38"}:
        raise ValueError(f"Unsupported Qwen backend family: {model_family!r}")
    # Each caller owns its nested template options, including remote engines.
    kwargs: dict[str, Any] = {
        "tool_call_parser": "qwen3_coder",
        "reasoning_parser": "qwen3",
        "chat_template_kwargs": {"preserve_thinking": True},
    }
    if model_family == "qwen38":
        kwargs["native_reasoning_effort"] = True
    return kwargs


def _qwen36_config(model: str) -> ullm.backends.config_local.ModelConfig:
    """Qwen3.6 family (HuggingFace chat template, qwen3_coder tool parser).

    Reasoning is split via vLLM's `qwen3` reasoning parser (shared with
    Qwen3.5; the parser handles both template styles).

    `model` accepts a bare size tag (e.g. "35B-A3B") or a full HF repo id /
    local path. Bare tags resolve to "Qwen/Qwen3.6-<tag>"."""
    ullm.backends.config_local.set_env()

    if "/" in model:
        model_path = model
    else:
        model_path = f"Qwen/Qwen3.6-{model}"

    # Native context cap from the HF config's text_config.max_position_embeddings
    # (= 262144). Higher would require explicit rope_scaling / YARN config.
    max_model_len = 262144
    engine_args = vllm.AsyncEngineArgs(
        model=model_path,
        dtype="auto",
        max_model_len=max_model_len,
        kv_cache_dtype="fp8_e4m3",
        # Engine-side reasoning config (mirrors the backend reasoning_parser):
        # populates reasoning_config so SamplingParams.thinking_token_budget is enforced.
        reasoning_parser="qwen3",
        # 35B-A3B's Mamba-cache-align block_size is 2096, > vLLM's default
        # max_num_batched_tokens (2048), which trips a sizing assert at engine
        # init. Bumping to 4096 leaves headroom without quadrupling activation
        # memory. Same constraint doesn't bite the 27B (smaller block_size).
        max_num_batched_tokens=4096,
        # A checkpoint whose tokenizer_config.json carries a `max_length` (an
        # HF Trainer artifact, e.g. vrfai/Qwen3.6-27B-FP8) makes it the default
        # for every processor call: the expanded <|image_pad|> run is cut and
        # `_check_special_mm_tokens` rejects the request. Explicit per-call
        # options override the tokenizer's own defaults, and these travel with
        # the engine args, so they hold for the in-process engine and for the
        # OpenAI server rebuilt from the same fields.
        mm_processor_kwargs={"truncation": False, "max_length": None},
        # NOTE: SimpleCPUOffloadConnector is incompatible with this model's
        # hybrid (full attention + gated-delta-network) architecture in
        # vLLM 0.19 — `_mamba_block_aligned_split` asserts
        # num_external_computed_tokens == 0 with a "External KV connector
        # is not verified yet" message, crashing the engine the moment a
        # CPU-cached prefix returns a hit. Re-evaluate when vLLM lifts that.
    )
    # Qwen3.6 model-card values for "general tasks" (thinking mode):
    # temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5.
    # (The "precise coding" preset lowers temperature to 0.6 and
    # presence_penalty to 0.0 — use that for narrow code-completion work.)
    sampling_params = ullm.backend.GenericSamplingParameters(
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_model_len,
        extra_params={"top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    )
    return ullm.backends.config_local.ModelConfig(
        engine_args=engine_args,
        sampling_params=sampling_params,
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs=_qwen_backend_kwargs("qwen36"),
    )


QWEN38_OFFICIAL_REPO = "Qwen/Qwen3.8-27B-FP8"
QWEN38_OFFICIAL_REVISION = "017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
FLASHNEXT_SOURCE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
FLASHNEXT_SOURCE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
#: The checkpoint the Flash-Next engine serves on the cluster: the source
#: above with 60% of each layer's experts kept, cut by
#: tools/checkpoints/prune_flashnext_experts.py. Its ``pruning.json`` names the
#: source revision, the plan and the rule it was cut by. The Kaggle model
#: below holds this directory's files, so both read the same data.
FLASHNEXT_E307_CHECKPOINT = "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform"
#: The same checkpoint as a Kaggle model, one version of it, since a kernel
#: has no cluster storage and no internet to fetch 100 GB over. A model rather
#: than a dataset (2026-10-24): a dataset can mount over GCSFUSE,
#: which stalls the load for hours, and a model's mount is expected to load
#: better. A model version is immutable, so the mount needs no revision pin.
FLASHNEXT_E307_KAGGLE_MODEL = "jeroencottaar/qwen3-8-flash-next-nvfp4-e307/Transformers/default/1"
#: The drafter the Flash-Next engine drafts with: the e307 checkpoint's MTP
#: head fine-tuned on ARC2 and ARC3 play (arc3-platypus ``drafter-sft``, head
#: ``v1``), its routed experts quantized to NVFP4 as the target's are, and its
#: LM head cut to the 16,384 ids generated most in that play
#: (``tools/checkpoints/reduce_drafter_vocab.py``; ``d2t`` maps them back).
#: Valid for the e307 target only.
FLASHNEXT_E307_DRAFTER = (
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-e307-normcond-uniform-drafter-v1-nvfp4-dv16k"
)
#: The same drafter as a Kaggle model, mounted beside the checkpoint's: the
#: ``<model>-drafter`` rule has no directory to find on a model's mount.
FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL = "jeroencottaar/qwen3-8-flash-next-e307-mtp-v1-nvfp4-dv16k/Transformers/default/1"
#: A sweep option, never served by default (competitor-tricks T1): e307's
#: experts (its ``pruning.json`` plan, applied by expert id) from Intel's
#: AutoRound W4A16 quantization of the base model -- INT4, group 128, tuned
#: rounding; vLLM reads the format as its ``inc`` quantization -- with
#: e307's FP8 PLE table in place of Intel's BF16 one, twice the size
#: (``tools/checkpoints/graft_ple_table.py``). Its ``pruning.json`` and
#: ``graft.json`` name both sources.
FLASHNEXT_E307_AUTOROUND_CHECKPOINT = (
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-W4A16-AutoRound-e307-normcond-uniform-fp8ple"
)
#: A sweep option, never served by default (competitor-tricks T7): e307's
#: experts from local-inference-lab's quantization-aware distillation of the
#: base model (NVFP4 experts, MXFP8 shared experts and attention; ModelOpt's
#: ``MIXED_PRECISION``, vLLM's ``modelopt_mixed``), with e307's FP8 PLE table
#: in place of its NVFP4 one, which no engine of ours reads. Its NVFP4 table
#: is the base table rounded to NVFP4 in 99.5-99.7% of its values, so the FP8
#: table is the same table at finer precision; everything else distillation
#: trained is the release's own.
FLASHNEXT_E307_QAD_CHECKPOINT = (
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-QAD-e307-normcond-uniform-fp8ple"
)
#: The fork's switch that puts FlashInfer's CuTe DSL NVFP4 MoE in its W4A16
#: mode, bf16 activations rather than quantized ones; the B200 cell sets it.
CUTEDSL_W4A16_ENV = "VLLM_CUTEDSL_MOE_FP4_FORCE_A16"


def _qwen38_config(model: str) -> ullm.backends.config_local.ModelConfig:
    """Qwen3.8 family (dense; Qwen3.5 architecture, qwen3 parsers).

    ``model`` with a ``/`` is an exact path or repo id; anything else is a
    size suffix under the official org (``Qwen/Qwen3.8-{model}``).

    Differences from :func:`_qwen36_config`, each deliberate:

    - No ``max_num_batched_tokens=4096`` — that cap exists for the 35B-A3B's
      mamba block size, which no dense 27B has.
    - No ``mm_processor_kwargs`` truncation override — that works around a
      Trainer artifact in the ``vrfai`` 3.6 checkpoint's tokenizer config,
      which the official upload does not carry. Re-add if a boot smoke says
      otherwise.
    - The official 27B repository and tokenizer use one immutable revision;
      path-shaped custom models remain caller-owned and unpinned.
    - ``native_reasoning_effort`` is on: this family's chat template reads
      ``reasoning_effort`` (low/medium/xhigh) as a kwarg, so the
      ``SystemPrompt`` setting reaches the engine — the first local family
      where it does anything at all.
    """
    ullm.backends.config_local.set_env()
    model_path = model if "/" in model else f"Qwen/Qwen3.8-{model}"
    revision = QWEN38_OFFICIAL_REVISION if model_path == QWEN38_OFFICIAL_REPO else None
    max_model_len = 262144  # native; YaRN beyond it is not configured here
    engine_args = vllm.AsyncEngineArgs(
        model=model_path,
        revision=revision,
        tokenizer_revision=revision,
        dtype="auto",
        max_model_len=max_model_len,
        kv_cache_dtype="fp8_e4m3",
        reasoning_parser="qwen3",
    )
    # Qwen3.8 model-card values for thinking mode: temperature=1.0,
    # top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
    # repetition_penalty=1.0. The non-thinking preset differs sharply
    # (0.7 / 0.80 / presence 1.5) and is not offered here.
    # repetition_penalty is vLLM's default, stated for card fidelity.
    sampling_params = ullm.backend.GenericSamplingParameters(
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_model_len,
        extra_params={"top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    )
    return ullm.backends.config_local.ModelConfig(
        engine_args=engine_args,
        sampling_params=sampling_params,
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs=_qwen_backend_kwargs("qwen38"),
    )


def _flashnext_config(model: str) -> ullm.backends.config_local.ModelConfig:
    """Qwen3.8-Flash-Next NVFP4: a qwen4_exp MoE (512 experts, top-10) that
    ships Qwen3.8's chat template and card sampling byte for byte, so the
    client wiring is :func:`_qwen38_config`'s. What differs is the engine, and
    each difference is the checkpoint's:

    - ``quantization="modelopt_fp4"`` with ``dtype="bfloat16"``: NVFP4 body,
      the PLE n-gram table stored FP8 and excluded from body quantization.
      The fork's loader selection is what reads that declaration.
    - ``kv_cache_dtype="fp8_per_token_head"``: the sparse-attention layers'
      K/V in e4m3 with a scale per token and head, computed as each row is
      written, because the checkpoint carries no calibrated KV scales. The
      indexer's own caches stay bf16. It roughly doubles the pool in tokens,
      and doubles the block the hybrid alignment picks with it
      (``flash_optimization.md``, FP8 KV cache).
    - ``mamba_ssm_cache_dtype="bfloat16"``: the recurrent state in bf16,
      where the checkpoint's float32 is the engine's default. The state sets
      the page every cache group shares, so halving it halves the attention
      block (3,136 to 1,600 tokens on both Kaggle cards) and with it the
      fixed pool a running request holds: +36% generated on the RTX's
      traffic replay, +24% on the four L4s'. It is rounded at every token
      and carried the whole context; its score held in ARC2's sweep.
    - ``mamba_cache_mode="align"``: the one recurrent cache mode this model
      supports beside prefix caching.
    - The input embedding in pinned host memory (``cpu_offload_params``
      naming ``embed_tokens``, competitor-tricks T3): the second-largest
      table on the card (248,320 rows of 2,560 bf16, 1.18 GiB), read like the
      PLE table, one five-kilobyte row a token. The fork's Qwen4Exp method
      (a public notebook's) allocates it pinned and gathers rows through UVA; with
      no ``cpu_offload_gb`` budget, naming it selects that method and builds
      none of vLLM's generic offloader. The MTP drafter shares the target's
      module, so it reads the same table. On every box: the card's 1.18 GiB
      goes to the KV pool, and host memory holds it wherever the engine runs.
    - No MoE backend: a backend's support check is a capability family, so
      the pin is the pair's and each cell names its own (``settings.md``,
      U-EV). Never "auto", which walks a priority list and logs its
      rejections at debug, so a demotion reads like a choice.
    - Drafting with its MTP head (``speculative_config``), four tokens a
      step (2026-09-28): the head is fine-tuned and cut to a 16k-id
      draft vocabulary (``FLASHNEXT_E307_DRAFTER``), its experts NVFP4, so it
      runs on the cell's MoE pin with the target's quantization. Drafts are
      sampled and verified against their probabilities: vLLM's argmax
      default is verified as one-hot, which caps acceptance at the target's
      top-1 probability. GDN RecoverSSM verifies from the step's starting
      state and recovers the accepted one, so no recurrent group holds a
      state block per draft token (+17.6% pool on the B200); the prefix
      block a drafter's lookup would drop is kept, since a continuation
      shares the token its cached row depends on (upstream #53388). The
      head's path is the cell's, as the checkpoint's is.
      ``flash_optimization.md`` ("Speculative decoding") holds the
      measurements.
    - Conversations kept live between turns (``kv_offload.md``, park mode):
      prefix-cache keys every 80 tokens (``prefix_match_unit``), a decode
      tail at the last of them a reply passed
      (``prefix_cache_decode_tail``), and each conversation's thread named
      by the client (``prefix_cache_context_threads``), which is what puts
      the host tier in park mode. The unit must divide the attention block,
      which the drafter's width sets: 1,840 tokens at four drafts (every
      cell here, the four L4s' two-rank stages included), 1,792 at three (the
      duck's RTX cell, which sets 64). 64 does not divide the first, and the
      engine refuses to start on a unit that does not divide it.

    ``model`` is the source repository or a checkpoint directory derived from
    it; only the repository carries a revision pin.
    """
    ullm.backends.config_local.set_env()
    revision = FLASHNEXT_SOURCE_REVISION if model == FLASHNEXT_SOURCE_REPO else None
    max_model_len = 262144  # native; YaRN beyond it is not configured here
    engine_args = vllm.AsyncEngineArgs(
        model=model,
        revision=revision,
        tokenizer_revision=revision,
        quantization="modelopt_fp4",
        dtype="bfloat16",
        max_model_len=max_model_len,
        kv_cache_dtype="fp8_per_token_head",
        mamba_ssm_cache_dtype="bfloat16",
        mamba_cache_mode="align",
        cpu_offload_params={"embed_tokens"},
        speculative_config={
            "method": "mtp",
            "num_speculative_tokens": 4,
            "quantization": "modelopt_fp4",
            "draft_sample_method": "probabilistic",
            "gdn_recoverssm": True,
            "disable_eagle_block_drop": True,
        },
        reasoning_parser="qwen3",
        prefix_match_unit=80,
        prefix_cache_decode_tail=True,
        prefix_cache_context_threads=True,
    )
    # Qwen3.8's card values, which this checkpoint's card repeats.
    sampling_params = ullm.backend.GenericSamplingParameters(
        temperature=1.0,
        top_p=0.95,
        max_tokens=max_model_len,
        extra_params={"top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0},
    )
    return ullm.backends.config_local.ModelConfig(
        engine_args=engine_args,
        sampling_params=sampling_params,
        backend_cls=ullm.backends.vllmx.VLLM,
        backend_kwargs=_qwen_backend_kwargs("qwen38"),
    )


def _draft_from(engine_args: vllm.AsyncEngineArgs, drafter: str) -> None:
    """Where this cell holds the engine's drafter; how it drafts is the
    engine's (``speculative_config``)."""
    if not engine_args.speculative_config:
        raise ValueError("the engine does not draft")
    engine_args.speculative_config = {**engine_args.speculative_config, "model": drafter}


def _capture_sizes(engine_args: vllm.AsyncEngineArgs, sizes: tuple[int, ...]) -> None:
    """Name the engine's CUDA graph ladder (``cudagraph_capture_sizes``),
    which the factory then leaves alone (``dense_cuda_capture``). One list
    feeds every graph set: a rung is a PIECEWISE graph of that many tokens,
    a FULL decode graph rounded up to the verify width, and a draft-step
    graph of that many requests."""
    compilation = engine_args.compilation_config
    if not isinstance(compilation, vllm.config.CompilationConfig):
        compilation = vllm.config.CompilationConfig(**(compilation or {}))
        engine_args.compilation_config = compilation
    compilation.cudagraph_capture_sizes = list(sizes)


def offload_ple_table(engine_args: vllm.AsyncEngineArgs) -> None:
    """Serve Flash-Next's PLE n-gram table from pinned host memory.

    The table is the largest thing on the card and is read by one gather per
    token, a few kilobytes, so it is the part of this model a PCIe link can
    carry. vLLM's Engram offload (#54371) creates the table in pinned host
    memory and looks its rows up through UVA on a side stream, started while
    the layer before runs; no budget or name selector, since the setting
    names the table itself. Host memory then holds the table beside whatever
    else the run claims there, and loading it there takes minutes longer
    than loading it onto the card.
    """
    engine_args.engram_config = vllm.config.EngramConfig(cpu_offload=True)


def _one_graph_capture(engine_env: dict[str, str]) -> None:
    """Capture the CUDA graphs once (competitor-tricks S2).

    By default the memory profiler captures a sample of FULL graphs itself to
    estimate what the real capture takes and holds that back from the KV
    pool: a second capture, ~124 s of a B200 boot. Off, the real capture
    comes out of the headroom outside the budget, so the cell lowers its
    budget by what the estimate held, which the boot line with the estimate
    on names ("is equivalent to --gpu-memory-utilization=..."): the same
    pool and the same margins.
    """
    engine_env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "0"


def drop_vision_tower(engine_args: vllm.AsyncEngineArgs) -> None:
    """Serve a multimodal checkpoint as the language model alone.

    Two switches, because neither implies the other. The first refuses every
    modality, which is also what stops the tower being built and its weights
    being read: the loader is handed a missing-layer marker for it and drops
    those tensors at load, so the card never holds them. The second releases
    the cache that holds processed pixels between the front end and the
    engine, sized in whole gigabytes of shared memory and claimed whether or
    not an image ever arrives.

    The third is belt and braces against the order the other two take effect
    in: with no modality to profile the memory profiler already skips the
    encoder, but a profiler that did reach for it would now find nothing
    built, which fails a boot rather than costing it memory.

    For a caller that has decided its task sends no images. The capability is
    real -- the checkpoint carries the tower, and a column that wants it back
    simply does not call this.
    """
    engine_args.language_model_only = True
    engine_args.mm_processor_cache_gb = 0
    engine_args.skip_mm_profiling = True


def backend_kwargs(engine: str) -> dict[str, Any]:
    """The client wiring for one engine: its parsers and template options.

    Public because the two hand-built backends below are not the only way to
    reach a model, and a caller assembling their own should not have to
    reproduce a checkpoint's parser names.
    """
    if engine == "qwen3.8-27b":
        return _qwen_backend_kwargs("qwen38")
    elif engine == "qwen3.6-27b":
        return _qwen_backend_kwargs("qwen36")
    elif engine == "qwen3.8-flash-next":
        return _qwen_backend_kwargs("qwen38")  # the same template, byte for byte
    else:
        raise ValueError(f"unknown engine {engine!r}; this module serves {list(ENGINES)}.")


def client_kwargs(engine: str) -> dict[str, Any]:
    """``OpenAIServer`` keywords for reaching this engine over HTTP.

    Only ``chat_template_kwargs`` is the engine's; the rest are facts about
    talking to a vLLM server at all. ``reasoning_history_field`` because the
    server runs a reasoning parser, so prior thinking comes back in a separate
    field rather than in content; ``token_limit_param`` because current vLLM
    rejects the legacy ``max_tokens`` wire name; ``streaming`` off because the
    SDK's accumulator drops the ``return_token_ids`` extras that tracing and
    token continuation read.
    """
    return {
        "parse_think_tags": True,
        "reasoning_history_field": "reasoning",
        "token_limit_param": "max_completion_tokens",
        "chat_template_kwargs": {
            **backend_kwargs(engine).get("chat_template_kwargs", {}),
            "enable_thinking": True,
        },
        "streaming": False,
    }


#: The one cell that serves a checkpoint the caller names (``build``'s
#: ``checkpoint``): one on shared storage, as that cell's own checkpoint is.
#: A Kaggle cell's checkpoint is a Kaggle model (``serve_kaggle_checkpoint``).
CHECKPOINT_CELLS = (("qwen3.8-flash-next", "slurm-b200"),)


def build(
    engine: str,
    environment: str,
    *,
    serving: Literal["direct", "server"] = "direct",
    checkpoint: str | None = None,
) -> Deployment:
    """Everything it takes to serve ``engine`` in ``environment``.

    The checkpoint's own facts come first, because they are what builds the
    argument object at all; the corrections to vLLM's wrong-for-everyone
    defaults are written over them, then the box's facts, then the pair's.
    Writing the corrections over rather than under is the one place a more
    specific bucket does not win (``settings.md``, "Layering").
    ``serving`` is the caller's choice of topology over one set of engine args
    -- an in-process engine, or vLLM's HTTP server in this process.

    ``checkpoint`` serves that directory in place of the cell's own
    checkpoint, on the cells in ``CHECKPOINT_CELLS``. The argument object is
    built around it from the start rather than around the Hub repository it
    was cut from: under ``HF_HUB_OFFLINE`` vLLM resolves the model the object
    is built with to a cached snapshot, which an offline launcher without
    that repository's cache does not have, and a local directory needs none.
    Every other fact of the cell is the same.
    """
    if checkpoint is not None and (engine, environment) not in CHECKPOINT_CELLS:
        raise ValueError(
            f"checkpoint names a directory to serve on {list(CHECKPOINT_CELLS)}, not on "
            f"({engine!r}, {environment!r}); a Kaggle cell serves a Kaggle model (serve_kaggle_checkpoint)."
        )
    # --- the engine: a checkpoint's own facts -------------------------
    if engine == "qwen3.8-27b":
        model = QWEN38_OFFICIAL_REPO
        config = _qwen38_config(model)
    elif engine == "qwen3.6-27b":
        model = "vrfai/Qwen3.6-27B-FP8"
        config = _qwen36_config(model)
    elif engine == "qwen3.8-flash-next":
        config = _flashnext_config(checkpoint or FLASHNEXT_SOURCE_REPO)
    else:
        raise ValueError(f"unknown engine {engine!r}; this module serves {list(ENGINES)}.")

    # --- neither coordinate: vLLM defaults wrong for everyone ---------
    # Over the checkpoint's arguments rather than under them: each of these is
    # wrong in vLLM rather than unstated by a checkpoint, so a builder setting
    # one would be restating a vLLM default.
    #
    # Take vLLM's own generation defaults rather than the ones shipped in a
    # checkpoint.
    config.engine_args.generation_config = "vllm"
    # CUDA graphs are worth their capture time on every engine we run long
    # enough to serve games, and prefix caching is what makes a block
    # offloadable at all -- a block is only spillable once full and hashed.
    config.engine_args.enforce_eager = False
    config.engine_args.enable_prefix_caching = True
    # No image processor cache. Every request resends each image of its
    # context, and the shared-memory cache evicts oldest first, a hit not
    # refreshing an image: once the images in use outgrow it, each miss
    # evicts another still in use and the hit rate falls and stays down,
    # while the cache holds shared memory against the job's cap. The
    # renderer's workers (the environment's, below) process every image of
    # every request instead.
    config.engine_args.mm_processor_cache_gb = 0
    # For a cell that turns the cache back on: one ref-counted multimodal
    # cache instead of the default per-process 'lru' mirror, whose split
    # brain evicts an item the EngineCore later expects and crashes the
    # request (vllm#31404).
    config.engine_args.mm_processor_cache_type = "shm"
    config.engine_args.watermark = ENGINE_ADMISSION_WATERMARK
    # Say it rather than let it be inferred. Unset, vLLM enables async
    # scheduling unless something is incompatible -- and then disables it
    # with a warning, which is how a run quietly loses it. Set, the same
    # incompatibilities raise instead. Nothing we serve is on that list:
    # it excludes pooling models and every speculative method outside the
    # EAGLE family, which includes MTP.
    config.engine_args.async_scheduling = True
    # Cache every recurrent state block, not only the boundaries the engine
    # can prove are reachable. The engine keeps the latest replay boundary
    # and shared-prefix junctions; a continuation resumes at the boundary its
    # own PREVIOUS turn ended on, which is neither, so under the engine's
    # default the state at the resume point is never cached and every
    # continuation arrives as a divergent hit: an attention prefix with no
    # state under it, recomputed from the last boundary that has one. Every
    # engine here is a hybrid.
    config.engine_args.prefix_cache_retention_interval = None

    # --- the environment: the box and the deploy target ---------------
    kaggle_engine: ullm.kaggle.KaggleEngine | None = None
    engine_env = dict(UNIVERSAL_ENGINE_ENV)
    # A CUDA graph per decode width (``settings.md``, U-00) is captured once
    # per card under tensor parallelism, on cards where capture memory is
    # what sets the headroom; the four-L4 box keeps the engine's own ladder
    # until a boot there measures what the dense one costs.
    dense_cuda_capture = True
    if environment == "slurm-b200":
        # One B200 per job, weights over the cluster's network.
        config.engine_args.tensor_parallel_size = 1
        offline = False
        # The renderer's threads process every image of every request before
        # it reaches the engine, with no image cache to skip any (above), and
        # one falls behind a full job's requests; a cluster job has the CPUs
        # to spare beside its games (2026-10-17).
        config.engine_args.renderer_num_workers = 4
    elif environment == "kaggle-rtx":
        # A kernel on one RTX 6000 PRO, offline once the engine has warmed its
        # caches: a worker re-importing on first generate must not reach out.
        config.engine_args.tensor_parallel_size = 1
        offline = True
        kaggle_engine = ullm.kaggle.VLLM_FORK
        # The renderer's threads, as on the cluster, but fewer: a kernel has
        # fewer CPUs for its games to share (2026-10-17).
        config.engine_args.renderer_num_workers = 2
    elif environment == "kaggle-4xl4":
        # The four-L4 kernel. Pinned, not probed: the launcher that builds
        # these args usually has no GPU, and a probe there reads zero.
        config.engine_args.tensor_parallel_size = 4
        offline = True
        kaggle_engine = ullm.kaggle.VLLM_FORK
        dense_cuda_capture = False
        # The renderer's threads, as on the other kernel (2026-10-17).
        config.engine_args.renderer_num_workers = 2
        # The driver's hardware work queues at their maximum, for any model
        # here: a pipeline stage's token receive spins on a side stream until
        # the other stage broadcasts, and at the default eight queues a CUDA
        # graph's branch can sit behind it in one, waiting for nothing it
        # needs (~35 ms a step on the drafted Flash-Next cell, +14.5% with
        # 32; ``flash_optimization_arc2.md``). This box only
        # (2026-10-03): no other box here runs pipeline stages, and none has
        # measured it.
        engine_env["CUDA_DEVICE_MAX_CONNECTIONS"] = "32"
    else:
        raise ValueError(f"unknown environment {environment!r}; this module serves on {list(ENVIRONMENTS)}.")
    if kaggle_engine is not None:
        # Every Kaggle kernel installs its kernels prebuilt from the wheelhouse
        # (``ullm.kaggle``); refuse to reach for any that are missing. There
        # is no network to reach over, and the fallback is not an error but a
        # local build -- one ``cicc`` per parallel job, several GiB each,
        # beside a model that already holds most of the box.
        engine_env["FLASHINFER_NO_DOWNLOAD"] = "1"
        # What a kernel compiles anyway -- a FlashInfer kernel the wheelhouse
        # carries for no card of its own -- runs one ``cicc`` of several GiB
        # per parallel job, in a cgroup the model and the host tier already
        # fill. ``set_env`` sizes that parallelism to the box's cores, which
        # on a kernel is a memory bill nothing there can pay; four jobs is
        # the compile's speed bounded by what the cgroup has left.
        engine_env["MAX_JOBS"] = "4"

    # --- the pair: what only the two together decide ------------------
    # gpu_memory_utilization is measured on the pairing, so a cell nobody has
    # served says so rather than starting an engine that profiles a pool sized
    # for another card.
    kaggle_dataset_sources: tuple[str, ...] = ()
    kaggle_model_sources: tuple[str, ...] = ()
    cell = (engine, environment)
    if cell in (("qwen3.8-27b", "slurm-b200"), ("qwen3.6-27b", "slurm-b200")):
        # Above the 0.92 this deployment started from: the extra headroom is
        # KV pool, and pool headroom is what keeps a game's cached prefix alive
        # between its turns. Sized for one GPU at the default per-step token
        # budget -- a wider budget or a multi-GPU high-concurrency run grows
        # the activation workspace out of the same headroom and OOMs.
        config.engine_args.gpu_memory_utilization = 0.98
        # A 25-game A/B measured autotune ~20-25% slower on this pairing, and
        # SM90+ resolves an unset value to True. What it tunes is kernel
        # selection for these weights on this card, so it is the pair's call.
        config.engine_args.enable_flashinfer_autotune = False
    elif cell == ("qwen3.8-27b", "kaggle-rtx"):
        # Version 1 of this dataset was checked once against
        # QWEN38_OFFICIAL_REVISION above (the record is in the duck's
        # evidence.md). A published version is immutable, so a kernel mounts it
        # without rescanning the 31 GB at every startup. The mount path and the
        # source are one fact: the path IS the ref, prefixed.
        kaggle_dataset_sources = ("driessmit1/qwen3-8-27b-fp8-hf-017b9c7a",)
        config.engine_args.model = f"/kaggle/input/datasets/{kaggle_dataset_sources[0]}"
        # Less headroom than the B200 leaves, for the smaller card.
        config.engine_args.gpu_memory_utilization = 0.95
        config.engine_args.enable_flashinfer_autotune = False
    elif cell == ("qwen3.6-27b", "kaggle-rtx"):
        kaggle_dataset_sources = ("driessmit1/qwen3-6-27b-fp8-hf-snapshot",)
        config.engine_args.model = f"/kaggle/input/datasets/{kaggle_dataset_sources[0]}"
        config.engine_args.gpu_memory_utilization = 0.95
        config.engine_args.enable_flashinfer_autotune = False
    elif cell in (("qwen3.8-27b", "kaggle-4xl4"), ("qwen3.6-27b", "kaggle-4xl4")):
        # The same published snapshots the RTX cells mount; one engine, one set
        # of bytes wherever it lands.
        kaggle_dataset_sources = (
            "driessmit1/qwen3-8-27b-fp8-hf-017b9c7a"
            if engine == "qwen3.8-27b"
            else "driessmit1/qwen3-6-27b-fp8-hf-snapshot",
        )
        config.engine_args.model = f"/kaggle/input/datasets/{kaggle_dataset_sources[0]}"
        # Lower than the RTX cells' 0.95 because of the V2 model runner every
        # environment here pins: it sizes the KV pool without reserving for
        # CUDA graph capture (V1 logs "Estimated CUDA graph memory: 2.21 GiB
        # total" and holds it back; V2 logs no such line and hands the whole
        # budget to the pool), so capture then asks for ~2 GiB that is already
        # gone and the engine dies partway through its 35 piecewise graphs.
        # Measured on the four L4s: at 0.95 the pool takes 13.16 GiB per card
        # and startup fails, at 0.85 it takes enough less that capture fits.
        # The headroom IS the reserve V2 does not make, so this moves only
        # with the runner -- and not by enforce_eager to dodge it, the graphs
        # are worth their memory.
        config.engine_args.gpu_memory_utilization = 0.85
        config.engine_args.enable_flashinfer_autotune = False
    elif cell == ("qwen3.8-flash-next", "slurm-b200"):
        # The cluster serves the pruned derivative from shared storage, or
        # the caller's checkpoint; a mounted directory carries no revision.
        config.engine_args.model = checkpoint or FLASHNEXT_E307_CHECKPOINT
        config.engine_args.revision = None
        config.engine_args.tokenizer_revision = None
        # Measured on this pair, as on the Kaggle card: 0.98 starts, sizes its
        # pool, and dies in the CUDA graph capture that runs after the pool
        # is allocated, which needs more of the card than 0.98 leaves outside
        # the budget; 0.97 captures and plays. The figures:
        # ``flash_optimization.md``, "The B200's ceiling". That is with the
        # memory profiler's own graph capture holding its estimate back from
        # the pool; one capture instead (below) lowers the budget by exactly
        # that, which the boot line names: 0.97 there is 0.9488 here. The
        # W4A16 MoE below takes more CUDA-graph memory outside the budget, the
        # more the more slots a harness runs: at this cell's slots it leaves
        # more to spare than Marlin's graphs did at 220, which this held, and
        # a harness running more states its own budget, as ARC2's does
        # (``flash_optimization.md``, "The MoE backend").
        config.engine_args.gpu_memory_utilization = 0.9488
        _one_graph_capture(engine_env)
        config.engine_args.enable_flashinfer_autotune = False
        # FlashInfer's CuTe DSL MoE in its W4A16 mode (2026-10-15):
        # Marlin's arithmetic -- the 4-bit expert weights over bf16
        # activations, no activation scale -- on this card's own tensor cores,
        # where Marlin runs the older warp-level ones. Its accuracy reads as
        # Marlin's; the RTX keeps Marlin, since the kernel is the 10.x
        # family's alone, so the boxes run the same arithmetic in different
        # kernels (``flash_optimization.md``, "The MoE backend"). Without the
        # fork's variable the backend quantizes each expert's input (W4A4), so
        # an engine that lacks it is refused here rather than run that way.
        # Like Marlin it serves no bf16 experts, so a drafter's unquantized
        # head needs its own backend (``speculative_config["moe_backend"]``).
        # Named rather than walked to, so losing it raises here instead of
        # demoting.
        if CUTEDSL_W4A16_ENV not in vllm.envs.environment_variables:
            raise ValueError(
                f"this engine has no {CUTEDSL_W4A16_ENV}, so its CuTe DSL MoE would quantize activations; "
                "the B200 cell needs the fork revision that carries it"
            )
        config.engine_args.kernel_config.moe_backend = "flashinfer_cutedsl"
        engine_env[CUTEDSL_W4A16_ENV] = "1"
        # The table on the host: measured at the same rate as on the card,
        # and the card then holds twice the contexts.
        offload_ple_table(config.engine_args)
        _draft_from(config.engine_args, FLASHNEXT_E307_DRAFTER)
    elif cell == ("qwen3.8-flash-next", "kaggle-rtx"):
        kaggle_model_sources = (FLASHNEXT_E307_KAGGLE_MODEL, FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL)
        config.engine_args.model = ullm.kaggle.model_mount(FLASHNEXT_E307_KAGGLE_MODEL)
        drafter = ullm.kaggle.model_mount(FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL)
        _draft_from(config.engine_args, drafter)
        # Measured on this pair rather than borrowed: 0.99 cannot start (the
        # card's context and driver hold ~1 GiB), 0.98 starts and dies in
        # play because the runtime footprint outgrows the start-up profile by
        # about a GiB, with or without expandable segments; 0.975 played a
        # replay window but the nine-hour submission at it ended an hour
        # early (``flash_optimization.md``, "The utilization ceiling"). 0.97
        # has played the hours, and a hundredth buys ~0.7% of pool, since the
        # budget bounds the whole footprint: nothing above it is worth a death.
        # Those are budgets with the profiler's graph estimate held back; with
        # one capture (below) the same footprint is 0.97's boot-line
        # equivalent, 0.9537, which has played an hour and nine hours. The
        # throughput stack (below) takes a hundredth of that back: its CUDA
        # graphs are ~0.65 GiB larger, and at 0.9537 its engine runs out of
        # memory in warm-up, where 0.9437 played the 80-lane traffic replay.
        config.engine_args.gpu_memory_utilization = 0.9437
        _one_graph_capture(engine_env)
        # PyTorch's expandable segments, against the fragmentation that ends
        # a long run: the allocator's failed-and-retried requests went from
        # ten a replay to none, at no throughput cost. vLLM refuses expandable
        # segments beside a KV connector (the host tier is one) unless its
        # cumem allocator holds the KV cache, so the two come together. One
        # rank here, so no custom all-reduce IPC for expandable memory to
        # conflict with -- which is what keeps this off the four L4s.
        engine_env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        config.engine_args.enable_cumem_allocator = True
        # The indexer's prefill scoring bound, as on the four L4s (that cell
        # describes the buffer): measured free beside expandable segments on
        # the replay, and it keeps a new largest buffer from being a long
        # run's first failed allocation.
        engine_env["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "64"
        # The Flash-Next throughput stack with KN, on the card it was measured
        # on: 2.84K -> 3.36K completed-request tokens/s (+18.3%) in native play
        # on an RTX PRO 6000 against the same harness's stock engine. The
        # competition kernel serves it; the fork keeps each kernel's shape,
        # dtype and SM120 fallbacks.
        engine_env.update(FLASHNEXT_THROUGHPUT_ENV)
        # One BLAS thread in the engine process, as in both measured arms.
        engine_env["OPENBLAS_NUM_THREADS"] = "1"
        config.engine_args.enable_flashinfer_autotune = False
        # The TRT-LLM kernels the B200 column pins are gated to the 10.x
        # capability family; this card is 12.0, so the pair needs its own.
        # Marlin runs the experts as 4-bit weights over bf16 activations,
        # where the native FP4 kernels also quantize the activations. Decode
        # here is bound by reading weights, not by the multiply -- a handful
        # of sequences spread over hundreds of experts leaves each expert a
        # token or two per step, too little for FP4 arithmetic to pay for
        # quantizing its input -- and that is the regime Marlin was built
        # for. Measured against `flashinfer_cutlass` on this pair, it was
        # faster both before and after the host tier engages, did not score
        # worse, and costs about a minute of repacking at load
        # (``flash_optimization.md``, "The MoE backend"). It also leaves the
        # grouped CUTLASS FP4 path, which has an open report of corrupt output
        # on this card family. Named rather than walked to, so losing it
        # raises here.
        config.engine_args.kernel_config.moe_backend = "marlin"
        # As on the cluster: the table is the largest thing on the card and
        # the one part a PCIe link can carry, and this card has a third less
        # to spend.
        offload_ple_table(config.engine_args)
        # Read the checkpoint into page cache ahead of the loader rather than
        # faulting it in over the network mount: faulting in costs minutes per
        # boot, and a boot is the floor of every experiment here (another
        # team's run of this architecture on this card prefetches in 182 s,
        # then loads its weights in 28). The reader
        # is the kernel's first setup command (``ullm.kaggle_prefetch``), so
        # it reads while the engine installs, and it keeps a bounded window
        # ahead of the shard the loader has open, dropping the shards it has
        # passed. The engine's own prefetch reads every shard at once with no
        # bound, from load on, into a page cache the pinned table and the
        # host tier leave smaller than the checkpoint; and the engine would
        # not choose it by itself, since the table is pinned before its
        # free-RAM check runs. So the engine maps lazily what the reader has
        # cached. Target first, then drafter: the order the runner loads them.
        config.engine_args.safetensors_load_strategy = "lazy"
        assert kaggle_engine is not None
        kaggle_engine = dataclasses.replace(
            kaggle_engine,
            prefetch=(config.engine_args.model, drafter),
        )
    elif cell == ("qwen3.8-flash-next", "kaggle-4xl4"):
        # The ARC2 harness's submission box, which ARC3 never runs on. The
        # runs behind each value: ``flash_optimization_arc2.md``.
        kaggle_model_sources = (FLASHNEXT_E307_KAGGLE_MODEL, FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL)
        config.engine_args.model = ullm.kaggle.model_mount(FLASHNEXT_E307_KAGGLE_MODEL)
        # The engine's drafting, as on the other cells: four drafts with the
        # head that holds its experts in NVFP4, so it runs on the cell's
        # Marlin pin. Drafting three alone added 12% to this box's replay
        # (run C against the undrafted run B); the settings below were tuned
        # around it, together 803 tokens a second against B's 536 (run L).
        _draft_from(config.engine_args, ullm.kaggle.model_mount(FLASHNEXT_E307_DRAFTER_KAGGLE_MODEL))
        # Below the single-card Flash cells' 0.97 because what play needs past
        # the start-up profile is an absolute amount -- the prefill kernels'
        # working sets above the dummy run's, allocator fragmentation, and
        # ~1 GiB a process outside torch -- and on a 22 GiB card a hundredth
        # is 0.22 GiB. With the drafter, one graph capture and both graph kinds
        # on the ladder below, 0.92 ran out of memory on the second stage
        # after 23 minutes of serving (58 MiB asked, 46 free, 373 MiB held by
        # torch in pieces too small), where 0.91 played the replay window with
        # ~50 MiB of those cards free. With FULL decode graphs alone (below)
        # the cards served at 0.92 with more free than 0.91 left with both,
        # and the pool held more requests at the same step cost (runs km2
        # against km1b, the 230-game replay at ARC2's 512-token steps and the
        # rings' own pool). 0.935 from the ladder above that, 0.925 to 0.945
        # on the same replay (runs km3 to km7; 2026-10-28).
        config.engine_args.gpu_memory_utilization = 0.935
        _one_graph_capture(engine_env)
        # The sparse-attention indexer scores a prefill's queries against the
        # whole compressed context in one float32 buffer of queries x
        # context/4 columns, chunked only above this many MiB (512 by default):
        # ~126 MiB for a 3,136-token block at a 42k context, ~200 MiB at 65k.
        # The start-up profile runs a short context and reserves none of it,
        # and a new largest buffer must come from the card rather than the
        # allocator's cache, which is where the longer run died (126 MiB asked,
        # 75 free, 352 held in smaller pieces). 64 keeps each piece near the
        # step's other buffers for at most four chunks a call at this step.
        engine_env["VLLM_SPARSE_INDEXER_MAX_LOGITS_MB"] = "64"
        # The throughput stack's portable part, the rest of it being out of
        # this box's reach: the engine refuses the fused commit under the
        # pipeline stages below, and the CUDA images load only on capability
        # 12.0 (``flash_optimization_arc2.md``, "The throughput stack on the
        # four L4s").
        engine_env.update(FLASHNEXT_THROUGHPUT_PORTABLE_ENV)
        # One BLAS thread in each engine process, as the RTX cell runs beside
        # the stack.
        engine_env["OPENBLAS_NUM_THREADS"] = "1"
        config.engine_args.enable_flashinfer_autotune = False
        # Marlin is the only NVFP4 MoE backend this card admits: its floor is
        # capability 7.5, where the FlashInfer backends are the 10.x and 12.x
        # families and the L4 is 8.9. Named rather than walked to, so losing
        # it raises here instead of demoting.
        config.engine_args.kernel_config.moe_backend = "marlin"
        # Two pipeline stages of two tensor ranks rather than a four-way
        # tensor split: four ranks pad this checkpoint's 640-wide experts by a
        # fifth (Marlin's tile on a 160-row slice) and store the attention
        # share of every cached token twice (2 KV heads over 4 ranks); two
        # ranks split both, though some part of the cache still does not
        # halve over them (~5.4 KiB a cached token per card against four
        # stages' 4.85; untraced). The layers repeat GDN, GDN, GDN, attention
        # twelve times, and the one PLE layer (index 2) and its table sit on
        # the first stage, the fork's condition for pipelining this model; the
        # split is below. Against four stages
        # of one card it holds the same pool within 1% (Kaggle version 38:
        # 1,376,256 tokens at 0.96, against 1,365,333 at 0.95), and generates
        # ~20% more by the host's count (494 against 411 tok/s over the play
        # window): a stage runs half the running set per micro-batch rather
        # than a quarter, for two all-reduces a layer over the pair's PCIe.
        # Paid in per-request latency, which this task does not price.
        config.engine_args.tensor_parallel_size = 2
        config.engine_args.pipeline_parallel_size = 2
        # Twenty-seven layers on the first stage, twenty-one on the second:
        # the second also holds the lm head, the sampling and the drafter, so
        # at twenty-four each it is the busier stage and the smaller pool.
        # 27/21 is where the pool peaks with the input embedding on the host:
        # the second stage's block is a page larger, and moving layer 26 (a
        # GDN layer) leaves the first stage's block as it was, where a
        # twenty-eighth would move attention layer 27 and make the first
        # stage the smaller pool (+3.9% pool and +1.8% generated against
        # 26/22, runs I and H).
        engine_env["VLLM_PP_LAYER_PARTITION"] = "27,21"
        # The table on the host, as on both Flash cells. It is a
        # vocab-parallel embedding, so each first-stage worker pins its
        # shard.
        offload_ple_table(config.engine_args)
        # The engine's own prefetch, every shard read at once from load on,
        # not the RTX cell's bounded reader (``ullm.kaggle_prefetch``): the
        # reader moves with the loaders it has seen, and here the second
        # stage's pass the first stage's shards before the first stage starts,
        # so it drops what those have yet to read (535 s of load against 252,
        # runs H and G).
        config.engine_args.safetensors_load_strategy = "prefetch"
        # The CUDA graphs this traffic replays, not the engine's ladder: every
        # request count from 12 to 20 a micro-batch, and 28, exact for both
        # the verify rows (five tokens a request) and the draft steps; coarse
        # elsewhere; nothing above 256 tokens, where a batch runs eagerly and
        # its GPU work hides the launches. ~0.4 GiB a card less than the
        # engine's ladder, and no padding (runs K, L).
        _capture_sizes(
            config.engine_args,
            (1, 2, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 24, 28, 32, 40, 48)
            + (60, 65, 70, 75, 80, 85, 90, 95, 100, 120, 140, 160, 192, 256),
        )
        # FULL decode graphs alone (2026-10-28): a step carrying a
        # prefill runs eagerly. The PIECEWISE graphs held about half of the
        # cards' graph memory and bought little here (run X's 512-token one:
        # the mixed steps' launches sit off the critical path), so the pool
        # takes what they held (the budget above).
        compilation = config.engine_args.compilation_config
        assert isinstance(compilation, vllm.config.CompilationConfig)
        compilation.cudagraph_mode = vllm.config.CUDAGraphMode.FULL_DECODE_ONLY
    else:
        raise ValueError(
            f"no measured pairing for engine {engine!r} in environment {environment!r}. "
            "gpu_memory_utilization is measured on the pair, so serving one that has never been "
            "measured needs its own branch here rather than a neighbour's figure."
        )
    # A drafting engine's cell names where its drafter is; without that vLLM
    # reads the target's own head out of the checkpoint, a second pass over
    # every shard beside a model already resident.
    speculative = config.engine_args.speculative_config
    if speculative and not speculative.get("model"):
        raise ValueError(f"{cell}: the engine drafts but the cell names no drafter path")
    # A revision pin belongs to the published repository, not to a mounted
    # copy, which is immutable by being published.
    if kaggle_dataset_sources or kaggle_model_sources:
        config.engine_args.revision = None
        config.engine_args.tokenizer_revision = None

    if serving == "direct":
        factory: ullm.factory.BackendFactory = ullm.backends.vllm_factory.EngineFactory.from_config(
            config, mode="direct", env_overrides=engine_env, hf_hub_offline_after_start=offline
        )
    else:
        # An engine that expects its client to name each conversation's thread
        # puts a host tier in park mode, and the HTTP client names none and
        # sends no closes: every finish would skip its park, and with no band
        # the tier would hold almost nothing, said only in the park log
        # (roadmap.md, "Park mode over the server topology").
        if config.engine_args.prefix_cache_context_threads:
            raise ValueError(
                f"{engine} cannot be served over the server topology: its engine expects the client to name "
                "each conversation's thread (prefix_cache_context_threads), which puts the host tier in park "
                "mode, and the HTTP client names none, so the tier would park nothing. Serve it direct."
            )
        # The server parses tool calls and reasoning server-side, so what feeds
        # the in-process backend splits: the tool parser goes to server args,
        # the matching reasoning parser already rides in engine args. It never
        # streams -- the SDK's accumulator drops the return_token_ids fields
        # tracing and token continuation read.
        factory = ullm.backends.vllm_factory.InprocessServerFactory(
            engine_args=config.engine_args,
            env_overrides=engine_env,
            hf_hub_offline_after_start=offline,
            serve_overrides={
                "enable_auto_tool_choice": True,
                "tool_call_parser": config.backend_kwargs["tool_call_parser"],
            },
            backend_kwargs={
                "parse_think_tags": True,
                "chat_template_kwargs": {
                    **config.backend_kwargs.get("chat_template_kwargs", {}),
                    "enable_thinking": True,
                },
                "streaming": False,
            },
        )
    factory.dense_cuda_capture = dense_cuda_capture
    return Deployment(
        factory=factory,
        kaggle_engine=kaggle_engine,
        kaggle_dataset_sources=kaggle_dataset_sources,
        kaggle_model_sources=kaggle_model_sources,
    )


def serve_kaggle_checkpoint(
    deployment: Deployment,
    model: ullm.kaggle.KaggleCheckpoint,
    drafter: ullm.kaggle.KaggleCheckpoint | None = None,
) -> None:
    """Serve ``model``, with ``drafter`` where the engine drafts, on a Kaggle
    cell in place of the cell's own pair: the engine's paths, the datasets and
    models the kernel mounts, and its setup -- a split checkpoint assembled
    before anything reads it, the read-ahead pointed at the new paths.

    For a task with a checkpoint of its own (ARC2's prune, the seal's
    joint205), which then names only its own sources: the datasets and model
    versions that hold it. Refuses a cell whose engine reads anything but its
    mounts, one to one: replacing them would leave a path unmounted or a mount
    unread.
    """
    engine_args = getattr(deployment.factory, "engine_args", None)
    if engine_args is None or deployment.kaggle_engine is None:
        raise ValueError("not a Kaggle cell this process builds: no engine arguments, or no Kaggle engine")
    speculative = engine_args.speculative_config
    paths = [str(engine_args.model), *([str(speculative["model"])] if speculative else [])]
    mounts = [
        *(f"{ullm.kaggle.KAGGLE_DATASET_MOUNT}/{source}" for source in deployment.kaggle_dataset_sources),
        *(ullm.kaggle.model_mount(source) for source in deployment.kaggle_model_sources),
    ]
    if sorted(paths) != sorted(mounts):
        raise ValueError(f"the cell reads {paths} and mounts {mounts}: not a pair to replace")
    if (drafter is not None) != bool(speculative):
        raise ValueError("a drafting cell takes a drafter beside its checkpoint, and only a drafting cell does")
    checkpoints = [model, *([drafter] if drafter is not None else [])]
    replaced = {old: new.path for old, new in zip(paths, checkpoints, strict=True)}
    engine_args.model = model.path
    if drafter is not None:
        engine_args.speculative_config = {**speculative, "model": drafter.path}
    deployment.kaggle_dataset_sources = tuple(
        dict.fromkeys(ref for checkpoint in checkpoints for ref in checkpoint.sources)
    )
    deployment.kaggle_model_sources = tuple(
        dict.fromkeys(ref for checkpoint in checkpoints for ref in checkpoint.model_sources)
    )
    deployment.kaggle_engine = dataclasses.replace(
        deployment.kaggle_engine,
        prefetch=tuple(replaced.get(path, path) for path in deployment.kaggle_engine.prefetch),
        assemble=tuple(checkpoint for checkpoint in checkpoints if checkpoint.shards_from),
    )
