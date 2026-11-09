"""The deployment columns: what they serve, and what they claim about it.

Everything here is config-level — no engine is built. The columns are only
truly exercised by booting one, which the gpu tier does; what this file
protects is the things that go wrong silently at config time and are
expensive to discover on the box.

The sampling is compared against ``analyzer_sampling_profiles`` (at the
standard cap) rather than against its values. What every column owes is the
SHARED profile — an environment that quietly rolled its own is the
regression worth catching, and the numbers themselves are single-sourced in
``presets.py`` with their reasons in ``evidence.md``.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, cast

import pytest
import ullm.kaggle as ullm_kaggle
from duck import CheckMode, DuckAgent, presets

deployment = pytest.importorskip("ullm.backends.deployment", reason="needs the vllm extra (linux)")

pytestmark = [
    # Building an engine config imports the GPU stack, which probes CUDA and
    # warns (not errors) on a GPU-less or old-driver box.
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
    pytest.mark.filterwarnings("ignore:CUDA initialization"),
]


def _factory(agent: DuckAgent) -> Any:
    """The engine factory the column built, typed loosely: it arrives through
    ullm's coordinate tables, which type as Any and narrow nothing."""
    factory = agent.llm_config.backend_factory
    assert factory is not None
    return cast("Any", factory)


def test_a_bare_build_serves_flash_next_on_the_cluster() -> None:
    """Flash-Next is the default engine: the submission's, and the cluster
    column a bare build boots."""
    bare = DuckAgent.build()
    assert _factory(bare).engine_args.model == _factory(DuckAgent.build(engine="qwen3.8-flash-next")).engine_args.model
    assert bare.run_config.max_concurrent == DuckAgent.build(engine="qwen3.8-flash-next").run_config.max_concurrent


def test_the_38_column_keeps_the_analyzer_sampling() -> None:
    """Official 3.8 weights, analyzer sampling and request cap, effort live."""
    ret = DuckAgent.build(engine="qwen3.8-27b")
    factory = _factory(ret)

    assert factory.engine_args.model == "Qwen/Qwen3.8-27B-FP8"
    assert ret.llm_config.sampling_profiles["play"] == presets.analyzer_sampling_profiles()["play"]
    assert factory.backend_kwargs["native_reasoning_effort"] is True
    assert ret.reasoning_effort == "xhigh", "the swept default rides the column"


def test_effort_is_column_scoped_not_a_class_default() -> None:
    """The class default is provider-neutral — this field reaches whatever
    backend a hand-built duck is paired with, and "xhigh" is a level only
    the 3.8 family has. The engine layer injects xhigh where it was swept,
    and an explicit override still beats the injection."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).reasoning_effort
        == "high"
    )
    assert DuckAgent.build(engine="qwen3.8-27b").reasoning_effort == "xhigh"
    assert DuckAgent.build(engine="qwen3.8-27b", reasoning_effort="low").reasoning_effort == "low"
    # The environment does not change which engine's effort scale applies.
    assert DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4").reasoning_effort == "xhigh"
    assert (
        DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4", reasoning_effort="low").reasoning_effort
        == "low"
    )
    assert DuckAgent.build(engine="qwen3.6-27b").reasoning_effort == DuckAgent.reasoning_effort


def test_the_engine_layer_scopes_the_summary_marker_ban_to_its_tokenizer() -> None:
    """Every column of the Qwen family knows the marker token; generic agents
    do not. A fact about the checkpoints' shared tokenizer, so it rides the
    engine layer and reaches no agent built for another model."""
    expected = (presets.QWEN_TOOL_CALL_TOKEN_ID,)
    assert presets.QWEN_TOOL_CALL_TOKEN_ID == 248058
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).compaction_forbidden_token_ids
        == ()
    )
    assert DuckAgent.build(engine="qwen3.8-27b").compaction_forbidden_token_ids == expected
    assert DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4").compaction_forbidden_token_ids == expected
    assert DuckAgent.build(engine="qwen3.6-27b").compaction_forbidden_token_ids == expected


def test_kaggle_follows_the_engine_the_same_way() -> None:
    """Kaggle's column is the same move as SLURM's: builder, snapshot, profile.

    Deployment sizing (gpu_util 0.85, the host KV tier at its Kaggle size,
    the Kaggle concurrency, tp=4) is Kaggle's own and must not move with
    the engine.
    """
    ret = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4")
    factory = _factory(ret)
    assert factory.cpu_kv_offload
    # The mount path IS the dataset ref, prefixed -- the kernel cannot have
    # one without the other, and ullm keeps them in one branch for that reason.
    sources = ret.run_config.kaggle_dataset_sources
    assert sources and factory.engine_args.model.endswith(sources[0])
    assert factory.engine_args.model.startswith("/kaggle/input/datasets/")
    assert ret.llm_config.sampling_profiles["play"] == presets.analyzer_sampling_profiles()["play"]
    # 0.85, not the 0.95 a single big card takes: the headroom is the CUDA
    # graph capture reserve the V2 model runner does not make for itself.
    assert factory.engine_args.gpu_memory_utilization == 0.85
    assert factory.engine_args.tensor_parallel_size == 4, "27B at FP8 does not fit one L4"
    assert ret.run_config.shuffle_games is True, "a submission's play order should be a random sample"


def test_kaggle_builds_the_shipped_configuration_on_the_forked_engine() -> None:
    """Kaggle carries the fork, so no field of the agent is refused there.

    ``ullm.vllm_overlay`` replaces the stock engine the wheels install with
    the pinned revision, and refuses on the kernel rather than run a
    half-overlaid one; nothing is left for the constructor to guard. The
    shipped defaults are pinned here because a default that quietly turned
    the blank back on would change what a submission serves.
    """
    agent = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4")
    assert agent.zero_recurrent is False
    assert (
        agent.inject_text_on_first_prompt
        == DuckAgent(
            reply_cap_tokens=None,
        ).inject_text_on_first_prompt
    )
    assert DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4", zero_recurrent=True).zero_recurrent is True


def test_the_kaggle_column_links_the_forked_engine() -> None:
    """The engine is said once, in ullm, and linked: the column names no
    install command of its own, so the two cannot drift apart, and only the
    wheelhouse is ARC2's, ullm's wheels with the record an ARC2 kernel makes.
    The fork rides along because the host tier's connector answers a
    scheduler hook only the fork has."""
    run_config = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4").run_config
    assert run_config.kaggle_engine == dataclasses.replace(ullm_kaggle.VLLM_FORK, wheels=presets.KAGGLE_WHEELS)
    assert run_config.kaggle_setup_commands == ()
    assert run_config.kaggle_kernel_sources == ()
    assert DuckAgent.build(engine="qwen3.8-27b").run_config.kaggle_engine is None, "a SLURM job installs nothing"


def test_the_engine_levers_are_set_on_both_columns() -> None:
    """Upstream's levers, each with a reason that survives here: the shm
    multimodal cache (every game opens with the same legend image —
    vllm#31404's trigger), autotune off (measured slower), the v2 runner
    pinned — the only runner that honours the recurrent-state blank — and
    the scheduler shape both columns share: the step budget and the
    admission reserve that makes the engine queue rather than preempt. The
    Kaggle box alone refuses to download FlashInfer kernels -- a kernel has no
    network, and the fallback is a local build beside the model -- bounds
    that build's parallelism, since each job is a multi-GiB compiler in a
    cgroup the model and the tier already fill, and widens the driver's
    hardware work queues, for any model there, so no CUDA graph branch waits
    behind a pipeline stage's spinning receive."""
    kaggle_env = {
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "FLASHINFER_NO_DOWNLOAD": "1",
        "MAX_JOBS": "4",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    }
    for environment, env in (("slurm-b200", {"VLLM_USE_V2_MODEL_RUNNER": "1"}), ("kaggle-4xl4", kaggle_env)):
        factory = _factory(DuckAgent.build(engine="qwen3.8-27b", environment=environment))
        engine_args = factory.engine_args
        assert engine_args.mm_processor_cache_type == "shm", environment
        assert engine_args.enable_flashinfer_autotune is False, environment
        assert engine_args.max_num_batched_tokens == 8192, environment
        assert engine_args.watermark == deployment.ENGINE_ADMISSION_WATERMARK, environment
        assert factory.env_overrides == env, environment
        assert factory.verify_serving_bug is False, environment


def test_the_blank_needs_no_kv_connector() -> None:
    """The blank is asked for through the sampling parameter, not obtained
    from a connector's serving side effect: a factory built with the tier
    off asks for no host tier and is a valid duck. The tier is the column's
    default, so off is a field a caller turns off.
    """
    bare = DuckAgent.build(engine="qwen3.8-27b")
    _factory(bare).cpu_kv_offload = False
    assert bare.zero_recurrent is DuckAgent.zero_recurrent
    assert not _factory(bare).cpu_kv_offload
    assert _factory(DuckAgent.build(engine="qwen3.8-27b")).cpu_kv_offload


def test_each_column_ships_the_host_tier_and_its_own_concurrency() -> None:
    """Both deployments run more games than their pool holds contexts, so each
    turns the victim tier on and runs a concurrency measured for its own cell.
    Sized differently: SLURM's B200 saturates deeper than four L4s, and the
    Kaggle tier is capped by a cgroup the harness shares."""
    from ullm.backends import config_local, vllm_factory

    def tier_of(agent: DuckAgent) -> typing.Any:
        agent_factory = agent.llm_config.backend_factory
        assert isinstance(agent_factory, vllm_factory.EngineFactory)
        return agent_factory

    slurm, kaggle = (
        DuckAgent.build(engine="qwen3.8-27b"),
        DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4"),
    )
    for agent in (slurm, kaggle):
        # Stated on the factory; the connector itself is built on the worker
        # at start, from these fields.
        assert tier_of(agent).cpu_kv_offload and tier_of(agent).cpu_kv_offload_victim is not False
    # The connector reports to the job directory's trace; its engine-log copy
    # is the connector's own default (off), which no preset touches. Writing
    # it is what the engine blocked on in the twelve-hour Kaggle run of
    # 2026-09-19.
    assert tier_of(slurm).cpu_kv_offload_extra == tier_of(kaggle).cpu_kv_offload_extra == {}
    # None, not a number: config_local's shipped size is the one place it lives.
    assert tier_of(slurm).cpu_kv_offload_gib is None
    assert config_local.DEFAULT_CPU_KV_OFFLOAD_GIB > 0, "and it is a real size"
    assert tier_of(kaggle).cpu_kv_offload_gib == 60.0, "Kaggle sizes its own, against the cgroup"
    assert (slurm.run_config.max_concurrent, kaggle.run_config.max_concurrent) == (80, 40)

    off = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4")
    tier_of(off).cpu_kv_offload = False
    assert not tier_of(off).cpu_kv_offload, "the tier is a field a caller turns off"


def test_the_flash_engine_layer_is_the_38s_with_no_served_opening() -> None:
    """Flash-Next ships Qwen3.8's template and tokenizer, so it takes the
    3.8 column's effort level and the family's summary-marker ban; what it
    does not take is the served opening, which halves what this model
    reasons per action. The shared profile still reaches it."""
    for environment in ("slurm-b200", "kaggle-4xl4"):
        flash = DuckAgent.build(engine="qwen3.8-flash-next", environment=environment)
        base = DuckAgent.build(engine="qwen3.8-27b", environment=environment)
        assert flash.reasoning_effort == base.reasoning_effort == "xhigh", environment
        assert flash.compaction_forbidden_token_ids == base.compaction_forbidden_token_ids, environment
        assert (flash.inject_text_on_first_prompt, flash.inject_text_on_later_prompts) == ("", ""), environment
        assert base.inject_text_on_first_prompt, "the 27B keeps its opening"
        assert flash.llm_config.sampling_profiles == presets.analyzer_sampling_profiles(), environment


def test_flash_next_keeps_its_recurrent_state_in_bf16_and_the_27bs_do_not() -> None:
    """The engine's own (ullm's ``_flashnext_config``), beside its FP8 KV
    cache: every Flash-Next column serves a bf16 state, so the B200 that
    scores ARC2 serves what the four L4s do; the 27Bs keep the checkpoint's
    own."""
    for environment in ("slurm-b200", "kaggle-4xl4"):
        flash = _factory(DuckAgent.build(engine="qwen3.8-flash-next", environment=environment))
        assert flash.engine_args.mamba_ssm_cache_dtype == "bfloat16", environment
        for engine in ("qwen3.8-27b", "qwen3.6-27b"):
            other = _factory(DuckAgent.build(engine=engine, environment=environment))
            assert other.engine_args.mamba_ssm_cache_dtype == "auto", (engine, environment)


def test_flash_next_serves_arc2s_reap307_with_the_combined_drafter_on_every_cell() -> None:
    """ARC2's checkpoint, not ullm's e307, and the combined ARC2/ARC3 drafter
    beside it: on the cluster from shared storage; on Kaggle from their
    Kaggle models, the checkpoint whole, so nothing is put together. Every
    source the engine reads is mounted, no dataset, and nothing of ullm's
    pair. The 27Bs keep their checkpoints."""
    slurm = _factory(DuckAgent.build(engine="qwen3.8-flash-next"))
    assert slurm.engine_args.model == presets.E307_ARC2_CHECKPOINT
    assert slurm.engine_args.speculative_config["model"] == presets.ARC23_FP8_DRAFTER
    kaggle = DuckAgent.build(engine="qwen3.8-flash-next", environment="kaggle-4xl4")
    engine_args = _factory(kaggle).engine_args
    assert engine_args.model == presets.E307_ARC2_KAGGLE.path
    assert engine_args.speculative_config["model"] == presets.ARC23_FP8_DRAFTER_KAGGLE.path
    assert kaggle.run_config.kaggle_dataset_sources == ()
    assert kaggle.run_config.kaggle_model_sources == (
        *presets.E307_ARC2_KAGGLE.model_sources,
        *presets.ARC23_FP8_DRAFTER_KAGGLE.model_sources,
    )
    engine = kaggle.run_config.kaggle_engine
    assert engine is not None and engine.wheels == presets.KAGGLE_WHEELS
    assert engine.assemble == ()
    assert _factory(DuckAgent.build(engine="qwen3.8-27b")).engine_args.model == "Qwen/Qwen3.8-27B-FP8"


def test_flash_next_caches_the_indexer_keys_in_fp8() -> None:
    """On both Flash-Next cells; the 27Bs have no sparse-attention indexer and
    keep the engine's own setting."""
    for environment in ("slurm-b200", "kaggle-4xl4"):
        flash = _factory(DuckAgent.build(engine="qwen3.8-flash-next", environment=environment))
        assert flash.engine_args.attention_config.indexer_kv_dtype == "fp8", environment
        other = _factory(DuckAgent.build(engine="qwen3.8-27b", environment=environment))
        assert other.engine_args.attention_config.indexer_kv_dtype == "auto", environment


def test_the_flash_cells_state_their_own_concurrency_and_tier() -> None:
    """The B200 holds twice the contexts of this model per card and runs more
    games than the 27B's column; the four L4s hold half as many, and run
    fewer. The tier is sized for the queue each leaves outside the pool."""
    slurm = DuckAgent.build(engine="qwen3.8-flash-next")
    kaggle = DuckAgent.build(engine="qwen3.8-flash-next", environment="kaggle-4xl4")
    assert (slurm.run_config.max_concurrent, _factory(slurm).cpu_kv_offload_gib) == (180, 96.0)
    assert (kaggle.run_config.max_concurrent, _factory(kaggle).cpu_kv_offload_gib) == (64, 45.0)
    # The step budget follows each cell's recurrent block: several blocks and
    # the decodes on one card, under one drafted block over the four L4s'
    # stages, where the rings also take a pool of their own. The cache-hit
    # unit is the engine layer's: Flash-Next keys every 80 tokens on every
    # cell (ullm's ``_flashnext_config``, beside the decode tail and context
    # threads it keeps conversations live with); the 27Bs set none.
    assert _factory(slurm).engine_args.max_num_batched_tokens == 8192
    assert _factory(kaggle).engine_args.max_num_batched_tokens == 512
    assert not _factory(slurm).engine_args.transient_kv_pool
    assert _factory(kaggle).engine_args.transient_kv_pool
    # The four L4s capture FULL decode graphs alone and hand the pool what
    # the PIECEWISE ones held; the B200 keeps vLLM's graphs.
    assert _factory(slurm).engine_args.compilation_config.cudagraph_mode is None
    assert _factory(kaggle).engine_args.compilation_config.cudagraph_mode.name == "FULL_DECODE_ONLY"
    assert _factory(kaggle).engine_args.gpu_memory_utilization == 0.935
    for environment in ("slurm-b200", "kaggle-4xl4"):
        engine_args = _factory(DuckAgent.build(engine="qwen3.8-27b", environment=environment)).engine_args
        assert engine_args.max_num_batched_tokens == 8192, environment
    for engine in presets.ENGINES:
        for environment in ("slurm-b200", "kaggle-4xl4"):
            engine_args = _factory(DuckAgent.build(engine=engine, environment=environment)).engine_args
            unit = 80 if engine == "qwen3.8-flash-next" else None
            assert engine_args.prefix_match_unit == unit, (engine, environment)
    assert slurm.run_config.max_concurrent > DuckAgent.build(engine="qwen3.8-27b").run_config.max_concurrent
    # On the cluster a figure past what most games spend, short of the sweeps' million.
    assert slurm.max_tokens_per_game == 500_000
    # The Kaggle backstop is the 27B's; Flash-Next plays without one there,
    # and an explicit cap still beats the cell.
    assert kaggle.max_tokens_per_game is None
    assert DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4").max_tokens_per_game == 70_000
    capped = DuckAgent.build(engine="qwen3.8-flash-next", environment="kaggle-4xl4", max_tokens_per_game=5)
    assert capped.max_tokens_per_game == 5


def test_the_kaggle_wheelhouse_is_a_stock_engine_of_this_line() -> None:
    """The offline closure and the engine this repo pins must be the same
    vLLM release, or the bundled source runs against a different engine than
    it was tested on. Stated as major.minor: the fork's pin carries a dev
    suffix and a local revision, and it is the release line that has to
    match, not the revision."""
    import vllm

    wheel = ullm_kaggle.KAGGLE_VLLM_REQUIREMENTS[0]
    assert wheel.startswith("vllm==")
    installed = vllm.__version__.split("+")[0].split(".dev")[0]
    assert wheel.removeprefix("vllm==").split(".")[:2] == installed.split(".")[:2], (
        f"Kaggle installs {wheel} while this checkout runs vllm {vllm.__version__}"
    )


def test_the_window_must_fit_the_engine() -> None:
    """Caught when the configuration settles rather than at the first
    over-long request: the engine only says so once a game has run far enough
    to build a big prompt. With the derivation on, the engine follows the
    window; a caller who owns the pairing gets the refusal instead."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.derive_max_model_len = False
    agent.context_window_tokens = _factory(agent).engine_args.max_model_len + 1
    agent.compact_to_tokens = 4096
    with pytest.raises(ValueError, match="max_model_len"):
        agent.prepare_for_backend()


def test_agent_fields_win_over_the_column() -> None:
    """The column sets configs and agent fields; an explicit field must
    replace one rather than collide with it."""
    agent = DuckAgent.build(engine="qwen3.8-27b", max_tokens_per_game=123, image_scale=2)
    assert agent.max_tokens_per_game == 123
    assert agent.image_scale == 2


def test_the_reply_cap_is_the_reply_reserve() -> None:
    """One number in two roles: raising the cap narrows the prompt budget by
    the same amount, so they must not be able to disagree."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    assert agent._reply_reserve_tokens() == DuckAgent.reply_cap_tokens


def test_check_images_require_the_check_oracle() -> None:
    """A visible sweep coordinate cannot silently degrade to task+show."""
    with pytest.raises(ValueError, match="requires check_mode ENABLED or MANDATORY"):
        DuckAgent(reply_cap_tokens=None, images="task+show+check", check_mode=CheckMode.DISABLED)


def test_columns_do_not_pin_the_run_seed() -> None:
    """A seed belongs to the run, not the column: pinning one in a column
    makes every deployment replay the same draws and hides it in a place
    nobody reads when comparing runs."""
    for environment in ("slurm-b200", "kaggle-4xl4"):
        assert DuckAgent.build(engine="qwen3.8-27b", environment=environment).run_config.rng_seed is None, environment


def test_only_kaggle_ranks_the_slots() -> None:
    """A submission has more games than slots and a hard deadline, so the
    Kaggle column grants slots by spend and keeps its per-game cap as a
    backstop above the SLURM figure. The SLURM column keeps play order and
    its cap — off a deadline a game's spend should not depend on where in
    the run it was scheduled — and an explicit cap still beats either
    column."""
    kaggle = DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4")
    assert kaggle.run_config.slot_policy == "least_tokens"
    assert kaggle.max_tokens_per_game == 70_000
    assert kaggle.run_config.max_runtime_s_per_game is None, "the submission's own clock is the budget"
    assert (
        DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4", max_tokens_per_game=5).max_tokens_per_game == 5
    )
    slurm = DuckAgent.build(engine="qwen3.8-27b")
    assert slurm.run_config.slot_policy == "queue_order"
    assert slurm.run_config.max_runtime_s_per_game == presets.GAME_RUNTIME_CAP_S
    assert (
        slurm.max_tokens_per_game
        == DuckAgent(
            reply_cap_tokens=None,
        ).max_tokens_per_game
        == 65_000
    )
