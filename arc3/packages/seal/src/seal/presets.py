"""What ARC3 asks of a deployment, over ullm's engine/environment coordinates.

``ullm.backends.deployment`` holds the task-free half — which checkpoint, on
which box, and what the two together decide about memory. This holds the
seal's half: the analyzer sampling, the context and budget shape, the
concurrency and the scheduling. :func:`configure` walks them onto a constructed
agent, one column at a time, so the agent's own file stays about how the seal
plays. Nothing here is model-visible except the sampling profile.

Which half a setting belongs to, and which of the two coordinates decides it,
is ``settings.md`` at the repo root.

Every vllm import is function-local — building engine args pulls in vllm, and
CPU-only boxes import ``seal`` to inspect an agent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import ullm.backend

from platypus import RunConfig

# The coordinates a bare ``SealAgent.build()`` serves. Flash-Next rather than
# the duck's 27B: the seal's runs are on that engine, and a bare build is what
# its milestone boots.
DEFAULT_ENGINE = "qwen3.8-flash-next"
DEFAULT_ENVIRONMENT = "slurm-b200"

# Every column ARC3 runs. Narrower than ullm's: ullm describes boxes and
# checkpoints, this says which of them a run may happen on, so a coordinate
# ullm can serve and the arena does not offer is refused rather than
# configured and never used.
ENGINES = ("qwen3.8-27b", "qwen3.6-27b", "qwen3.8-flash-next")
ENVIRONMENTS = ("slurm-b200", "kaggle-rtx")


def _known_boxes() -> tuple[str, ...]:
    """Every environment ullm can serve, for a refusal that says which kind of
    mistake was made — a typo, or a box ARC3 does not run on."""
    return tuple(_deployment().ENVIRONMENTS)


def _deployment() -> Any:
    """ullm's engine/environment tables, imported late (they pull in vllm)."""
    from ullm.backends import deployment as _deployment_module

    return _deployment_module


def _flashnext_kv_caches(engine_args: Any, *, block_size: int) -> None:
    """The KV caches ARC3 plays Flash-Next with, the duck's
    (``duck.presets._flashnext_kv_caches``, which says why each): the
    sparse-attention cache in INT4, the indexer's compressed-key cache in FP8,
    the compressor rings in their own pool, and the cell's block. The recurrent
    state stays bf16. Seal sweep 2 read INT4 in and the FP8 state out on its
    own play (2026-10-12)."""
    engine_args.kv_cache_dtype = "int4_per_token_head"
    engine_args.attention_config.indexer_kv_dtype = "fp8"
    engine_args.transient_kv_pool = True
    engine_args.block_size = block_size


# A per-game wall NET, for the ENDING rather than the detection: a game
# spending hours inside the kernel costs no tokens, so the per-level budget
# never retires it and the job's walltime CANCELS it, scoring nothing and
# taking its whole pass out of the analysis. Nine hours (2026-09-27):
# at full concurrency a game's wall stretches with its engine's load, and six
# began to end games that were still scoring (the duck's ``evidence.md``,
# "The per-game wall net").
GAME_RUNTIME_CAP_S = 9 * 60 * 60

# The token id of the ``<tool_call>`` marker in the Qwen3.8 and Qwen3.6
# tokenizers (one vocabulary of 248077; checked on both published repos),
# which the summarizing compaction bans while the model writes a summary: a
# summary that opens a tool call is a summary that ends the request as a
# tool turn. A fact about the checkpoints' tokenizer, so it rides the engine
# layer of ``configure`` and reaches no agent built for another model.
QWEN_TOOL_CALL_TOKEN_ID = 248058


# The share of each 27B cell's slots the seal runs: the cells' figures are the
# duck's, sized for the duck's window, and the seal's longer one
# (``SealAgent.context_window_tokens``) holds more cache a game, so fewer games
# fill the same pool and tier (2026-10-12). About the old budget over
# the new, as the 27Bs' slots were scaled when the duck's budget grew. The
# Flash-Next cells state the seal's own, for its own checkpoint.
WINDOW_SLOT_SHARE = 0.85

# The checkpoint the seal plays Flash-Next on, in place of ullm's e307:
# our joint205, REAP-307 for ARC-3 cut unevenly to fewer routed experts
# a layer, its routers, shared expert and attention trained with the cut. On
# one card it plays a little worse than e307, and its smaller weights make the
# card faster by more (2026-10-28; ``evidence.md``, "joint205's
# columns"). Drafted by the combined head below, in place of e307's drafter,
# which the sweeps played it beside.
JOINT205_CHECKPOINT = "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307-dials-joint205-u800"
# joint205 on Kaggle: its own part a private model, every tensor but those of
# a ten-shard Flash-Next's PLE-only shards, which it holds byte for byte and
# takes from another model's (``ullm.kaggle_assemble``), since the whole cut
# does not fit the account's room.
JOINT205_KAGGLE_MODEL = "jeroencottaar/qwen3-8-flash-next-nvfp4-joint205-u800/Transformers/default/1"
# Where joint205 takes those shards: our ARC2 REAP-307, whose four are
# byte for byte the ones joint205's index names. e307's model, the first
# source, was deleted (2026-11-02), and the ARC2 submission serves this one,
# so it stays on Kaggle as long as that submission does.
JOINT205_PLE_KAGGLE_MODEL = "driessmit1/qwen3-8-flash-next-nvfp4-arc2-reap307/PyTorch/nvfp4/1"
# joint205's drafter (2026-10-29): our combined ARC2/ARC3 MTP
# head, trained against joint205 and ARC2's REAP-307 with its own KV in INT4,
# the format the seal serves its drafter in, beside the main model's. In
# e307's drafter format (NVFP4 experts, a 16k draft vocabulary). On his replay
# check against joint205 it led every head at INT4 drafter KV, e307's v1 head
# in both formats (``arc_3_results/shared/drafter-final-20261029``). On Kaggle
# the account's model of it, its files byte for byte.
ARC23_INT4_DRAFTER = (
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc23-final-20261029-int4-drafter-nvfp4-dv16k"
)
ARC23_INT4_DRAFTER_KAGGLE_MODEL = "jeroencottaar/qwen3-8-flash-next-arc23-int4-mtp-nvfp4-dv16k/Transformers/default/1"


def _serve_joint205(served: Any, environment: str, *, checkpoint: str | None = None) -> None:
    """joint205 and its drafter in place of ullm's pair on a Flash-Next
    column; a ``checkpoint`` the caller named, already the cell's model
    (``configure``), in joint205's place, drafted by the same head."""
    deployment = _deployment()
    if environment == "kaggle-rtx":
        import ullm.kaggle

        deployment.serve_kaggle_checkpoint(
            served,
            ullm.kaggle.KaggleCheckpoint(JOINT205_KAGGLE_MODEL, shards_from=(JOINT205_PLE_KAGGLE_MODEL,)),
            ullm.kaggle.KaggleCheckpoint(ARC23_INT4_DRAFTER_KAGGLE_MODEL),
        )
    elif environment == "slurm-b200":
        engine_args = served.factory.engine_args
        if checkpoint is None:
            engine_args.model = JOINT205_CHECKPOINT
        engine_args.speculative_config = {**engine_args.speculative_config, "model": ARC23_INT4_DRAFTER}
    else:
        raise ValueError(f"no joint205 for environment {environment!r}: say where it mounts there")


def seal_run_config(**overrides: Any) -> RunConfig:
    """Every seal ``RunConfig``, so the wall net has one definition."""
    return replace(RunConfig(max_runtime_s_per_game=GAME_RUNTIME_CAP_S), **overrides)


def analyzer_sampling_profiles() -> dict[str, ullm.backend.SamplingParameters]:
    """The analyzer sampling every column shares: temperature, and a top-k
    cut kept for now.

    The nucleus and the presence penalty are off, stated rather than left to
    the engine's defaults (2026-09-26): Flash-Next's sweep 13 read the
    penalty as a cost where the grid travels as an image, and the nucleus at
    nothing (the duck's ``evidence.md``, "Sweeps 12 and 13"); both also cap a
    speculative drafter's acceptance, which vLLM samples at temperature
    alone. ``top_k`` 194 is sweep-3's value, kept until its own call.
    ``temperature`` is a round number where sweep 6 found the damage above it
    stops (the duck's ``evidence.md``).

    """
    return {
        "play": ullm.backend.GenericSamplingParameters(
            max_tokens=None,  # The agent stamps its cap on each call's copy.
            temperature=0.6,
            top_p=1.0,
            extra_params={"top_k": 194, "presence_penalty": 0.0},
        )
    }


def configure(
    agent: Any,
    *,
    engine: str = DEFAULT_ENGINE,
    environment: str = DEFAULT_ENVIRONMENT,
    serving: Literal["direct", "server"] = "direct",
    backend_factory: Any = None,
    checkpoint: str | None = None,
) -> None:
    """Configure a constructed seal for one engine in one environment.

    The layers apply in precedence order, top to bottom: what ARC3 asks
    everywhere, then the engine's, then the environment's, then the cell's.
    A later layer overwriting an earlier one is how this is meant to read —
    ``settings.md`` at the repo root says which layer owns what.

    Each chain is exhaustive over its own coordinate and refuses anything
    else, so a cell nobody has configured fails here rather than running on
    whatever the layer above happened to leave.

    ``backend_factory`` is the caller's, for a model this process does not
    start: an externally served one, or an engine they already hold. Left
    unset, the coordinates build it. ``ullm.backends.deployment`` supplies the
    engine facts a caller needs to construct one -- ``client_kwargs`` for a
    served model, ``backend_kwargs`` for an engine they hold.

    ``checkpoint`` is a checkpoint directory to play in place of the cell's,
    every other fact of the cell kept, on the cells ullm serves one on
    (``deployment.CHECKPOINT_CELLS``: Flash-Next on the B200), where it takes
    joint205's place beside the same drafter. The engine arguments are built
    around it, so an offline launcher (``HF_HUB_OFFLINE``) needs no Hub cache
    (``deployment.build``).
    """
    if engine not in ENGINES:
        raise ValueError(f"ARC3 has no engine {engine!r}; it runs {list(ENGINES)}.")
    if environment not in ENVIRONMENTS:
        elsewhere = " ullm can serve it, but ARC3 does not run there." if environment in _known_boxes() else ""
        raise ValueError(f"ARC3 has no environment {environment!r}; it runs in {list(ENVIRONMENTS)}.{elsewhere}")

    if backend_factory is not None and checkpoint is not None:
        raise ValueError(
            "checkpoint is for the engine the coordinates build; a supplied backend_factory serves its own."
        )
    if backend_factory is not None:
        # A model this process does not start brings no deploy assets: nothing
        # for a kernel to mount, no wheelhouse of ours behind it.
        agent.llm_config.backend_factory = backend_factory
    else:
        served = _deployment().build(engine, environment, serving=serving, checkpoint=checkpoint)
        if engine == "qwen3.8-flash-next":
            _serve_joint205(served, environment, checkpoint=checkpoint)
        agent.llm_config.backend_factory = served.factory
        if serving == "server":
            # The server topology continues a history over /v1/completions,
            # which takes no image input, so its column sends the board as text
            # alone (2026-09-27); an image asked of it after the build is
            # refused at ``prepare_for_backend``.
            agent.multimodal_context = ""
        # Deploy assets, from the environment that mounts them.
        agent.run_config.kaggle_engine = served.kaggle_engine
        agent.run_config.kaggle_dataset_sources = served.kaggle_dataset_sources
        agent.run_config.kaggle_model_sources = served.kaggle_model_sources

    # --- P-00: what ARC3 asks of any engine anywhere ------------------
    # Every column serves vLLM, which answers token_count exactly (the
    # id-ledger length on continued calls, its own render otherwise), so the
    # context decision runs on that weight rather than the bill+bound
    # arithmetic cloud backends get -- and it continues a served history by
    # the sampled ids, so the seal can promise its steps nest (U5): an
    # undeclared rewrite is an error, not a quiet cache miss.
    agent.token_native_context = True
    agent.llm_config.exact_history = True
    # The engine's per-step token budget. Run high: this task trades latency
    # for prefill throughput, so several prefills advance per step instead of
    # one blocking the rest. What it costs is activation workspace out of the
    # headroom the KV pool is left, which makes the pool the engine profiles
    # at startup the number to read after a change. An engine in ``align``
    # mode constrains it further: a prefill chunk must end on a recurrent
    # block boundary.
    engine_args = getattr(agent.llm_config.backend_factory, "engine_args", None)
    if engine_args is not None:
        # The window the engine was built with, before the agent settles its
        # own onto it (``prepare_for_backend``): the family's native context,
        # which ``compaction_strategy="summarize"`` runs under.
        agent.engine_ceiling_tokens = engine_args.max_model_len
        engine_args.max_num_batched_tokens = 8192
        # The concurrency is set above what the pool holds, so contexts live
        # in the host tier between turns and come back through it rather than
        # being recomputed.
        agent.llm_config.backend_factory.cpu_kv_offload = True

    # --- P-E0: the engine ---------------------------------------------
    if engine == "qwen3.8-27b":
        # The swept level, and one only this family's template renders. The
        # class default is provider-neutral because it reaches whatever backend
        # a hand-built seal is paired with; a level the template has no branch
        # for makes the engine refuse the request rather than fall back.
        agent.reasoning_effort = "xhigh"
    elif engine == "qwen3.6-27b":
        pass  # no xhigh in its template; the agent's own default stands
    elif engine == "qwen3.8-flash-next":
        agent.reasoning_effort = "xhigh"  # Qwen3.8's template, so its levels
        # No served opening on this model: its runs so far were played
        # without one, and the register it sets halves what the model
        # reasons per action here.
        agent.inject_text_on_first_prompt = ""
        agent.inject_text_on_later_prompts = ""
    else:
        raise ValueError(
            f"no ARC3 engine layer for {engine!r}. An engine added to ENGINES says what the task asks "
            "of it here, even if that is nothing, rather than inheriting another engine's."
        )
    # All three checkpoints share the tokenizer, so the marker banned while a
    # summary is written is one id for the family.
    agent.compaction_forbidden_token_ids = (QWEN_TOOL_CALL_TOKEN_ID,)

    # --- P-0V: the environment ----------------------------------------
    if environment == "slurm-b200":
        pass  # the standard column; every layer above it already applies
    elif environment == "kaggle-rtx":
        # A submission: its own clock bounds the run, so cutting a game early
        # would forfeit score the arena would still have counted.
        agent.run_config.max_runtime_s_per_game = None
        # More games than slots against a deadline is the one shape where who
        # holds a slot is worth asking. A per-level cap frees a slot by ENDING
        # a game and forfeiting what it would still have scored; ranking frees
        # the same slot by deprioritising, at no cost -- so the cap comes off
        # exactly where the ranking is in play. Ranked by each game's whole
        # spend, as the duck's, so the games share the machine evenly
        # (2026-10-12; ``settings.md``, P-0V, says why).
        agent.run_config.slot_policy = "least_game_tokens"
        agent.max_tokens_per_level = None
        # The same notebook submitted repeatedly varies its play order.
        agent.run_config.shuffle_games = True
        # A replay the game does not reproduce leaves the game playing on from
        # its real state rather than crashing it: a submission scores what it
        # can (``platypus.replay_shadow``).
        agent.run_config.replay_shadow_mismatch = "continue"
        # Games start 10 s apart rather than 0.5 s: a cohort started together
        # builds its contexts together, and the KV peak of that launch wave
        # is what spreading the starts flattens.
        agent.run_config.start_gap_s = 10.0
        # The confinement guards the cluster's readings and its shared
        # storage; a submission needs neither (2026-10-01).
        agent.confine_filesystem = False
        # An agent that sends its frames as text alone leaves the vision tower
        # on both of these checkpoints as weight the card holds and never
        # reads, so it is dropped for one. Dropped HERE and not on the
        # cluster: the tower is a capability worth keeping reachable on a box
        # with room to play with it, and this is the box where what it costs
        # is a live constraint. An agent sending the grid as an image keeps
        # it, and one asking for an image after a build that dropped it is
        # refused at ``prepare_for_backend``.
        engine_args = getattr(agent.llm_config.backend_factory, "engine_args", None)
        if engine_args is not None and not agent.multimodal_context.strip():
            _deployment().drop_vision_tower(engine_args)
    else:
        raise ValueError(
            f"no ARC3 environment layer for {environment!r}. An environment added to ENVIRONMENTS says "
            "how a run is scheduled there, even if that is nothing, rather than inheriting the "
            "standard column's."
        )

    # --- P-EV: the cell -----------------------------------------------
    cell = (engine, environment)
    if cell in (("qwen3.8-27b", "slurm-b200"), ("qwen3.6-27b", "slurm-b200")):
        # Sized to saturate the engine rather than to fill the pool: generated
        # tokens per second stop rising once the running set is deep enough,
        # and this is past that point while still above what the pool holds.
        # 80 scaled with the mean context to the 125,000-token prompt budget:
        # times 56,832 / 125,000, the old budget over the new, since the
        # context per generated token runs a steady share of the budget
        # (the duck's ``evidence.md``, "Sweeps 12 and 13"), so the cache the games hold
        # stays where it was (2026-09-27).
        agent.run_config.max_concurrent = 36
    elif cell in (("qwen3.8-27b", "kaggle-rtx"), ("qwen3.6-27b", "kaggle-rtx")):
        # 40 scaled with the mean context to the 125,000-token prompt budget:
        # times 56,832 / 125,000, the old budget over the new, since the
        # context per generated token runs a steady share of the budget
        # (the duck's ``evidence.md``, "Sweeps 12 and 13"), so the cache the games hold
        # stays where it was (2026-09-27).
        agent.run_config.max_concurrent = 18
        # Claimed eagerly and counted against the same cgroup as the harness
        # and its game processes, so it takes what those leave. On our own
        # engine alone: a caller who supplied a backend owns its memory, and
        # the field is not even on their factory.
        if engine_args is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = 60.0
    elif cell == ("qwen3.8-flash-next", "slurm-b200"):
        # Sized to keep the pool full with a standing queue: at the line the
        # engine runs what the pool holds at these context lengths and admits
        # from the queue as contexts finish, the tier carrying what leaves.
        # Bounded by the tier, not the pool: the tier holds the queue, and a
        # queue it cannot hold is evicted and recomputed on every return. 150
        # came from a long run on the float32 state whose tier held through
        # the first compaction wave, the slot counts above it saturating the
        # tier within the hour; 200 is that scaled to the bf16 state's larger
        # pool (ullm's engine; 2026-09-21), and 180 that scaled again
        # to the pool drafting leaves (7.95M -> 7.08M tokens, 0.89;
        # 2026-09-26). The job asks for 32 GiB more with it
        # (``notebooks/demo.ipynb``).
        # Scaled with the mean context when the prompt budget went from 56,832
        # to 125,000 (2026-09-27), so the cache the games hold stays
        # where it was: context per generated token runs a steady 0.59 of the
        # budget (the duck's ``evidence.md``, "Sweeps 12 and 13"), held in
        # 2,112-token blocks plus about five recurrent-state blocks a request
        # whatever its length, so a game's blocks go from 20.9 to 39.9 (x 0.523).
        # 100 under park mode (2026-09-28): at 94 the pool bound in
        # 43% of minutes (about 73 running when bound, every slot when not),
        # while 120 lanes of the traffic replay overflowed the 96 GiB tier
        # with some 52 parked threads of ~1.5 GiB each.
        # The seal's own for joint205 (2026-10-28): that figure, set
        # on e307 with the FP8 cache, scaled to the INT4 caches' tokens per
        # byte, which the tier's parked threads hold too, to joint205's larger
        # pool and to the seal's window share. Derived rather than run
        # (``evidence.md``, "joint205's columns").
        agent.run_config.max_concurrent = 167
        # Games start 12 s apart, so the opening fill of these slots spreads
        # over 20 minutes and the games reach their compactions at different
        # times rather than in one wave (2026-09-28).
        agent.run_config.start_gap_s = 12.0
        # Below the duck's, sized from the tier's reuse depth (``kv_offload.md``,
        # "Sizing the tier") on the seal's own play (2026-10-09;
        # ``evidence.md``, "The engine columns", holds the reading). The job
        # asks for the memory on top. Our own engine alone, as on every cell
        # that sizes the tier.
        if engine_args is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = 64.0
        # The KV caches at this cell's block. The step budget stays P-00's:
        # at four drafts this cell's decodes alone would fill the RTX cell's
        # (``packages/ullm/flash_optimization.md``, "512 tokens a step").
        if engine_args is not None:
            _flashnext_kv_caches(engine_args, block_size=3_120)
        # Games here play about as many tokens as they do in a Kaggle
        # submission, which lifts the cap and is bounded by its clock: this is
        # where settings are tuned and Kaggle is where they are scored, so the
        # two should play games of the same length. Eyeballed against the
        # Kaggle runs, not derived; the per-level cap stays the
        # mechanism, only its size is this cell's.
        agent.max_tokens_per_level = (175_000,)
        # The cache-hit unit is the engine layer's (ullm's
        # ``_flashnext_config``), with the decode tail and context threads that
        # keep conversations live; the block above is a multiple of it. What a
        # unit buys on this box: ``flash_optimization.md``, "Cache hits inside
        # a block".
    elif cell == ("qwen3.8-flash-next", "kaggle-rtx"):
        # Held a little above what the pool runs, so the pool stays full with a
        # short standing queue, and no further: past the pool's line extra
        # slots only queue in the tier, which shares the cgroup with the PLE
        # table here, and they dilute the slot ranking, which pays by choosing
        # the most promising games for the slots there are. 60 sat over the
        # float32 state's ~52 running; the bf16 state (ullm's engine) runs
        # ~66 on the traffic replay, and 80 over it kept the queue at 0-15
        # (2026-09-20). Drafting leaves ~0.85 of that pool (2.64M -> ~2.23M
        # tokens), and 68 is 80 scaled with it: drafted at 80 slots the
        # replay ran ~52 with 11-26 waiting (2026-09-26).
        # Scaled with the mean context when the prompt budget went from 56,832
        # to 125,000 (2026-09-27), so the cache the games hold stays
        # where it was: context per generated token runs a steady 0.59 of the
        # budget (the duck's ``evidence.md``, "Sweeps 12 and 13"), held in
        # 1,840-token blocks plus about five recurrent-state blocks a request
        # whatever its length, so a game's blocks go from 23.2 to 45.1 (x 0.515).
        # 40 (2026-09-29): at 35, the 9-hour runs (Kaggle E and F)
        # queued for capacity in only 77-83% of their minutes, with KV at
        # 87-88% even in the rest. Park mode refreshing only the siblings a
        # thread needs frees the tier room the extra parked threads take.
        # 80 with the INT4 caches, the rings' pool and 512-token steps below
        # (2026-10-12), a little above what the pool then runs on the
        # traffic replay (``packages/ullm/flash_optimization.md``, "The RTX
        # column on the traffic replay").
        # The seal's own for joint205, to revise on a run
        # (2026-10-28): above that figure at the seal's window share, short of
        # it scaled with joint205's larger pool (``evidence.md``, "joint205's
        # columns").
        agent.run_config.max_concurrent = 80
        # Games start 30 s apart, so the opening fill of these slots spreads
        # over 20 minutes and the games reach their compactions at different
        # times rather than in one wave (2026-09-28).
        agent.run_config.start_gap_s = 30.0
        # Three drafts where the engine layer drafts four: the official games
        # are easier than the private set, and lower acceptance erodes four
        # drafts' edge first (2026-09-29; ``flash_optimization.md``,
        # "Speculative decoding"). The cache-hit unit is the one read on this
        # card ("Cache hits inside a block"); the block below is a multiple of
        # it.
        if engine_args is not None:
            engine_args.speculative_config["num_speculative_tokens"] = 3
            engine_args.prefix_match_unit = 64
        # The KV caches at this cell's block, and steps of at most the largest
        # CUDA graph this card captures, so none runs eagerly
        # (``packages/ullm/flash_optimization.md``, "512 tokens a step").
        if engine_args is not None:
            _flashnext_kv_caches(engine_args, block_size=3_008)
            engine_args.max_num_batched_tokens = 512
        # A balance of the tier's robustness against host memory, which this
        # engine spends twice: the PLE table pins 47.68 GiB and cannot be
        # moved, since the checkpoint is larger than the card, and the tier
        # sits beside it in the cgroup the game processes grow in. Sized
        # from the tier's trace in the job directory, whose reuse-depth line
        # says how deep a hit actually lands at this concurrency, so a change
        # of concurrency moves it (``packages/ullm/flash_optimization.md``,
        # "Host memory", holds the reading).
        if engine_args is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = 50.0
    else:
        raise ValueError(
            f"no column for engine {engine!r} in environment {environment!r}. Concurrency and tier "
            "size are measured on the pair, so a cell nobody has run states its own rather than "
            "inheriting a neighbour's figures."
        )

    # The seal's longer window: a share of the cell's slots where they are the
    # duck's.
    if engine != "qwen3.8-flash-next":
        agent.run_config.max_concurrent = round(agent.run_config.max_concurrent * WINDOW_SLOT_SHARE)
    agent.llm_config.sampling_profiles = analyzer_sampling_profiles()
