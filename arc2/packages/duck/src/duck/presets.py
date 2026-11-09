"""What ARC2 asks of a deployment, over ullm's engine/environment coordinates.

``ullm.backends.deployment`` holds the task-free half — which checkpoint, on
which box, and what the two together decide about memory. This holds the
duck's half: the analyzer sampling, the context and budget shape, the
concurrency and the scheduling. :func:`configure` walks them onto a constructed
agent, one column at a time, so the agent's own file stays about how the duck
plays. Nothing here is model-visible except the sampling profile.

Which half a setting belongs to, and which of the two coordinates decides it,
is ``settings.md`` at the repo root.

Every vllm import is function-local — building engine args pulls in vllm, and
CPU-only boxes import ``duck`` to inspect an agent.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal

import ullm.backend
import ullm.kaggle

from platypus import RunConfig

# The coordinates a bare ``DuckAgent.build()`` serves.
DEFAULT_ENGINE = "qwen3.8-flash-next"
DEFAULT_ENVIRONMENT = "slurm-b200"

# Every column ARC2 runs. Narrower than ullm's: ullm describes boxes and
# checkpoints, this says which of them a run may happen on, so a coordinate
# ullm can serve and this task does not run on is refused rather than
# configured and never used. Two 27B checkpoints of one family, which differ
# in what the kernel mounts and in their effort scale and nowhere the pair
# decides, and Flash-Next, which does differ and states its own figures; the
# cells are keyed by the pair regardless.
ENGINES = ("qwen3.8-27b", "qwen3.6-27b", "qwen3.8-flash-next")
ENVIRONMENTS = ("slurm-b200", "kaggle-4xl4")

# The wheelhouse an ARC2 kernel installs the engine from: ullm's wheels, with
# the record of installing them on an ARC2 kernel (P-0V).
KAGGLE_WHEELS = "jeroencottaar/arc2-site-0290"


def _known_boxes() -> tuple[str, ...]:
    """Every environment ullm can serve, for a refusal that says which kind of
    mistake was made — a typo, or a box this task does not run on."""
    return tuple(_deployment().ENVIRONMENTS)


def _deployment() -> Any:
    """ullm's engine/environment tables, imported late (they pull in vllm)."""
    from ullm.backends import deployment as _deployment_module

    return _deployment_module


# our REAP-307 of the unpruned Flash-Next: its routed experts cut to 307
# on the model's own ARC-2 play. It has e307's format (ModelOpt NVFP4, 307
# experts), so it moves only the path off ullm's e307. ARC2's own: ARC3 reads
# its cuts separately. Sweep 9 read it +1.24 over e307 (p=0.006) at the same
# tokens (``evidence.md``, "Duck sweeps 8 and 9").
E307_ARC2_CHECKPOINT = "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc2-reap307"
# The same on Kaggle, as a Kaggle model: the checkpoint whole, as our
# Kaggle model of ARC3's REAP-307 holds that one, named where his ARC2 upload is
# to land (2026-10-28: not up yet, so a kernel cannot mount it until it
# is).
E307_ARC2_KAGGLE = ullm.kaggle.KaggleCheckpoint("driessmit1/qwen3-8-flash-next-nvfp4-arc2-reap307/PyTorch/nvfp4/1")
# The drafter (2026-10-29): our combined ARC2/ARC3 MTP head, trained
# against this checkpoint and ARC3's with its own KV in FP8, the format this
# task serves its drafter in, beside the main model's. In e307's drafter format
# (NVFP4 experts, a 16k draft vocabulary). On his replay check it led the
# INT4-trained head on ARC2 in both drafter KV formats and drew level with
# e307's v1 head (``arc_3_results/shared/drafter-final-20261029``); against
# REAP-307's own head it reads level in Kaggle real play.
# On Kaggle the account's model of it, its files byte for byte.
ARC23_FP8_DRAFTER = (
    "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc23-final-20261029-fp8-drafter-nvfp4-dv16k"
)
ARC23_FP8_DRAFTER_KAGGLE = ullm.kaggle.KaggleCheckpoint(
    "jeroencottaar/qwen3-8-flash-next-arc23-fp8-mtp-nvfp4-dv16k/Transformers/default/1"
)


# A per-game wall NET, for the ENDING rather than the detection: a game
# spending hours inside the kernel costs no tokens, so the per-game budget
# never retires it and the job's walltime CANCELS it, scoring nothing and
# taking its whole pass out of the analysis. Eight hours (2026-10-06):
# clear of the wall clock a game needs to reach its token budget, which the
# SLURM column's concurrency and token cap both stretch -- at 500k a game and
# the column's slots in flight, a game generates a few tens of tokens a
# second -- and inside the twelve hours a B200 job gets on ``dev``.
#
# Off on Kaggle, where the submission's own clock is the only budget that
# matters — see the environment layer of ``configure``.
GAME_RUNTIME_CAP_S = 8 * 60 * 60

# The token id of the ``<tool_call>`` marker in the Qwen3.8 and Qwen3.6
# tokenizers (one vocabulary of 248077; checked on both published repos),
# which the summarizing compaction bans while the model writes a summary: a
# summary that opens a tool call is a summary that ends the request as a
# tool turn. A fact about the checkpoints' tokenizer, so it rides the engine
# layer of ``configure`` and reaches no agent built for another model.
QWEN_TOOL_CALL_TOKEN_ID = 248058


def duck_run_config(**overrides: Any) -> RunConfig:
    """Every duck ``RunConfig``, so the wall net has one definition.

    Kaggle passes ``max_runtime_s_per_game=None`` through here to turn it off,
    which reads as the deliberate choice it is.
    """
    return replace(RunConfig(max_runtime_s_per_game=GAME_RUNTIME_CAP_S), **overrides)


def analyzer_sampling_profiles() -> dict[str, ullm.backend.GenericSamplingParameters]:
    """The analyzer sampling every column shares: the whole distribution past
    ``top_k``, at 0.8, with no presence penalty.

    ``top_p`` 1.0 (2026-10-06): drafting and training on the duck's
    own play are easier against the plain distribution, and nothing showed a
    cost -- sweep 8 read 1.0 against upstream's 0.889 at +0.15 (p=0.81).
    ``top_k`` is upstream's sweep-3 best arm, kept at full precision because it
    is a validated coordinate rather than an aesthetic. The presence penalty is
    off: upstream shipped 1.5 against rumination, and ARC2's sweep 5 read the
    knob at -10.7 points over 0 to 1.45 (p=0.003), monotone, with no token
    saving -- the shipped end was the worst. It stays in ``extra_params`` at
    0.0 so a sweep knob has a field to reach. ``temperature`` 0.8, for every
    phase: sweep 9 read the play's reasoning flat to 0.9 and falling past it
    (-3.4 at 0.9-1.0), and from 0.6 to 0.8 it spends 1.6k fewer effective
    tokens a game at the same score (+0.50 throughput-adjusted, p<0.001); the
    other four temperatures it drew read level (``evidence.md``, "Duck sweeps
    8 and 9").
    """
    return {
        "play": ullm.backend.GenericSamplingParameters(
            max_tokens=None,  # The agent stamps its cap on each call's copy.
            temperature=0.8,
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
) -> None:
    """Configure a constructed duck for one engine in one environment.

    The layers apply in precedence order, top to bottom: what ARC2 asks
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
    """
    if engine not in ENGINES:
        raise ValueError(f"ARC2 has no engine {engine!r}; it runs {list(ENGINES)}.")
    if environment not in ENVIRONMENTS:
        elsewhere = " ullm can serve it, but ARC2 does not run there." if environment in _known_boxes() else ""
        raise ValueError(f"ARC2 has no environment {environment!r}; it runs in {list(ENVIRONMENTS)}.{elsewhere}")

    if backend_factory is not None:
        # A model this process does not start brings no deploy assets: nothing
        # for a kernel to mount, no wheelhouse of ours behind it.
        agent.llm_config.backend_factory = backend_factory
    else:
        served = _deployment().build(engine, environment, serving=serving)
        agent.llm_config.backend_factory = served.factory
        # Deploy assets, from the environment that mounts them.
        agent.run_config.kaggle_engine = served.kaggle_engine
        agent.run_config.kaggle_dataset_sources = served.kaggle_dataset_sources
        agent.run_config.kaggle_model_sources = served.kaggle_model_sources

    # --- P-00: what ARC2 asks of any engine anywhere ------------------
    # The live summary rather than pruning (sweep 6: +11.4 over prune, with
    # about 11 points fewer give-ups; ``evidence.md``). Here rather
    # than as the class default because a summary needs a real reply
    # reserve, which every column's cap gives it and an agent built
    # uncapped does not have.
    agent.compaction_strategy = "summarize"
    # Every column serves vLLM, which answers token_count exactly (the
    # id-ledger length on continued calls, its own render otherwise), so the
    # context decision runs on that weight rather than the bill+bound
    # arithmetic cloud backends get -- and it continues a served history by
    # the sampled ids, so the duck can promise its steps nest: an undeclared
    # rewrite is an error, not a quiet cache miss.
    agent.token_native_context = True
    agent.llm_config.exact_history = True
    engine_args = getattr(agent.llm_config.backend_factory, "engine_args", None)
    if engine_args is not None:
        # The window the engine was built with, before the agent settles its
        # own onto it (``prepare_for_backend``): the family's native context,
        # which ``summarize`` runs under.
        agent.engine_ceiling_tokens = engine_args.max_model_len
        # The concurrency is set above what the pool holds, so contexts live
        # in the host tier between turns and come back through it rather than
        # being recomputed.
        agent.llm_config.backend_factory.cpu_kv_offload = True

    # --- P-E0: the engine ---------------------------------------------
    if engine == "qwen3.8-27b":
        # The swept level, and one only this family's template renders. The
        # class default is provider-neutral because it reaches whatever backend
        # a hand-built duck is paired with; a level the template has no branch
        # for makes the engine refuse the request rather than fall back.
        agent.reasoning_effort = "xhigh"
    elif engine == "qwen3.6-27b":
        pass  # no xhigh in its template; the agent's own default stands
    elif engine == "qwen3.8-flash-next":
        agent.reasoning_effort = "xhigh"  # Qwen3.8's template, so its levels
        # No served opening on this model: its runs so far were played
        # without one, and the register it sets halves what the model
        # reasons per action there. The opening's reading here is the 27B's.
        agent.inject_text_on_first_prompt, agent.inject_text_on_later_prompts = ("", "")
        if engine_args is not None:
            # The sparse-attention indexer's compressed keys in plain fp8
            # rather than bf16: half the bytes a cached key, the drafter's
            # layer included. +5.9% generated and +7.1% pool on the four
            # L4s' traffic replay, and no score cost in sweeps 8 and 9 (-0.10
            # in sweep 9, n.s.; ``evidence.md``, "Duck sweeps 8 and 9"). The
            # weights are untouched.
            engine_args.attention_config.indexer_kv_dtype = "fp8"
    else:
        raise ValueError(
            f"no ARC2 engine layer for {engine!r}. An engine added to ENGINES says what the task asks "
            "of it here, even if that is nothing, rather than inheriting another engine's."
        )
    # All three checkpoints share the tokenizer, so the marker banned while a
    # summary is written is one id for the family.
    agent.compaction_forbidden_token_ids = (QWEN_TOOL_CALL_TOKEN_ID,)

    # --- P-0V: the environment ----------------------------------------
    if environment == "slurm-b200":
        pass  # the standard column; every layer above it already applies
    elif environment == "kaggle-4xl4":
        # A submission: its own clock bounds the run, so cutting a game early
        # would forfeit score the arena would still have counted.
        agent.run_config.max_runtime_s_per_game = None
        # More games than slots against a deadline is the one shape where who
        # holds a slot is worth asking: the machine follows the games that
        # have spent little, and a game grinding its task slides behind the
        # queue. Only here -- a run that fits its games in its slots would
        # just be paying for the swaps.
        agent.run_config.slot_policy = "least_tokens"
        # The per-game cap stays, as a backstop rather than the allocator: the
        # ranking frees a slot by deprioritising, so a game that has spent its
        # way to this has already been behind every cheaper game for a while,
        # and the cap only ends what the queue would otherwise starve. Above
        # the 65k the SLURM column measures at, since a submission has no
        # wall net under it. Upstream takes the cap off where the ranking is
        # in play; the twelve-hour runs here fired it 18 and 29 times of 240
        # (`evidence.md`, "The narrowed context window"), which is the reading
        # that keeps it.
        agent.max_tokens_per_game = 70_000
        # The same notebook submitted repeatedly varies its play order.
        agent.run_config.shuffle_games = True
        # No watchdog over the engine (2026-11-01, for the final
        # submission): a stall it ended would forfeit the rest of the clock
        # had the engine come back, and a dead engine adds no score whether
        # the run stops at once or at its soft deadline.
        agent.run_config.engine_stall_timeout_s = None
        # Games start 10 s apart rather than 0.5 s: a cohort started together
        # builds its contexts together, and the KV peak of that launch wave
        # is what spreading the starts flattens.
        agent.run_config.start_gap_s = 10.0
        # ullm's wheelhouse records its install after the ARC3 competition's
        # `arc-agi` install, which an ARC2 kernel never runs (its competition
        # mount is data only), so an ARC2 kernel refuses that record and
        # installs with pip. ARC2's holds the same wheels with the record an
        # ARC2 kernel makes.
        if agent.run_config.kaggle_engine is not None:
            agent.run_config.kaggle_engine = replace(agent.run_config.kaggle_engine, wheels=KAGGLE_WHEELS)
    else:
        raise ValueError(
            f"no ARC2 environment layer for {environment!r}. An environment added to ENVIRONMENTS says "
            "how a run is scheduled there, even if that is nothing, rather than inheriting the "
            "standard column's."
        )

    # --- P-EV: the cell -----------------------------------------------
    cell = (engine, environment)
    if cell in (("qwen3.8-27b", "slurm-b200"), ("qwen3.6-27b", "slurm-b200")):
        # Sized to saturate the engine rather than to fill the pool: generated
        # tokens per second stop rising once the running set is deep enough,
        # and this is past that point while still above what the pool holds.
        agent.run_config.max_concurrent = 80
        # The engine's per-step token budget, stated per cell because it
        # follows the engine's recurrent block on the pair. In ``align`` mode
        # a prefill chunk ends on a block boundary, so the budget says how
        # many blocks of prefill advance per step: run high, since this task
        # trades latency for prefill throughput, at the cost of activation
        # workspace out of the KV pool's headroom. This engine's block is
        # 1,568, so 8,192 is five of them. A caller-supplied engine
        # (``engine_args`` is None then) states its own.
        if engine_args is not None:
            engine_args.max_num_batched_tokens = 8192
    elif cell in (("qwen3.8-27b", "kaggle-4xl4"), ("qwen3.6-27b", "kaggle-4xl4")):
        agent.run_config.max_concurrent = 40
        # As on the B200 cell.
        if engine_args is not None:
            engine_args.max_num_batched_tokens = 8192
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
        # (``notebooks/demo.ipynb``). Under the tier's park mode the GPU holds
        # running conversations only, and at 180 every slot ran with the KV
        # cache at 70-79% and never a capacity wait (the 150k check,
        # 2026-09-28). 180 for now (2026-10-16): the value is being
        # tuned separately, in sweep work.
        agent.run_config.max_concurrent = 180
        # A third above the shipped size, for the queue that concurrency
        # leaves outside the pool. The job asks for the memory on top. Our
        # own engine alone, as on every cell that sizes the tier.
        if engine_args is not None:
            # The REAP-307 checkpoint and its drafter, from shared storage.
            engine_args.model = E307_ARC2_CHECKPOINT
            engine_args.speculative_config = {**engine_args.speculative_config, "model": ARC23_FP8_DRAFTER}
            agent.llm_config.backend_factory.cpu_kv_offload_gib = 96.0
            # Below the engine layer's budget: at 230 sequences, with the
            # memory profiler's own graph capture holding its estimate back
            # from the pool, 0.97 ran out of memory by a few MB in the CUDA
            # graphs' capture after the pool is sized (the long run,
            # 2026-09-28; 300 lanes at 0.97 before it), and 0.95 started and
            # ran (2026-09-28). The engine layer captures the graphs
            # once, so the budget is 0.95 less that estimate, as the long
            # run's boot line names it at 230 sequences: 0.9174, which the
            # column's fewer sequences, and so smaller graphs, leave more room
            # under. ullm's B200 cell's W4A16 MoE takes more in those graphs,
            # at 230 sequences inside what this leaves above where 0.97 failed
            # (arc3-platypus ``flash_optimization.md``, "The MoE backend");
            # unbooted.
            engine_args.gpu_memory_utilization = 0.9174
        # A cluster figure: past what most games spend, and well short of the
        # million the sweeps let a game run to. On the 1,200-game eval run at
        # this column's settings, 8 games went past it, and what they spent
        # past it was 1.4% of the run's tokens (``evidence.md``, "Duck sweeps 8
        # and 9"). At this concurrency a game generates a few tens of tokens a
        # second, so a long game takes hours to reach it, which the wall net
        # (``GAME_RUNTIME_CAP_S``) leaves room for. Kaggle, where the score is
        # what counts and the ranking already deprioritises a long game, runs
        # uncapped (see its cell).
        agent.max_tokens_per_game = 500_000
        # The budget, as on the 27B cells (see there). This engine's block is
        # 3,136 on one card: 8,192 holds two blocks and the decodes.
        if engine_args is not None:
            engine_args.max_num_batched_tokens = 8192
    elif cell == ("qwen3.8-flash-next", "kaggle-4xl4"):
        # 64 (2026-10-28), set with ullm's cell at utilization 0.935.
        # The twelve-hour eval runs on the drafted
        # cell (2026-10-04 and 06) read the pool at 30-37 requests, since the
        # drafter keeps its own KV for every token, where 50 kept a standing
        # queue of about 8-10 at the pool-bound p10. The pool has grown since
        # (512 tokens a step, the rings' own pool, FULL decode graphs alone at
        # a higher utilization): on the 230-game replay at 50 the running set
        # reached the slots with a few waiting (run km2), so more slots give the
        # pool its queue back, over the smaller resident set 50 left the slot
        # ranking to choose the most promising game from. Slots gate the
        # engine only: every game's process runs from the start, so this is
        # not a host-memory figure (the tier's size is).
        agent.run_config.max_concurrent = 64
        # The REAP-307 checkpoint and its drafter, as on the cluster, from
        # their Kaggle models, which the kernel mounts in place of ullm's pair.
        if engine_args is not None:
            served = _deployment().Deployment(
                factory=agent.llm_config.backend_factory,
                kaggle_engine=agent.run_config.kaggle_engine,
                kaggle_dataset_sources=agent.run_config.kaggle_dataset_sources,
                kaggle_model_sources=agent.run_config.kaggle_model_sources,
            )
            _deployment().serve_kaggle_checkpoint(served, E307_ARC2_KAGGLE, ARC23_FP8_DRAFTER_KAGGLE)
            agent.run_config.kaggle_engine = served.kaggle_engine
            agent.run_config.kaggle_dataset_sources = served.kaggle_dataset_sources
            agent.run_config.kaggle_model_sources = served.kaggle_model_sources
        # 512 tokens a step (2026-10-28), under the drafted cell's
        # 1,840-token block: a prefill advances in chunks of the budget, which
        # the fork's scheduler admits beside the decodes. The step's smaller
        # activation workspace goes to the pool, 5.96 GiB a card against 5.37
        # at 3,300, and the drafted replay at 512 (run Z, FP8 KV, the
        # 230-game workload) generated 2.3-2.7% more than those at 3,300 past
        # what that pool alone gives (arc2-experiments
        # ``arm_budget/box.py``, the box arm sweep 2 was read on).
        if engine_args is not None:
            engine_args.max_num_batched_tokens = 512
        # The compressor rings in a block pool of their own
        # (2026-10-28): a ring holds a few kilobytes but takes a whole shared
        # block for every running request, about a twentieth of a request's
        # pool at the cell's reference context. The pool holds four rings a
        # sequence in each ring group and needs the V2 model runner, which
        # every Flash-Next cell runs.
        if engine_args is not None:
            engine_args.transient_kv_pool = True
        # The host tier holds the conversations parked between requests. The
        # same runs' ``kv_park.log`` (at 60 slots and 60 GiB, 1,953 slots of
        # tier) read the deepest key a resume needed at p99 ~1,150 slots,
        # parked threads holding at most 861 (~26 GiB), and no parked thread
        # evicted; a 1,200-slot tier (37 GiB) would have re-prefilled part of
        # 0.1-0.3% of resumes. 45 GiB keeps headroom for longer games over
        # that, and the host needs the rest: the PLE table pins ~48 GB across
        # the workers and cannot be moved, since the checkpoint is larger than
        # the cards, and at 60 the unreclaimable memory sat at 95-99% of the
        # 173 GiB cgroup through the pool-bound hours, where a spike from the
        # games killed the engine.
        if engine_args is not None:
            agent.llm_config.backend_factory.cpu_kv_offload_gib = 45.0
        # No per-game cap, over the environment layer's backstop: the ranking
        # frees a slot by deprioritising a game, and this engine spends fewer
        # tokens per solve than the 27B the backstop was measured on, so a cap
        # here would only end games the queue was already handling. The
        # twelve-hour run bears that out -- it finished 236 of 240 games with
        # the engine idle for its last hours, and every cap costs score there
        # (upstream's ``flash_optimization_arc2.md``, "What a per-game cap
        # costs"). Playing every task in takes brings none back either: the
        # ranking's handicaps between takes (``RunConfig.take_handicap_tokens``,
        # and ``TieredTakeAgent``'s categories) are what balance them.
        agent.max_tokens_per_game = None
    else:
        raise ValueError(
            f"no column for engine {engine!r} in environment {environment!r}. Concurrency and tier "
            "size are measured on the pair, so a cell nobody has run states its own rather than "
            "inheriting a neighbour's figures."
        )

    agent.llm_config.sampling_profiles = analyzer_sampling_profiles()


__all__ = [
    "ARC23_FP8_DRAFTER",
    "ARC23_FP8_DRAFTER_KAGGLE",
    "DEFAULT_ENGINE",
    "DEFAULT_ENVIRONMENT",
    "E307_ARC2_CHECKPOINT",
    "E307_ARC2_KAGGLE",
    "ENGINES",
    "ENVIRONMENTS",
    "GAME_RUNTIME_CAP_S",
    "QWEN_TOOL_CALL_TOKEN_ID",
    "analyzer_sampling_profiles",
    "configure",
    "duck_run_config",
]
