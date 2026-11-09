"""ARC-3 task construction and outcome translation."""

from dataclasses import replace
import os
from pathlib import Path

from harness_support.agent_settings import attach_agent_settings, record_agent_settings
from harness_support.augmentation import arc3_game
from harness_support.arc3.results import (
    enrich_arc3_result,
    progress_snapshot as progress_snapshot,
)
from training.rollout_slots import _rng_seed_per_game, _slot_game_ids


def build_benchmark(spec: dict):
    """Construct ordinary ARC-3 games through the shared training setup."""
    from duck.agent import DuckAgent
    from serving_contract import (
        ContractLLMAgentSolver,
        install_contract_backend,
    )
    from taaf.benchmark import Benchmark
    from taaf.game_api import GameAPI

    names = list(spec["games"])
    game_ids = _slot_game_ids(names)
    slot_seeds = spec.get("slot_seeds")
    slot_rng_seeds = (
        _rng_seed_per_game(game_ids, slot_seeds) if slot_seeds is not None else {}
    )
    summarizing = spec["max_summary_tokens"] > 0
    if summarizing and not hasattr(DuckAgent, "preserve_failed_draws"):
        raise ValueError(
            "ARC-3 summarisation training requires harness support for "
            "preserve_failed_draws; the pinned harness retries failed summaries "
            "and violates training lineage (arc3-platypus#100)."
        )
    overrides: dict = {
        "file_tools": "none",
        "compaction_strategy": "summarize" if summarizing else "prune",
        "journal": not summarizing,
        "summary_thinking": False,
        "engine_ceiling_tokens": spec["max_served_sequence_tokens"],
        "max_tokens_per_game": spec.get("max_episode_tokens"),
    }
    # The remote server owns image processor limits; no local engine arguments
    # exist to derive them onto. Older harness pins have no such control.
    if hasattr(DuckAgent, "derive_max_pixels"):
        overrides["derive_max_pixels"] = False
    if summarizing:
        overrides["preserve_failed_draws"] = True
    # Image mode is explicit in the frozen training recipe; tools, their
    # responses and ordinary Duck text feedback remain part of the context.
    overrides["multimodal_context"] = spec.get("multimodal_context", "")
    if spec.get("reasoning_effort") is not None:
        overrides["reasoning_effort"] = spec["reasoning_effort"]
    # Window contract: the trainer's sequence budget bounds what a training
    # sample may hold, so the duck's context window (and compaction target)
    # must come from the spec rather than the 65536 deployment default.
    if spec.get("context_window_tokens") is not None:
        overrides["context_window_tokens"] = int(spec["context_window_tokens"])
    if spec.get("compact_to_tokens") is not None:
        overrides["compact_to_tokens"] = int(spec["compact_to_tokens"])
    # The trainer resolves the pinned preset margin and sends it explicitly.
    if spec.get("safety_margin_tokens") is not None:
        overrides["safety_margin_tokens"] = int(spec["safety_margin_tokens"])
    # Reject a spec that leaves malformed-reply retries enabled below.
    if spec.get("malformed_retries") is not None:
        overrides["malformed_retries"] = int(spec["malformed_retries"])

    from training_backend import configure_training_agent
    from ullm.backends.engine_proxy import EngineProxy
    from training_backend import (
        prompt_injection_settings,
        qwen_engine_factory,
        training_backend_factory,
        training_sampling,
    )

    engine = spec.get("engine", "qwen3.8-27b")
    agent = DuckAgent.build(
        engine=engine,
        environment="slurm-b200",
        backend_factory=qwen_engine_factory(
            EngineProxy(
                base_url=spec["base_url"],
                model=spec["served_model"],
                api_key=os.environ.get("SERVER_API_KEY", ""),
                tokenizer_path=spec.get("tokenizer_id"),
                # Engine metadata owns the tokenizer revision: HF serving
                # pins it, while a local merged checkpoint has no revision.
                # EngineProxy still verifies the full tokenizer fingerprint.
            ),
            engine=engine,
        ),
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        reply_cap_tokens=spec.get("max_completion_tokens", 8192),
        **overrides,
    )
    agent.run_config = replace(
        agent.run_config,
        process_mode="subprocess",
        min_request_gap_s=0.0,
        max_concurrent=int(spec["max_concurrent"]),
        max_runtime_s_per_game=spec.get("max_runtime_s_per_game"),
        # The deployment's policy (platypus's RunConfig default): an agent
        # exception ends only its own game, which Benchmark teardown marks
        # "crashed" with the levels it had completed, and the window goes on.
        # One Duck bug in one game must not end a run. Serving failures still
        # stop the window: the gateway poisons it on any failed generation.
        on_agent_error="crash_game",
        # Explicit per-slot seeds survive scheduling order; unseeded specs
        # derive model RNG streams from the window seed and game index.
        rng_seed=int(spec["rng_seed"]) if spec.get("rng_seed") is not None else 0,
        rng_seed_per_game=slot_rng_seeds,
    )
    if "sampling" not in spec:
        raise ValueError(
            "spec lacks 'sampling' — training requires explicit sampling parameters"
        )
    sampling = spec["sampling"]
    configure_training_agent(
        agent,
        backend_factory=training_backend_factory(
            source_factory=agent.llm_config.backend_factory
        ),
        sampling=training_sampling(sampling),
        max_concurrent=int(spec["max_concurrent"]),
    )
    agent.prepare_for_backend()
    trigger = spec.get("compaction_trigger_tokens")
    if type(trigger) is not int or trigger != agent.context_budget_tokens():
        raise ValueError(
            "harness compaction trigger differs from the trainer's context contract"
        )
    if agent.compact_target_tokens() != spec.get("compact_to_tokens"):
        raise ValueError(
            "harness compaction target differs from the trainer's context contract"
        )
    forbidden = tuple(agent.compaction_forbidden_token_ids) if summarizing else ()
    if tuple(spec["summary_forbidden_token_ids"]) != forbidden:
        raise ValueError(
            "harness token bans differ from the trainer's context contract"
        )
    summary_cap = 2 * agent._summary_target_tokens() if summarizing else 0
    if spec["max_summary_tokens"] != summary_cap:
        raise ValueError(
            "harness completion cap differs from the trainer's context contract"
        )
    if summarizing:
        agent._require_summary_fits(spec["max_served_sequence_tokens"])
    elif spec["max_served_sequence_tokens"] != agent.context_window_tokens:
        raise ValueError(
            "harness served ceiling differs from the trainer's context contract"
        )
    if agent.malformed_retries != 0 or spec.get("accept_malformed_output") is not True:
        raise ValueError(
            "training requires single-attempt generation and retained malformed replies"
        )
    # The per-level completion-token budget is the Duck preset's own, never
    # overridden here; every run reports what it was (enrich_results).
    resolved_per_level = (
        None if agent.max_tokens_per_level is None else list(agent.max_tokens_per_level)
    )
    install_contract_backend(agent)

    # A separate ordinary GameAPI keeps every rollout at the game's normal
    # initial state while giving its model RNG a distinct identity.
    games = [
        arc3_game(GameAPI, spec, env_name=name, external_game_id=game_id)
        for name, game_id in zip(names, game_ids, strict=True)
    ]
    job_dir = Path(spec["job_dir"])
    # What the agent plays under, as the harness resolved it: every run
    # reports it (enrich_results).
    record_agent_settings(
        job_dir,
        {"max_tokens_per_level": resolved_per_level, **prompt_injection_settings(agent)},
    )
    return Benchmark(
        games=games,
        solver=ContractLLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=job_dir,
        trace_root=job_dir / "traces",
    )


def enrich_results(spec: dict, runs: list[dict], originals: list[dict]) -> None:
    """Report scored ARC-3 outcomes, terminal progress and the agent settings."""
    runs[:] = [
        enrich_arc3_result(run, max_runtime_seconds=spec["max_runtime_s_per_game"])
        for run in runs
    ]
    attach_agent_settings(Path(spec["job_dir"]), runs)
    for run in runs:
        run["terminal_progress"] = {
            "version": 1,
            "after_nonce": run["steps"][-1]["nonce"] if run["steps"] else None,
            "score": run["episode_return"],
            "stage": run["levels_completed"],
            "data": {"arc_score": run["final_arc_score"]},
        }
