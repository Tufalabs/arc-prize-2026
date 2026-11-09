"""ARC-2 worker construction in its independent Duck/Platypus/TAAF environment."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Any

from harness_support.agent_settings import attach_agent_settings, record_agent_settings
from harness_support.augmentation import augment_arc2_task
from harness_support.arc2.results import (
    enrich_arc2_result,
    progress_snapshot as progress_snapshot,
)
from training.rollout_slots import _rng_seed_per_game, _slot_game_ids


def add_response_reports(
    run: dict[str, Any], *, job_dir: Path, max_runtime_seconds: float | None
) -> None:
    """Join processed-response penalties to host traces using the exact nonce.

    Only a run's last draw may lack its journal record, and only when it
    authorized no action and the run was cut there: by the runtime deadline,
    or by a crash between the host recording the draw and the agent
    processing it. That draw is dropped from the report; the gateway admits
    one unreported terminal draw for exactly these runs.
    """
    path = job_dir / "processed_responses" / f"{run['run_id']}.jsonl"
    records = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    by_nonce = {record["nonce"]: record for record in records}
    if len(by_nonce) != len(records):
        raise ValueError("ARC-2 processed-response journal contains duplicate nonces")
    processed = []
    for index, step in enumerate(run["steps"]):
        record = by_nonce.pop(step["nonce"], None)
        if record is None:
            cut = run["state"] == "crashed" or (
                # ARC-2 states describe the standing submission even when
                # the runtime deadline stops the session.
                run["state"] in {"cancelled", "gave_up", "won", "submitted"}
                and max_runtime_seconds is not None
                and run["final_wallclock_seconds"] >= max_runtime_seconds
            )
            if (
                index != len(run["steps"]) - 1
                or not cut
                or step["action_index"] is not None
            ):
                raise ValueError(
                    "ARC-2 completed trace lacks processed-response penalty evidence"
                )
            continue
        step.update(record)
        processed.append(step)
    if by_nonce:
        raise ValueError("ARC-2 processed response is absent from host trace")
    run["steps"] = processed


class _SummaryRequested(Exception):
    """The probe's stop at the Duck's summary request."""


def duck_summary_request_caps(agent: Any) -> dict[str, int]:
    """The completion caps the Duck sets on a compaction summary request.

    Runs the Duck's own ``_summarize_live_history`` on a copy of the prepared
    agent up to the request it builds (``_write_summary``), so ``cap`` (the
    summary written without reasoning, what training serves) and
    ``reasoning_cap`` (its retry with reasoning) are the Duck's numbers, not
    re-derived here. The copy reaches neither the engine nor a kernel, and
    nothing is awaited, so no event loop is needed.
    """
    import copy

    probe = copy.deepcopy(agent)
    caps: dict[str, int] = {}

    async def requested(_ctx: Any, _instruction: Any, *, cap: int, reasoning_cap: int, **_: Any):
        caps.update(cap=cap, reasoning_cap=reasoning_cap)
        raise _SummaryRequested

    probe._write_summary = requested
    # The file shown beside a summary lives in the game's kernel, which the
    # copy must not start; it changes the instruction, not the caps.
    probe._summary_candidate_source = lambda: None
    summary = probe._summarize_live_history(None, 0, call_limit=None)
    try:
        summary.send(None)
    except _SummaryRequested:
        return caps
    except StopIteration:
        pass
    else:
        summary.close()
    raise RuntimeError("the Duck's compaction did not request a summary first")


def served_image_max_pixels(spec: dict[str, Any]) -> int | None:
    """The served image processor's pixel ceiling: the count above which it
    shrinks an image, or None when the server has images disabled, as the
    native server states it on ``/metadata`` (``image_max_pixels``, through
    the gateway)."""
    import urllib.parse
    import urllib.request

    key = os.environ.get("SERVER_API_KEY", "")
    request = urllib.request.Request(
        f"{spec['base_url'].rstrip('/')}/metadata?"
        + urllib.parse.urlencode({"model": spec["served_model"]}),
        headers={"Authorization": f"Bearer {key}"} if key else {},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        metadata = json.loads(response.read())
    if "image_max_pixels" not in metadata:
        raise RuntimeError(
            "the server does not state its image pixel ceiling (/metadata image_max_pixels)"
        )
    ceiling = metadata["image_max_pixels"]
    if ceiling is not None and (type(ceiling) is not int or ceiling <= 0):
        raise RuntimeError(f"the server states an invalid image pixel ceiling: {ceiling!r}")
    return ceiling


def check_served_image_ceiling(agent: Any, spec: dict[str, Any]) -> None:
    """Refuse a server whose image processor would shrink an image the Duck sends.

    The deployed Duck sets the engine's ``max_pixels`` to its own largest
    render (``derive_max_pixels``); training turns that off, because the
    engine is the server's, so the requirement is checked here instead, with
    the Duck's own function for it.
    """
    if not agent._images_enabled():
        return
    from duck import images

    needed = images.largest_image_pixels(
        scale=agent._image_scale(), gridlines=agent.image_gridlines
    )
    served = served_image_max_pixels(spec)
    if served is None:
        raise ValueError("the ARC-2 Duck sends images, and the server has images disabled")
    if needed > served:
        raise ValueError(
            f"the server's image processor shrinks images above {served} pixels; "
            f"the ARC-2 Duck's largest render needs {needed}"
        )


def load_tasks(dataset: str | list[str]) -> dict[str, Any]:
    """Every puzzle of the named TAAF sets, by id; the trainer has checked that
    the sets are disjoint and hold every listed game."""
    from harness_support.arc2.dataset import dataset_names
    from taaf.datasets import load

    return {task.game_id: task for name in dataset_names(dataset) for task in load(name)}


def build_benchmark(spec: dict[str, Any]):
    """Build one independent full Duck session for each seeded puzzle slot."""
    from harness_support.arc2.contract import (
        ProcessedResponseObserver,
        ContractARC2LLMAgentSolver,
    )
    from training_backend import configure_training_agent
    from serving_contract import install_contract_backend
    from taaf.benchmark import Benchmark
    from taaf.game import ArcGame
    from training_backend import (
        prompt_injection_settings,
        qwen_engine_factory,
        training_backend_factory,
        training_sampling,
    )
    from ullm.backends.engine_proxy import EngineProxy

    tasks = load_tasks(spec["dataset"])
    engine = spec.get("engine", "qwen3.8-27b")
    names = list(spec["games"])
    slot_ids = _slot_game_ids(names)
    if spec.get("slot_seeds") is None:
        raise ValueError("ARC-2 requires an explicit independent seed per rollout slot")
    slot_rng_seeds = _rng_seed_per_game(slot_ids, spec["slot_seeds"])
    factory = qwen_engine_factory(
        EngineProxy(
            base_url=spec["base_url"],
            model=spec["served_model"],
            api_key=os.environ.get("SERVER_API_KEY", ""),
            tokenizer_path=spec["tokenizer_id"],
            # A local checkpoint (Flash) has no revision: engine metadata owns
            # it then, as in the ARC-3 worker. The fingerprint is still verified.
            tokenizer_revision=spec["model_revision"] or None,
        ),
        engine=engine,
    )
    factory.backend_kwargs["classify_tool_calls"] = True
    from duck.agent import DuckAgent

    # The Duck's malformed-reply nudge drops a reply the backend refused and
    # re-prompts; training refuses nothing (accept_malformed_output keeps
    # every draw), so it would never fire, and the Duck requires it off beside
    # preserve_failed_draws. A retained empty reply is nudged by the analyzer
    # entry instead, unless terminate_on_empty_reply ends the game on it
    # (docs/arc2_training.md). Keep the worker compatible with harness pins
    # that predate the nudge control.
    retention_overrides = (
        {"malformed_nudges": 0} if hasattr(DuckAgent, "malformed_nudges") else {}
    )
    agent = DuckAgent.build(
        engine=engine,
        environment="slurm-b200",
        backend_factory=factory,
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        reply_cap_tokens=spec["max_completion_tokens"],
        context_window_tokens=spec["context_window_tokens"],
        engine_ceiling_tokens=spec["max_served_sequence_tokens"],
        summary_thinking=False,
        compact_to_tokens=spec["compact_to_tokens"],
        safety_margin_tokens=spec["safety_margin_tokens"],
        max_tokens_per_game=spec.get("max_episode_tokens"),
        reasoning_effort=spec["reasoning_effort"],
        malformed_retries=0,
        perturb_retry_prompt=False,
        preserve_failed_draws=True,
        classify_tool_calls=True,
        response_observer=ProcessedResponseObserver(),
        derive_max_pixels=False,
        terminate_on_empty_reply=bool(spec.get("terminate_on_empty_reply", False)),
        **retention_overrides,
    )
    agent.run_config = replace(
        agent.run_config,
        process_mode="subprocess",
        min_request_gap_s=0.0,
        max_concurrent=int(spec["max_concurrent"]),
        max_runtime_s_per_game=spec["max_runtime_s_per_game"],
        # The deployment's policy (platypus's RunConfig default, which the
        # Duck's presets keep): an agent exception ends only its own game,
        # which Benchmark teardown marks "crashed" and scores on its standing
        # answer, and the window goes on. One Duck bug in one game's kernel
        # must not end a run (2026-10-11: "raise" ended a 40-minute Flash run
        # on one housekeeping failure). Serving failures still stop the
        # window: the gateway poisons it on any failed generation.
        on_agent_error="crash_game",
        rng_seed=int(spec["rng_seed"]),
        rng_seed_per_game=slot_rng_seeds,
    )
    configure_training_agent(
        agent,
        backend_factory=training_backend_factory(source_factory=factory),
        sampling=training_sampling(spec["sampling"]),
        max_concurrent=spec["max_concurrent"],
    )
    agent.prepare_for_backend()
    # The engine preset's prompt injection, served as the harness serves it;
    # the trainer pins it in the checkpoint identity from the first window.
    injection = prompt_injection_settings(agent)
    if agent.context_budget_tokens() != spec["compaction_trigger_tokens"]:
        raise ValueError("ARC-2 compaction trigger differs from trainer contract")
    # The trainer re-derives the Duck's served-ceiling bound and summary cap
    # (harness_support/arc2/trainer.py); these read the Duck's own, through
    # private methods until it has public accessors (docs/arc2_training.md,
    # "Where the trainer can drift from the Duck"). The Duck refuses a ceiling
    # below window + reply reserve + its summary cap with reasoning.
    served = spec["max_served_sequence_tokens"]
    if agent._derived_engine_window() != served:
        raise ValueError("ARC-2 engine ceiling differs from trainer contract")
    caps = duck_summary_request_caps(agent)
    if caps["cap"] != spec["max_summary_tokens"]:
        raise ValueError(
            f"ARC-2 summary cap differs from trainer contract: the Duck requests "
            f"{caps['cap']}, the trainer expects {spec['max_summary_tokens']}"
        )
    if (
        agent.context_window_tokens + agent._reply_reserve_tokens() + caps["reasoning_cap"]
        > served
    ):
        raise ValueError(
            "ARC-2 served ceiling cannot hold the Duck's summary request with reasoning"
        )
    if agent.compact_target_tokens() != spec["compact_to_tokens"]:
        raise ValueError("ARC-2 compaction target differs from trainer contract")
    if agent.context_observer.reply_cap_tokens != spec["max_completion_tokens"]:
        raise ValueError("ARC-2 reply cap differs from trainer contract")
    if agent.summary_thinking is not False:
        raise ValueError("ARC-2 summary thinking differs from trainer contract")
    if tuple(agent.compaction_forbidden_token_ids) != tuple(
        spec["summary_forbidden_token_ids"]
    ):
        raise ValueError("ARC-2 compaction token bans differ from trainer contract")
    check_served_image_ceiling(agent, spec)
    install_contract_backend(agent)
    job_dir = Path(spec["job_dir"])
    # Every run reports what its agent played under (enrich_results).
    record_agent_settings(job_dir, injection)
    os.environ["ARC2_TRAINING_RESPONSE_DIR"] = str(job_dir / "processed_responses")
    return Benchmark(
        games=[
            ArcGame(
                spec_data=augment_arc2_task(
                    replace(tasks[name], game_id=slot), puzzle=name, spec=spec
                )
            )
            for name, slot in zip(names, slot_ids, strict=True)
        ],
        solver=ContractARC2LLMAgentSolver(agent=agent),
        raise_solver_errors=True,
        n_passes=1,
        job_dir=job_dir,
        trace_root=job_dir / "traces",
    )


def enrich_results(spec: dict, runs: list[dict], originals: list[dict]) -> None:
    """Join ARC-2 processed responses, seeded slots, scored submissions and
    the agent settings."""
    tasks = load_tasks(spec["dataset"])
    counts = {game_id: len(task.to_answer) for game_id, task in tasks.items()}
    seeds = _rng_seed_per_game(_slot_game_ids(spec["games"]), spec["slot_seeds"])
    for run, original in zip(runs, originals, strict=True):
        add_response_reports(
            run,
            job_dir=Path(spec["job_dir"]),
            max_runtime_seconds=spec["max_runtime_s_per_game"],
        )
        slot = str(original["game_id"])
        run.update(
            rollout_slot_id=f"{slot}:{run['run_id']}",
            rollout_seed=seeds[slot],
            behavior_policy_snapshot=spec["behavior_policy_snapshot"],
            serving_adapter=spec["serving_adapter"],
        )
        enrich_arc2_result(
            run,
            benchmark_run=original,
            test_output_count=counts[run["game_id"]],
            malformed_call_penalty=spec["malformed_call_penalty"],
            max_episode_penalty=spec["max_episode_penalty"],
        )
    attach_agent_settings(Path(spec["job_dir"]), runs)
