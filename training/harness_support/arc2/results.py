"""Terminal ARC-2 correctness and malformed-call shaping for worker callbacks."""

from __future__ import annotations

import math
from typing import Any


def progress_snapshot(run: Any, *, start_level: int = 0) -> dict[str, Any]:
    """Expose no correctness before finalization, including standing submissions."""
    del run, start_level
    return {"version": 1, "score": 0.0, "stage": "episode", "data": {}}


def charged_penalty(penalty: float, cap: float | None) -> tuple[bool, float]:
    """What an episode's uncapped ``penalty`` costs it under ``cap``
    (``training.max_episode_penalty``, None for none): whether the cap bound
    (the penalty reached it) and the amount charged, ``min(cap, penalty)``.
    The worker and the trainer's re-derivation both call this, so the two
    agree to the bit."""
    capped = cap is not None and penalty >= cap
    return capped, cap if capped else penalty


# How an ARC-2 game can end, each scored on its standing answer: the harness
# finishes it (won/submitted answered; gave_up never answered), the deadline
# cancels it, or its agent raised and Benchmark teardown marked it crashed
# (``on_agent_error="crash_game"``, harness_support/arc2/worker.py).
TERMINAL_STATES = frozenset({"won", "submitted", "gave_up", "cancelled", "crashed"})


def enrich_arc2_result(
    run: dict[str, Any],
    *,
    benchmark_run: dict[str, Any],
    test_output_count: int,
    malformed_call_penalty: float,
    max_episode_penalty: float | None,
) -> dict[str, Any]:
    """Translate the final standing submission to normalized exact-match reward.

    The reward is correctness minus ``malformed_call_penalty`` per malformed
    call, at most ``max_episode_penalty`` in all when that is set, terminal
    (``malformed_penalty`` is the uncapped charge, ``penalty_capped`` whether
    the cap was charged instead). A reply the backend kept as malformed counts
    at least one malformed call: an empty reply (no text, no tool call), which
    ullm's turn manager counts as one and a summary's leaves at zero, or raw
    text a parser rejected. Each step also reports its processed response's
    ``empty_reply`` (harness_support/arc2/contract.py); the outcome counts
    them (``empty_reply_count``), which changes no reward.

    A ``crashed`` run is scored the same way: TAAF scores the answer standing
    when its agent raised, and its malformed calls are charged as usual
    (``termination_category`` says how it ended).
    """
    if benchmark_run.get("finished") is not True:
        raise ValueError("ARC-2 result was reported before final scoring")
    if run["state"] not in TERMINAL_STATES:
        raise RuntimeError(f"ARC-2 infrastructure failure: {run['state']}")
    if type(test_output_count) is not int or test_output_count < 1:
        raise ValueError("ARC-2 tasks must have at least one test output")
    result = benchmark_run.get("result")
    pairs = result.get("pairs") if isinstance(result, dict) else None
    if pairs is None:
        if benchmark_run.get("history"):
            raise ValueError("standing ARC-2 submission lacks terminal scoring")
        success = [False] * test_output_count
    else:
        if len(pairs) != test_output_count or any(
            type(pair.get("correct")) is not bool for pair in pairs
        ):
            raise ValueError(
                "ARC-2 terminal scoring must cover every known test output"
            )
        success = [pair["correct"] for pair in pairs]
    correctness = sum(success) / test_output_count
    score = benchmark_run.get("final_score")
    if type(score) not in (float, int) or not math.isclose(
        score / 100.0, correctness, abs_tol=1e-9
    ):
        raise ValueError("ARC-2 final score disagrees with per-output correctness")
    malformed = empty_replies = 0
    for step in run["steps"]:
        count = step.get("malformed_call_count")
        if type(count) is not int or count < 0:
            raise ValueError("ARC-2 steps require explicit malformed-call counts")
        empty = step.get("empty_reply")
        if type(empty) is not bool:
            raise ValueError("ARC-2 steps require an explicit empty-reply indicator")
        count = max(count, int(step.get("stop_reason") == "malformed"))
        step["malformed_call_count"] = count
        malformed += count
        empty_replies += empty
        step["reward"] = 0.0
    malformed_penalty = malformed_call_penalty * malformed
    capped, charged = charged_penalty(malformed_penalty, max_episode_penalty)
    reward = correctness - charged
    run.update(
        environment="arc2",
        episode_return=reward,
        terminal_reward=reward,
        arc2_outcome={
            "correctness": correctness,
            "per_output_success": success,
            "malformed_call_count": malformed,
            "malformed_penalty": malformed_penalty,
            "max_episode_penalty": max_episode_penalty,
            "penalty_capped": capped,
            "empty_reply_count": empty_replies,
            "termination_category": run["state"],
        },
        terminal_progress={
            "version": 1,
            "after_nonce": run["steps"][-1]["nonce"] if run["steps"] else None,
            "score": reward,
            "stage": "episode",
            "data": {
                "correctness": correctness,
                "malformed_call_count": malformed,
                "empty_reply_count": empty_replies,
            },
        },
    )
    return run
