"""ARC-2 terminal scoring, strict result evidence and legacy gateway regression."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from harness_support.arc2.results import enrich_arc2_result, progress_snapshot
from harness_support.arc2.dataset import dataset_manifest
from training.config import DEFAULT_MALFORMED_CALL_PENALTY
from training.gateway_progress import ProgressLedger
from training.platypus_rollout import build_window_trajectories
from training.rollout_gateway import (
    GatewayError,
    RunResult,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)

HAS_ARC2_DATA = (
    Path(__file__).resolve().parents[1]
    / "arc2-platypus/packages/taaf/src/taaf/data/arc2-train.npz"
).exists()


def result_payload(
    success=(True, False),
    malformed=0,
    state="submitted",
    *,
    penalty=DEFAULT_MALFORMED_CALL_PENALTY,
    empty_reply=False,
    cap=None,
):
    """Build a callback with independently checkable final and draw evidence.

    ``empty_reply`` makes the one draw an empty reply, as its processed
    response journals it (harness_support/arc2/contract.py): kept as
    malformed, so it counts at least one malformed call."""
    payload = dict(
        run_id="abc",
        game_id="puzzle",
        state=state,
        levels_completed=int(any(success)),
        number_of_levels=1,
        actions_per_level=[1],
        base_actions_per_level=[],
        final_wallclock_seconds=900.0,
        action_wallclock_seconds=[3.0],
        rollout_slot_id="puzzle#s0:abc",
        rollout_seed=17,
        behavior_policy_snapshot={"name": "policy", "id": 1, "path": "/adapter"},
        serving_adapter={"name": "policy", "id": 1, "path": "/adapter"},
        steps=[
            dict(
                nonce="first",
                action_index=0,
                output_len=2,
                output_crc32=token_ids_crc32([3, 4]),
                continued=False,
                history_epoch=0,
                exact_history=True,
                malformed_call_count=malformed,
                empty_reply=empty_reply,
                **({"stop_reason": "malformed"} if empty_reply else {}),
            )
        ],
    )
    benchmark = dict(
        finished=True,
        history=[{}],
        result={"pairs": [{"correct": value} for value in success]},
        final_score=100 * sum(success) / len(success),
    )
    enriched = enrich_arc2_result(
        payload,
        benchmark_run=benchmark,
        test_output_count=len(success),
        malformed_call_penalty=penalty,
        max_episode_penalty=cap,
    )
    # The 27B Duck preset's prompt injection, as the worker reports it.
    enriched["agent_settings"] = dict(AGENT_SETTINGS)
    return enriched


AGENT_SETTINGS = {
    "inject_text_on_first_prompt": "We need solve",
    "inject_text_on_later_prompts": "We need",
    "prompt_injection_schedule": "game",
}


@pytest.mark.skipif(not HAS_ARC2_DATA, reason="Requires the private ARC-2 submodule")
def test_arc2_dataset_manifests_are_complete_disjoint_and_content_bound():
    """Named datasets resolve repository contents, never a sibling checkout."""
    train, evaluation = dataset_manifest("arc2-train"), dataset_manifest("arc2-eval")
    assert len(train["puzzle_ids"]) == 1000
    assert len(evaluation["puzzle_ids"]) == 120
    assert set(train["puzzle_ids"]).isdisjoint(evaluation["puzzle_ids"])
    assert sum(train["test_outputs"].values()) == 1076
    assert sum(evaluation["test_outputs"].values()) == 167
    assert train["sha256"] != evaluation["sha256"]


@pytest.mark.skipif(not HAS_ARC2_DATA, reason="Requires the private ARC-2 submodule")
def test_real_harness_exact_pixels_dimensions_attempts_and_standing_submission():
    """Exercise the actual ARC-2 scorer in an isolated import namespace."""
    root = Path(__file__).resolve().parents[1]
    program = """
import json
from taaf.datasets import spec_from_task_json
from taaf.game import ArcGame, Submission
from taaf.grid import Grid
task = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[0]], "output": [[1, 2], [3, 4]]}, {"input": [[0]], "output": [[5]]}]}
spec = spec_from_task_json("puzzle", task)
def answer(first, second):
    return Submission(attempts=tuple(tuple(Grid.from_lists(g) for g in pair) for pair in [first, second]))
wrong = [[0]]
correct = [[1, 2], [3, 4]]
scenarios = [
    answer([wrong, correct], [wrong, [[5]]]),
    answer([[[1, 2], [3, 0]], wrong], [wrong, wrong]),
    answer([[[1, 2, 3, 4]], wrong], [wrong, wrong]),
    answer([correct, correct], [wrong, wrong]),
]
scores=[]
for submitted in scenarios:
    game=ArcGame(spec_data=spec)
    game.start_game()
    assert game.execute_action(submitted) is None
    assert game.game_run.result is None
    game.finish_game()
    scores.append(game.game_run.final_score)
game=ArcGame(spec_data=spec)
game.start_game()
game.execute_action(scenarios[0])
game.execute_action(scenarios[1])
game.finish_game()
assert game.game_run.final_score == 0.0
print(json.dumps(scores))
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "arc2-platypus/packages/taaf/src"),
    }
    output = subprocess.run(
        [sys.executable, "-c", program],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(output.stdout.splitlines()[-1]) == [100.0, 0.0, 0.0, 50.0]


@pytest.mark.parametrize(
    "success,malformed", [((True, False), 2), ((False,), 3), ((True,), 0)]
)
def test_terminal_reward_is_fractional_unclamped_and_separate(success, malformed):
    """The optimizer receives shaped reward while metrics retain correctness."""
    run = RunResult.from_payload(result_payload(success, malformed))
    capture = StepCapture(
        "abc",
        "first",
        [1, 2],
        [3, 4],
        [-0.1, -0.2],
        "stop",
        0,
        native_token_prompt=True,
        adapter=run.behavior_policy_snapshot,
    )
    window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
    trajectory = build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=900.0
    )[0]
    assert trajectory.task_id == "puzzle"
    assert (
        trajectory.trajectory_reward == sum(success) / len(success) - 0.05 * malformed
    )
    assert trajectory.metadata["final_arc_score"] == sum(success) / len(success)
    assert trajectory.termination_reason == "COMPLETED_RUN"


def _trajectory(run):
    capture = StepCapture(
        "abc",
        "first",
        [1, 2],
        [3, 4],
        [-0.1, -0.2],
        "stop",
        0,
        native_token_prompt=True,
        adapter=run.behavior_policy_snapshot,
    )
    window = WindowCapture(captures={("abc", "first"): capture}, results={"abc": run})
    return build_window_trajectories(
        window, policy_iteration=1, max_runtime_seconds=900.0
    )[0]


@pytest.mark.parametrize("journaled", [1, 0])
def test_an_empty_reply_is_one_malformed_call(journaled):
    """ullm counts an ordinary empty reply one malformed call, a summary's
    none; either way it is charged one, and counted as an empty reply."""
    payload = result_payload(malformed=journaled, empty_reply=True, penalty=0.3)
    assert payload["steps"][0]["malformed_call_count"] == 1
    outcome = payload["arc2_outcome"]
    assert (outcome["malformed_call_count"], outcome["empty_reply_count"]) == (1, 1)
    assert payload["terminal_reward"] == pytest.approx(0.5 - 0.3)
    run = RunResult.from_payload(payload, malformed_call_penalty=0.3)
    trajectory = _trajectory(run)
    assert trajectory.trajectory_reward == pytest.approx(0.2)
    assert trajectory.steps[-1].reward == pytest.approx(0.2)
    assert trajectory.metadata["malformed_call_count"] == 1
    assert trajectory.metadata["malformed_penalty"] == pytest.approx(0.3)
    assert trajectory.metadata["empty_reply_count"] == 1
    step = trajectory.steps[0].metadata
    assert (step["empty_reply"], step["malformed_call_count"], step["format_break"]) == (
        True,
        1,
        True,
    )
    # The coefficient is the trainer's: a worker that scored with another fails.
    with pytest.raises(GatewayError, match="malformed_penalty disagrees"):
        RunResult.from_payload(payload)


def _many_penalties(success, cap, *, penalty=0.1):
    """Three malformed tool calls on one draw and an empty reply on another:
    four malformed calls at PENALTY each, scored with CAP."""
    payload = result_payload(success, malformed=3, penalty=penalty, cap=cap)
    empty = {**payload["steps"][0], "nonce": "second", "malformed_call_count": 1,
             "empty_reply": True, "stop_reason": "malformed", "action_index": None,
             "continued": True}
    payload["steps"].append(empty)
    # Score the two draws again, as the worker does.
    benchmark = dict(
        finished=True,
        history=[{}],
        result={"pairs": [{"correct": value} for value in success]},
        final_score=100 * sum(success) / len(success),
    )
    return enrich_arc2_result(
        payload,
        benchmark_run=benchmark,
        test_output_count=len(success),
        malformed_call_penalty=penalty,
        max_episode_penalty=cap,
    )


@pytest.mark.parametrize(
    ("cap", "capped", "charged"),
    [(None, False, 0.4), (1.0, False, 0.4), (0.4, True, 0.4), (0.25, True, 0.25)],
)
def test_the_episode_penalty_cap_binds_only_above_it(cap, capped, charged):
    """reward = correctness - min(cap, malformed_call_penalty * malformed calls);
    the outcome keeps the uncapped penalty, the cap and whether it bound."""
    payload = _many_penalties((True, False), cap)
    outcome = payload["arc2_outcome"]
    assert (outcome["malformed_call_count"], outcome["empty_reply_count"]) == (4, 1)
    assert outcome["malformed_penalty"] == pytest.approx(0.4)
    assert outcome["max_episode_penalty"] == cap
    assert outcome["penalty_capped"] is capped
    assert payload["terminal_reward"] == pytest.approx(0.5 - charged)
    run = RunResult.from_payload(payload, malformed_call_penalty=0.1, max_episode_penalty=cap)
    capture = StepCapture(
        "abc", "first", [1, 2], [3, 4], [-0.1, -0.2], "stop", 0,
        native_token_prompt=True, adapter=run.behavior_policy_snapshot,
    )
    second = StepCapture(
        "abc", "second", [1, 2, 3, 4, 5], [3, 4], [-0.1, -0.2], "stop", 1,
        native_token_prompt=True, adapter=run.behavior_policy_snapshot,
    )
    window = WindowCapture(
        captures={("abc", "first"): capture, ("abc", "second"): second},
        results={"abc": run},
    )
    (trajectory,) = build_window_trajectories(window, policy_iteration=1, max_runtime_seconds=900.0)
    assert trajectory.trajectory_reward == pytest.approx(0.5 - charged)
    assert trajectory.steps[-1].reward == pytest.approx(0.5 - charged)
    assert trajectory.metadata["penalty_capped"] is capped
    assert trajectory.metadata["malformed_penalty"] == pytest.approx(0.4)


def test_no_cap_leaves_the_reward_as_it_was():
    """Without a cap the reward is correctness - coefficient x malformed calls,
    the same float as before the cap."""
    payload = _many_penalties((True, False), None, penalty=0.05)
    assert payload["terminal_reward"] == 0.5 - 0.05 * 4


def test_a_success_with_penalties_above_the_cap_still_beats_no_success():
    """our case: penalties past the cap cost only the cap, so a solved
    puzzle scores correctness - 0.5 and still outscores an unsolved one."""
    solved = _many_penalties((True,), 0.5, penalty=0.5)
    assert solved["arc2_outcome"]["malformed_penalty"] == pytest.approx(2.0)
    assert solved["terminal_reward"] == pytest.approx(1.0 - 0.5)
    unsolved_clean = result_payload((False,), cap=0.5)
    assert unsolved_clean["terminal_reward"] == 0.0
    assert solved["terminal_reward"] > unsolved_clean["terminal_reward"]
    # Uncapped, the same episode would have scored below zero.
    assert _many_penalties((True,), None, penalty=0.5)["terminal_reward"] < 0


@pytest.mark.parametrize(
    ("worker_cap", "trainer_cap"), [(0.5, None), (None, 0.5), (0.5, 0.4), (1.0, 0.5)]
)
def test_the_cap_is_the_trainers(worker_cap, trainer_cap):
    payload = _many_penalties((True, False), worker_cap)
    with pytest.raises(GatewayError, match="max_episode_penalty disagrees"):
        RunResult.from_payload(payload, malformed_call_penalty=0.1, max_episode_penalty=trainer_cap)


def test_a_capped_outcome_must_say_so():
    payload = _many_penalties((True, False), 0.25)
    payload["arc2_outcome"]["penalty_capped"] = False
    with pytest.raises(GatewayError, match="penalty_capped disagrees"):
        RunResult.from_payload(payload, malformed_call_penalty=0.1, max_episode_penalty=0.25)
    payload = _many_penalties((True, False), 0.25)
    # An uncapped reward under a binding cap is refused too.
    payload["episode_return"] = payload["terminal_reward"] = 0.5 - 0.4
    payload["steps"][-1]["reward"] = 0.0
    with pytest.raises(GatewayError, match="reward disagrees"):
        RunResult.from_payload(payload, malformed_call_penalty=0.1, max_episode_penalty=0.25)


@pytest.mark.parametrize("tamper", ["uncharged", "missing", "not_bool"])
def test_trainer_refuses_inconsistent_empty_reply_evidence(tamper):
    payload = result_payload(malformed=1, empty_reply=True)
    step = payload["steps"][0]
    if tamper == "uncharged":
        # An empty reply reported without its malformed call, with the
        # outcome made to agree: the trainer still refuses it.
        step["malformed_call_count"] = 0
        payload["arc2_outcome"]["malformed_call_count"] = 0
        payload["arc2_outcome"]["malformed_penalty"] = 0.0
        payload["episode_return"] = payload["terminal_reward"] = 0.5
        match = "empty replies are malformed calls"
    elif tamper == "missing":
        del step["empty_reply"]
        match = "empty-reply counts disagree"
    else:
        step["empty_reply"] = 1
        match = "empty-reply counts disagree"
    with pytest.raises(GatewayError, match=match):
        RunResult.from_payload(payload)


def test_pending_deadline_draw_does_not_erase_terminal_standing_score():
    """Live snapshots stay zero and final scoring occurs after the pending draw."""
    ledger = ProgressLedger()

    def snapshot(after):
        return {**progress_snapshot(None), "after_nonce": after}

    ledger.admit("abc", "first", snapshot(None), previous_nonce=None)
    ledger.admit("abc", "pending", snapshot("first"), previous_nonce="first")
    run = RunResult.from_payload(result_payload(state="cancelled"))
    final = ledger.reconcile(
        "abc",
        run.terminal_progress,
        run.steps,
        run.episode_return,
        terminal_reward=run.terminal_reward,
    )
    assert final["score"] == 0.5
    assert ledger.close("abc", ["first"])["unprocessed_nonces"] == ["pending"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("correctness", 0.75),
        ("malformed_penalty", 1),
        ("malformed_call_count", True),
        ("empty_reply_count", True),
        ("empty_reply_count", 1),
        ("penalty_capped", True),
        ("penalty_capped", 0),
        ("max_episode_penalty", 0.5),
        ("per_output_success", [1, 0]),
    ],
)
def test_corrupt_terminal_evidence_is_rejected(field, value):
    """Callbacks cannot replace exact binary scoring or typed penalty counts."""
    payload = result_payload()
    payload["arc2_outcome"][field] = value
    with pytest.raises(GatewayError):
        RunResult.from_payload(payload)


@pytest.mark.parametrize("tool_summary", [False, True])
def test_served_compaction_admission_and_join(tool_summary):
    from training.context_contract import resolve_context_contract
    from training.gateway_lineage import LineageLedger
    from training.rollout_gateway import SamplingPolicy

    context = resolve_context_contract(1200, 176, 100, 0, 256, 1800, (2,))
    policy = SamplingPolicy("model", 1.0, 1.0, -1, context=context)
    ledger = LineageLedger()
    captures = []
    payload = result_payload()
    first = payload["steps"][0]
    payload["steps"] = []
    draws = [
        ("first", [1, 2], 0, "start", 176),
        ("summary", [1, 2, 3, 4] + [5] * 1296, 0, "continue", 256),
    ]
    if not tool_summary:
        draws.append(("reset", [90, 91], 2, "reset", 176))
    for index, (nonce, prompt, epoch, kind, maximum) in enumerate(draws):
        stamp = dict(
            version=1, epoch=epoch, kind=kind, compaction_prompt_token_ids=None
        )
        policy.validate(
            dict(
                model="model",
                user=f"abc#{nonce}",
                prompt=prompt,
                max_tokens=maximum,
                **({"logit_bias": {2: -100.0}} if nonce == "summary" else {}),
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                min_p=0.0,
                min_tokens=0,
            )
        )
        ledger.admit(
            "abc",
            nonce,
            prompt,
            stamp,
            context.lineage_policy(),
            native_token_prompt=True,
            summary_request=nonce == "summary",
        )
        evidence = ledger.complete("abc", nonce, prompt, [3, 4])
        capture = StepCapture(
            "abc",
            nonce,
            prompt,
            [3, 4],
            [-0.1, -0.2],
            "stop",
            index,
            native_token_prompt=True,
            adapter=payload["serving_adapter"],
            lineage=evidence,
            logit_bias={2: -100.0} if nonce == "summary" else {},
            summary_request=nonce == "summary",
        )
        captures.append(capture)
        payload["steps"].append(
            {
                **first,
                "nonce": nonce,
                "action_index": 0 if index == 0 else None,
                "continued": kind == "continue",
                "history_epoch": epoch,
                "has_tool_calls": tool_summary and nonce == "summary",
            }
        )
    payload["terminal_progress"]["after_nonce"] = draws[-1][0]
    ledger.report("abc", [capture.nonce for capture in captures])
    run = RunResult.from_payload(payload)
    window = WindowCapture(
        captures={("abc", c.nonce): c for c in captures}, results={"abc": run}
    )
    (trajectory,) = build_window_trajectories(
        window, policy_iteration=0, max_runtime_seconds=900.0
    )
    assert [step.metadata["segment_start"] for step in trajectory.steps] == (
        [True, False] if tool_summary else [True, False, True]
    )
    assert [step.completion_token_ids for step in trajectory.steps] == [[3, 4]] * len(
        draws
    )
    assert trajectory.steps[1].reward == (0.5 if tool_summary else 0.0)
    assert trajectory.trajectory_reward == 0.5

    assert [step.metadata["summary_tool_call"] for step in trajectory.steps] == (
        [False, True] if tool_summary else [False, False, False]
    )
    from training.dataset import build_samples
    from training.rewards import StepSignal

    samples = build_samples(
        [trajectory],
        per_rollout_signals=[
            [
                StepSignal(reward=0, return_to_go=0.5, advantage=1)
                for _ in trajectory.steps
            ]
        ],
    )
    assert samples[0].token_logit_bias == {}
    if not tool_summary:
        assert samples[2].token_logit_bias == {}
    assert samples[1].token_logit_bias == {2: [0.0] * 1299 + [-100.0, -100.0]}
