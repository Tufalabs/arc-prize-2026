"""A real Benchmark in which one game's Duck raises mid-game, hosted as the
training worker hosts it: each game in its own subprocess, every draw kept and
journaled (harness_support/arc2/worker.py), ullm's stub engine behind the
host socket.

The crash is a Duck bug the model's own code reaches, as on 2026-10-11 (job
2932277, a ``kernel housekeeping failed`` in one game of 239): the model
changes into a subdirectory, the Duck's end-of-cell sweep removes it, and the
next cell's housekeeping (``kernel_workspace.housekeeping``) cannot read the
working directory, so ``PythonTool._housekeeping`` raises. Under the harness's
``on_agent_error="crash_game"``, which the workers run since then, the
benchmark completes: that game is ``crashed`` and scored on the answer it had
submitted, the other game is unaffected, and ``benchmark.json`` carries both
for the worker's results. Under ``"raise"`` the benchmark fails, as the run
did.

Runs where the ARC-2 Duck is installed (the ARC-2 worker environment; the
harness-environment test runs on kander use it); CI has no Duck. The CPU
tests of the worker's report and the trainer's handling of a crashed run are
tests/test_arc2_crashed_game.py.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
from typing import Any

import pytest

duck_agent = pytest.importorskip("duck.agent")
if not hasattr(duck_agent.DuckAgent, "terminate_on_empty_reply") or not hasattr(
    duck_agent.DuckAgent, "_uses_context_summary"
):
    pytest.skip("requires the ARC-2 Duck environment", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import taaf.benchmark  # noqa: E402 - requires the ARC-2 environment
import ullm.backend  # noqa: E402
import ullm.backends.stub  # noqa: E402
import ullm.conv  # noqa: E402
import ullm.factory  # noqa: E402
import ullm.tools.python  # noqa: E402
from platypus import LLMAgentSolver, LLMConfig, RunConfig  # noqa: E402
from taaf.game_examples import flip_game  # noqa: E402

from harness_support.arc2.contract import ProcessedResponseObserver  # noqa: E402
from harness_support.arc2.results import enrich_arc2_result  # noqa: E402
from training.inference import _platypus_window_outcomes  # noqa: E402
from training.platypus_rollout import build_window_trajectories  # noqa: E402
from training.rollout_gateway import (  # noqa: E402
    RunResult,
    StepCapture,
    WindowCapture,
    token_ids_crc32,
)

STEADY, CRASHY = "steady", "crashy"
SNAPSHOT = {"name": "policy", "id": 1, "path": "/adapter"}
AGENT_SETTINGS = {
    "inject_text_on_first_prompt": "",
    "inject_text_on_later_prompts": "",
    "prompt_injection_schedule": "game",
}


def _python(code: str) -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.ListContent(
            elements=[
                ullm.conv.ToolCallContent(tool_name="python", arguments={"code": code})
            ]
        ),
        stop_reason="tool_use",
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=50, completion_tokens=5, num_requests=1
        ),
    )


def _done() -> ullm.conv.LLMResponse:
    return ullm.conv.LLMResponse(
        final_content=ullm.conv.StringContent(text="Done."),
        stop_reason="stop",
        token_usage=ullm.backend.TokenUsage(
            prompt_tokens=50, completion_tokens=3, num_requests=1
        ),
    )


# One script for both games, by turn. Both answer at turn 0 (each task's
# answer is its question flipped left-right). At turn 1 only the crashing
# game's model (its 3x3 task) changes into a subdirectory, which the Duck's
# sweep removes after the cell; its turn-2 cell then fails the Duck's
# housekeeping. The steady game runs the same cells and finishes.
SCRIPT = [
    _python("import numpy as np\ntest_out[0][0] = np.fliplr(np.array(test_in[0]))\n"),
    _python(
        "import os\nimport numpy as np\n"
        "if np.array(test_in[0]).shape == (3, 3):\n"
        "    os.makedirs('work', exist_ok=True)\n"
        "    os.chdir('work')\n"
        "print('ok')\n"
    ),
    _python("print('again')\n"),
    _done(),
]


class _Engine(ullm.backends.stub.StubBackend):
    """The host's engine for every game: a request gets the reply its turn
    names (the replies already in its conversation), so games interleaving on
    the one host each play the whole script, and the reply is stamped
    ``run_id#n<turn>``, the identity the processed-response journal keys on."""

    def _prepare_input(
        self,
        sampling_params: Any,
        turn: Any,
        conversation: Any,
        send_back_reasoning: str,
    ) -> Any:
        replies = sum(
            isinstance(part, ullm.conv.LLMResponse)
            for element in conversation.elements
            if isinstance(element, ullm.conv.ResponseTurn)
            for part in element.elements
        )
        return (self.responses[min(replies, len(self.responses) - 1)], replies, None)

    def _stamp_llm_input(self, llm_input: Any, run_id: str) -> Any:
        response, replies, _ = llm_input
        return (response, replies, run_id)

    async def _generate(self, sampling_params: Any, llm_input: Any):
        yield llm_input

    def _process_output(
        self,
        sampling_params: Any,
        llm_output: Any,
        conversation: Any,
        turn: Any,
        element: Any,
    ) -> None:
        response, replies, run_id = llm_output
        self._fill_into(
            response,
            element,
            conversation,
            self.resolve_response_injection(sampling_params),
        )
        element.generation_request_id = f"{run_id}#n{replies}"
        self._call_index += 1

    def _process_partial_output(
        self, sampling_params: Any, llm_output: Any, intermediate: Any
    ) -> None:
        response, _, _ = llm_output
        self._fill_into(
            response,
            intermediate,
            ullm.conv.Conversation(),
            self.resolve_response_injection(sampling_params),
        )


def _benchmark(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> taaf.benchmark.Benchmark:
    if not ullm.tools.python._unshare_usable():
        pytest.skip(
            "the game kernels run in their own subprocesses, which need user namespaces here"
        )
    # Inherited by each game's child, where the observer journals.
    monkeypatch.setenv(
        "ARC2_TRAINING_RESPONSE_DIR", str(tmp_path / "processed_responses")
    )
    agent = duck_agent.DuckAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=4096,
        images="off",
        show_changed_answers=False,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(
                backend=_Engine(responses=SCRIPT)
            ),
            sampling_profiles={
                "play": ullm.backend.GenericSamplingParameters(max_tokens=None)
            },
        ),
        # The worker's hosting: a child per game, its error policy.
        run_config=RunConfig(
            rng_seed=1,
            process_mode="subprocess",
            on_agent_error=policy,  # type: ignore[arg-type]
            max_concurrent=2,
            status_interval_s=None,
        ),
        # The training worker's retention settings.
        malformed_retries=0,
        perturb_retry_prompt=False,
        preserve_failed_draws=True,
        malformed_nudges=0,
        classify_tool_calls=True,
        response_observer=ProcessedResponseObserver(),
    )
    return taaf.benchmark.Benchmark(
        games=[flip_game(STEADY), flip_game(CRASHY, size=3, seed=1)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=tmp_path,
        raise_solver_errors=True,
    )


def _journal(tmp_path: Path, run_id: str) -> list[dict]:
    path = tmp_path / "processed_responses" / f"{run_id}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _report(record: dict, journal: list[dict]) -> dict:
    """The worker's report of one benchmark.json run (scripts/
    platypus_rollout_worker.py and harness_support/arc2/worker.py), its steps
    as the host trace would list them: each journaled draw, the first one
    authorizing the run's one answer."""
    steps = [
        {
            **entry,
            "action_index": 0 if index == 0 else None,
            "output_len": 1,
            "output_crc32": token_ids_crc32([index]),
            "continued": index > 0,
            "history_epoch": 0,
            "exact_history": True,
        }
        for index, entry in enumerate(journal)
    ]
    report = {
        "run_id": record["run_id"],
        "game_id": record["game_id"],
        "state": record["state"],
        "levels_completed": record["levels_completed"],
        "number_of_levels": record["number_of_levels"],
        "actions_per_level": record["actions_per_level"],
        "base_actions_per_level": [],
        "final_wallclock_seconds": record["final_wallclock_seconds"],
        "action_wallclock_seconds": [
            entry["wallclock_seconds"] for entry in record["history"]
        ],
        "steps": steps,
        "solver_note": record["solver_note"],
        "rollout_slot_id": f"{record['game_id']}:{record['run_id']}",
        "rollout_seed": 1,
        "behavior_policy_snapshot": SNAPSHOT,
        "serving_adapter": SNAPSHOT,
    }
    enrich_arc2_result(
        report,
        benchmark_run=record,
        test_output_count=1,
        malformed_call_penalty=0.05,
        max_episode_penalty=0.5,
    )
    report["agent_settings"] = dict(AGENT_SETTINGS)
    return report


def test_a_crashed_game_is_scored_on_its_answer_and_the_benchmark_completes(
    tmp_path, monkeypatch
):
    benchmark = _benchmark(tmp_path, monkeypatch, "crash_game")
    asyncio.run(asyncio.wait_for(benchmark.run(), timeout=600))
    runs = {run.game_id: run for run in benchmark.game_runs}

    crashed = runs[CRASHY]
    note = crashed.solver_note or ""
    assert crashed.state == "crashed", note
    assert crashed.finished is True
    assert note.startswith("agent error:"), note
    assert "kernel housekeeping failed" in note, note
    # Scored on the answer it submitted before its agent raised.
    assert len(crashed.history) == 1
    assert crashed.result is not None and [
        pair.correct for pair in crashed.result.pairs
    ] == [True]
    assert crashed.final_score == 100.0
    # The other game played its whole script.
    steady = runs[STEADY]
    assert (steady.state, steady.final_score, steady.solver_note) == (
        "won",
        100.0,
        None,
    )

    # What the worker reads: benchmark.json, written at teardown after the
    # crashed run was marked and scored.
    records = {
        record["game_id"]: record
        for record in json.loads((tmp_path / "benchmark.json").read_text())["game_runs"]
    }
    record = records[CRASHY]
    assert (record["state"], record["finished"], record["final_score"]) == (
        "crashed",
        True,
        100.0,
    )
    assert record["solver_note"] == crashed.solver_note
    # Every draw was journaled before its tool ran: the crashed game's three,
    # the last the cell that failed housekeeping; the steady game's whole
    # script (and the Duck's audit turn after it).
    crashed_journal = _journal(tmp_path, crashed.run_id)
    assert [entry["nonce"] for entry in crashed_journal] == ["n0", "n1", "n2"]
    steady_journal = _journal(tmp_path, steady.run_id)
    assert [entry["nonce"] for entry in steady_journal] == [
        f"n{i}" for i in range(len(steady_journal))
    ]
    assert len(steady_journal) >= len(SCRIPT)

    # The worker's report and the trainer's handling of it.
    reports = {
        game: _report(records[game], _journal(tmp_path, runs[game].run_id))
        for game in (STEADY, CRASHY)
    }
    outcome = reports[CRASHY]["arc2_outcome"]
    assert outcome["termination_category"] == "crashed"
    assert outcome["per_output_success"] == [True]
    assert reports[CRASHY]["terminal_reward"] == 1.0
    results = {
        report["run_id"]: RunResult.from_payload(
            report, malformed_call_penalty=0.05, max_episode_penalty=0.5
        )
        for report in reports.values()
    }
    captures = {}
    for run in results.values():
        prompt = [1, 2]
        for index, step in enumerate(run.steps):
            captures[(run.run_id, step["nonce"])] = StepCapture(
                run.run_id,
                step["nonce"],
                list(prompt),
                [index],
                [-0.1],
                "stop",
                len(captures),
                native_token_prompt=True,
                adapter=SNAPSHOT,
            )
            prompt = prompt + [index, 7]
    rollouts = build_window_trajectories(
        WindowCapture(captures=captures, results=results),
        policy_iteration=0,
        max_runtime_seconds=None,
    )
    by_game = {rollout.task_id: rollout for rollout in rollouts}
    assert by_game[STEADY].termination_reason == "COMPLETED_RUN"
    assert by_game[CRASHY].termination_reason == "CRASHED"
    assert by_game[CRASHY].trajectory_reward == 1.0
    # The note's head names the agent error, its tail the failure inside the
    # kernel: housekeeping could not read the deleted working directory.
    assert by_game[CRASHY].metadata["crash_note"].startswith("agent error:")
    tail = by_game[CRASHY].metadata["crash_note_tail"]
    assert "housekeeping" in tail and "FileNotFoundError" in tail, tail
    outcomes = _platypus_window_outcomes(
        "w", rollouts, log_path=tmp_path / "worker.log"
    )
    assert (outcomes["crashed"], outcomes["errors"]) == (1, 0)


def test_under_raise_the_same_crash_fails_the_benchmark(tmp_path, monkeypatch):
    """The workers' policy until 2026-10-11: the one crash ends the run."""
    benchmark = _benchmark(tmp_path, monkeypatch, "raise")
    with pytest.raises(RuntimeError, match="agent error in game subprocess") as raised:
        asyncio.run(asyncio.wait_for(benchmark.run(), timeout=600))
    assert "kernel housekeeping failed" in str(raised.value)
    assert {run.game_id: run.state for run in benchmark.game_runs}[CRASHY] == "crashed"
