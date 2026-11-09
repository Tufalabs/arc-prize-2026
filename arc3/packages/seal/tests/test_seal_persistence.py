"""Kernel persistence modes and what they cost in snapshot cadence.

The four modes differ in one observable way — how long the model's own
definitions survive — and in one structural way: a kernel that outlives a
move cannot be pickled at a move boundary (A4), so the longer-lived modes
must narrow where the agent can be snapshotted.
"""

from __future__ import annotations

import json
from typing import Any, Literal

import pytest
import taaf.benchmark
import ullm.backends.stub
import ullm.factory

from platypus import LLMAgentSolver, LLMConfig, RunConfig
from seal import SealAgent
from seal.repl import GameREPL


async def _run_snippets(tool: GameREPL, *snippets: str) -> list[str]:
    # The seal prelude calls back into the agent for runtime state; standalone
    # here, a minimal stand-in is enough since these rows only care about what
    # survives between snippets.
    def _get_state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [],
            "available_actions": [],
        }

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {"queued": 0}}

    tool.host_functions = {"get_state": _get_state, "action": _action}
    out: list[str] = []
    try:
        for code in snippets:
            content = await tool._execute({"code": code})
            out.append(content.to_string())
    finally:
        tool.reset(keep_tmpdir_contents=False)
    return out


async def test_round_mode_forgets_between_snippets() -> None:
    """The original seal's process-per-call sandbox: a name bound in one
    snippet is gone in the next."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=True)
    _, second = await _run_snippets(tool, "carried = 1234", "print(carried)")
    assert "NameError" in second, f"round mode must not carry state; got {second[:300]}"


async def test_persistent_mode_carries_state_between_snippets() -> None:
    """The point of the longer-lived modes: helpers and results the model
    builds stay available instead of being rewritten every snippet."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False)
    _, second = await _run_snippets(tool, "carried = 1234", "print(carried)")
    assert "NameError" not in second, f"persistent mode must carry state; got {second[:300]}"
    assert "1234" in second


async def test_persistent_mode_restarts_after_memory_exhaustion() -> None:
    """A caught MemoryError must not leave later snippets in a poisoned worker."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False)
    first, second = await _run_snippets(
        tool,
        "carried = 1234\njunk = [None] * 300_000_000",
        "print(carried)",
    )
    assert "MemoryError" in first
    assert "kernel was restarted" in first
    assert "variables and definitions were cleared" in first
    assert "NameError" in second


async def test_persistent_mode_refreshes_reserved_runtime_names() -> None:
    """User helpers persist; host-owned runtime names cannot stay shadowed.

    `history` is the one that matters: the canonical object lives outside the
    model's namespace, so a cell that rebinds the name loses only its own
    copy and the next cell is handed the real one back."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False)
    _, second = await _run_snippets(
        tool,
        "helper = 17\nhistory = 'stale'\nsubmit_action = 'stale'",
        "print(helper, type(history).__name__, callable(submit_action))",
    )
    assert "17 History True" in second


async def test_submit_action_globals_do_not_publish_the_raw_host_dispatcher() -> None:
    """The advertised function's globals are the surface module's, not the
    store's: neither the host primitive nor the store's adapters are there."""
    tool = GameREPL(timeout=30.0, reset_before_each_call=False)
    (output,) = await _run_snippets(
        tool,
        "g = submit_action.__globals__\n"
        "print('_submit' in g, '__host_call_trusted__' in g, '_rs' in g)\n"
        "try:\n"
        "    g['_submit']([])\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__)\n",
    )
    assert "False False False" in output
    assert "KeyError" in output
    assert tool.pending_host_exception is None


@pytest.mark.parametrize(
    ("persistence", "expected"),
    [
        ("round", {"move", "level", "off"}),
        ("move", {"move", "level", "off"}),
        ("level", {"level", "off"}),
        ("always", {"off"}),
    ],
)
def test_snapshot_cadence_narrows_with_persistence(
    persistence: Literal["round", "move", "level", "always"], expected: set[str]
) -> None:
    """A4: a kernel must not outlive a checkpoint boundary. Keeping kernel
    state across moves forfeits move-boundary snapshots, and keeping it for
    the whole game forfeits all of them."""
    assert set(SealAgent(persistence=persistence).supported_checkpointing()) == expected


def test_cadence_is_not_decided_at_construction() -> None:
    """The report must follow configuration, so it cannot be a class
    constant fixed before the agent is configured — an agent built with
    defaults and then reconfigured must report the NEW answer."""
    agent = SealAgent()
    assert set(agent.supported_checkpointing()) == {"level", "off"}  # the "level" default
    agent.persistence = "always"
    assert set(agent.supported_checkpointing()) == {"off"}
    agent.persistence = "move"
    assert set(agent.supported_checkpointing()) == {"move", "level", "off"}


async def test_incompatible_checkpointing_is_rejected_at_run_not_construction() -> None:
    """Constructing an agent whose persistence rules out the configured
    checkpointing must not raise — the pieces are only knowably inconsistent
    once the run starts and both are settled. Starting the run must raise."""
    agent = SealAgent(
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        reply_cap_tokens=None,  # must not raise
        compaction_strategy="prune",
        persistence="always",
        run_config=RunConfig(checkpointing="move", process_mode="inprocess"),
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.StubBackend(responses=[])),
        ),
    )
    import arcengine
    from taaf.game_examples import ExampleGame

    game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]], label="persist_game")
    benchmark = taaf.benchmark.Benchmark(games=[game], solver=LLMAgentSolver(agent=agent), n_passes=1)
    with pytest.raises(ValueError, match="checkpointing='move' not supported"):
        await benchmark.run()


async def _pretend_a_snippet_ran(agent: SealAgent) -> None:
    """Put the agent in the state a mid-game reset happens from: a live
    kernel that has run something. Nothing is announced before that, because
    there is no scratch work to have lost."""
    tool = agent._ensure_repl()

    def _state() -> dict[str, Any]:
        return {
            "level": 1,
            "turns": [],
            "available_actions": [],
        }

    def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
        return {"ack": {"queued": 0}}

    tool.host_functions = {"get_state": _state, "action": _action}
    await tool._execute({"code": "scratch = 1"})
    assert not tool.is_fresh


async def test_reset_notice_is_pending_only_for_the_middle_modes() -> None:
    """ "round" resets before every snippet and "always" never resets, so
    neither has anything to announce; the two modes that reset at a boundary
    the model can't see must say so, or scratch state vanishes silently."""
    modes: list[tuple[Literal["round", "move", "level", "always"], bool]] = [
        ("round", False),
        ("move", True),
        ("level", True),
        ("always", False),
    ]
    for mode, expected in modes:
        agent = SealAgent(reply_cap_tokens=None, compaction_strategy="prune", persistence=mode)
        await _pretend_a_snippet_ran(agent)
        agent._reset_repl("this move started")
        assert bool(agent.repl_reset_pending) is expected, f"{mode} pending={agent.repl_reset_pending!r}"


async def test_a_session_whose_kernel_died_still_gets_its_notice() -> None:
    """The model built a session, its kernel died between calls, and the level
    changed before another call: the session is gone all the same, and the
    model is told so."""
    agent = SealAgent(reply_cap_tokens=None, compaction_strategy="prune", persistence="level")
    await _pretend_a_snippet_ran(agent)
    tool = agent.repl
    assert tool is not None and tool._process is not None
    tool._process.kill()
    tool._process.wait()
    assert tool.is_fresh
    agent._reset_repl("a new level started")
    assert agent.repl_reset_pending == "a new level started"
    agent.repl_reset_pending = ""
    agent._reset_repl("a new level started")
    assert agent.repl_reset_pending == "", "a reset with no call since the last one has nothing to announce"


def test_no_reset_notice_before_the_first_snippet() -> None:
    """The game's opening prompt must not announce that earlier definitions
    are gone: nothing has run, so there were none. The notice is about work
    the model actually did and lost."""
    agent = SealAgent(persistence="move")
    agent._reset_repl("this move started")
    assert agent.repl_reset_pending == ""
    assert "session was cleared" not in agent._build_user_prompt_text()


async def test_reset_notice_reaches_the_prompt_once() -> None:
    """The notice must appear in the next user prompt and then clear —
    repeating it every turn would tell the model its state is gone long
    after it rebuilt it."""
    agent = SealAgent(persistence="move")
    await _pretend_a_snippet_ran(agent)
    agent._reset_repl("this move started")
    first = agent._build_user_prompt_text()
    assert "session was cleared" in first and "this move started" in first
    assert "session was cleared" not in agent._build_user_prompt_text()


def test_system_prompt_states_the_actual_lifetime() -> None:
    """Each mode must describe what really survives. Telling the model its
    code is not saved when it IS saved wastes every snippet rewriting
    helpers; the reverse loses work it expected to keep."""
    round_text = SealAgent(persistence="round")._system_prompt_text()
    assert "not saved between calls" in round_text
    persistent_modes: list[Literal["move", "level", "always"]] = ["move", "level", "always"]
    for mode in persistent_modes:
        text = SealAgent(persistence=mode)._system_prompt_text()
        assert "not saved between calls" not in text, f"{mode} still claims snippets are ephemeral"
        assert "keeps your definitions" in text

    default_text = SealAgent()._system_prompt_text()
    assert "keeps your definitions" in default_text
    assert "cleared when a new level starts" in default_text  # the "level" default
    assert "cleared when a new level starts" not in SealAgent(persistence="always")._system_prompt_text()


def test_tool_description_lists_frame_shape() -> None:
    """The system prompt and runtime both expose frame shape, so the tool
    description in the same request must not claim a narrower surface."""
    assert ".grid" in SealAgent()._ensure_repl().description
    assert ".diff_sparse" in SealAgent()._ensure_repl().description
    assert "submit_action(actions)" in SealAgent()._ensure_repl().description
    assert "`memory()`" not in SealAgent()._ensure_repl().description
    assert "segment(" in SealAgent()._ensure_repl().description


def test_tool_description_is_not_contradicted_by_persistence() -> None:
    """The tool's own description and parameter help also claim
    ephemerality; a persistent mode that leaves them alone contradicts the
    system prompt in the same context window."""
    agent = SealAgent(persistence="always")
    tool = agent._ensure_repl()
    blob = json.dumps({"description": tool.description, "parameters": tool.parameters})
    assert "ephemeral" not in blob and "not saved across tool calls" not in blob
    assert "game-scoped" in tool.description


if __name__ == "__main__":
    pytest.main([__file__])
