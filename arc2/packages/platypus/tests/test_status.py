"""Per-game status reporting: the registry's state machine, the report
the monitor prints, and the label reaching the parent in both process
modes."""

from __future__ import annotations

import pytest
from infra_fixtures import StatusPublishingAgent

from platypus import AgentSolver, RunConfig
from platypus.status import (
    FINISHED,
    RUNNING,
    WAITING_FOR_SLOT,
    GameStatus,
    StatusHandle,
    StatusRegistry,
    format_status_lines,
)
from taaf.benchmark import Benchmark
from taaf.game_examples import flip_game


def _publications() -> tuple[list[str], StatusHandle]:
    published: list[str] = []
    return published, StatusHandle(published.append)


# --- the handle -----------------------------------------------------------


def test_nested_labels_restore_the_one_underneath() -> None:
    # A kernel run happens inside a turn, so its label must not survive it:
    # otherwise a game reads "running kernel" for the whole of the LLM call
    # that follows, which is the wrong half of the report to trust.
    published, status = _publications()
    status.set_base("other")
    with status("waiting for LLM response"):
        with status("running kernel"):
            pass
    assert published == ["other", "waiting for LLM response", "running kernel", "waiting for LLM response", "other"]


def test_an_unchanged_label_is_not_republished() -> None:
    # Every publication is a pipe message out of a game subprocess, and the
    # duck re-enters these labels thousands of times per game.
    published, status = _publications()
    with status("running kernel"):
        pass
    with status("running kernel"):
        pass
    assert published == ["running kernel", "running", "running kernel", "running"]

    published.clear()
    for _ in range(5):
        status.set("running kernel")
    assert published == ["running kernel"]


# --- the registry ---------------------------------------------------------


def test_a_finished_game_stops_reporting_what_it_was_doing() -> None:
    # The agent's label describes work that has ended. Carrying it past the
    # coarse transition would leave finished games counted as "running
    # kernel" forever — and the count of stuck kernels is the number this
    # report exists to make trustworthy.
    registry = StatusRegistry()
    registry.register("0", "ls20")
    registry.set_coarse("0", RUNNING)
    registry.set_detail("0", "running kernel")
    assert registry.snapshot()[0].label == "running kernel"

    registry.set_coarse("0", FINISHED)
    assert registry.snapshot()[0].label == FINISHED


def test_a_label_published_before_the_slot_does_not_mask_the_queue() -> None:
    # A game queued behind max_concurrent must read as queued, whatever its
    # agent last said — the two are different problems for the reader.
    registry = StatusRegistry()
    registry.register("0", "ls20")
    registry.set_detail("0", "waiting for LLM response")
    assert registry.snapshot()[0].label == WAITING_FOR_SLOT


# --- the report -----------------------------------------------------------


def test_the_report_names_the_game_that_has_been_stuck_longest() -> None:
    # The whole point: a run with one game wedged for forty minutes has the
    # same counts as a healthy one, so the report has to name it.
    now = 1_000.0
    games = [
        GameStatus(game_id="fast", coarse=RUNNING, detail="running kernel", since=now - 5),
        GameStatus(game_id="wedged", coarse=RUNNING, detail="waiting for LLM response", since=now - 2_400),
        GameStatus(game_id="queued", coarse=WAITING_FOR_SLOT, since=now - 3_000),
        GameStatus(game_id="done", coarse=FINISHED, since=now - 1),
    ]
    coarse, running = format_status_lines(games, now)

    assert coarse == "4 games: 2 running, 1 finished, 1 waiting for slot"
    assert "1 running kernel, 1 waiting for LLM response" in running
    # Not "queued", which has waited longer but is waiting on us, not stuck.
    assert "longest: wedged waiting for LLM response 40.0m" in running


def test_the_report_omits_the_detail_line_when_nothing_runs() -> None:
    games = [GameStatus(game_id="a", coarse=WAITING_FOR_SLOT, since=0.0)]
    assert format_status_lines(games, 0.0) == ["1 games: 1 waiting for slot"]


# --- end to end -----------------------------------------------------------


@pytest.mark.parametrize("process_mode", ["inprocess", "subprocess"])
async def test_a_hosted_game_reports_what_its_agent_is_doing(
    process_mode: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole chain, and for ``subprocess`` the only test that can catch
    the label failing to cross the pipe: agent -> ctx.status -> (pipe) ->
    parent registry -> printed line."""
    agent = StatusPublishingAgent(
        label="waiting for LLM response",
        hold_s=1.2,
        run_config=RunConfig(process_mode=process_mode, status_interval_s=0.25),  # type: ignore[arg-type]
    )
    benchmark = Benchmark(games=[flip_game("g")], solver=AgentSolver(agent=agent))
    await benchmark.run()

    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("[games]")]
    assert any("1 waiting for LLM response" in line for line in lines), lines


async def test_reporting_off_leaves_the_agent_a_working_no_op_handle(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # status_interval_s=None has to be genuinely free — no monitor, and in
    # subprocess mode nothing on the control pipe — while agent code that
    # publishes labels keeps running unchanged.
    agent = StatusPublishingAgent(
        label="waiting for LLM response",
        hold_s=0.1,
        run_config=RunConfig(process_mode="subprocess", status_interval_s=None),
    )
    benchmark = Benchmark(games=[flip_game("g")], solver=AgentSolver(agent=agent))
    await benchmark.run()

    assert benchmark.game_runs[0].state == "gave_up"
    assert not [line for line in capsys.readouterr().out.splitlines() if line.startswith("[games]")]
