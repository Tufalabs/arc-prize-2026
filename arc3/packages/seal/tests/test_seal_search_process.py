"""The search child: it must contain what it runs.

Search executes model-written code tens of thousands of times per call, so
"the model wrote something pathological" is the normal case, not the
exceptional one. These tests use models that are pathological on purpose —
one that hangs, one that allocates without bound, one that raises — and
assert the game survives each with a usable answer. Each spawns a real
process, so they cost a second or so apiece.
"""

from __future__ import annotations

import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pytest

from platypus.contract import Action
from seal.model import Prediction, WorldModel
from seal.move_lister import MoveLister
from seal.planner import BFSPlanner, NoPlan, Plan
from seal.search_process import PreparationRefused, SearchResult, run_search
from seal.trace import LevelTrace, Step

FRAME = np.zeros((4, 4), dtype=np.int8)
ACTIONS = ("ACTION1",)


class LineModel(WorldModel):
    """A corridor of ``length`` states ending in a win.

    ``pathology`` makes the model misbehave the way a real model-written one
    might: hang forever inside a single move, allocate without bound, or
    raise.
    """

    def __init__(self, length: int = 3, pathology: str = "none") -> None:
        super().__init__()
        self.length = length
        self.pathology = pathology

    def _initial_state(self) -> dict[str, Any]:
        return {"outcome": "playing", "n": 0}

    def _step(self, state: dict[str, Any], action: Action) -> Prediction:
        if self.pathology == "hang":
            while True:  # exactly what the between-expansions timeout cannot catch
                time.sleep(0.05)
        if self.pathology == "balloon":
            self._eat = getattr(self, "_eat", [])
            while True:
                self._eat.append(bytearray(64 * 1024 * 1024))
        if self.pathology == "raise":
            raise RuntimeError("model exploded")
        state["n"] += 1
        state["outcome"] = "won" if state["n"] >= self.length else "playing"
        return Prediction(kind="claim")

    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        return np.full((4, 4), state["n"] % 10, dtype=np.int8)


def trace_for(model: WorldModel) -> LevelTrace:
    return LevelTrace(
        steps=[Step(action=None, frame=FRAME, outcome="playing")],
        states=[model.initial_state()],
    )


@dataclass
class Given:
    """A job whose model and trace exist already, which is all the planner's
    side of the child needs; ``delay`` stands in for a slow load, and
    ``refusal`` for a load the harness refuses."""

    model: WorldModel
    trace: LevelTrace
    delay: float = 0.0
    refusal: str | None = None

    def prepare(self, announce: Callable[[dict[str, str]], None]) -> tuple[WorldModel, LevelTrace]:
        announce({"world_model.py": "0" * 64})
        time.sleep(self.delay)
        if self.refusal is not None:
            raise PreparationRefused(self.refusal)
        return self.model, self.trace


def planner() -> BFSPlanner:
    return BFSPlanner(move_lister=MoveLister(), energy_label=None, allow_repetition=None, mode_max=0)


async def search(model: WorldModel, **kw: Any) -> Any:
    kw.setdefault("hard_timeout", 60.0)
    return await run_search(planner(), Given(model, trace_for(model)), ACTIONS, seed=1, **kw)


async def test_a_search_result_comes_back_across_the_process_boundary() -> None:
    # The baseline the rest depends on: a Plan built in the child, with its
    # actions and diagnostics, survives being pickled home.
    result = await search(LineModel(length=3))
    assert result.ok, result.describe()
    assert isinstance(result.value, Plan)
    assert (result.value.target, len(result.value.actions)) == ("win", 3)
    assert result.value.diagnostics.nodes_explored > 0


async def test_a_plan_comes_home_with_what_the_model_predicts_for_each_move() -> None:
    # Predicted in the child, by the model that searched: the kernel checks
    # its own replay of the path against exactly these boards and outcomes.
    result = await search(LineModel(length=3))
    assert [(turn.outcome, int(turn.frame[0, 0])) for turn in result.predicted if turn.frame is not None] == [
        ("playing", 1),
        ("playing", 2),
        ("won", 3),
    ]
    assert result.boundary_reason is None


async def test_a_boundary_plan_ends_its_predictions_before_the_move_that_raised() -> None:
    # The last move of a boundary path is the one the model makes no claim
    # for, so there is nothing to predict for it — and why is what the model
    # is told.
    result = await search(LineModel(pathology="raise"))
    assert isinstance(result.value, Plan)
    assert len(result.value.actions) == 1
    assert result.predicted == ()
    assert "model exploded" in (result.boundary_reason or "")


async def test_a_model_that_hangs_inside_one_move_is_killed_and_reported() -> None:
    # The case the planner's own max_duration provably cannot catch: it is
    # checked between node expansions, and this never leaves one. Without
    # the hard timeout the game would hang forever.
    started = time.monotonic()
    result = await search(LineModel(pathology="hang"), hard_timeout=3.0)
    assert result.outcome == "timeout"
    assert "hard timeout" in (result.error or "")
    # The call it hung in, for the report to name.
    assert result.stuck_in == f"do_move(state, {Action('ACTION1')!r})"
    # Killed near the deadline, not merely eventually.
    assert time.monotonic() - started < 25.0


async def test_a_model_that_allocates_without_bound_takes_only_the_child_down() -> None:
    # An in-process OOM would take the game with it, and under
    # process_mode="inprocess" the whole run. The limit is deliberately low
    # so this stays quick and cannot disturb the machine.
    result = await search(LineModel(pathology="balloon"), hard_timeout=60.0, memory_limit_bytes=512 * 2**20)
    assert result.outcome in ("died", "error"), result.describe()
    # Whatever the kernel did to it, the parent has something to report.
    assert result.error


async def test_the_memory_budget_is_on_top_of_what_the_imports_reserve() -> None:
    # OpenBLAS and friends reserve address space they never touch; a job's
    # memory limit does not count it, and neither does the search's budget.
    # A budget far below what the child reserves after its imports still
    # searches.
    from seal.search_process import _address_space_reserved  # pyright: ignore[reportPrivateUsage]

    budget = 64 * 2**20
    assert _address_space_reserved() > budget
    result = await search(LineModel(length=3), memory_limit_bytes=budget)
    assert result.ok, result.describe()


async def test_a_model_whose_do_move_raises_is_a_boundary_not_a_failure() -> None:
    # Worth pinning because it is the line between the two mechanisms: a
    # model that raises inside do_move is EXPECTED, and `WorldModel.step`
    # turns it into an `error` prediction that search treats as a boundary
    # target. The child does not fail, and the search comes back with a
    # usable path to the thing that broke.
    result = await search(LineModel(pathology="raise"))
    assert result.ok, result.describe()
    assert isinstance(result.value, Plan)
    assert result.value.diagnostics.boundary_type == "error"


async def test_a_failure_in_the_search_itself_comes_back_as_a_value() -> None:
    # Not as an exception: the caller decides what to do, and it needs the
    # traceback to say so. A raise would also strand the diagnostics on the
    # exception object, which is the pattern this port removes.
    #
    # The strict energy contract is a genuine instance -- `key_fn` raises
    # when a state lacks the energy field, and nothing in the planner
    # catches it.
    p = BFSPlanner(move_lister=MoveLister(), energy_label="energy", allow_repetition=None, mode_max=0)
    model = LineModel(length=3)  # its states carry no "energy" key
    result = await run_search(p, Given(model, trace_for(model)), ACTIONS, seed=1, hard_timeout=60.0)
    assert result.outcome == "error"
    assert "energy" in (result.error or "")


async def test_the_time_budget_covers_the_preparation() -> None:
    # The model's load, test and replay run in the child before the search,
    # inside the same budget: a preparation that spends all of it leaves the
    # planner nothing, and the search reports running out of time rather than
    # running past its budget. What the job read comes home either way.
    p = planner()
    p.max_duration = 0.5
    model = LineModel(length=3)
    result = await run_search(p, Given(model, trace_for(model), delay=1.0), ACTIONS, seed=1, hard_timeout=60.0)
    assert result.ok, result.describe()
    assert isinstance(result.value, NoPlan) and result.value.diagnostics.exit_reason == "timeout"
    assert result.digests == {"world_model.py": "0" * 64}


async def test_a_refused_preparation_comes_back_as_its_own_outcome() -> None:
    # A harness fault, which the caller ends the game on, apart from a search
    # that failed, which the model is told about.
    model = LineModel(length=3)
    result = await run_search(
        planner(), Given(model, trace_for(model), refusal="two harnesses disagree"), ACTIONS, seed=1, hard_timeout=60.0
    )
    assert (result.outcome, result.error) == ("refused", "two harnesses disagree")
    assert result.digests == {"world_model.py": "0" * 64}


def test_a_long_stderr_is_read_from_its_tail(tmp_path: Path) -> None:
    # A model that logs without bound fills the child's stderr; the parent
    # keeps the last lines for the report and must not load the rest to
    # find them.
    from seal.search_process import _STDERR_TAIL_CHARS, _read_stderr  # pyright: ignore[reportPrivateUsage]

    log = tmp_path / "stderr.txt"
    log.write_text("x" * 8 * 2**20 + "the end\n")
    tracemalloc.start()
    try:
        text = _read_stderr(str(log))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert text.startswith("...<truncated>...") and text.endswith("the end\n")
    assert len(text) <= _STDERR_TAIL_CHARS + len("...<truncated>...\n")
    assert peak < 2**20, f"reading the tail allocated {peak} bytes"


@pytest.mark.parametrize(
    "result, expected",
    [
        (SearchResult(outcome="ok", duration_s=1.24), "search ok in 1.2s"),
        (
            SearchResult(outcome="timeout", duration_s=3.0, error="hard timeout"),
            "search timeout after 3.0s: hard timeout",
        ),
        # A child killed before it could report anything leaves no error.
        (SearchResult(outcome="died", duration_s=0.5), "search died after 0.5s"),
    ],
)
def test_describe_names_the_outcome_and_its_detail(result: SearchResult, expected: str) -> None:
    """Callers assert on ``result.ok`` and print this when it is False, so a
    failed search is only as diagnosable as this line. Constructed directly:
    the failure outcomes it has to render cost a spawned process each to
    reach, and none of them is what this checks."""
    assert result.describe() == expected
