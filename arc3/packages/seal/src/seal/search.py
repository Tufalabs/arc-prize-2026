"""The search a passing ``test_world_model()`` earns, host side.

After a python call whose last ``test_world_model()`` passed and which played
no move, the agent runs a chain outside the call's time limit (see
``SealAgent._search_after_call``). This module is the part that needs no
kernel: the search in a child process, which reads the files, tests them
again and searches, all under the search's time and memory caps, so none of
the model's code runs in the game's process; and the report of what it
found — for the model to read, and for the kernel to install as
``suggested_actions`` / ``suggested_history``.

The whale searched after every learn, from an agent loop with no LLM in it.
The seal searches only when the model asked for a test and got a pass, and
reports back into the same turn; the model decides what to do with a path.
"""

from __future__ import annotations

import base64
import contextlib
import io
import os
import re
import sys
import traceback
from dataclasses import dataclass
from types import CodeType
from typing import Any, Callable, Generator, Sequence

import numpy as np
import numpy.typing as npt

from platypus.status import NULL_STATUS, StatusHandle
from seal.action_names import model_move, to_engine_action
from seal.harness import game as harness_game
from seal.harness import test as harness_test
from seal.model import WorldModel
from seal.move_lister import MoveLister
from seal.palette import AsciiLayout
from seal.planner import BFSPlanner, NoPlan, Plan
from seal.py_model import PyWorldModel
from seal.rng import derive_seed
from seal.search_process import PreparationRefused, SearchResult, run_search
from seal.state import TimelineEntry
from seal.trace import LevelTrace, Step

#: Beyond the search's budget, what the child may take before it is killed:
#: spawning it and importing the harness, and one node expansion the
#: planner's between-expansions check cannot interrupt. The model's load, its
#: test and the level's replay are inside the budget.
SPAWN_GRACE_S = 30.0

#: How far under the search's memory budget the planner's own memory check
#: fires. The check polls resident memory between expansions and ends the
#: search with a report; the address-space cap (what the child had reserved
#: after its imports, plus the budget) refuses an allocation outright and the
#: child dies without one. Set equal, the allocation that trips the check is
#: the one the cap refuses, so the graceful exit could never win. The margin
#: covers what the imports and the model already hold resident, and what an
#: allocator reserves beyond what it touches.
MEMORY_HEADROOM_BYTES = 512 * 2**20

#: The click candidates per searched state (``MoveLister``): the biggest
#: same-colour regions, one representative pixel each. The report names it,
#: because a search that found nothing in a MOUSE game found nothing among
#: these clicks only.
MAX_CLICKS = MoveLister().max_clicks


#: The note a call gets when its last ``test_world_model()`` passed and the
#: harness's second test, after the call, did not: the model changed its files
#: after the pass, which may well be what it meant to do.
RETEST_FAILED_NOTE = (
    "[search] Your world model passed during this call but not after it, so it was not searched. "
    "(Files changed after the test are tested as they are now.)"
)

#: The line a report adds when its call looked the path up before the search
#: had run, and the search then bound one: the names exist only once a call has
#: ended, which a model reading a NameError cannot tell.
ASKED_EARLY_NOTE = (
    "This call looked up suggested_actions or suggested_history before the search had run — the search "
    "only runs once a call has ended. Both are available now, from your next call on."
)

#: What a call's output shows when it looked the path up too early: the name
#: unbound, or looked for on an object that never carries it.
_ASKED_EARLY = re.compile(
    r"name 'suggested_(?:actions|history)' is not defined|has no attribute 'suggested_(?:actions|history)'"
)


def asked_early(output: str) -> bool:
    """Whether a call's output shows it looking up the path its own search
    had not bound yet."""
    return _ASKED_EARLY.search(output) is not None


#: The planner's reason for ordering by path length when the model defines no
#: ``distance_to_win``: the ordinary case, and not worth a line.
_NO_HEURISTIC = "distance_to_win is not defined by this model"


class SearchHarnessError(RuntimeError):
    """Two runs of the harness over the same files disagreed: the search
    process's test failed on a model the REPL just passed, or the REPL could
    not reproduce a path the search predicted. A seal bug, never the model's;
    the game ends on it rather than handing the model a result it cannot
    trust."""


def make_planner(seconds: float, memory_bytes: int) -> BFSPlanner:
    """The planner a search runs: a win or a boundary only (``mode_max=0``),
    bounded by ``seconds`` and by memory under the child's cap.

    The repetition check is off (``allow_repetition=None``): it throws away a
    win whose path shows one board more than three times while hidden state
    changes, which a real game can require, and the seal has no evidence yet
    that the paths it rejects are wrong more often than right
    (2026-09-17)."""
    return BFSPlanner(
        mode_max=0,
        allow_repetition=None,
        max_duration=seconds,
        memory_limit_bytes=memory_bytes - MEMORY_HEADROOM_BYTES,
        move_lister=MoveLister(max_clicks=MAX_CLICKS),
    )


@dataclass(frozen=True)
class Provenance:
    """What a search searched: the files, the level, and where it started."""

    level: int
    #: The length of ``history`` the search started from; its paths begin
    #: after ``history[history_length - 1]``.
    history_length: int
    #: ``{file name: sha256 hex}`` for every file in the working directory.
    digests: dict[str, str]
    mouse: bool


@dataclass(frozen=True)
class SearchReport:
    """One search's outcome, ready for the model and the kernel."""

    provenance: Provenance
    result: SearchResult

    @property
    def plan(self) -> Plan | None:
        """The path found, or None."""
        return self.result.value if isinstance(self.result.value, Plan) else None

    def kernel_payload(self) -> dict[str, Any] | None:
        """What the kernel installs, or None when there is no path: the moves
        as the model knows them, each predicted turn as a history entry, and
        the digests of the files searched, which the kernel keeps the path
        bound only as long as its files match."""
        plan = self.plan
        if plan is None:
            return None
        turns = [
            {
                **model_move(turn.action),
                "outcome": turn.outcome,
                "frame": None if turn.frame is None else _frame_payload(turn.frame, self.provenance.level),
            }
            for turn in self.result.predicted
        ]
        return {
            "actions": [model_move(action) for action in plan.actions],
            "turns": turns,
            "digests": dict(self.provenance.digests),
        }

    def text(self, *, asked_early: bool = False) -> str:
        """The report as the model reads it, appended to the call's output.
        ``asked_early``: the call looked the path up before this search ran
        (:func:`asked_early`), which the report answers when it binds one."""
        prov = self.provenance
        lines = [f"[search] Your world model passed, so it was searched from history[{prov.history_length - 1}]."]
        plan = self.plan
        if plan is not None:
            n = len(plan.actions)
            moves = "move" if n == 1 else "moves"
            if plan.target == "win":
                fewest = " — the fewest your model allows" if plan.optimal else ""
                lines.append(f"Found: a win in {n} {moves}{fewest}.")
            elif plan.diagnostics.boundary_type == "render_error":
                lines.append(
                    f"Found: a board your world model cannot draw, after the last of {n} {moves}. "
                    "The search could not list clicks there, so it went no further; the last turn of "
                    "suggested_history has an all-wildcard board."
                )
                if self.result.boundary_reason:
                    lines.append(self.result.boundary_reason)
                lines.append("To reproduce: world_model.render(copy.deepcopy(suggested_history[-1].state))")
                lines.append(_boundary_win_line(self.result, mouse=prov.mouse))
            else:
                lines.append(f"Found: a move your world model does not predict, the last of {n} {moves}.")
                if self.result.boundary_reason:
                    lines.append(self.result.boundary_reason)
                lines.append(
                    "To reproduce: world_model.do_move(copy.deepcopy(suggested_history[-1].state), "
                    "suggested_actions[-1])"
                )
                lines.append(_boundary_win_line(self.result, mouse=prov.mouse))
            lines.append(
                "suggested_actions holds the path; suggested_history is history as your model predicts it "
                "after those moves."
            )
            if asked_early:
                lines.append(ASKED_EARLY_NOTE)
        elif isinstance(self.result.value, NoPlan):
            no_plan = self.result.value
            lines.append(
                _NO_PLAN_LINES[no_plan.tag].format(reason=no_plan.reason, n_moves=no_plan.detail.get("n_moves", ""))
            )
        else:
            lines.append("Found: nothing.")
        lines.append(f"Search space: {_coverage(self.result, mouse=prov.mouse)}")
        stuck = self.result.stuck_in
        if stuck and self.result.outcome == "timeout":
            lines.append(f"It was stopped inside {stuck} in your world model, a call that had not returned.")
        elif stuck and self.result.outcome == "died":
            lines.append(f"It died inside {stuck} in your world model.")
        if plan is None:
            if _complete(self.result):
                lines.append(_NOTHING_COMPLETE_MOUSE if prov.mouse else _NOTHING_COMPLETE)
            else:
                lines.append(_NOTHING_INCOMPLETE)
        diagnostics = self.result.value.diagnostics if self.result.value is not None else None
        reason = diagnostics.best_first_disable_reason if diagnostics is not None else None
        if reason is not None and reason != _NO_HEURISTIC:
            lines.append(f"Order: {reason}, so the search stopped using it and went by path length.")
        if prov.mouse:
            lines.append(_mouse_line(self.result))
        turns = "turn" if prov.history_length == 1 else "turns"
        # The file digests stay off the report: they are the kernel's, which
        # keeps the path bound while its files match them.
        lines.append(f"Searched: level {prov.level}, {prov.history_length} {turns} of history.")
        return "\n".join(lines)


#: The "Found" line for each kind of nothing the planner reports.
_NO_PLAN_LINES = {
    "exhausted": "Found: no win, and no move your world model does not predict.",
    "timeout": "Found: nothing before the search stopped.",
    "repetition": (
        "Found: a win, not suggested: its path returns to the same board {n_moves} times in a row of moves that "
        "change only what the board does not show — usually a rule that makes a move repeat for no visible effect."
    ),
    "invisible_edges": "Found: nothing: {reason}.",
}


#: What to do next when the whole space was searched and held no target.
_NOTHING_COMPLETE = (
    "The state space was fully searched, and no win or unknown state exists. Some rules have not been "
    "discovered yet. Update your world model or explore with moves."
)

#: The same, in a game with MOUSE: complete only for the clicks the search
#: tried, so a win may still need one it did not.
_NOTHING_COMPLETE_MOUSE = (
    "The states those clicks reach were fully searched, and no win or unknown state is among them. Some rules "
    "have not been discovered yet, or a win needs a click the search did not try. Update your world model or "
    "explore with moves."
)

#: What to do next when the search stopped early with no target.
_NOTHING_INCOMPLETE = (
    "The state space may or may not have a win or unknown state. If you believe BFS is the best algorithm, "
    "make your world model faster or more memory efficient, or update distance_to_win. If it's not the best "
    "algorithm, write your own planner/searcher."
)


def _boundary_win_line(result: SearchResult, *, mouse: bool) -> str:
    """What a boundary plan says about a win: a boundary is returned only when
    no win was found, and the search never looks past a boundary. In a game
    with MOUSE, only for the clicks the search tried."""
    if _complete(result):
        clicks = " with the clicks tried" if mouse else ""
        return (
            f"No win is reachable{clicks} without passing a move your model does not predict or a board it cannot "
            "draw; the search does not look past those."
        )
    return "A win may still exist: the search stopped before it could rule one out."


def _mouse_line(result: SearchResult) -> str:
    """Which clicks the search tried: one per same-colour region. After a
    complete search, whether that was every region of every board it
    searched; after one that stopped early, only the cap, since how far it got
    is the machine's and the report must read the same on every machine."""
    diagnostics = result.value.diagnostics if result.ok and result.value is not None else None
    if not _complete(result) or diagnostics is None:
        return (
            f"MOUSE: each searched state tried one click per same-color region, on its {MAX_CLICKS} "
            "largest regions — not every pixel."
        )
    if diagnostics.boards_over_click_cap == 0:
        return (
            "MOUSE: each searched state tried one click per same-color region, on every region of its board — "
            "not every pixel."
        )
    return (
        f"MOUSE: each searched state tried one click per same-color region, on its {MAX_CLICKS} largest regions "
        f"only — not every pixel. Some searched boards had more regions than that, up to "
        f"{diagnostics.most_click_regions}, and their other regions were never clicked."
    )


def _complete(result: SearchResult) -> bool:
    """Whether the search ran until nothing was left to search."""
    return (
        result.ok and result.value is not None and result.value.diagnostics.exit_reason in ("win_optimal", "exhausted")
    )


def _coverage(result: SearchResult, *, mouse: bool) -> str:
    """Whether every state reachable under the model was searched, or why not;
    in a game with MOUSE, complete only for the clicks the search tried.

    Nothing measured goes in — no duration, node count or memory figure — so
    the same search reads the same on every machine and replays."""
    if result.outcome == "timeout":
        return "not complete — the search ran past its time limit and was stopped."
    if result.outcome == "died":
        return "not complete — the search process died, most often from running out of memory."
    if not result.ok or result.value is None:
        last = (result.error or "").strip().splitlines()
        return f"not complete — the search failed: {last[-1] if last else 'no error was reported'}."
    reason = result.value.diagnostics.exit_reason
    if mouse and reason == "win_optimal":
        return "complete for the clicks tried — every shorter path using them was ruled out."
    if mouse and reason == "exhausted":
        return "complete for the clicks tried — every state they reach was searched."
    return {
        "win_optimal": "complete — every shorter path was ruled out.",
        "exhausted": "complete — every reachable state was searched.",
        "timeout": "not complete — the search ran out of time.",
        "timeout_after_win": "not complete — out of time before a shorter win was ruled out.",
        "memory_limit": "not complete — the search ran out of memory.",
        "memory_limit_after_win": "not complete — out of memory before a shorter win was ruled out.",
        "expansions": "not complete — the search hit its expansion budget.",
        "expansions_after_win": "not complete — the expansion budget ran out before a shorter win was ruled out.",
        "invisible_edges": "not complete — a move changed the state without changing the board.",
    }.get(reason, f"not complete ({reason}).")


def _frame_payload(frame: npt.NDArray[np.int8], level: int) -> dict[str, Any]:
    """A predicted board in the shape the kernel builds a turn from."""
    grid = np.ascontiguousarray(frame, dtype=np.int8)
    return {
        "grid_b64": base64.b64encode(grid.astype(np.uint8).tobytes()).decode("ascii"),
        "shape": [int(grid.shape[0]), int(grid.shape[1])],
        "step": 0,
        "level": level,
    }


def _grid(entry: TimelineEntry) -> npt.NDArray[np.int8]:
    """An observed settled frame as an array."""
    grid = harness_game._grid_from_payload(entry.frame.payload())  # pyright: ignore[reportPrivateUsage]
    assert grid is not None, "a recorded frame is well-formed"
    return grid[0]


@dataclass
class ModelJob:
    """A search of the working directory's model, as the search's process
    builds it (``search_process.SearchJob``): plain data, so nothing of the
    model's runs before that process's caps are on.

    ``entries`` and ``turns`` are the level's history, as the host keeps it
    and as the kernel receives it; ``model_actions`` the names the model's
    code sees as ``available_actions``; ``ascii_layout`` and ``matching``
    the kernel's, so the model loads and is held to its frames as the REPL
    held it."""

    directory: str
    entries: list[TimelineEntry]
    turns: list[dict[str, Any]]
    model_actions: list[str]
    ascii_layout: AsciiLayout
    matching: dict[str, Any]

    def prepare(self, announce: Callable[[dict[str, str]], None]) -> tuple[WorldModel, LevelTrace]:
        """Read the files, load the model from them, test it as the REPL did,
        and replay the level under it. A load or a test that fails on files
        the REPL passed is :class:`PreparationRefused`, unless the search ran
        out of memory on the way: that is its budget, and raises as
        ``MemoryError``, as any search failure does."""
        files = harness_test.working_files(self.directory)
        announce(harness_test.file_digests(files))
        with _memory_errors() as exhausted:
            try:
                model = PyWorldModel(files, self.turns, self.model_actions, ascii_layout=self.ascii_layout)
            except MemoryError:
                raise
            except Exception as error:
                if exhausted:
                    raise MemoryError from error
                raise PreparationRefused(
                    "the REPL passed world_model.py, and loading the same files in the search's process raised:\n"
                    + "".join(traceback.format_exception(error))
                ) from error
            # The test loads ``world_model.py`` afresh, outside the model's own
            # load, so it needs the kernel's alphabet stated: a sprite read with
            # ``from_ascii_grid`` at load reads in the alphabet it was written in.
            with harness_game._ascii_layout_scope(self.ascii_layout):  # pyright: ignore[reportPrivateUsage]
                tested = harness_test.validate_world_model(
                    os.path.join(model.directory, "world_model.py"),
                    model.history(),
                    list(self.model_actions),
                    rules=harness_test.MatchRules(**self.matching),
                )
            if tested and exhausted:
                raise MemoryError
        if tested:
            raise PreparationRefused(
                f"the REPL passed world_model.py, and the search's process's test of the same files failed:\n{tested!r}"
            )
        return model, _trace(model, self.entries)


@contextlib.contextmanager
def _memory_errors() -> Generator[list[bool]]:
    """Whether a ``MemoryError`` was raised inside the block, caught or not.

    The harness's test catches what the model's code raises and reports it as
    a failed result, which in the search's process reads as two harnesses
    disagreeing. Under the search's memory cap a ``MemoryError`` is the
    budget instead, so the caller reports it as the search failing, not the
    game. Watched with ``sys.monitoring``, which sees a raise however it is
    caught. A process with every tool id taken watches nothing. Only a flag
    is kept: holding the exceptions would hold their frames, and memory is
    short there.
    """
    monitoring = sys.monitoring
    tool = next((tool for tool in range(6) if monitoring.get_tool(tool) is None), None)
    raised: list[bool] = []
    if tool is None:
        yield raised
        return

    def on_raise(code: CodeType, offset: int, exception: BaseException) -> None:
        if isinstance(exception, MemoryError) and not raised:
            raised.append(True)

    monitoring.use_tool_id(tool, "seal search memory")
    monitoring.register_callback(tool, monitoring.events.RAISE, on_raise)
    monitoring.set_events(tool, monitoring.events.RAISE)
    try:
        yield raised
    finally:
        monitoring.set_events(tool, monitoring.events.NO_EVENTS)
        monitoring.register_callback(tool, monitoring.events.RAISE, None)
        monitoring.free_tool_id(tool)


async def search_level(
    *,
    directory: str,
    entries: Sequence[TimelineEntry],
    turns: list[dict[str, Any]],
    model_actions: list[str],
    game_actions: list[str],
    level: int,
    seconds: float,
    memory_bytes: int,
    ascii_layout: AsciiLayout,
    matching: dict[str, Any] | None = None,
    status: StatusHandle = NULL_STATUS,
) -> SearchReport:
    """Search the working directory's model from the end of ``entries``, in a
    process of its own (:class:`ModelJob`): it tests the model again first,
    and ``seconds`` and ``memory_bytes`` bound that test as they bound the
    search.

    ``turns`` are the kernel payloads of ``entries`` — the history the model's
    code reads. ``model_actions`` are the names its code sees as
    ``available_actions``; ``game_actions`` the game's own action set as the
    model is taught it, which is what the search enumerates.
    ``ascii_layout`` is the kernel's, which the model's code runs under in the
    search's process. ``matching`` is the matching settings in force
    (``harness.test.MatchRules``), so the test holds the frames to what the
    REPL's did. Raises :class:`SearchHarnessError` when that test fails on
    files the REPL passed.
    """
    job = ModelJob(
        directory=directory,
        entries=list(entries),
        turns=turns,
        model_actions=list(model_actions),
        ascii_layout=ascii_layout,
        matching=dict(matching or {}),
    )
    engine_actions = [name for name in (to_engine_action(action) for action in game_actions) if name is not None]
    result = await run_search(
        make_planner(seconds, memory_bytes),
        job,
        engine_actions,
        seed=derive_seed("search", level, len(entries)),
        hard_timeout=seconds + SPAWN_GRACE_S,
        memory_limit_bytes=memory_bytes,
        status=status,
    )
    if result.outcome == "refused":
        raise SearchHarnessError(result.error or "the search's process refused the model")
    provenance = Provenance(
        level=level,
        history_length=len(entries),
        digests=dict(result.digests),
        mouse="MOUSE" in game_actions,
    )
    return SearchReport(provenance=provenance, result=result)


def _trace(model: PyWorldModel, entries: Sequence[TimelineEntry]) -> LevelTrace:
    """The level as the search reads it: each observed step, and the model's
    state after it, replayed here under the model that searches — its states
    must pickle against this process's ``world_model`` module."""
    steps: list[Step] = []
    states: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    with _quiet(), harness_game._model_process_settings(), model.active():  # pyright: ignore[reportPrivateUsage]
        for entry in entries:
            if entry.action is None:
                state = model.initial_state()
            else:
                model.step(state, entry.action)
            steps.append(Step(action=entry.action, frame=_grid(entry), outcome=entry.outcome))  # pyright: ignore[reportArgumentType]
            states.append(harness_test._safe_deepcopy(state) or {})  # pyright: ignore[reportPrivateUsage]
    return LevelTrace(steps=steps, states=states)


@contextlib.contextmanager
def _quiet() -> Generator[None]:
    """Swallow what the model's code prints while this process runs it."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield
