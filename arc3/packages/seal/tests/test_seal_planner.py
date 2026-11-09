"""Behavioural tests for :class:`seal.planner.BFSPlanner`.

Covers the target priority hierarchy (win > boundary > untried-action >
unvisited > untried-pair), the sticky mode floor, energy-aware Pareto
dedup, best-first ordering and its auto-disable, the repetition guard, and
the give-up tags.

``GraphModel`` is a real ``WorldModel`` over a hand-rolled finite directed
graph. The actual search runs against it — the real dedup, the real
cascade, the real heap — so a scenario is a graph literal rather than a
pile of per-test mocking. A missing (node, action) entry raises, which is
how error boundaries get made.
"""

from __future__ import annotations

import copy
import os
import pickle
import zlib
from typing import Any, Sequence

import numpy as np
import numpy.typing as npt

from platypus.contract import Action
from seal.harness.game import Unknown
from seal.model import Prediction, WorldModel
from seal.move_lister import MoveLister
from seal.planner import (
    BFSPlanner,
    NoPlan,
    Plan,
    _current_rss_bytes,
    _deepcopy_state,
    action_signature,
)
from seal.rng import derive_seed
from seal.trace import LevelTrace, Step

#: ``{node: {action name: (next node, kind, outcome[, energy delta])}}``
Graph = dict[str, dict[str, tuple[Any, ...]]]

SEED = derive_seed(20260725, "plan", 0)
FRAME = np.zeros((4, 4), dtype=np.int8)


class GraphModel(WorldModel):
    """A world model whose state space is a finite directed graph.

    State is ``{'node': str, 'outcome': str[, 'energy': int]}``. Each node
    renders to a flat 4x4 grid of a per-node colour, so distinct nodes give
    distinct frames. An action with no graph entry raises, and the base
    class turns that into an ``error`` prediction — which is how a test
    manufactures an error boundary.
    """

    def __init__(self, graph: Graph, start: str = "S", energy_start: int | None = None) -> None:
        super().__init__()
        self._graph = graph
        self._start = start
        self._energy_start = energy_start
        # Deterministic per-node colour. sandbox's fixture uses
        # `abs(hash(node)) % 15 + 1`, which is PYTHONHASHSEED-dependent and
        # can collide across nodes; crc32 is stable across processes and
        # runs, so a scenario means the same thing every time.
        self._colours = {node: zlib.crc32(node.encode()) % 15 + 1 for node in graph}

    def _initial_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {"node": self._start, "outcome": "playing"}
        if self._energy_start is not None:
            state["energy"] = self._energy_start
        return state

    def _step(self, state: dict[str, Any], action: Action) -> Prediction:
        entry = self._graph[state["node"]][action.name]
        nxt, kind, outcome = entry[0], entry[1], entry[2]
        denergy = entry[3] if len(entry) > 3 else -1
        if kind == "error":
            raise RuntimeError(f"injected error from {state['node']} action={action.name}")
        if kind == "unknown":
            raise Unknown(f"never seen from {state['node']}")
        state["node"] = nxt
        state["outcome"] = outcome
        if "energy" in state:
            state["energy"] += denergy
        return Prediction(kind="claim")

    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        return np.full((4, 4), self._colours[state["node"]], dtype=np.int8)


def make_trace(states: list[dict[str, Any]], actions: Sequence[Action | None] = ()) -> LevelTrace:
    """A trace standing in for observed history.

    ``states[i]`` is the model's predicted state after ``steps[i]``; the
    first step is the level-initial observation and carries no action.
    Built directly rather than through ``observe`` because these scenarios
    are about search, not about the consistency check.
    """
    if not actions:
        actions = [None] * len(states)
    steps = [Step(action=a, frame=FRAME, outcome="playing") for a in actions]
    return LevelTrace(steps=steps, states=list(states))


def planner(**kw: Any) -> BFSPlanner:
    """A planner with the features each test does not care about off.

    ``energy_label`` defaults to ``None`` so graphs with no energy field do
    not trip the strict Pareto contract, and ``allow_repetition`` to
    ``None`` because most scenarios cycle deliberately.
    """
    kw.setdefault("move_lister", MoveLister())
    kw.setdefault("allow_repetition", None)
    kw.setdefault("energy_label", None)
    kw.setdefault("prune_invisible_edges", False)
    return BFSPlanner(**kw)


def run(
    model: GraphModel,
    plan_with: BFSPlanner,
    *,
    available: Sequence[str] = ("ACTION1", "ACTION2"),
    trace: LevelTrace | None = None,
) -> Plan | NoPlan:
    if trace is None:
        trace = make_trace([model.initial_state()])
    return plan_with.plan(model, trace, available, seed=SEED)


def names(outcome: Plan | NoPlan) -> list[str]:
    assert isinstance(outcome, Plan), f"expected a plan, got {outcome}"
    return [a.name for a in outcome.actions]


# --- wins -----------------------------------------------------------------


def test_shortest_win_is_returned_and_proven_optimal() -> None:
    # Two winning paths at different lengths. The search records the first
    # win rather than returning it, keeps going until nothing cheaper is
    # left, and only then reports it — which is what makes `optimal` a
    # claim rather than a hope.
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("B", "claim", "playing")},
        "A": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("A", "claim", "playing")},
        "B": {"ACTION1": ("B2", "claim", "playing"), "ACTION2": ("B", "claim", "playing")},
        "B2": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("B2", "claim", "playing")},
        "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("WIN", "claim", "won")},
    }
    outcome = run(GraphModel(graph), planner())
    assert names(outcome) == ["ACTION1", "ACTION1"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.optimal) == ("win", True)


def test_an_unknown_transition_is_the_boundary_when_no_win_exists() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
        "A": {"ACTION1": ("B", "unknown", "playing"), "ACTION2": ("A", "claim", "playing")},
        "B": {"ACTION1": ("B", "claim", "playing"), "ACTION2": ("B", "claim", "playing")},
    }
    outcome = run(GraphModel(graph), planner())
    assert names(outcome) == ["ACTION1", "ACTION1"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.boundary_type) == ("boundary", "unknown")
    # A fallback target makes no optimality claim: it is the first hit, not
    # a searched-for shortest.
    assert outcome.optimal is False


def test_an_action_whose_code_raises_is_an_error_boundary() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("S", "error", "playing"), "ACTION2": ("S", "claim", "playing")},
    }
    outcome = run(GraphModel(graph), planner())
    assert names(outcome) == ["ACTION1"]
    assert isinstance(outcome, Plan)
    assert outcome.diagnostics.boundary_type == "error"


class UndrawableModel(GraphModel):
    """A graph whose node ``X`` cannot be drawn."""

    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        if state["node"] == "X":
            raise ValueError("no board for X")
        return super()._render(state)


UNDRAWABLE: Graph = {
    "S": {"ACTION1": ("X", "claim", "playing"), "ACTION6": ("S", "claim", "playing")},
    "X": {"ACTION1": ("WIN", "claim", "won"), "ACTION6": ("X", "claim", "playing")},
    "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION6": ("WIN", "claim", "won")},
}


def test_in_a_click_game_a_board_the_model_cannot_draw_is_a_target_of_its_own() -> None:
    # Clicks are listed from the board, so a state that does not draw cannot
    # be expanded: the path to it is the target, marked apart from Unknown
    # and from code that raised, and the win behind it is not reached.
    outcome = run(UndrawableModel(UNDRAWABLE), planner(), available=("ACTION1", "ACTION6"))
    assert names(outcome) == ["ACTION1"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.boundary_type) == ("boundary", "render_error")


def test_a_game_without_clicks_never_draws_a_state_while_searching() -> None:
    # Nothing is listed from the board, so the undrawable state is expanded
    # like any other and the win behind it is found.
    outcome = run(UndrawableModel(UNDRAWABLE), planner(), available=("ACTION1",))
    assert names(outcome) == ["ACTION1", "ACTION1"]
    assert isinstance(outcome, Plan)
    assert outcome.target == "win"


# --- the fallback cascade -------------------------------------------------


def test_nothing_reachable_at_any_mode_is_no_plan() -> None:
    graph: Graph = {"S": {"ACTION1": ("S", "claim", "playing"), "ACTION2": ("S", "claim", "playing")}}
    outcome = run(GraphModel(graph), planner(mode_max=0))
    assert isinstance(outcome, NoPlan)
    assert outcome.tag == "exhausted"


def test_mode_one_returns_an_action_type_never_tried() -> None:
    graph: Graph = {
        "S": {
            "ACTION1": ("S", "claim", "playing"),
            "ACTION2": ("S", "claim", "playing"),
            "ACTION3": ("S", "claim", "playing"),
        }
    }
    p = planner(mode_max=1, tried_action_signatures={("ACTION1",), ("ACTION2",)})
    outcome = run(GraphModel(graph), p, available=("ACTION1", "ACTION2", "ACTION3"))
    assert names(outcome) == ["ACTION3"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("untried_action", 1)


def test_action_signatures_split_clicks_by_the_colour_they_land_on() -> None:
    # Clicking a blue cell and a red one are different experiments; two
    # clicks on the same colour are the same one. Without this, the first
    # click anywhere would mark all clicking as tried.
    frame = np.zeros((8, 8), dtype=np.int8)
    frame[5, 3] = 4
    frame[2, 6] = 9
    blue = Action("ACTION6", row=5, col=3)
    red = Action("ACTION6", row=2, col=6)
    assert action_signature(blue, frame) == ("ACTION6", 4)
    assert action_signature(red, frame) == ("ACTION6", 9)

    frame[0, 0] = 4
    assert action_signature(Action("ACTION6", row=0, col=0), frame) == action_signature(blue, frame)

    assert action_signature(Action("ACTION1"), frame) == ("ACTION1",)
    # No frame to read the colour from: degrade to name-only rather than
    # inventing a signature.
    assert action_signature(blue, None) == ("ACTION6",)


def test_mode_one_treats_a_click_on_a_new_colour_as_untried() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("S", "claim", "playing"), "ACTION6": ("S", "claim", "playing")},
    }
    model = GraphModel(graph)
    node_colour = int(model.render({"node": "S", "outcome": "playing"})[0, 0])
    other_colour = (node_colour + 1) % 16
    p = planner(mode_max=1, tried_action_signatures={("ACTION1",), ("ACTION6", other_colour)})
    outcome = run(model, p, available=("ACTION1", "ACTION6"))
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("untried_action", 1)
    assert outcome.actions[-1].name == "ACTION6"


def test_mode_two_returns_a_state_never_visited_for_real() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
        "A": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("A", "claim", "playing")},
    }
    model = GraphModel(graph)
    p = planner(mode_max=2, tried_action_signatures={("ACTION1",), ("ACTION2",)})
    outcome = run(model, p, trace=make_trace([model.initial_state()]))
    assert names(outcome) == ["ACTION1"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("unvisited", 2)


def test_mode_three_returns_a_state_action_pair_never_tried() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
        "A": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
    }
    model = GraphModel(graph)
    s = {"node": "S", "outcome": "playing"}
    a = {"node": "A", "outcome": "playing"}
    # History walked S -[1]-> A -[2]-> S, so both nodes are visited and both
    # action types tried; only the untried pairs remain.
    trace = make_trace([dict(s), dict(a), dict(s)], [None, Action("ACTION1"), Action("ACTION2")])
    p = planner(mode_max=3, tried_action_signatures={("ACTION1",), ("ACTION2",)})
    outcome = run(model, p, trace=trace)
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("untried_pair", 3)


def test_a_boundary_outranks_every_novelty_fallback() -> None:
    # Priority is not "whatever was found first": a boundary is more
    # informative than a novelty probe, so it wins even at the top mode.
    graph: Graph = {"S": {"ACTION1": ("S", "unknown", "playing"), "ACTION2": ("S", "claim", "playing")}}
    model = GraphModel(graph)
    s = {"node": "S", "outcome": "playing"}
    trace = make_trace([dict(s), dict(s), dict(s)], [None, Action("ACTION1"), Action("ACTION2")])
    outcome = run(model, planner(mode_max=3), trace=trace)
    assert names(outcome) == ["ACTION1"]
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("boundary", 0)


# --- the sticky mode floor ------------------------------------------------


def test_a_raised_floor_stops_the_search_at_the_first_target_it_holds() -> None:
    # What the floor buys: at floor 2 the previous call already established
    # this model has nothing below mode 2, so the first unvisited state in
    # hand ends the search rather than the queue emptying.
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
        "A": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("A", "claim", "playing")},
    }
    model = GraphModel(graph)
    kw: dict[str, Any] = {"mode_max": 2, "tried_action_signatures": {("ACTION1",), ("ACTION2",)}}

    exhaustive = run(model, planner(**kw))
    assert isinstance(exhaustive, Plan)
    assert (exhaustive.diagnostics.mode_used, exhaustive.diagnostics.exit_reason) == (2, "exhausted")

    early = run(model, planner(min_mode=2, **kw))
    assert isinstance(early, Plan)
    assert early.diagnostics.exit_reason == "early_exit"
    assert early.diagnostics.nodes_explored < exhaustive.diagnostics.nodes_explored


def test_the_floor_does_not_hide_a_better_target_that_turned_up_anyway() -> None:
    # A raised floor is a guess about what this model has. The cascade is
    # still consulted top-down at the end, so a boundary found while running
    # at floor 3 outranks the novelty probe the floor was aiming at.
    graph: Graph = {"S": {"ACTION1": ("S", "unknown", "playing"), "ACTION2": ("S", "claim", "playing")}}
    s = {"node": "S", "outcome": "playing"}
    trace = make_trace([dict(s), dict(s), dict(s)], [None, Action("ACTION1"), Action("ACTION2")])
    outcome = run(GraphModel(graph), planner(mode_max=3, min_mode=3), trace=trace)
    assert isinstance(outcome, Plan)
    assert (outcome.target, outcome.diagnostics.mode_used) == ("boundary", 0)


# --- dedup ----------------------------------------------------------------


def test_without_energy_a_second_arrival_at_a_known_state_is_pruned() -> None:
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("BOOST", "claim", "playing")},
        "BOOST": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("BOOST", "claim", "playing")},
        "A": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("A", "claim", "playing")},
    }
    p = planner(energy_label=None, mode_max=0)
    outcome = run(GraphModel(graph), p)
    assert isinstance(outcome, NoPlan)
    assert outcome.diagnostics.seen_states == 3  # S, A, BOOST


def test_with_energy_a_longer_path_survives_if_it_arrives_richer() -> None:
    # The Pareto rule: reaching the same place later is worth exploring if
    # you get there with more fuel, because what you can do next differs.
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("BOOST", "claim", "playing", +10)},
        "BOOST": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("BOOST", "claim", "playing")},
        "A": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("A", "claim", "playing")},
    }
    p = planner(energy_label="energy", mode_max=0)
    model = GraphModel(graph, energy_start=10)
    outcome = run(model, p)
    assert isinstance(outcome, NoPlan)
    # One node more than the plain-dedup case above: A is re-expanded from
    # BOOST because that arrival carries more energy.
    assert outcome.diagnostics.nodes_explored == 4


# --- best-first ordering --------------------------------------------------


class HeuristicModel(GraphModel):
    """A graph model with a model-written ``distance_to_win``."""

    def __init__(self, graph: Graph, distances: dict[str, Any], **kw: Any) -> None:
        super().__init__(graph, **kw)
        self._distances = distances
        self.calls = 0

    def _distance_to_win(self, state: dict[str, Any]) -> Any:
        self.calls += 1
        value = self._distances[state["node"]]
        if isinstance(value, Exception):
            raise value
        return value


def test_a_heuristic_that_misbehaves_disables_best_first_without_failing_the_search() -> None:
    # The heuristic is model-written, so it is expected to be wrong or
    # missing sometimes. That must cost the search its ordering, not its
    # result — and the reason has to be recorded, or a silently-degraded
    # search looks the same as a healthy one.
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("S", "claim", "playing")},
        "A": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("A", "claim", "playing")},
        "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("WIN", "claim", "won")},
    }
    for distances, expected in [
        ({"S": ValueError("model bug"), "A": 1, "WIN": 0}, "raised"),
        ({"S": "close", "A": 1, "WIN": 0}, "non-numeric"),
    ]:
        model = HeuristicModel(graph, distances)
        outcome = run(model, planner())
        assert names(outcome) == ["ACTION1", "ACTION1"]
        assert isinstance(outcome, Plan)
        assert outcome.diagnostics.best_first is False
        assert expected in (outcome.diagnostics.best_first_disable_reason or "")
        # Disabled for the rest of the call, not retried per node.
        assert model.calls == 1


def test_a_model_with_no_heuristic_falls_back_without_pretending_to_have_one() -> None:
    # Absence is the common case, not an error. sandbox surfaces it
    # as a KeyError on the first call and routes it through the same
    # auto-disable; here it is a None, and must land in the same place.
    graph: Graph = {
        "S": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("S", "claim", "playing")},
        "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("WIN", "claim", "won")},
    }
    outcome = run(GraphModel(graph), planner())
    assert isinstance(outcome, Plan)
    assert outcome.diagnostics.best_first is False
    assert "not defined" in (outcome.diagnostics.best_first_disable_reason or "")


def test_best_first_still_returns_the_shortest_win() -> None:
    # The point of the anytime-optimal hunt: a heuristic that leads to a
    # long win first must not make that the answer. Here the heuristic
    # prefers the 3-step branch, and the 2-step win must still come back.
    graph: Graph = {
        "S": {"ACTION1": ("LONG", "claim", "playing"), "ACTION2": ("SHORT", "claim", "playing")},
        "LONG": {"ACTION1": ("LONG2", "claim", "playing"), "ACTION2": ("LONG", "claim", "playing")},
        "LONG2": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("LONG2", "claim", "playing")},
        "SHORT": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("SHORT", "claim", "playing")},
        "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("WIN", "claim", "won")},
    }
    distances = {"S": 10, "LONG": 1, "LONG2": 1, "SHORT": 9, "WIN": 0}
    outcome = run(HeuristicModel(graph, distances), planner())
    assert isinstance(outcome, Plan)
    assert outcome.diagnostics.best_first is True
    assert (len(outcome.actions), outcome.optimal) == (2, True)


# --- the post-win budget --------------------------------------------------

#: A heuristic that walks into the 3-step win first, with a 2-step win
#: elsewhere — so "what the residual search buys you" is observable.
_TWO_WINS: Graph = {
    "S": {"ACTION1": ("LONG", "claim", "playing"), "ACTION2": ("SHORT", "claim", "playing")},
    "LONG": {"ACTION1": ("LONG2", "claim", "playing"), "ACTION2": ("LONG", "claim", "playing")},
    "LONG2": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("LONG2", "claim", "playing")},
    "SHORT": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("SHORT", "claim", "playing")},
    "WIN": {"ACTION1": ("WIN", "claim", "won"), "ACTION2": ("WIN", "claim", "won")},
}
_TOWARDS_LONG = {"S": 10, "LONG": 1, "LONG2": 1, "SHORT": 9, "WIN": 0}


def test_the_post_win_budget_stops_the_residual_search_and_costs_optimality() -> None:
    # The whole point of the cap: past the first win the search is only
    # looking for something shorter, and at depth that can eat the entire
    # budget. Capping it returns the win in hand — which is the trade, and
    # the reason the result must stop claiming to be optimal.
    unbounded = run(HeuristicModel(_TWO_WINS, _TOWARDS_LONG), planner())
    assert isinstance(unbounded, Plan)
    assert (len(unbounded.actions), unbounded.optimal) == (2, True)

    capped = run(HeuristicModel(_TWO_WINS, _TOWARDS_LONG), planner(max_expansions_after_first_win=0))
    assert isinstance(capped, Plan)
    assert capped.target == "win"
    assert len(capped.actions) == 3  # the long win, found first
    assert (capped.optimal, capped.diagnostics.exit_reason) == (False, "expansions_after_win")


def test_the_post_win_wall_clock_does_the_same() -> None:
    # sandbox spends this budget in seconds (60s against a 600s
    # total), so the clock is the mechanism of record; the expansion count
    # above is its deterministic companion. Zero seconds is the one value
    # that makes the check fire without a timing race.
    capped = run(HeuristicModel(_TWO_WINS, _TOWARDS_LONG), planner(max_duration_after_first_win=0.0))
    assert isinstance(capped, Plan)
    assert len(capped.actions) == 3
    assert (capped.optimal, capped.diagnostics.exit_reason) == (False, "timeout_after_win")


def test_a_later_win_does_not_extend_the_post_win_budget() -> None:
    # Stamped once. If later wins reset it, a model that keeps finding
    # marginally-shorter wins could hold the search open indefinitely,
    # which is exactly what the cap exists to prevent.
    p = planner(max_expansions_after_first_win=6)
    outcome = run(HeuristicModel(_TWO_WINS, _TOWARDS_LONG), p)
    assert isinstance(outcome, Plan)
    # Six more expansions is enough to reach the shorter win, and the
    # budget must still be measured from the first one.
    assert outcome.diagnostics.nodes_explored <= 7


# --- resource bounds ------------------------------------------------------


def test_an_expansion_budget_ends_the_search_as_a_timeout() -> None:
    # The deterministic companion to the wall clock: a bound that does not
    # depend on how fast the machine is, so a recorded run and its replay
    # explore the same space. It shares the `timeout` tag with the clock and
    # the memory limit -- the tag drives speculate escalation -- while
    # `exit_reason` keeps which bound fired, because only the machine-
    # dependent ones make a run unreproducible.
    graph: Graph = {f"N{i}": {"ACTION1": (f"N{i + 1}", "claim", "playing")} for i in range(500)}
    graph["N500"] = {"ACTION1": ("N500", "claim", "playing")}
    p = planner(mode_max=0, max_expansions=10)
    outcome = run(GraphModel(graph, start="N0"), p, available=("ACTION1",))
    assert isinstance(outcome, NoPlan)
    assert (outcome.tag, outcome.diagnostics.exit_reason) == ("timeout", "expansions")
    assert outcome.diagnostics.nodes_explored == 10


# --- the repetition guard -------------------------------------------------


class TreadmillModel(WorldModel):
    """Ticks a counter and burns energy without changing the picture — the
    "do this a thousand times to win" failure mode the model is prone to.

    ``visible`` controls whether the counter reaches the render, and
    ``changes_state`` whether anything outside energy moves at all.
    """

    def __init__(self, *, visible: bool = False, changes_state: bool = True) -> None:
        super().__init__()
        self.visible = visible
        self.changes_state = changes_state

    def _initial_state(self) -> dict[str, Any]:
        return {"outcome": "playing", "energy": 100, "x": 0}

    def _step(self, state: dict[str, Any], action: Action) -> Prediction:
        if self.changes_state:
            state["x"] += 1
        state["energy"] -= 1
        return Prediction(kind="claim")

    def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
        # Energy reaches the frame, so without the energy normalisation
        # every frame would look distinct and the guard would never fire.
        colour = ((state["x"] if self.visible else 0) & 0x1F) << 2 | (state["energy"] & 0x03)
        return np.full((8, 8), colour & 0x7F, dtype=np.int8)


def check_path(model: WorldModel, steps: int, allow: int | None) -> tuple[int | None, NoPlan | None]:
    p = BFSPlanner(allow_repetition=allow, energy_label="energy")
    return p._check_repetition(model, model.initial_state(), [Action("ACTION1")] * steps)


def test_a_path_that_changes_the_picture_every_step_passes() -> None:
    max_rep, rejection = check_path(TreadmillModel(visible=True), steps=5, allow=10)
    assert (max_rep, rejection) == (1, None)


def test_a_path_that_never_changes_the_picture_is_rejected() -> None:
    max_rep, rejection = check_path(TreadmillModel(), steps=12, allow=10)
    assert rejection is not None
    assert rejection.tag == "repetition"
    # Between N+1 identical frames there are N transitions that advanced
    # state without changing the render — the count the prompt calls out.
    assert max_rep is not None
    assert rejection.detail["n_moves"] == max_rep - 1


def test_the_energy_bar_alone_does_not_make_frames_look_distinct() -> None:
    # Energy reaches the render, so without normalising it to its initial
    # value every frame differs and the guard silently never fires. This is
    # the case that makes the guard useless if it regresses.
    _, rejection = check_path(TreadmillModel(), steps=12, allow=10)
    assert rejection is not None


def test_the_threshold_accepts_exactly_n_repeats_and_rejects_n_plus_one() -> None:
    # 4 steps -> 5 identical frames, which is the allowance exactly.
    max_rep, rejection = check_path(TreadmillModel(), steps=4, allow=5)
    assert (max_rep, rejection) == (5, None)
    max_rep, rejection = check_path(TreadmillModel(), steps=5, allow=5)
    assert max_rep == 6
    assert rejection is not None


def test_repetition_that_only_burns_energy_is_not_a_model_bug() -> None:
    # The frames pile up, but no non-energy state moved: these are dedup
    # misses, a search inefficiency rather than something to ask the model
    # about. Without this condition the guard fires on healthy models.
    max_rep, rejection = check_path(TreadmillModel(changes_state=False), steps=15, allow=3)
    assert max_rep is not None and max_rep > 3
    assert rejection is None


def test_the_guard_can_be_turned_off() -> None:
    max_rep, rejection = check_path(TreadmillModel(), steps=15, allow=None)
    assert (max_rep, rejection) == (None, None)


def test_a_rejected_path_comes_back_as_no_plan() -> None:
    # End to end: the guard sits between the search and the caller, so a
    # repetitive win must not escape as a plan.
    # State really moves (the node flips) but the picture never does, which
    # is the shape the guard exists to catch.
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing")},
        "A": {"ACTION1": ("S", "claim", "playing")},
    }

    class Stuck(GraphModel):
        def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
            return np.zeros((4, 4), dtype=np.int8)

    model = Stuck(graph)
    p = planner(mode_max=1, allow_repetition=0, tried_action_signatures=set())
    outcome = run(model, p, available=("ACTION1",))
    assert isinstance(outcome, NoPlan)
    assert outcome.tag == "repetition"


# --- invisible edges ------------------------------------------------------


def test_an_edge_that_moves_hidden_state_without_moving_the_picture_stops_the_search() -> None:
    # If do_move mutates state the render never shows, no target found by
    # searching it means anything — so the search aborts rather than
    # spending its budget on a model that is known to be wrong.
    class Hidden(WorldModel):
        def _initial_state(self) -> dict[str, Any]:
            return {"outcome": "playing", "counter": 0}

        def _step(self, state: dict[str, Any], action: Action) -> Prediction:
            state["counter"] += 1
            return Prediction(kind="claim")

        def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
            return np.zeros((4, 4), dtype=np.int8)

    p = planner(prune_invisible_edges=True, mode_max=0)
    model = Hidden()
    outcome = p.plan(model, make_trace([model.initial_state()]), ("ACTION1",), seed=SEED)
    assert isinstance(outcome, NoPlan)
    assert outcome.tag == "invisible_edges"
    assert "counter" in outcome.detail["changed_keys"]


def test_the_invisible_edge_check_is_off_by_default() -> None:
    class Hidden(WorldModel):
        def _initial_state(self) -> dict[str, Any]:
            return {"outcome": "playing", "counter": 0}

        def _step(self, state: dict[str, Any], action: Action) -> Prediction:
            state["counter"] += 1
            return Prediction(kind="claim")

        def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
            return np.zeros((4, 4), dtype=np.int8)

    model = Hidden()
    p = planner(mode_max=0, max_expansions=20)
    outcome = p.plan(model, make_trace([model.initial_state()]), ("ACTION1",), seed=SEED)
    assert isinstance(outcome, NoPlan)
    assert outcome.tag == "timeout"


def test_a_fat_state_value_is_truncated_before_it_reaches_a_prompt() -> None:
    # These details are spliced into what the model is asked next, so one
    # fat field would swamp the message. A plain list, not an ndarray:
    # numpy summarises its own repr, so it would never reach the cap.
    class Fat(WorldModel):
        def _initial_state(self) -> dict[str, Any]:
            return {"outcome": "playing", "blob": list(range(2000)), "counter": 0}

        def _step(self, state: dict[str, Any], action: Action) -> Prediction:
            state["counter"] += 1
            return Prediction(kind="claim")

        def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
            return np.zeros((4, 4), dtype=np.int8)

    model = Fat()
    p = planner(prune_invisible_edges=True, mode_max=0)
    outcome = p.plan(model, make_trace([model.initial_state()]), ("ACTION1",), seed=SEED)
    assert isinstance(outcome, NoPlan)
    assert "truncated" in outcome.detail["example_state_repr"]


def test_the_changed_keys_a_prompt_lists_come_out_sorted() -> None:
    # This dict is listed to the model line by line in the speculate prompt,
    # so its order is prompt bytes. The union of two key sets iterates in
    # PYTHONHASHSEED order, which nothing pins — an unsorted walk makes the
    # same states produce a different prompt in the next process.
    class Hidden(WorldModel):
        def _initial_state(self) -> dict[str, Any]:
            return {"outcome": "playing", "zulu": 0, "alpha": 0, "mike": 0}

        def _step(self, state: dict[str, Any], action: Action) -> Prediction:
            for key in ("zulu", "alpha", "mike"):
                state[key] += 1
            return Prediction(kind="claim")

        def _render(self, state: dict[str, Any]) -> npt.NDArray[np.int8]:
            return np.zeros((4, 4), dtype=np.int8)

    model = Hidden()
    p = planner(prune_invisible_edges=True, mode_max=0)
    outcome = p.plan(model, make_trace([model.initial_state()]), ("ACTION1",), seed=SEED)
    assert isinstance(outcome, NoPlan)
    assert list(outcome.detail["changed_keys"]) == ["alpha", "mike", "zulu"]


def test_the_hard_cap_leaves_the_soft_monitor_room_to_win() -> None:
    """The two bounds must not sit on the same number.

    ``BFSPlanner.memory_limit_bytes`` is polled between node expansions and
    exits with a ``memory_limit`` reason the report can name; ``RLIMIT_AS``
    is the kernel refusing an allocation outright. Equal, the allocation that
    trips the monitor is the one that dies, so the graceful exit is
    unreachable and the search ends in a raw death.
    """
    from seal.agent import SealAgent
    from seal.search import make_planner

    hard = SealAgent.__dataclass_fields__["search_memory_bytes"].default
    soft = make_planner(60.0, hard).memory_limit_bytes
    assert isinstance(hard, int) and soft is not None
    assert 0 < soft < hard


# --- a bound firing with a target already in hand --------------------------


def _boundary_plus_a_long_corridor() -> Graph:
    """A boundary one step from the root, and a corridor to get lost in.

    ``ACTION2`` from the root is unknown, which is a mode-0 boundary found
    in the first expansion. ``ACTION1`` walks a chain long enough that any
    bound fires well before it ends.
    """
    graph: Graph = {f"N{i}": {"ACTION1": (f"N{i + 1}", "claim", "playing")} for i in range(2000)}
    graph["N2000"] = {"ACTION1": ("N2000", "claim", "playing")}
    graph["N0"]["ACTION2"] = ("B", "unknown", "playing")
    graph["B"] = {"ACTION1": ("B", "claim", "playing"), "ACTION2": ("B", "unknown", "playing")}
    return graph


def test_an_expansion_budget_still_returns_the_boundary_it_found() -> None:
    """The bound ends the hunt for something better; it does not throw away
    what the search already holds. Every bound test before this one used a
    graph with no boundary in it, so "hit a bound" and "came back
    empty-handed" were never told apart -- and a replay whose SIGKILL landed
    on the budget's heels lost a boundary found at depth 1 and asked the
    model to speculate instead."""
    outcome = run(GraphModel(_boundary_plus_a_long_corridor(), start="N0"), planner(mode_max=0, max_expansions=20))

    assert isinstance(outcome, Plan), f"expected the boundary, got {outcome}"
    assert outcome.diagnostics.search_result == "boundary"
    assert outcome.diagnostics.exit_reason == "expansions"
    assert names(outcome) == ["ACTION2"]
    # The expansion budget stands in for all three bounds: they differ only
    # in which flag the loop sets, and every one falls through to the same
    # cascade below it. The clock cannot be pinned to a node count without
    # flaking, and the memory poll reads whole-process RSS -- any ceiling
    # low enough to fire is already crossed before the root is expanded, so
    # there is nothing in hand to keep.


def test_a_bound_with_nothing_in_hand_is_still_a_give_up() -> None:
    """The converse, so the two tests above cannot pass by never bounding."""
    corridor: Graph = {f"N{i}": {"ACTION1": (f"N{i + 1}", "claim", "playing")} for i in range(2000)}
    corridor["N2000"] = {"ACTION1": ("N2000", "claim", "playing")}

    outcome = run(GraphModel(corridor, start="N0"), planner(mode_max=0, max_expansions=20), available=("ACTION1",))

    assert isinstance(outcome, NoPlan)
    assert (outcome.tag, outcome.diagnostics.exit_reason) == ("timeout", "expansions")


# --- the state copier and the memory probe --------------------------------
#
# Both are invisible in a search's result until they are wrong, and then they
# are wrong quietly: a copier that breaks object sharing changes the dedup key
# and so the whole search tree, and a probe reading another process's memory
# ends a search on a limit that was never reached.


def test_the_state_copier_is_indistinguishable_from_deepcopy() -> None:
    """A copy whose pickle differs from ``copy.deepcopy``'s is a different
    state to the dedup check, which keys on exactly those bytes.

    The shapes below are the ones where a hand-written copier drifts: shared
    references (pickle emits a back-reference for the second sighting), a
    tuple of immutables (``copy.deepcopy`` returns it unchanged rather than
    rebuilding it), a self-reference, and the types that look atomic but are
    rebuilt -- ``set``, ``frozenset``, ``bytearray``.

    A tuple reached again while its own items are being copied is the one that
    needs care: it has to end up as a single shared object, or the cycle is
    replaced by two tuples and the pickle grows a member the original had as a
    back-reference.
    """
    shared: list[int] = [1, 2, 3]
    shared_tuple = (shared, shared)
    recursive: dict[str, Any] = {"outcome": "playing"}
    recursive["self"] = recursive
    cyclic_tuple: tuple[list[Any]] = ([],)
    cyclic_tuple[0].append(cyclic_tuple)

    cases: list[dict[str, Any]] = [
        {"row": 3, "col": 4, "outcome": "playing", "energy": 12.5},
        {"a": shared, "b": shared},
        {"t": shared_tuple, "same": shared_tuple},
        {"atoms": (1, 2, 3), "mutables": ([1], [2])},
        {"grid": np.arange(16, dtype=np.int8).reshape(4, 4)},
        {"s": {1, 2}, "fs": frozenset({3}), "ba": bytearray(b"xy")},
        {"nested": {"deep": [{"x": (1, 2)}, {"y": [3, 4]}]}},
        {"cycle": cyclic_tuple},
        recursive,
    ]
    for state in cases:
        mine = _deepcopy_state(state, {})
        theirs = copy.deepcopy(state)
        assert pickle.dumps(mine, protocol=pickle.HIGHEST_PROTOCOL) == pickle.dumps(
            theirs, protocol=pickle.HIGHEST_PROTOCOL
        ), f"copy diverged for {state!r}"


def test_the_memory_probe_reads_this_process_not_the_one_that_forked_it() -> None:
    """``/proc/self`` resolves at open time, so a descriptor held across a fork
    reports the parent's memory in the child -- and a search in that child
    would stop on a limit it never reached, or never stop at all.
    """

    def independent_read() -> int:
        """This process's RSS, opened fresh, so it cannot be a stale handle."""
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    _current_rss_bytes()  # open the descriptor here, so the child inherits it
    parent = independent_read()

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns to pytest
        try:
            os.close(read_fd)
            # Move the child's RSS well clear of the parent's, so reading the
            # wrong process is a large unambiguous error rather than drift.
            ballast = bytearray(256 * 2**20)
            ballast[::4096] = b"x" * len(ballast[::4096])
            os.write(write_fd, f"{_current_rss_bytes()},{independent_read()}".encode())
        finally:
            os._exit(0)

    os.close(write_fd)
    with os.fdopen(read_fd) as fh:
        probed, actual = (int(part) for part in fh.read().split(","))
    os.waitpid(pid, 0)

    # Absolute sizes are the machine's business; what must hold is that the
    # probe answered for the child rather than for the process it forked from.
    tolerance = 16 * 2**20
    assert abs(probed - actual) < tolerance, f"probe said {probed}, the child's own statm says {actual}"
    # And the two processes must be far enough apart that the check above
    # could not have passed on the parent's number by coincidence.
    assert abs(actual - parent) > 2 * tolerance, f"ballast did not separate the processes ({actual} vs {parent})"


def test_a_cheaper_route_to_a_seen_state_is_not_thrown_away() -> None:
    """Dedup must not drop the cheaper arrival at an already-generated state.

    Two routes reach ``M``: ``S-C-M`` in two actions and ``S-A-B-M`` in
    three. The heuristic likes ``A`` and ``B`` and dislikes ``C``, so the
    long route generates ``M`` first. Keying ``seen`` on the state alone then
    discards the cheap arrival, and the three-action win is never built.

    The stamp is the visible half but not the damage: win-finding here is
    anytime-optimal — the first win is recorded, the frontier is pruned below
    its cost, and an empty queue is what proves shortest. A route deleted
    before it was ever queued cannot be found by that phase, so the queue
    empties honestly and the agent plays the longer path. Actions and energy
    are both budgeted per level, so the extra one is spent, not cosmetic.
    """
    graph: Graph = {
        "S": {"ACTION1": ("A", "claim", "playing"), "ACTION2": ("C", "claim", "playing")},
        "A": {"ACTION1": ("B", "claim", "playing"), "ACTION2": ("A", "claim", "playing")},
        "B": {"ACTION1": ("M", "claim", "playing"), "ACTION2": ("B", "claim", "playing")},
        "C": {"ACTION1": ("M", "claim", "playing"), "ACTION2": ("C", "claim", "playing")},
        "M": {"ACTION1": ("W", "claim", "won"), "ACTION2": ("M", "claim", "playing")},
        "W": {"ACTION1": ("W", "claim", "won"), "ACTION2": ("W", "claim", "won")},
    }
    # C is made unattractive so the long route runs first; everything on it
    # looks one step from the win.
    distances = {"S": 5, "A": 1, "B": 1, "M": 1, "C": 9, "W": 0}
    outcome = run(HeuristicModel(graph, distances), planner())

    assert names(outcome) == ["ACTION2", "ACTION1", "ACTION1"], "the cheaper route to M was discarded"
    assert isinstance(outcome, Plan) and outcome.optimal
