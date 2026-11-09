"""Golden vector for the search's move ordering.

``tests/data/rng_contract_vector.json`` came from sandbox with the
whale's search, and it pins what makes a recorded search replay: the seed
derivation and the per-node orderings it produces. A change here shows up in
a reference as a search that takes a different path — several LLM calls
before anything looks wrong — so this is the cheap early signal.

Regenerate deliberately, never to make a red test green. (The vector also
carries the whale's premove sweeps, which the seal does not have.)
"""

from __future__ import annotations

import json
from pathlib import Path

import arcengine
import numpy as np
import numpy.typing as npt
import pytest

from platypus.contract import Action
from seal.move_lister import MoveLister
from seal.rng import derive_rng, derive_seed

VECTOR = json.loads((Path(__file__).parent / "data" / "rng_contract_vector.json").read_text())


def _name(action: Action) -> str:
    """Render an action the way the vector spells it. The vector was
    generated against sandbox's ``arcengine.ActionInput``, whose
    click data is ``{'x': col, 'y': row}``; ``Action`` carries the same
    coordinate under the (row, col) names the platypus contract fixes."""
    if action.name == "ACTION6":
        return f"CLICK({action.col},{action.row})"
    return action.name


def _actions(action_ids: list[int]) -> tuple[str, ...]:
    """The vector stores engine action ids; the agent contract speaks
    names, in the same order."""
    return tuple(arcengine.GameAction.from_id(i).name for i in action_ids)


def _frame() -> npt.NDArray[np.int8]:
    return np.array(VECTOR["frame"], dtype=np.int8)


def test_derive_seed_is_stable() -> None:
    # The primitive itself: blake2b over "|"-joined parts, big-endian. If
    # this drifts, everything below drifts with it and no amount of matching
    # call structure will keep the two repos in step.
    for case in VECTOR["derive_seed"]:
        assert derive_seed(*case["parts"]) == case["seed"], case["parts"]


@pytest.mark.parametrize("case", VECTOR["nodes"], ids=lambda c: f"plan{c['plan_index']}_click{c['with_frame']}")
def test_node_orderings_match_vector(case: dict[str, object]) -> None:
    plan_seed = derive_seed(VECTOR["action_seed"], "plan", case["plan_index"])
    assert plan_seed == case["plan_seed"]
    frame = _frame() if case["with_frame"] else None
    for k, expected in enumerate(case["orders"]):  # type: ignore[arg-type]
        lister = MoveLister(max_clicks=4, max_clicks_per_color=1)
        moves = lister.list_moves(_actions(case["available_actions"]), frame, derive_rng(plan_seed, "node", k))  # type: ignore[arg-type]
        assert [_name(a) for a in moves] == expected, f"node {k}"


def test_node_orderings_differ_across_nodes_and_plans() -> None:
    # The reason per-node derivation exists at all: a single stream per
    # search would give every node the same permutation, and a single seed
    # per game would repeat the whole search. Both would be reproducible and
    # both would lose the exploration spread.
    orders = [tuple(o) for case in VECTOR["nodes"] if not case["with_frame"] for o in case["orders"]]
    assert len(set(orders)) > 1
