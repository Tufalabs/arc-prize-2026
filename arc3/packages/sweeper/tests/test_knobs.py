"""How values are drawn: the range, the shape, and the independence between
knobs that lets results recorded under different knob sets pool."""

from __future__ import annotations

import pickle
import random
import statistics

import pytest

from sweeper.knobs import Choice, Field, Knob, LogUniformInt, UniformInt, check_picklable, draw_values, knob_rng


def test_log_uniform_stays_in_range_and_is_log_shaped() -> None:
    """A uniform draw over 1..1000 would put 90% of the arms above 100 and
    answer nothing about the small end; log-uniform splits the arms evenly
    between the decades, so 1-vs-10 gets as many as 100-vs-1000."""
    draw = LogUniformInt(1, 1000)
    values = [draw(random.Random(seed)) for seed in range(2000)]

    assert min(values) >= 1
    assert max(values) <= 1000
    # Half the draws below the geometric mean (~31.6), not below the
    # arithmetic one (~500) — the property that separates the two shapes.
    assert 20 <= statistics.median(values) <= 50
    per_decade = [sum(1 for v in values if lo <= v < hi) for lo, hi in ((1, 10), (10, 100), (100, 1001))]
    assert all(count > len(values) / 5 for count in per_decade), per_decade


def test_a_knob_keeps_its_value_when_another_knob_is_added() -> None:
    """The property that makes `missing` usable: a job recorded before a knob
    existed drew the SAME values for the knobs it did have, so old and new
    jobs describe one design rather than two. A shared RNG stream would
    redraw everything the moment the knob set changed."""
    first = {"a": Knob(draw=LogUniformInt(1, 1000), apply=Field("inner.a"))}
    second = dict(first) | {"b": Knob(draw=Choice((True, False)), apply=Field("inner.b"))}

    for seed in range(20):
        assert draw_values(first, seed)["a"] == draw_values(second, seed)["a"]


def test_declaration_order_does_not_change_the_design() -> None:
    """Two experiment files listing the same knobs in different orders are
    the same experiment, and must produce the same arms."""
    a = Knob(draw=LogUniformInt(1, 100), apply=Field("inner.a"))
    b = Knob(draw=Choice((1, 2, 3)), apply=Field("inner.b"))
    assert draw_values({"a": a, "b": b}, 7) == draw_values({"b": b, "a": a}, 7)


def test_a_lambda_knob_is_refused_where_it_can_still_be_explained() -> None:
    """Knobs travel into game subprocesses with the agent. Left to pickle,
    this surfaces from inside a spawned child with no mention of the knob —
    and only in subprocess mode, so every in-process test would pass."""
    knobs = {"bad": Knob(draw=lambda rng: rng.random(), apply=Field("inner.x"))}
    with pytest.raises(ValueError, match="'bad' does not pickle"):
        check_picklable(knobs, batch="arm")


def test_the_helpers_pickle() -> None:
    """The reason the helpers are frozen dataclasses rather than closures."""
    knob = Knob(draw=LogUniformInt(1, 1000), apply=Field("inner.max_moves"), missing=100)
    assert pickle.loads(pickle.dumps(knob)) == knob


def test_field_walks_a_dotted_path() -> None:
    """One knob reaching a nested config object, without a closure."""

    class Leaf:
        def __init__(self) -> None:
            self.value = 0

    class Root:
        def __init__(self) -> None:
            self.leaf = Leaf()

    root = Root()
    Field("leaf.value")(root, "ignored", 42)
    assert root.leaf.value == 42


def test_a_path_can_reach_through_a_mapping_key() -> None:
    """Sampling profiles live under a name, not on an attribute, so a knob
    that cannot index a mapping cannot reach the one place per-call sampling
    is configured."""

    class Params:
        def __init__(self) -> None:
            self.max_tokens = 0

    class Config:
        def __init__(self) -> None:
            self.sampling_profiles = {"play": Params()}

    class Agent:
        def __init__(self) -> None:
            self.llm_config = Config()

    agent = Agent()
    Field("llm_config.sampling_profiles[play].max_tokens")(agent, "ignored", 4096)
    assert agent.llm_config.sampling_profiles["play"].max_tokens == 4096


def test_uniform_int_covers_both_ends_evenly() -> None:
    """The linear sibling of LogUniformInt: ends included, no low-end bunching."""
    draws = [UniformInt(30, 300)(knob_rng(seed, "t")) for seed in range(4000)]
    assert min(draws) == 30 and max(draws) == 300
    assert all(isinstance(d, int) for d in draws)
    # Uniform means mass in the top half ~ half; log-uniform would put ~25% there.
    top_half = sum(d > 165 for d in draws) / len(draws)
    assert 0.45 < top_half < 0.55, top_half
    with pytest.raises(ValueError, match="low <= high"):
        UniformInt(5, 4)
