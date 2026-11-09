"""Which configuration a game gets, and which knobs are allowed to decide it.

The arm is chosen from ``ctx.pass_index``, so these cover the mapping
(pass to seed to values), the refusal of knobs aimed at values the solver
reads before any game starts, the delegation to the wrapped agent, and what
an LLM agent's arms are checked against and carry into a game child.
"""

from __future__ import annotations

import pickle
import random
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest

import taaf.game
from platypus import Agent, AgentSolver, GiveUp, LLMAgent, LLMAgentSolver, LLMConfig, MoveContext, RunConfig
from platypus.context import Produced
from platypus.reference_agents import ANSWERS
from sweeper.agent import LLMSweepAgent, SweepAgent
from sweeper.collect import parse_sweep_lines
from sweeper.knobs import RECORD, Choice, Design, Field, Knob, LogUniformInt
from taaf.benchmark import Benchmark
from taaf.game_examples import flip_game

#: Two wrong answers, then the mirror: a lane wins at its third submission
#: and holds the win from there.
WIN_SCRIPT = ("black", "black") + ("flip",) * 4


@dataclass
class ScriptedAgent(Agent):
    """Submits a fixed answer list, in order. ``cap`` bounds how many of
    them it will submit, which is what an arm knob here varies."""

    script: tuple[str, ...] = WIN_SCRIPT
    cap: int = 100
    flag: bool = False
    #: A mapping a knob can index into (``name[key]`` segments), mirroring
    #: the shape of an agent's sampling profiles.
    table: dict[str, int] = field(default_factory=lambda: {"play": 1})
    played: list[str] = field(default_factory=lambda: list[str]())

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """The scripted answers, up to the cap; a cap of zero gives up unanswered."""
        if self.cap == 0:
            return GiveUp("nothing to try")
        for name in self.script[: self.cap]:
            self.played.append(name)
            ctx.submit(ANSWERS[name](view))
        return None


ARM = {"cap": Knob(draw=LogUniformInt(1, 6), apply=Field("inner.cap"), missing=100)}
RUN = {"flag": Knob(draw=Choice((True, False)), apply=Field("inner.flag"), missing=False)}
DESIGN = Design(run=RUN, arm=ARM)


def _sweeper(**overrides: object) -> SweepAgent:
    """A sweep agent over the scripted agent, silent."""
    kwargs: dict[str, object] = {
        "inner": ScriptedAgent(),
        "design": DESIGN,
        "run_seed": 3,
        "arm_seeds": (11, 12, 13),
        "announce": False,
    }
    kwargs.update(overrides)
    return SweepAgent(**kwargs)  # type: ignore[arg-type]


def _discard(submission: taaf.game.Submission, *, produced: Produced | None = None) -> None:
    """A submit channel that keeps nothing; arming never answers."""
    del submission, produced


def _context(pass_index: int) -> MoveContext:
    """A context that reports the given pass."""
    return MoveContext(game_id="g", history=[], rng=random.Random(0), submit=_discard, pass_index=pass_index)


def test_the_run_batch_is_applied_at_construction() -> None:
    """Run knobs must land before the solver reads the template, which is
    what makes engine settings reachable at all."""
    agent = _sweeper(run_seed=3)
    assert agent.inner.flag == agent.run_values["flag"]  # type: ignore[attr-defined]
    assert set(agent.run_values) == {"flag"}


def test_each_pass_gets_its_own_arm_and_the_same_pass_always_gets_the_same_one() -> None:
    """The arm IS the pass: reproducible from the seed list, and different
    between passes or there is no sweep."""
    values: list[int] = []
    for pass_index in range(3):
        agent = _sweeper()
        agent._arm(_context(pass_index))
        values.append(int(agent.arm_values["cap"]))
        assert agent.arm_seed == (11, 12, 13)[pass_index]

    again = _sweeper()
    again._arm(_context(1))
    assert again.arm_values["cap"] == values[1]
    assert len(set(values)) > 1, f"three arms drew the same value: {values}"


def test_one_seed_covers_every_pass() -> None:
    """The Kaggle shape: a submission is one design point, and its passes are
    replicates rather than different configurations."""
    agent = _sweeper(arm_seeds=(99,))
    agent._arm(_context(0))
    first = agent.arm_values["cap"]
    other = _sweeper(arm_seeds=(99,))
    other._arm(_context(7))
    assert other.arm_values["cap"] == first


def test_a_pass_with_no_arm_is_refused() -> None:
    """Silently reusing an arm would make two passes look like one
    configuration measured twice."""
    agent = _sweeper(arm_seeds=(1, 2))
    with pytest.raises(ValueError, match="pass_index 2 has no arm"):
        agent._arm(_context(2))


def test_an_arm_knob_aimed_at_a_template_only_value_is_refused() -> None:
    """The failure this prevents does not look like a bug: the knob varies,
    nothing happens, and the analysis reports that it does not matter."""
    bad = Design(arm={"conc": Knob(draw=Choice((1, 2)), apply=Field("run_config.max_concurrent"))})
    with pytest.raises(ValueError, match="cannot vary per arm"):
        _sweeper(design=bad)

    # max_runtime_s_per_game reads per GAME but not per MOVE: run_game freezes
    # it into its budget before do_move, and the arm batch is applied inside
    # that call. Refused for the same reason as the rest.
    per_game = Design(arm={"budget": Knob(draw=Choice((1.0, 2.0)), apply=Field("run_config.max_runtime_s_per_game"))})
    with pytest.raises(ValueError, match="cannot vary per arm"):
        _sweeper(design=per_game)


def test_the_printed_design_parses_back() -> None:
    """A Kaggle submission's log is the only record of what it ran, so the
    format is load-bearing and must not drift from its reader."""
    agent = _sweeper()
    parsed = parse_sweep_lines(agent.describe())
    assert parsed["run_seed"] == 3
    assert parsed["run_values"] == agent.run_values
    assert sorted(parsed["arms"]) == [0, 1, 2]
    assert parsed["arms"][1]["seed"] == 12


async def test_the_arm_reaches_the_game_it_was_drawn_for() -> None:
    """End to end, the assertion that matters: the value recorded for an arm
    is the value that governed play. Recording without applying, or applying
    the template's value everywhere, both pass every unit test above."""
    agent = _sweeper(run_config=RunConfig(process_mode="inprocess", max_concurrent=1))
    games: list[taaf.game.Game] = [flip_game("a")]
    benchmark = Benchmark(label="arm-reaches", games=games, solver=AgentSolver(agent=agent), n_passes=3)
    await benchmark.run()

    caps = [DESIGN.arm_values_for(seed)["cap"] for seed in (11, 12, 13)]
    for run, cap in zip(benchmark.game_runs, caps, strict=True):
        assert sum(run.actions_per_level) <= cap, f"pass played past its arm's cap of {cap}"
    assert max(caps) > min(caps), "the arms must differ or this proves nothing"


def widen_context(values: Mapping[str, Any]) -> dict[str, Any]:
    """A coordinate transform: two independent knobs onto two coupled fields.

    Module-level because it rides into a game subprocess with the design — a
    closure or a local function could not make the trip.
    """
    window = values["without_margin"] + values["margin"]
    return {"inner.cap": window, "inner.script": ("flip",) * min(6, values["margin"])}


DERIVED = Design(
    arm={
        "without_margin": Knob(draw=Choice((2, 4)), apply=RECORD, missing=2),
        "margin": Knob(draw=Choice((1, 2)), apply=RECORD, missing=1),
    },
    derive_arm=widen_context,
)


def test_a_derivation_turns_coordinates_into_coupled_fields() -> None:
    """The knobs vary independently and record independently; the fields they
    imply are written once, after the whole batch is drawn — so a constrained
    pair of fields becomes an unconstrained pair of knobs without reshaping
    the agent to suit the experiment."""
    agent = _sweeper(design=DERIVED, arm_seeds=(11, 12, 13))
    agent._arm(_context(0))

    values = agent.arm_values
    assert set(values) == {"without_margin", "margin"}, "the coordinates are what the analysis regresses on"
    assert agent.inner.cap == values["without_margin"] + values["margin"]  # type: ignore[attr-defined]


def test_recording_knobs_without_a_derivation_are_refused() -> None:
    """They would vary in every table and configure nothing — the same silent
    nothing the target check exists to prevent, from the other direction."""
    orphaned = Design(arm={"x": Knob(draw=Choice((1, 2)), apply=RECORD)})
    with pytest.raises(ValueError, match="record only"):
        _sweeper(design=orphaned)


def test_a_derivation_aimed_at_a_template_only_field_is_refused() -> None:
    """The derivation is checked like a knob, and at construction: a path that
    cannot take effect per arm should cost an error, not an allocation."""

    def to_template(values: Mapping[str, Any]) -> dict[str, Any]:
        """Aims at a value the solver reads before any game starts."""
        return {"run_config.max_concurrent": values["x"]}

    bad = Design(arm={"x": Knob(draw=Choice((1, 2)), apply=RECORD)}, derive_arm=to_template)
    with pytest.raises(ValueError, match="cannot vary per arm"):
        _sweeper(design=bad)


def test_a_knob_aimed_at_a_field_that_does_not_exist_is_refused() -> None:
    """The silent null this apparatus cannot otherwise see.

    `setattr` invents an attribute rather than failing, so a knob pointed at a
    renamed field draws, records, fills its column and changes nothing — which
    reads as a measured null. The other target checks cannot catch it: they
    ask whether a path MAY vary per arm, not whether it names anything.
    """
    design = Design(arm={"typo": Knob(draw=Choice((1, 2)), apply=Field("inner.no_such_field"))})
    with pytest.raises(ValueError, match="does not exist on the agent"):
        SweepAgent(inner=ScriptedAgent(), design=design, run_seed=1, arm_seeds=(11,), announce=False)


def test_a_knob_aimed_at_a_real_field_is_accepted() -> None:
    """The other half: the check must not refuse the paths that do resolve,
    including one indexing a mapping."""
    design = Design(
        arm={
            "cap": Knob(draw=Choice((1, 2)), apply=Field("inner.cap")),
            "keyed": Knob(draw=Choice((3, 4)), apply=Field("inner.table[play]")),
        }
    )
    SweepAgent(inner=ScriptedAgent(), design=design, run_seed=1, arm_seeds=(11,), announce=False)


def test_a_knob_aimed_at_a_missing_mapping_key_is_refused() -> None:
    """``set_by_path`` on a bracketed last segment would CREATE the key — an
    entry beside the real ones that nothing reads, the same silent null one
    segment deeper than a misspelled attribute."""
    design = Design(arm={"typo": Knob(draw=Choice((1, 2)), apply=Field("inner.table[typo]"))})
    with pytest.raises(ValueError, match="does not exist on the agent"):
        SweepAgent(inner=ScriptedAgent(), design=design, run_seed=1, arm_seeds=(11,), announce=False)


def _derive_to_missing(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """A derivation whose output path names nothing on the agent."""
    return {"inner.no_such_field": values["x"]}


def _derive_to_new_key(values: Mapping[str, Any]) -> Mapping[str, Any]:
    """A derivation targeting a key that exists only in the mapping a run
    knob installs — valid in application order, invalid before it."""
    return {"inner.table[extra]": values["x"]}


def test_a_run_derivation_may_target_what_a_run_knob_just_created() -> None:
    """Validation follows application order: the direct run knobs land
    first, so a derivation may write into a container a knob just rebuilt.
    Checked against the construction graph instead, this design would be
    falsely refused."""
    design = Design(
        run={
            "table": Knob(draw=Choice(({"play": 1, "extra": 0},)), apply=Field("inner.table")),
            "x": Knob(draw=Choice((5,)), apply=RECORD),
        },
        derive_run=_derive_to_new_key,
    )
    agent = _sweeper(design=design)
    inner = agent.inner
    assert isinstance(inner, ScriptedAgent)
    assert inner.table["extra"] == 5, "the derived write must land in the knob-installed mapping"


def test_an_arm_derivation_may_target_what_an_arm_knob_just_created() -> None:
    """The arm mirror of the run case: the game's copy gets the arm knobs
    first, so a derivation may write into a container an arm knob just
    rebuilt. Simulated against the template's own graph instead of a
    knobbed copy, this valid design would be refused at construction."""
    design = Design(
        arm={
            "table": Knob(draw=Choice(({"play": 1, "extra": 0},)), apply=Field("inner.table")),
            "x": Knob(draw=Choice((5,)), apply=RECORD),
        },
        derive_arm=_derive_to_new_key,
    )
    agent = _sweeper(design=design)
    agent._arm(_context(0))
    inner = agent.inner
    assert isinstance(inner, ScriptedAgent)
    assert inner.table["extra"] == 5, "the derived write must land in the knob-installed mapping"


class _TwoFacedDerive:
    """Legal on its first call, template-only after — the impure ``Derive``
    the at-application check exists to catch: construction validates the map
    the derive returned THEN, the game applies the map it returns NOW."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        """One clean face for the construction probe, then the smuggle."""
        self.calls += 1
        if self.calls == 1:
            return {}
        return {"run_config.max_concurrent": values["x"]}


def test_an_impure_derivation_cannot_smuggle_a_template_only_path_into_an_arm() -> None:
    """Without per-arm legality on the map actually applied, the second face
    would set a field the solver read before the fan-out — a knob column
    that varies while nothing changes, the measured null again."""
    design = Design(arm={"x": Knob(draw=Choice((5,)), apply=RECORD)}, derive_arm=_TwoFacedDerive())
    agent = _sweeper(design=design, arm_seeds=(11,))

    with pytest.raises(ValueError, match="cannot vary per arm"):
        agent._arm(_context(0))


def test_a_run_derivation_aimed_at_a_missing_field_is_refused() -> None:
    """Derived paths bypass ``Knob.apply``, so the knob-target sweep cannot
    see them; they are validated against the built agent all the same."""
    bad = Design(run={"x": Knob(draw=Choice((1, 2)), apply=RECORD)}, derive_run=_derive_to_missing)
    with pytest.raises(ValueError, match="does not exist on the agent"):
        _sweeper(design=bad)


def test_an_arm_derivation_aimed_at_a_missing_field_is_refused() -> None:
    """The arm batch is applied a whole job later, in the game process — the
    refusal must land at construction, where it costs no allocation."""
    bad = Design(arm={"x": Knob(draw=Choice((1, 2)), apply=RECORD)}, derive_arm=_derive_to_missing)
    with pytest.raises(ValueError, match="does not exist on the agent"):
        _sweeper(design=bad)


@dataclass(kw_only=True)
class _WindowedAgent(LLMAgent):
    """An LLM agent with the duck's window rule: its window must fit the
    engine's, which it settles from its own unless told the engine is not its
    to write."""

    window: int = 100
    engine_window: int = 150

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Never played here."""
        del ctx, view
        return GiveUp("not played")

    def prepare_for_backend(self, *, settle_engine: bool = True) -> None:
        """Settle the engine window, or check against it."""
        if settle_engine:
            self.engine_window = self.window
        if self.window > self.engine_window:
            raise ValueError(f"window {self.window} exceeds the engine's {self.engine_window}")


def test_an_arm_the_wrapped_agent_refuses_is_refused_at_construction() -> None:
    """A combination the agent refuses would otherwise fail inside every game
    of its pass. The engine is the sweep's, so the arm is checked against it
    rather than allowed to widen it."""
    design = Design(arm={"window": Knob(draw=Choice((100, 400)), apply=Field("inner.window"), missing=100)})
    inner = _WindowedAgent()
    with pytest.raises(ValueError, match="is refused by the wrapped agent: window 400 exceeds the engine's 150"):
        LLMSweepAgent(
            inner=inner, design=design, arm_seeds=tuple(range(10)), llm_config=inner.llm_config, announce=False
        )


def test_the_child_copy_carries_no_engine_on_the_wrapped_agent_either() -> None:
    """Every game child unpickles what it is handed, and the wrapped agent's
    ``llm_config`` is its own reference to the engine spec."""
    import ullm.backend
    import ullm.backends.stub
    import ullm.factory

    backend = ullm.backends.stub.StubBackend.from_script("8", on_exhausted="loop")
    inner = _WindowedAgent(
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(max_tokens=50)},
        )
    )
    agent = LLMSweepAgent(inner=inner, llm_config=inner.llm_config, announce=False)
    child = pickle.loads(pickle.dumps(LLMAgentSolver(agent=agent)._child_agent(agent)))  # pyright: ignore[reportPrivateUsage]
    assert child.llm_config.backend_factory is None
    assert child.inner.llm_config.backend_factory is None
    assert agent.inner is inner and inner.llm_config.backend_factory is not None, "the template keeps its engine"
