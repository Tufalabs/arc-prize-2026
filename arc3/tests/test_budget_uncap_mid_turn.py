"""A per-turn token cap must stop binding the moment its reason expires.

The cap exists to hand a slot to the next game, so it is retired once nothing
is queued (`platypus.slots`, `Modeler.uncap_when_queue_drained`). The catch is
that the decision used to be taken once per *turn* — and a turn is many LLM
calls with tool work between them, so it can outlive the queue that justified
capping it by hours. A game was then cut short by a bound nobody would still
impose, and a truncated round cannot be salvaged: the work is simply lost.

Driven end-to-end rather than at the manager, because the thing under test is
a race between two subsystems: the solver draining its queue and publishing
the tally, and a turn already in flight reading it. A unit test would have to
assert the race away.

Marked ``local``: it starts six games in subprocesses and reads what the
solver's scheduler does with them, which a shared CI runner cannot be held
to. ``make test-local`` runs it, and so does ``test-full`` before a merge.

The games never end on their own — the backend answers every call with another
tool call — so a game ends only when something stops it. That makes the two
outcomes cleanly separable: a capped game dies on its cap, an uncapped one
runs to `max_tool_rounds`.
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import arcengine
import numpy as np
import pytest
import taaf.benchmark
import taaf.game
import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.factory
import ullm.tool

from platypus import Action, GiveUp, LLMAgent, LLMAgentSolver, LLMConfig, LLMMoveContext, Observation, RunConfig

#: Tokens the backend spends per call, before any cap applies. A range rather
#: than a constant so nothing under test can depend on a round number.
COST_RANGE = (700, 1300)

#: The per-turn allowance. Four-ish calls, so a capped game dies well inside
#: ``MAX_ROUNDS`` and the two outcomes cannot be confused.
CAP = 4_000

#: A turn that is never truncated ends here instead. Comfortably more calls
#: than the cap allows.
MAX_ROUNDS = 20

N_GAMES = 6
MAX_CONCURRENT = 2


@dataclass
class EndlessBackend(ullm.backends.stub.StubBackend):
    """Answers every call with another tool call, and charges for it.

    Two behaviours the stub does not have and this test needs. It **honours
    ``max_tokens``**: asked for less than a call would cost, it returns what
    fits and reports ``stop_reason="max_tokens"``, which is what the per-turn
    cap relies on to bind at all. And it **never volunteers an end** — every
    response carries a tool call — so a turn runs until something outside it
    decides otherwise, which is the situation the cap is for.

    Costs and delays are drawn from a seeded RNG per instance: a fixed cost
    would let an off-by-one in the cap arithmetic pass, and a fixed delay
    would make the queue drain at a reproducible round, which is exactly the
    coincidence this test must not depend on.
    """

    rng_seed: int = 0
    _rng: random.Random = field(default_factory=lambda: random.Random(0), init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self._rng = random.Random(self.rng_seed)

    async def _generate(
        self, sampling_params: ullm.backend.SamplingParameters, llm_input: Any
    ) -> AsyncGenerator[Any, None]:
        """One call: a little wall-clock, then the scripted tool call."""
        await asyncio.sleep(self._rng.uniform(0.005, 0.02))
        yield llm_input

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Charge the call, clipped to whatever ``max_tokens`` allows."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        wanted = self._rng.randint(*COST_RANGE)
        allowed = getattr(sampling_params, "max_tokens", None)
        if allowed is not None and allowed < wanted:
            element.token_usage = ullm.backend.TokenUsage(prompt_tokens=1, completion_tokens=allowed)
            element.stop_reason = "max_tokens"
        else:
            element.token_usage = ullm.backend.TokenUsage(prompt_tokens=1, completion_tokens=wanted)
            element.stop_reason = "tool_use"


@dataclass
class NoopTool(ullm.tool.Tool):
    """Does nothing, so a round costs one call and no more."""

    name: str = "noop"
    description: str = "Does nothing."
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Answer immediately: the tool is only here to keep the turn going."""
        del arguments
        return ullm.conv.StringContent(text="ok")


@dataclass
class NeverEndingGame(taaf.game.Game):
    """A board that never finishes, so only the agent can end the game."""

    label: str = field(default="endless", init=True)

    def _start_game(self, session: taaf.game.RunSession) -> taaf.game.GameState:
        del session
        self.game_id = self.label
        self.number_of_levels = 1
        self.base_actions_per_level = [10_000]
        self.grid_size = (64, 64)
        return self._build()

    def _execute_action(self, action: Any) -> taaf.game.GameState:
        del action
        return self._build()

    def _build(self) -> taaf.game.GameState:
        raw = arcengine.FrameDataRaw(
            game_id=self.game_id or self.label,
            state=arcengine.GameState.NOT_FINISHED,
            levels_completed=0,
            win_levels=1,
            available_actions=[1, 2, 3, 4],
        )
        raw.frame = [np.zeros((64, 64), dtype=np.int8)]
        return taaf.game.GameState(raw=raw)


#: Marks in the run's ``solver_note``, so the outcome is readable off the
#: benchmark rather than out of the agent.
CAPPED = "ended on the per-turn cap"
UNCAPPED = "ran out its rounds"


@dataclass
class BudgetedAgent(LLMAgent):
    """Runs one endless turn under a per-turn cap, and reports how it ended.

    ``honour_lift`` is the switch this test turns: with it the manager is told
    how to ask whether the cap still applies, without it the cap is decided
    once and holds for the turn — which is the behaviour before the fix.
    """

    honour_lift: bool = True

    async def _do_move(self, ctx: LLMMoveContext, obs: Observation) -> Action | GiveUp:
        """One turn, then a verdict. The turn is the whole test."""
        del obs
        manager = self.new_turn_manager(
            ctx,
            profile="play",
            system_prompt="Call the noop tool.",
            tools=[NoopTool()],
            max_tool_rounds=MAX_ROUNDS,
        )
        manager.max_generated_tokens_per_turn = CAP
        if self.honour_lift:
            manager.per_turn_cap_lifted = lambda: ctx.slots.drained
        manager.conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
        await manager.run()
        truncated = any(
            isinstance(element, ullm.conv.LLMResponse) and element.truncated_by_per_turn_cap
            for turn in manager.conversation.elements
            if isinstance(turn, ullm.conv.ResponseTurn)
            for element in turn.elements
        )
        return GiveUp(reason=CAPPED if truncated else UNCAPPED)


async def _run(honour_lift: bool, tmp_path: Path) -> list[str]:
    """Play the fleet and return each game's verdict, in start order."""
    backend = EndlessBackend.from_script(
        ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="noop", arguments={})),
        on_exhausted="repeat_last",
    )
    agent = BudgetedAgent(
        honour_lift=honour_lift,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=backend),
            sampling_profiles={"play": ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=2000)},
        ),
        run_config=RunConfig(max_concurrent=MAX_CONCURRENT),
    )
    benchmark = taaf.benchmark.Benchmark(
        games=[NeverEndingGame(label=f"g{index}") for index in range(N_GAMES)],
        solver=LLMAgentSolver(agent=agent),
        n_passes=1,
        job_dir=tmp_path,
    )
    await benchmark.run()
    return [str(run.solver_note or "") for run in benchmark.game_runs]


@pytest.mark.local
async def test_a_turn_that_outlives_the_queue_stops_being_capped(tmp_path: Path) -> None:
    """Once nothing is queued, a turn already running loses its cap.

    The last games to start are the ones the queue drains under, and they must
    run their rounds out. That some earlier game still died on the cap is the
    other half: a test where nothing is ever capped would pass with the cap
    removed altogether.
    """
    notes = await _run(honour_lift=True, tmp_path=tmp_path)
    assert any(CAPPED in note for note in notes), f"nothing was ever capped, so the cap is not being applied: {notes}"
    assert any(UNCAPPED in note for note in notes), f"the cap never lifted: {notes}"
    # The game that starts last cannot have begun before the queue drained.
    assert UNCAPPED in notes[-1], f"the last game to start was still capped: {notes}"


@pytest.mark.local
async def test_without_the_lift_every_game_dies_on_a_retired_cap(tmp_path: Path) -> None:
    """The same fleet, with the cap decided once per turn: nobody survives.

    This is what the fix is worth. The queue drains exactly as before and the
    tally says so, but a turn already in flight never asks again, so every
    game is cut short by a bound that stopped applying.
    """
    notes = await _run(honour_lift=False, tmp_path=tmp_path)
    assert all(CAPPED in note for note in notes), f"expected every game to die on the cap: {notes}"
