"""The per-turn token cap, and what happens when it is lifted mid-turn.

A turn is many calls with tool work between them, so it can run for hours —
long enough for the reason a cap was imposed to expire while the turn is still
going. ``per_turn_cap_lifted`` is how the manager finds out: asked before every
call, and again when a call stops on the cap.

The cap is enforced by lowering ``sampling_params.max_tokens`` per call, so
these drive a backend that honours it. Costs are exact rather than estimated,
because the cases differ by which of two ceilings binds.
"""

from dataclasses import dataclass, field
from typing import Any, Sequence, cast

import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tool

#: Per-call sampling ceiling, and what a call would spend unbidden. COST below
#: PER_CALL_MAX so only the per-turn cap can ever truncate a call.
PER_CALL_MAX = 100
COST = 60

#: Two calls at COST fit; the third is squeezed to 30 and truncated.
CAP = 150


@dataclass
class CostedBackend(ullm.backends.stub.StubBackend):
    """Spends :data:`COST` a call, or whatever ``max_tokens`` leaves it.

    A real backend stopping at ``max_tokens`` is the only thing that makes a
    per-turn cap bind, and the stub does not model it. ``lift_on_truncation``
    flips :attr:`lifted` the moment the cap first bites, which is how a test
    puts the lift exactly in the window between a truncated call and the
    manager's decision about it.
    """

    lift_on_truncation: bool = False
    lifted: list[bool] = field(default_factory=lambda: [False])
    spends: list[int] = field(default_factory=list[int])

    def _process_output(
        self,
        sampling_params: ullm.backend.SamplingParameters,
        llm_output: Any,
        conversation: ullm.conv.Conversation,
        turn: ullm.conv.ResponseTurn,
        element: ullm.conv.LLMResponse,
    ) -> None:
        """Charge the call, clipped to whatever the sampling ceiling allows."""
        super()._process_output(sampling_params, llm_output, conversation, turn, element)
        allowed = getattr(sampling_params, "max_tokens", None)
        spend = COST if allowed is None else min(COST, allowed)
        self.spends.append(spend)
        element.token_usage = ullm.backend.TokenUsage(prompt_tokens=1, completion_tokens=spend)
        if spend < COST:
            element.stop_reason = "max_tokens"
            if self.lift_on_truncation:
                self.lifted[0] = True
        else:
            element.stop_reason = "tool_use" if element.final_content.tool_calls() else "stop"


@dataclass
class Noop(ullm.tool.Tool):
    """A tool that answers instantly, so a round costs one call."""

    name: str = "noop"
    description: str = "does nothing"
    parameters: dict[str, Any] = field(default_factory=lambda: {"type": "object", "properties": {}})

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Answer at once — the tool is only here to keep the turn going."""
        del arguments
        return ullm.conv.StringContent(text="ok")


@dataclass
class ObservingTurnManager(ullm.manager.TurnManager):
    """Collect manager-owned discarded rounds for hook contract coverage."""

    discarded: list[tuple[str, list[ullm.conv.ConversationElement]]] = field(
        default_factory=lambda: list[tuple[str, list[ullm.conv.ConversationElement]]]()
    )

    def _observe_discarded_round(self, elements: Sequence[ullm.conv.ConversationElement], *, reason: str) -> None:
        """Retain a slice copy exactly as a diagnostics subclass would."""
        self.discarded.append((reason, list(elements)))


def _call() -> ullm.conv.LLMResponse:
    """A response that asks for another round."""
    return ullm.conv.LLMResponse(final_content=ullm.conv.ToolCallContent(tool_name="noop", arguments={}))


def _manager(
    backend: CostedBackend,
    *,
    manager_type: type[ullm.manager.TurnManager] = ullm.manager.TurnManager,
    **kwargs: Any,
) -> ullm.manager.TurnManager:
    """A turn over ``backend``, capped at :data:`CAP`."""
    tool = Noop()
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"), tools=[tool])
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    response_manager = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=PER_CALL_MAX),
    )
    return manager_type(
        response_manager=response_manager,
        conversation=conversation,
        max_generated_tokens_per_turn=CAP,
        **kwargs,
    )


def _backend(script: Sequence[ullm.conv.LLMResponse | str], **kwargs: Any) -> CostedBackend:
    """The scripted backend, repeating its last entry once the script runs out."""
    # from_script is typed on the base class, so it hands back a StubBackend.
    return cast(CostedBackend, CostedBackend.from_script(*script, on_exhausted="repeat_last", **kwargs))


def _responses(manager: ullm.manager.TurnManager) -> list[ullm.conv.LLMResponse]:
    """Every model response left in the finished turn."""
    turn = manager.conversation.elements[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    return [e for e in turn.elements if isinstance(e, ullm.conv.LLMResponse)]


async def test_a_cap_nobody_lifts_truncates_the_turn() -> None:
    """The baseline the rest is measured against: the cap binds and ends it."""
    backend = _backend([_call()])
    manager = _manager(backend)
    await manager.run()
    assert backend.spends == [COST, COST, CAP - 2 * COST], backend.spends
    assert any(r.truncated_by_per_turn_cap for r in _responses(manager)), "the cap never bit"


async def test_a_lifted_cap_stops_binding_before_the_next_call() -> None:
    """Asked before every call, so a turn already running is released.

    Every call spends its full cost, where the third would otherwise have been
    squeezed into what the cap had left.
    """
    backend = _backend([_call(), _call(), _call(), _call(), "done"])
    manager = _manager(backend, per_turn_cap_lifted=lambda: True)
    await manager.run()
    assert backend.spends == [COST] * 5, backend.spends
    assert not any(r.truncated_by_per_turn_cap for r in _responses(manager))


async def test_a_round_truncated_by_a_cap_since_lifted_is_run_again() -> None:
    """The window this exists for: the cap bites, and then stops applying.

    The lift lands between the truncated call and the manager's decision about
    it, so the round is discarded and re-run without the cap — a truncated
    round cannot be salvaged, and its output must not survive into the
    conversation the model reads next.
    """
    backend = _backend([_call(), _call(), _call(), "done"], lift_on_truncation=True)
    manager = cast(
        ObservingTurnManager,
        _manager(
            backend,
            manager_type=ObservingTurnManager,
            per_turn_cap_lifted=lambda: backend.lifted[0],
        ),
    )
    await manager.run()
    # Two calls fit, the third is squeezed to 30 and discarded, and a fourth
    # runs at full cost. The re-run draws the *next* scripted response rather
    # than repeating the discarded one — the stub advances per emission, where
    # a real backend would be asked the same question again. What is under
    # test is that a fourth call happened at all, and uncapped.
    assert backend.spends == [COST, COST, CAP - 2 * COST, COST], backend.spends
    assert not any(r.truncated_by_per_turn_cap for r in _responses(manager)), "the discarded round survived"
    assert not any(r.stop_reason == "max_tokens" for r in _responses(manager))
    assert len(manager.discarded) == 1
    reason, elements = manager.discarded[0]
    assert reason == "per_turn_cap_lifted"
    assert len(elements) == 1 and isinstance(elements[0], ullm.conv.LLMResponse)
    discarded = elements[0]
    assert discarded.token_usage is not None and discarded.token_usage.completion_tokens == CAP - 2 * COST
    assert discarded.stop_reason == "max_tokens"
    assert discarded.truncated_by_per_turn_cap is True


async def test_lift_predicate_sees_the_response_before_truncation_is_stamped() -> None:
    """Manager metadata must not change the predicate's pre-existing answer."""
    backend = _backend([_call()])
    manager = cast(
        ObservingTurnManager,
        _manager(backend, manager_type=ObservingTurnManager),
    )

    def lifted() -> bool:
        """Lift only if the current response was already marked truncated."""
        turn = manager.conversation.last_turn
        responses = [element for element in turn.elements if isinstance(element, ullm.conv.LLMResponse)]
        return bool(responses and responses[-1].truncated_by_per_turn_cap)

    manager.per_turn_cap_lifted = lifted
    await manager.run()

    responses = _responses(manager)
    assert backend.spends == [COST, COST, CAP - 2 * COST]
    assert responses[-1].stop_reason == "max_tokens"
    assert responses[-1].truncated_by_per_turn_cap is True
    assert manager.discarded == []


async def test_the_sampling_ceiling_comes_back_when_the_cap_lifts() -> None:
    """A cap that bound an earlier round must not go on binding later ones.

    ``max_tokens`` is lowered per call to enforce the cap, so a lift that left
    the lowered value in place would cap every remaining round at whatever the
    budget happened to have left — silently, since nothing would be flagged as
    truncated any more.
    """
    lifted = [False]
    backend = _backend([_call(), _call(), _call(), "done"])
    manager = _manager(backend, per_turn_cap_lifted=lambda: lifted[0])

    def _lift_after_two() -> None:
        """Retire the cap once two calls have been made."""
        if len(backend.spends) >= 2:
            lifted[0] = True

    manager.before_round = _lift_after_two
    await manager.run()
    # The third call is the one that would have been squeezed to 30.
    assert backend.spends == [COST, COST, COST, COST], backend.spends


async def test_a_retry_granted_on_the_last_round_still_runs() -> None:
    """A discarded round must not spend the turn's round allowance.

    The retry exists because a truncated round cannot be salvaged. If the
    discard also consumed a round, a lift arriving on the final one would
    delete the response and then have nowhere to re-run it — closing a turn
    whose last element is a ``ToolResult``, which is malformed input on the
    next request rather than merely a lost round.
    """
    backend = _backend([_call(), _call(), _call(), "done"], lift_on_truncation=True)
    manager = _manager(backend, per_turn_cap_lifted=lambda: backend.lifted[0])
    manager.max_tool_rounds = 3

    await manager.run()

    # Three surviving rounds, and the discarded third does not count as one.
    assert backend.spends == [COST, COST, CAP - 2 * COST, COST], backend.spends
    turn = manager.conversation.elements[-1]
    assert isinstance(turn, ullm.conv.ResponseTurn)
    assert isinstance(turn.elements[-1], ullm.conv.LLMResponse), "turn closed on a ToolResult"


async def test_a_re_armed_cap_ends_the_turn_rather_than_spinning() -> None:
    """One discard is the most a turn can take, without counting them.

    A truncated call spends whatever the cap had left, so a predicate that
    goes back to capping finds nothing to spend and the turn ends on the
    cap-exhausted placeholder. Pinned because the discard path has no counter
    guarding it, and this invariant is why it does not need one.
    """
    backend = _backend([_call()], lift_on_truncation=True)
    manager = _manager(backend, per_turn_cap_lifted=lambda: backend.lifted[0])

    def rearm() -> None:
        """Put the cap back before each call, so a retry is capped again."""
        backend.lifted[0] = False

    manager.before_round = rearm
    manager.max_tool_rounds = 50

    await manager.run()

    # Two full calls, one truncated to what was left, and no further call.
    assert backend.spends == [COST, COST, CAP - 2 * COST], backend.spends
    assert any(r.truncated_by_per_turn_cap for r in _responses(manager))
