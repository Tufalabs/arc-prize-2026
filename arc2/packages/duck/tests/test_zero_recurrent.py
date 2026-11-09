"""``zero_recurrent``: the recurrent-state blank as an option — off by
default here, refused where the engine cannot honour it, and asked for on
every call once the prompt reaches the position.

The fork that implements the blank is pinned at the workspace root
(``vllm_fork.md``), which is exactly the setting a parent workspace does not
inherit; and the Kaggle offline stack installs stock wheels. Both would run,
score and never blank if the agent did not refuse them up front.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backend
from duck.agent import DuckAgent

import platypus.diagnostics


def test_the_blank_is_off_unless_asked_for() -> None:
    """The shipped configuration is the no-blank treatment: where the blank
    should land on this prompt layout is unmeasured (roadmap)."""
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).zero_recurrent
        is False
    )
    assert (
        DuckAgent(
            reply_cap_tokens=None,
        ).zero_recurrent_at
        == 1568
    )


def test_zero_recurrent_refuses_an_engine_without_the_field(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pin that selects the fork is a workspace-ROOT setting, so a parent
    workspace consuming these packages resolves vllm by its own rules and can
    pair our agent with a stock build. Asking the installed engine is the only
    check that survives that, so it is the one that must hold."""
    import vllm

    class _StockSamplingParams:
        """A stand-in for a stock build: no zero_recurrent_state field.

        The real class is a msgspec Struct whose ``__struct_fields__`` is
        readonly, so the module attribute is swapped rather than the tuple.
        """

        __struct_fields__ = ("temperature", "top_p", "max_tokens")

    monkeypatch.setattr(vllm, "SamplingParams", _StockSamplingParams)
    with pytest.raises(ValueError, match="no SamplingParams.zero_recurrent_state"):
        DuckAgent(reply_cap_tokens=None, zero_recurrent=True).prepare_for_backend()
    # An agent that is not asking for the blank runs on a stock engine.
    agent = DuckAgent(reply_cap_tokens=None, zero_recurrent=False)
    agent.prepare_for_backend()
    assert agent.zero_recurrent is False


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_the_blank_refuses_the_legacy_model_runner() -> None:
    """Only the V2 runner implements the blank, and the fork refuses every
    blanked request without it — so this pairing does not score unblanked,
    it dies at the first LLM call of every game. Caught when the
    configuration settles, for the reason the window check is: the engine
    only says so once a game has paid for the hardware and the model load."""
    agent = DuckAgent.build(zero_recurrent=True)
    assert agent.llm_config.backend_factory.env_overrides["VLLM_USE_V2_MODEL_RUNNER"] == "1"  # type: ignore[union-attr]
    agent.llm_config.backend_factory.env_overrides["VLLM_USE_V2_MODEL_RUNNER"] = "0"  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="V2 model runner"):
        agent.prepare_for_backend()
    # Each half alone is a configuration someone may want.
    agent.zero_recurrent = False
    agent.prepare_for_backend()


def test_the_blank_is_asked_for_only_past_the_position() -> None:
    """The engine refuses a position the prompt does not reach, so the agent
    asks only once the measured prompt is past it — and asks on every such
    call, clearing a stale request first, because the engine is the side that
    knows whether its cache already holds state past the position."""
    agent = DuckAgent(reply_cap_tokens=None, zero_recurrent_at=1000)
    agent.zero_recurrent = True  # past construction, so a stock test engine is not asked to have the field
    params = ullm.backend.GenericSamplingParameters()
    agent._live_params = params

    class _Span:
        def kv(self, **values: Any) -> None:
            pass

    original = platypus.diagnostics.current_span
    platypus.diagnostics.current_span = lambda: _Span()  # type: ignore[assignment] — the kv sink is not under test
    try:
        agent._arm_zero_recurrent(999)
        assert "zero_recurrent_state" not in params.extra_params
        agent._arm_zero_recurrent(1001)
        assert params.extra_params["zero_recurrent_state"] is True
        assert params.extra_params["zero_recurrent_at"] == 1000
        agent._arm_zero_recurrent(1001)  # idempotent: one request's worth of fields, not two
        assert sum(1 for key in params.extra_params if key.startswith("zero_recurrent")) == 2
        agent.zero_recurrent = False
        agent._arm_zero_recurrent(1001)
        assert "zero_recurrent_state" not in params.extra_params, "an agent not asking must leave no stale request"
    finally:
        platypus.diagnostics.current_span = original  # type: ignore[assignment]


if __name__ == "__main__":
    pytest.main([__file__])
