"""``SealAgent.build``'s ``**overrides``: what it applies, and what it refuses.

The overrides land after the column, so an explicit value wins. The two
refusals exist because nothing freezes an agent: a name nobody reads would
otherwise be a setting that silently does nothing, and a nested object passed
whole would silently discard what the column composed inside it.
"""

from __future__ import annotations

from typing import Any

import pytest

from platypus import LLMConfig, RunConfig
from seal.agent import SealAgent

pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_an_override_beats_the_column() -> None:
    agent = SealAgent.build(reasoning_effort="low", compact_to_tokens=2048)
    assert (agent.reasoning_effort, agent.compact_to_tokens) == ("low", 2048)


def test_a_name_the_agent_does_not_have_is_refused() -> None:
    """A typo assigned blindly would be a knob nobody reads."""
    with pytest.raises(TypeError, match="has no field 'reasoning_efort'"):
        SealAgent.build(reasoning_efort="low")


@pytest.mark.parametrize("name", ["llm_config", "run_config"])
def test_a_composed_object_cannot_be_passed_whole(name: str) -> None:
    """The failure this prevents is silent: the column builds the backend
    factory and the deploy assets INTO these, so a replacement arrives after
    them and takes an agent with no engine to the run."""
    whole: dict[str, Any] = {"llm_config": LLMConfig(), "run_config": RunConfig()}
    with pytest.raises(TypeError, match="cannot be passed to build"):
        SealAgent.build(**{name: whole[name]})
