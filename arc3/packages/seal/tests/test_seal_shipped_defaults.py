"""What the seal ships from the duck's sweeps 12 and 13: summarize without the
journal, the grid as an image, and temperature sampling with the top-k cut
kept (2026-09-26 to 2026-09-28; the duck's ``evidence.md``) -- and the
vision tower that image needs, kept on the Kaggle box, with an image asked of
an engine that has none refused before it starts. The slots are the duck's
(``test_seal_context_is_the_ducks``).
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backends.deployment

from seal import SealAgent

pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")


@pytest.mark.parametrize("environment", ["slurm-b200", "kaggle-rtx"])
def test_a_column_runs_the_class_defaults(environment: str) -> None:
    """Memory and the image are the fields' own defaults, which no column
    restates; the sampling profile is every column's."""
    for agent in (SealAgent(), SealAgent.build(environment=environment)):
        assert (agent.compaction_strategy, agent.journal, agent.multimodal_context) == (
            "summarize",
            False,
            "current_grid",
        )
    play: Any = SealAgent.build(environment=environment).llm_config.sampling_profiles["play"]
    assert (play.temperature, play.top_p) == (0.6, 1.0)
    assert play.extra_params == {"top_k": 194, "presence_penalty": 0.0}


def test_summarize_refuses_an_uncapped_agent() -> None:
    """A summary needs a real reply reserve, which an agent without a reply
    cap does not have."""
    with pytest.raises(ValueError, match="requires a positive reply_cap_tokens"):
        SealAgent(reply_cap_tokens=None).prepare_for_backend()
    SealAgent(reply_cap_tokens=None, compaction_strategy="prune").prepare_for_backend()


def _engine_args(agent: SealAgent) -> Any:
    factory: Any = agent.llm_config.backend_factory
    assert factory is not None
    return factory.engine_args


def test_the_kaggle_column_keeps_the_vision_tower_its_image_needs() -> None:
    """The Kaggle box drops the tower only for an agent with no image to read."""
    engine_args = _engine_args(SealAgent.build(environment="kaggle-rtx"))
    assert engine_args.language_model_only is False
    assert engine_args.skip_mm_profiling is False


def test_the_server_column_sends_no_image() -> None:
    """The server topology continues a history over /v1/completions, which
    takes no image: its column sends the board as text, and an image asked
    of it afterwards is refused before the first request."""
    agent = SealAgent.build(engine="qwen3.8-27b", serving="server")
    assert agent.multimodal_context == ""
    agent.multimodal_context = "current_grid"
    with pytest.raises(ValueError, match="cannot continue a history with an image"):
        agent.prepare_for_backend()


def test_an_image_asked_of_an_engine_without_its_tower_is_refused() -> None:
    """Refused where the agent settles onto its engine, on any box, rather than
    on the first image request of a Kaggle run."""
    agent = SealAgent.build(environment="kaggle-rtx")
    ullm.backends.deployment.drop_vision_tower(_engine_args(agent))
    with pytest.raises(ValueError, match="without its vision tower"):
        agent.prepare_for_backend()
    agent.multimodal_context = ""
    agent.prepare_for_backend()
