"""What the duck ships: summarize without the journal, the grid as an image,
temperature sampling with the top-k cut kept, and the slots scaled to the
prompt budget (2026-09-26 and 2026-09-27; ``evidence.md``) -- and the
vision tower that image needs, kept on the Kaggle box, with an image asked of
an engine that has none refused before it starts.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backends.deployment

from duck import DuckAgent

pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")


def _engine_args(agent: DuckAgent) -> Any:
    factory: Any = agent.llm_config.backend_factory
    assert factory is not None
    return factory.engine_args


@pytest.mark.parametrize("environment", ["slurm-b200", "kaggle-rtx"])
def test_a_column_runs_the_class_defaults(environment: str) -> None:
    """Memory and the image are the fields' own defaults, which no column
    restates; the sampling profile is every column's."""
    for agent in (DuckAgent(), DuckAgent.build(environment=environment)):
        assert (agent.compaction_strategy, agent.journal, agent.multimodal_context) == (
            "summarize",
            False,
            "current_grid",
        )
    play: Any = DuckAgent.build(environment=environment).llm_config.sampling_profiles["play"]
    assert (play.temperature, play.top_p) == (0.6, 1.0)
    assert play.extra_params == {"top_k": 194, "presence_penalty": 0.0}


@pytest.mark.parametrize(
    ("engine", "environment", "slots"),
    [
        ("qwen3.8-27b", "slurm-b200", 36),
        ("qwen3.8-27b", "kaggle-rtx", 18),
    ],
)
def test_the_slots_moved_with_the_mean_context(engine: str, environment: str, slots: int) -> None:
    """The 27Bs' concurrency, scaled from the prompt budget of 56,832 to
    125,000 by the old budget over the new, so the cache their games hold
    stays where it was."""
    assert DuckAgent.build(engine=engine, environment=environment).run_config.max_concurrent == slots


def test_summarize_refuses_an_uncapped_agent() -> None:
    """A summary needs a real reply reserve, which an agent without a reply
    cap does not have."""
    with pytest.raises(ValueError, match="requires a positive reply_cap_tokens"):
        DuckAgent(reply_cap_tokens=None).prepare_for_backend()
    DuckAgent(reply_cap_tokens=None, compaction_strategy="prune").prepare_for_backend()


def test_the_kaggle_column_keeps_the_vision_tower_its_image_needs() -> None:
    """The Kaggle box drops the tower only for an agent with no image to read."""
    engine_args = _engine_args(DuckAgent.build(environment="kaggle-rtx"))
    assert engine_args.language_model_only is False
    assert engine_args.skip_mm_profiling is False


def test_the_server_column_sends_no_image() -> None:
    """The server topology continues a history over /v1/completions, which
    takes no image: its column sends the board as text, and an image asked
    of it afterwards is refused before the first request."""
    agent = DuckAgent.build(engine="qwen3.8-27b", serving="server")
    assert agent.multimodal_context == ""
    agent.multimodal_context = "current_grid"
    with pytest.raises(ValueError, match="cannot continue a history with an image"):
        agent.prepare_for_backend()


def test_an_image_asked_of_an_engine_without_its_tower_is_refused() -> None:
    """Refused where the agent settles onto its engine, on any box, rather than
    on the first image request of a Kaggle run."""
    agent = DuckAgent.build(environment="kaggle-rtx")
    ullm.backends.deployment.drop_vision_tower(_engine_args(agent))
    with pytest.raises(ValueError, match="without its vision tower"):
        agent.prepare_for_backend()
    agent.multimodal_context = ""
    agent.prepare_for_backend()
