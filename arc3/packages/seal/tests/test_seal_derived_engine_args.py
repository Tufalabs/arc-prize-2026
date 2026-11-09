"""Engine arguments that follow from the agent, settled at ``prepare_for_backend``.

The engine window follows ``context_window_tokens``, the sequence cap
follows ``run_config.max_concurrent``, and the image processor's pixel
ceiling follows what the seal renders. These pin the whole contract: WHEN it
happens (at the solver's last moment, so an override made after construction
is seen), what it produces per column, that a chosen value is refused rather
than overwritten, and that the flags turn it off.
"""

from __future__ import annotations

from typing import Any

import pytest
import ullm.backends.deployment as deployment
import ullm.backends.oai
import ullm.factory

from seal.agent import SealAgent, largest_image_pixels

presets = pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")


def _engine_args(agent: SealAgent) -> Any:
    factory: Any = agent.llm_config.backend_factory
    assert factory is not None
    return factory.engine_args


# --------------------------------------------------------------------------
# what a preset hands over
# --------------------------------------------------------------------------


def test_a_column_leaves_both_for_the_agent_to_settle() -> None:
    """A value on the column would be the second copy, and the derivation
    would then have to decide which of the two a run meant."""
    agent = SealAgent(
        reply_cap_tokens=None,
        compaction_strategy="prune",
    )
    presets.configure(agent)
    factory: Any = agent.llm_config.backend_factory
    assert factory.engine_args.max_num_seqs is None
    assert factory.engine_args.max_model_len == 262144, "the family builder's native ceiling"


# --------------------------------------------------------------------------
# what it settles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("environment", ["slurm-b200", "kaggle-rtx"])
def test_each_column_settles_onto_its_own_numbers(environment: str) -> None:
    """Each column's engine follows that column's agent, not a shared figure:
    a column summarizes, so its window is the engine's native ceiling rather
    than the agent's own."""
    agent = SealAgent.build(environment=environment)
    engine_args = _engine_args(agent)
    assert agent.compaction_strategy == "summarize"
    assert agent.engine_ceiling_tokens is not None
    assert engine_args.max_model_len == agent.engine_ceiling_tokens > agent.context_window_tokens
    assert engine_args.max_num_seqs == agent.run_config.max_concurrent


def test_the_window_is_settled_only_at_prepare_not_at_construction() -> None:
    """A caller who widens the agent's window gets an engine that accepts it,
    without having to know that a second field moves with it."""
    agent = SealAgent.build()
    agent.compaction_strategy = "prune"  # the one strategy whose engine window IS the agent's
    agent.context_window_tokens = 40960
    agent.compact_to_tokens = 4096  # keep the budget coordinate valid
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == 40960


def test_the_sequence_cap_follows_a_concurrency_override() -> None:
    agent = SealAgent.build()
    agent.run_config.max_concurrent = 12
    agent.prepare_for_backend()
    assert _engine_args(agent).max_num_seqs == 12


def test_settling_twice_is_settling_once() -> None:
    """The solver calls it; a caller inspecting the settled config may too."""
    agent = SealAgent.build()
    agent.prepare_for_backend()
    first = (_engine_args(agent).max_model_len, _engine_args(agent).max_num_seqs)
    agent.prepare_for_backend()
    assert (_engine_args(agent).max_model_len, _engine_args(agent).max_num_seqs) == first


def test_an_environment_arc3_does_not_run_in_is_refused() -> None:
    """ullm describes boxes; this describes the ones a run may happen on. An
    accelerator the arena does not offer has to fail at construction, not go
    quietly unconfigured."""
    with pytest.raises(ValueError, match="ARC3 does not run there"):
        SealAgent.build(environment="kaggle-4xl4")


def test_the_flash_next_kaggle_column_decides_what_the_card_changes() -> None:
    """The card is a third smaller than the cluster's and a different
    capability family, so the column names MoE kernels this card admits and
    drives a shallower pool, rather than taking either from the cluster."""
    kaggle = SealAgent.build(engine="qwen3.8-flash-next", environment="kaggle-rtx")
    slurm = SealAgent.build(engine="qwen3.8-flash-next", environment="slurm-b200")
    kaggle_args = _engine_args(kaggle)
    # Named, and not TRT-LLM's: those kernels are gated to the 10.x capability
    # family and this card is 12.0, while "auto" would walk to whatever admits.
    assert kaggle_args.kernel_config.moe_backend not in ("auto", "flashinfer_trtllm")
    # What the smaller pool does not change: the table still comes off the card.
    assert kaggle_args.engram_config is not None and kaggle_args.engram_config.cpu_offload
    assert kaggle.run_config.max_concurrent < slurm.run_config.max_concurrent


def test_a_backend_with_no_engine_of_ours_is_left_alone() -> None:
    """An external server owns its engine; there is nothing here to settle."""
    agent = SealAgent.build(
        engine="qwen3.8-27b",
        environment="slurm-b200",
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        backend_factory=ullm.factory.PrebuiltBackend(
            backend=ullm.backends.oai.OpenAIServer(
                base_url="http://127.0.0.1:8765/v1", model="qwen", **deployment.client_kwargs("qwen3.8-27b")
            )
        ),
    )
    agent.prepare_for_backend()  # must not raise
    assert not hasattr(agent.llm_config.backend_factory, "engine_args")


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


def test_a_chosen_window_is_refused_not_overwritten() -> None:
    """A caller who set the engine's window meant it: deriving over it would
    hand them a run whose engine is not the one they configured, and no sign."""
    agent = SealAgent.build()
    _engine_args(agent).max_model_len = 40960
    with pytest.raises(ValueError, match="derive_max_model_len"):
        agent.prepare_for_backend()


@pytest.mark.parametrize("axis_labels", [False, True])
def test_the_pixel_ceiling_is_the_largest_image_the_seal_sends(axis_labels: bool) -> None:
    """Every image the seal sends fits under the ceiling as the processor
    rounds it, so none is shrunk, the label margin included; and the ceiling
    is far under the checkpoint's own, which is what the start-up profile
    reserves for."""
    agent = SealAgent.build()
    agent.axis_labels = axis_labels
    agent.prepare_for_backend()
    ceiling = largest_image_pixels(upscale=agent.multimodal_upscale, axis_labels=axis_labels)
    assert _engine_args(agent).mm_processor_kwargs == {"max_pixels": ceiling}
    assert _engine_args(agent).limit_mm_per_prompt == {"video": 0}, "the seal sends no video"
    side = 64 * agent.multimodal_upscale + (32 if axis_labels else 0)
    assert ceiling >= (-(-side // 32) * 32) ** 2
    assert ceiling < 16384 * 32 * 32


def test_a_chosen_pixel_ceiling_is_refused_too() -> None:
    agent = SealAgent.build()
    _engine_args(agent).mm_processor_kwargs = {"max_pixels": 1 << 20}
    with pytest.raises(ValueError, match="derive_max_pixels"):
        agent.prepare_for_backend()


def test_a_chosen_sequence_cap_is_refused_too() -> None:
    agent = SealAgent.build()
    _engine_args(agent).max_num_seqs = 7
    with pytest.raises(ValueError, match="derive_max_num_seqs"):
        agent.prepare_for_backend()


def test_the_flag_is_how_a_caller_owns_the_pairing() -> None:
    agent = SealAgent.build()
    _engine_args(agent).max_model_len = 262144
    agent.derive_max_model_len = False
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == 262144, "theirs, kept"
    assert _engine_args(agent).max_num_seqs == agent.run_config.max_concurrent, "the other still settles"


def test_an_engine_too_short_for_the_agent_is_refused() -> None:
    """With the derivation off the two lengths can disagree again, and the
    agent would then build prompts the engine rejects."""
    agent = SealAgent.build()
    agent.derive_max_model_len = False
    _engine_args(agent).max_model_len = 4096
    with pytest.raises(ValueError, match="exceeds the engine's max_model_len"):
        agent.prepare_for_backend()
