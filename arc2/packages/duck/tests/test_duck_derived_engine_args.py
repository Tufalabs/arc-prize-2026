"""Engine arguments that follow from the agent, settled at ``prepare_for_backend``.

The engine window follows ``context_window_tokens``, the sequence cap
follows ``run_config.max_concurrent``, and the image processor's pixel
ceiling follows the largest image the agent renders. These pin the whole contract: WHEN it
happens (at the solver's last moment, so an override made after construction
is seen), what it produces per column, that a chosen value is refused rather
than overwritten, and that the flags turn it off.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import ullm.backends.oai
import ullm.factory
from duck.agent import DuckAgent
from ullm.backends import deployment

import taaf.grid

presets = pytest.importorskip("duck.presets", reason="needs the vllm extra (linux)")

pytestmark = [
    pytest.mark.filterwarnings("ignore:Can't initialize NVML"),
    pytest.mark.filterwarnings("ignore:CUDA initialization"),
]


def _engine_args(agent: DuckAgent) -> Any:
    factory: Any = agent.llm_config.backend_factory
    assert factory is not None
    return factory.engine_args


def _external_factory() -> ullm.factory.PrebuiltBackend:
    return ullm.factory.PrebuiltBackend(
        backend=ullm.backends.oai.OpenAIServer(
            base_url="http://127.0.0.1:8765/v1", model="qwen", **deployment.client_kwargs(presets.DEFAULT_ENGINE)
        )
    )


# --------------------------------------------------------------------------
# what a column hands over
# --------------------------------------------------------------------------


def test_a_column_leaves_both_for_the_agent_to_settle() -> None:
    """A value on the column would be the second copy, and the derivation
    would then have to decide which of the two a run meant."""
    agent = DuckAgent(
        reply_cap_tokens=None,
    )
    presets.configure(agent)
    factory: Any = agent.llm_config.backend_factory
    assert factory.engine_args.max_num_seqs is None
    assert factory.engine_args.max_model_len == 262144, "the family builder's native ceiling"
    assert agent.engine_ceiling_tokens == 262144, "recorded before the agent settles its own window onto it"


# --------------------------------------------------------------------------
# what it settles
# --------------------------------------------------------------------------


@pytest.mark.parametrize("engine", ["qwen3.8-27b", "qwen3.6-27b"])
@pytest.mark.parametrize("environment", ["slurm-b200", "kaggle-4xl4"])
def test_each_column_settles_onto_its_own_numbers(engine: str, environment: str) -> None:
    """Each column's sequence cap follows that column's agent, not a shared
    figure; its window is the family's ceiling, since every column summarizes
    and the summary request runs past the agent's window."""
    agent = DuckAgent.build(engine=engine, environment=environment)
    engine_args = _engine_args(agent)
    assert agent.compaction_strategy == "summarize"
    ceiling = agent.engine_ceiling_tokens
    assert ceiling is not None and engine_args.max_model_len == ceiling > agent.context_window_tokens
    assert engine_args.max_num_seqs == agent.run_config.max_concurrent


def test_the_window_is_settled_only_at_prepare_not_at_construction() -> None:
    """A caller who widens the agent's window gets an engine that accepts it,
    without having to know that a second field moves with it."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.compaction_strategy = "prune"  # the window follows the agent's under prune
    agent.context_window_tokens = 40960
    agent.compact_to_tokens = 4096  # keep the budget coordinate valid
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == 40960


def test_the_sequence_cap_follows_a_concurrency_override() -> None:
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.run_config.max_concurrent = 12
    agent.prepare_for_backend()
    assert _engine_args(agent).max_num_seqs == 12


def test_settling_twice_is_settling_once() -> None:
    """The solver calls it; a caller inspecting the settled config may too."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.prepare_for_backend()
    args = _engine_args(agent)
    first = (args.max_model_len, args.max_num_seqs, dict(args.mm_processor_kwargs))
    agent.prepare_for_backend()
    assert (args.max_model_len, args.max_num_seqs, dict(args.mm_processor_kwargs)) == first


def test_an_environment_arc2_does_not_run_in_is_refused() -> None:
    """ullm describes boxes; this describes the ones a run may happen on. A
    card the competition does not offer has to fail at construction, not go
    quietly unconfigured."""
    with pytest.raises(ValueError, match="ARC2 does not run there"):
        DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-rtx")


def test_a_backend_with_no_engine_of_ours_is_left_alone() -> None:
    """An external server owns its engine; there is nothing here to settle."""
    agent = DuckAgent.build(
        engine="qwen3.8-27b",
        environment="slurm-b200",
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        backend_factory=_external_factory(),
        images="off",
    )
    agent.prepare_for_backend()  # must not raise
    assert not hasattr(agent.llm_config.backend_factory, "engine_args")


# --------------------------------------------------------------------------
# what it refuses
# --------------------------------------------------------------------------


def test_a_chosen_window_is_refused_not_overwritten() -> None:
    """A caller who set the engine's window meant it: deriving over it would
    hand them a run whose engine is not the one they configured, and no sign."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    _engine_args(agent).max_model_len = 40960
    with pytest.raises(ValueError, match="derive_max_model_len"):
        agent.prepare_for_backend()


def test_the_pixel_ceiling_follows_the_render_scale() -> None:
    """The ceiling is the largest render at the agent's own scale, so a sweep
    that doubles the scale gets a ceiling that grew with it rather than one
    sized for the default -- which the processor would have shrunk to."""
    from duck import images

    small, large = (
        DuckAgent.build(engine="qwen3.8-27b", image_scale=4),
        DuckAgent.build(engine="qwen3.8-27b", image_scale=16),
    )
    for agent in (small, large):
        agent.prepare_for_backend()
        expected = images.largest_image_pixels(scale=agent.image_scale, gridlines=agent.image_gridlines)
        assert _engine_args(agent).mm_processor_kwargs == {"max_pixels": expected}
        assert _engine_args(agent).limit_mm_per_prompt["video"] == 0, "the duck sends no video"
    assert _engine_args(large).mm_processor_kwargs["max_pixels"] > _engine_args(small).mm_processor_kwargs["max_pixels"]


def test_the_pixel_ceiling_never_shrinks_an_image_the_agent_sends() -> None:
    """Against the processor's own resize rule: for every image the agent can
    send, at every scale a sweep uses, the ceiling resizes exactly as no
    ceiling would. The rule rounds sides to its patch multiple before
    comparing, which is what the rounding in the ceiling exists for.

    Both kinds of image, because the grids are not always the larger: the
    colour legend's size is fixed by its own layout, so below the shipped
    scale it is the legend a grid-only ceiling would shrink -- blurring the
    labels the model reads the colour names off."""
    from duck import images, palette
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    largest = taaf.grid.Grid(np.zeros((taaf.grid.MAX_SIDE, taaf.grid.MAX_SIDE), dtype=np.int8))
    legend_height, legend_width, _ = palette.legend_array().shape
    for scale in (1, 2, 3, 4, 8, 16):
        for gridlines in (False, True):
            ceiling = images.largest_image_pixels(scale=scale, gridlines=gridlines)
            pair_height, pair_width, _ = images.side_by_side([largest, largest], scale=scale, gridlines=gridlines).shape
            for height, width in ((pair_height, pair_width), (legend_height, legend_width)):
                capped = smart_resize(height, width, factor=32, min_pixels=1, max_pixels=ceiling)
                uncapped = smart_resize(height, width, factor=32, min_pixels=1, max_pixels=2**40)
                assert capped == uncapped, (scale, gridlines, height, width)


def test_a_chosen_pixel_ceiling_is_refused_too() -> None:
    from duck import images

    agent = DuckAgent.build(engine="qwen3.8-27b")
    _engine_args(agent).mm_processor_kwargs = {"max_pixels": 12345}
    with pytest.raises(ValueError, match="derive_max_pixels"):
        agent.prepare_for_backend()

    # Edited in place, after a first settle: the processor arguments are the
    # one derivable argument that is a dict, and a baseline holding the
    # engine's own object would follow the edit and read it as the value
    # nobody chose -- settling over a caller's number instead of refusing.
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.prepare_for_backend()
    _engine_args(agent).mm_processor_kwargs["max_pixels"] = 262144
    with pytest.raises(ValueError, match="derive_max_pixels"):
        agent.prepare_for_backend()
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.derive_max_pixels = False
    _engine_args(agent).mm_processor_kwargs = {"max_pixels": 12345}
    agent.prepare_for_backend()
    assert _engine_args(agent).mm_processor_kwargs == {"max_pixels": 12345}
    assert images.largest_image_pixels(scale=4, gridlines=True) != 12345


def test_a_chosen_sequence_cap_is_refused_too() -> None:
    agent = DuckAgent.build(engine="qwen3.8-27b")
    _engine_args(agent).max_num_seqs = 7
    with pytest.raises(ValueError, match="derive_max_num_seqs"):
        agent.prepare_for_backend()


def test_the_flag_is_how_a_caller_owns_the_pairing() -> None:
    agent = DuckAgent.build(engine="qwen3.8-27b")
    _engine_args(agent).max_model_len = 131072
    agent.derive_max_model_len = False
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == 131072, "theirs, kept"
    assert _engine_args(agent).max_num_seqs == agent.run_config.max_concurrent, "the other still settles"


def test_summarize_derives_the_engine_s_own_ceiling() -> None:
    """Its summary request exceeds the agent's window by design, and by how
    much is not the agent's to know, so the window is the ceiling the engine
    was built with -- not the agent's, which would under-size the engine."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    assert _engine_args(agent).max_model_len == agent.engine_ceiling_tokens == 262144, "the native ceiling"
    agent.compaction_strategy = "prune"
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == agent.context_window_tokens, "prune takes the agent's"
    agent.compaction_strategy = "summarize"
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == 262144, "and back: the ceiling is kept, not the settled window"


def test_a_hand_built_duck_that_summarizes_states_its_engine_window() -> None:
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.compaction_strategy = "summarize"
    agent.engine_ceiling_tokens = None
    with pytest.raises(ValueError, match="engine_ceiling_tokens"):
        agent.prepare_for_backend()


def test_summarize_refuses_a_ceiling_the_agent_s_own_numbers_overflow() -> None:
    """The window, the reply reserve and the summary's cap are the agent's
    own; when even those do not fit under the ceiling, no run can."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.compaction_strategy = "summarize"
    agent.context_window_tokens = 262144 - 4096
    with pytest.raises(ValueError, match="past the engine's ceiling"):
        agent.prepare_for_backend()


def test_summarize_runs_on_a_window_the_caller_states() -> None:
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.compaction_strategy = "summarize"
    agent.derive_max_model_len = False
    _engine_args(agent).max_model_len = agent.context_window_tokens + 16_384
    agent.prepare_for_backend()
    assert _engine_args(agent).max_model_len == agent.context_window_tokens + 16_384


def test_an_engine_too_short_for_the_agent_is_refused() -> None:
    """With the derivation off the two lengths can disagree again, and the
    agent would then build prompts the engine rejects."""
    agent = DuckAgent.build(engine="qwen3.8-27b")
    agent.derive_max_model_len = False
    _engine_args(agent).max_model_len = 4096
    with pytest.raises(ValueError, match="exceeds the engine's max_model_len"):
        agent.prepare_for_backend()
