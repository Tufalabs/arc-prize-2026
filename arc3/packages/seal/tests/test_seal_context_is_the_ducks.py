"""The seal's compaction target and reply reserve, its concurrency, its
engine's caches and how its slots are ranked are the duck's: the concurrency
as a share, for the seal's longer window, except on Flash-Next, where the
seal plays a checkpoint of its own (``presets.JOINT205_CHECKPOINT``).

The seal states each figure beside the duck's own reason for it, so what can
drift is the duck: a figure retuned on the duck and not ported leaves the seal
running a number the duck no longer ships. These hold every such figure to the
duck's current one, so that port cannot be missed.
"""

from __future__ import annotations

from typing import Any

import pytest

from duck import DuckAgent
from seal import SealAgent

presets = pytest.importorskip("seal.presets", reason="needs the vllm extra (linux)")


def test_the_compaction_target_and_the_reply_reserve_are_the_ducks() -> None:
    seal, duck = SealAgent(), DuckAgent()
    assert (seal.compact_to_tokens, seal.reply_cap_tokens) == (duck.compact_to_tokens, duck.reply_cap_tokens)


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
@pytest.mark.parametrize(
    ("engine", "environment"),
    [(engine, environment) for engine in presets.ENGINES for environment in presets.ENVIRONMENTS],
)
def test_every_column_serves_the_ducks_concurrency(engine: str, environment: str) -> None:
    """The slots, and how far apart the games start into them: both sized for
    the engine the two agents share, the slots in the share the seal's longer
    window leaves. On Flash-Next the checkpoint is the seal's own, and so are
    its slots."""
    seal = SealAgent.build(engine=engine, environment=environment)
    duck = DuckAgent.build(engine=engine, environment=environment)
    assert seal.run_config.start_gap_s == duck.run_config.start_gap_s
    if engine != "qwen3.8-flash-next":
        assert seal.run_config.max_concurrent == round(duck.run_config.max_concurrent * presets.WINDOW_SLOT_SHARE)


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
@pytest.mark.parametrize(
    ("engine", "environment"),
    [(engine, environment) for engine in presets.ENGINES for environment in presets.ENVIRONMENTS],
)
def test_every_column_serves_the_ducks_caches_step_and_slot_ranking(engine: str, environment: str) -> None:
    """The KV caches and their block, the step budget, and who gets a slot:
    chosen for the engine and the box, which the two agents share."""

    def served(agent: SealAgent | DuckAgent) -> tuple[object, ...]:
        factory: Any = agent.llm_config.backend_factory
        args = factory.engine_args
        return (
            args.kv_cache_dtype,
            args.attention_config.indexer_kv_dtype,
            args.transient_kv_pool,
            args.block_size,
            args.max_num_batched_tokens,
            agent.run_config.slot_policy,
        )

    seal = SealAgent.build(engine=engine, environment=environment)
    duck = DuckAgent.build(engine=engine, environment=environment)
    assert served(seal) == served(duck)


def test_duck_like_sets_every_duck_value_and_an_override_still_wins() -> None:
    from seal.agent import DUCK_LIKE

    agent = SealAgent.build(duck_like=True, python_timeout_s=120)
    for name, value in DUCK_LIKE.items():
        assert getattr(agent, name) == (120 if name == "python_timeout_s" else value), name
    assert SealAgent.build().use_world_model != "none", "duck_like is not the default"


def test_duck_like_holds_the_ducks_current_values() -> None:
    """Where a duck-like value is the duck's own, it is the duck's current
    one: its window, its kernel timeout, eviction chunk and tool-output cap,
    its colour-name legend, and its board, which prints one letter per colour
    with no row numbers."""
    import duck.grid_utils as duck_grid_utils
    from seal.agent import DUCK_LIKE
    from seal.palette import ASCII_SYMBOLS

    duck = DuckAgent()
    assert (DUCK_LIKE["python_timeout_s"], DUCK_LIKE["evict_turns"]) == (duck.python_timeout_s, duck.evict_turns)
    assert DUCK_LIKE["tool_output_tokens"] == duck.tool_output_tokens
    assert DUCK_LIKE["context_window_tokens"] == duck.context_window_tokens
    assert DUCK_LIKE["color_name_legend"] == duck.color_name_legend
    assert duck.frame_api == "ergonomic", "the duck's board is its letters"
    assert ASCII_SYMBOLS[DUCK_LIKE["ascii_symbols"]] == duck_grid_utils.ARC_COLOR_CHARS
