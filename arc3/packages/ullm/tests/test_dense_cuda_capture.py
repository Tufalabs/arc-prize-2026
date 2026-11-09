"""Dense CUDA-graph capture: a rung for every decode width, settled at start.

Covers the shape of the rungs, that they come from the width the engine will
actually admit however late it was decided, and the three cases that leave the
engine's own ladder alone.
"""

from __future__ import annotations

from typing import Any

import pytest

vllm = pytest.importorskip("vllm", reason="needs the vllm extra (linux)")

import ullm.backends.vllm_factory  # noqa: E402
import ullm.backends.vllmx  # noqa: E402


def _factory(max_num_seqs: int | None = 40, **fields: Any) -> "ullm.backends.vllm_factory.EngineFactory":
    """A factory over a throwaway engine spec, for the capture field alone."""
    return ullm.backends.vllm_factory.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m", max_num_seqs=max_num_seqs),
        backend_cls=ullm.backends.vllmx.VLLM,
        **fields,
    )


def _sizes(factory: "ullm.backends.vllm_factory.EngineFactory") -> Any:
    return getattr(factory.engine_args.compilation_config, "cudagraph_capture_sizes", None)


def test_every_decode_width_and_the_ladder_above() -> None:
    """A graph serves one exact size, so a batch between rungs is padded up
    and the model computes filler it discards. Decode is never dispatched
    above the width, so that range is the only one worth making dense -- and
    the ladder above it has to survive, because mixed batches land there and
    prefill overflows past it."""
    factory = _factory()
    factory._settle_at_start(factory.engine_args)
    sizes = _sizes(factory)
    assert [s for s in sizes if s <= 40] == list(range(1, 41)), "every decode width"
    assert max(sizes) > 40, "the mixed-batch ladder is left standing"
    assert sorted(sizes) == sizes and len(set(sizes)) == len(sizes)


def test_on_by_default_and_not_written_until_start() -> None:
    """Construction states it; start applies it, so a width changed after
    construction is the width the rungs follow."""
    factory = _factory()
    assert factory.dense_cuda_capture is True
    assert not _sizes(factory)
    factory.engine_args.max_num_seqs = 12
    factory._settle_at_start(factory.engine_args)
    assert [s for s in _sizes(factory) if s <= 12] == list(range(1, 13))


def test_an_unset_width_is_not_guessed() -> None:
    """vLLM's own default width depends on the serving topology, so any guess
    here would be wrong under one of them."""
    factory = _factory(max_num_seqs=None)
    factory._settle_at_start(factory.engine_args)
    assert not _sizes(factory)


def test_named_rungs_are_the_callers() -> None:
    factory = _factory()
    factory.engine_args.compilation_config = vllm.config.CompilationConfig(cudagraph_capture_sizes=[1, 7])
    factory._settle_at_start(factory.engine_args)
    assert _sizes(factory) == [1, 7]


def test_off_writes_nothing() -> None:
    factory = _factory(dense_cuda_capture=False)
    factory._settle_at_start(factory.engine_args)
    assert not _sizes(factory)


def test_a_drafter_widens_every_rung_to_its_query_length() -> None:
    """A drafted decode step is dispatched only at an exact multiple of
    1 + draft tokens, so those multiples are the dense rungs, up to the widest
    drafted batch."""
    factory = _factory(max_num_seqs=40)
    factory.engine_args.speculative_config = {"method": "mtp", "num_speculative_tokens": 3}
    factory._settle_at_start(factory.engine_args)
    sizes = _sizes(factory)
    assert set(range(4, 161, 4)) <= set(sizes), "every drafted decode width"
    assert 160 <= max(sizes), "the widest drafted batch has a graph"
    assert {1, 2, 8} <= set(sizes), "the ladder stays for non-uniform steps"
    assert sorted(sizes) == sizes and len(set(sizes)) == len(sizes)


def test_a_named_ceiling_bounds_the_rungs() -> None:
    """vLLM refuses a named ceiling that differs from the widest rung."""
    factory = _factory(max_num_seqs=40)
    factory.engine_args.speculative_config = {"method": "mtp", "num_speculative_tokens": 3}
    factory.engine_args.compilation_config = vllm.config.CompilationConfig(max_cudagraph_capture_size=96)
    factory._settle_at_start(factory.engine_args)
    sizes = _sizes(factory)
    assert max(sizes) == 96
    assert set(range(4, 97, 4)) <= set(sizes)
