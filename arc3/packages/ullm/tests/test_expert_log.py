"""The expert routing histogram: what it accumulates, in which units, what it writes,
and that the plugin and the factory's two fields stay inert unless asked for.
"""

from __future__ import annotations

import json
import os
import types
from pathlib import Path
from typing import Any

import pytest
import ullm.expert_log as E


def test_norm_accumulators_carry_the_right_units(tmp_path: Path) -> None:
    """gate*norm is a plain product of a weight and an L2 norm; only the _sq table is squared."""
    import torch

    hist = E.ExpertHistogram(tmp_path / "h.json", num_experts=4)
    # Two tokens, k=2. Rows are the kernel's permuted expert outputs; the map
    # says which row each (token, slot) got. Norms: row0=5, row1=10, row2=1.
    rows = torch.tensor([[3.0, 4.0], [6.0, 8.0], [1.0, 0.0], [0.0, 0.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=torch.bfloat16)
    permute_map = torch.tensor([0, 1, 2, 0], dtype=torch.int32)  # token0 -> rows 0,1; token1 -> rows 2,0
    ids = torch.tensor([[1, 3], [2, 1]])
    final = torch.tensor([[4.5, 6.0], [2.5, 3.0]], dtype=torch.bfloat16)  # the gate-weighted row sums
    # Equal logits at the chosen ids give softmax weights 0.5/0.5 for token 0;
    # token 1's derived weights differ from the kernel's on purpose, so the
    # weights check must report exactly that gap.
    hist.stash(0, torch.zeros(2, 4))
    hist.stash_norms(0, final, rows, weights, permute_map)
    hist.capture(0, ids)

    assert hist._norm_steps == 1 and hist._norm_failures == 0
    gn = hist._gate_norm[0].tolist()
    assert gn[1] == pytest.approx(0.5 * 5 + 0.75 * 5)  # expert 1: token0 slot0 (row0) + token1 slot1 (row0)
    assert gn[3] == pytest.approx(0.5 * 10)  # expert 3: token0 slot1 (row1)
    assert gn[2] == pytest.approx(0.25 * 1)  # expert 2: token1 slot0 (row2)
    sq = hist._gate_norm_sq[0].tolist()
    assert sq[1] == pytest.approx((0.5 * 5) ** 2 + (0.75 * 5) ** 2)
    assert sq[3] == pytest.approx((0.5 * 10) ** 2)
    assert hist._kernel_mass[0].tolist() == pytest.approx([0, 1.25, 0.25, 0.5])
    # derived weights are 0.5/0.5 for both tokens (equal logits), so squares are 0.25 each
    assert hist._mass_sq[0].tolist() == pytest.approx([0, 0.5, 0.25, 0.25])
    assert float(hist._finalize_err) == pytest.approx(0.0, abs=1e-6)
    assert float(hist._weights_err) == pytest.approx(0.25, abs=1e-6)  # kernel 0.75 vs derived 0.5

    hist.write()
    payload = json.loads((tmp_path / "h.json").read_text())
    assert payload["num_experts"] == 4
    assert payload["captures"] == 1
    assert payload["norm_captures"] == 1
    assert payload["counts_by_layer"] == {"0": {"1": 2, "2": 1, "3": 1}}
    assert payload["gate_norm_by_layer"]["0"]["3"] == pytest.approx(5.0)
    assert payload["norm_checks"]["finalize_max_rel_err"] == pytest.approx(0.0, abs=1e-6)
    assert "UNWEIGHTED" in payload["norm_units"]
    assert payload["mass_sq_by_layer"]["0"]["1"] == pytest.approx(0.5)
    assert payload["mass_method"].startswith("derived")


def test_reset_zeroes_in_place_and_a_non_finite_check_step_is_ignored(tmp_path: Path) -> None:
    """Warm-up's garbage forwards are counted (the hook precedes graph capture) and then wiped;
    a self-check on a non-finite kernel output must not poison the running maximum."""
    import torch

    hist = E.ExpertHistogram(tmp_path / "h.json", num_experts=4)
    rows = torch.tensor([[3.0, 4.0], [float("nan"), 0.0]], dtype=torch.bfloat16)
    weights = torch.tensor([[1.0, 0.0]], dtype=torch.bfloat16)
    final = torch.tensor([[float("nan"), float("nan")]], dtype=torch.bfloat16)
    # As in the engine: every capture runs inside the forward's inference
    # mode, where the accumulators come into being; the reset runs outside it.
    with torch.inference_mode():
        hist.stash(0, torch.zeros(1, 4))
        hist.stash_norms(0, final, rows, weights, torch.tensor([0, 1], dtype=torch.int32))
        hist.capture(0, torch.tensor([[1, 2]]))
    assert float(hist._finalize_err) == 0.0
    counts_row = hist._counts[0]
    assert counts_row.tolist() == [0, 1, 1, 0] and hist._steps == 1
    hist.reset()
    assert hist._counts[0] is counts_row, "the graphs hold this tensor; it is zeroed, not replaced"
    assert counts_row.tolist() == [0, 0, 0, 0] and hist._steps == 0 and hist._norm_steps == 0
    with torch.inference_mode():
        hist.capture(0, torch.tensor([[3, 3]]))
    assert counts_row.tolist() == [0, 0, 0, 2]


def test_padded_rows_stay_out_of_every_accumulator(tmp_path: Path) -> None:
    """A CUDA-graph batch carries padded rows the MoE routes too; with the
    runner's mask given, counts, mass, kernel mass and norms take only the
    live rows, and the JSON says how many rows were padding."""
    import torch

    padding = torch.tensor([False, True, True, False, False, False])  # the runner's buffer, longer than the batch
    hist = E.ExpertHistogram(tmp_path / "h.json", num_experts=4, padding=padding)
    hist._first_layer = 0
    ids = torch.tensor([[0, 1], [2, 3], [2, 3], [1, 0]])  # rows 1 and 2 are padding
    rows = torch.ones(8, 2, dtype=torch.bfloat16)
    weights = torch.full((4, 2), 0.5, dtype=torch.bfloat16)
    permute_map = torch.arange(8, dtype=torch.int32)
    final = torch.ones(4, 2, dtype=torch.bfloat16)
    hist.stash(0, torch.zeros(4, 4))
    hist.stash_norms(0, final, rows, weights, permute_map)
    hist.capture(0, ids)
    hist.capture(1, ids)  # a second layer: rows are tallied once per forward, at the first layer
    assert hist._counts[0].tolist() == [2, 2, 0, 0], "experts 2 and 3 were chosen by padded rows only"
    assert hist._mass[0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert hist._kernel_mass[0].tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])
    assert hist._gate_norm[0].tolist() == pytest.approx([0.5 * 2**0.5 * 2, 0.5 * 2**0.5 * 2, 0.0, 0.0])
    hist.write()
    payload = json.loads((tmp_path / "h.json").read_text())
    assert payload["padding"] == {"masked": True, "rows_seen": 4, "rows_padded": 2}
    unmasked = E.ExpertHistogram(tmp_path / "u.json", num_experts=4)
    unmasked.capture(0, ids)
    unmasked.write()
    assert json.loads((tmp_path / "u.json").read_text())["padding"] == {
        "masked": False,
        "rows_seen": None,
        "rows_padded": None,
    }


def test_a_padding_mask_is_bound_only_while_the_runner_maintains_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """The v2 runner's mask buffer exists whatever the flag says, but is
    refreshed only under VLLM_MOE_SKIP_PADDING; graph capture leaves it all
    True, so binding it with the flag off would drop every live row."""
    pytest.importorskip("vllm", reason="needs the vllm extra (linux)")
    from types import SimpleNamespace

    import torch

    runner = SimpleNamespace(input_buffers=SimpleNamespace(is_padding=torch.ones(4, dtype=torch.bool)))
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    assert E._padding_mask(runner) is runner.input_buffers.is_padding
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "0")
    with pytest.raises(RuntimeError, match="VLLM_MOE_SKIP_PADDING"):
        E._padding_mask(runner)
    assert E._padding_mask(SimpleNamespace()) is None, "the v1 runner keeps no mask; rows count as they come"


def test_capture_without_norms_writes_no_norm_tables(tmp_path: Path) -> None:
    """A plain histogram run carries counts and mass only; the norm tables appear with the norm mode."""
    import torch

    hist = E.ExpertHistogram(tmp_path / "h.json", num_experts=3)
    hist.stash(5, torch.tensor([[1.0, 0.0, 0.0]]))
    hist.capture(5, torch.tensor([[0, 2]]))
    hist.write()
    payload = json.loads((tmp_path / "h.json").read_text())
    assert payload["counts_by_layer"] == {"5": {"0": 1, "2": 1}}
    assert "gate_norm_by_layer" not in payload and "norm_units" not in payload
    assert payload["mass_by_layer"]["5"]["0"] > payload["mass_by_layer"]["5"]["2"]


def test_the_plugin_is_inert_without_a_log_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """vLLM runs every general plugin in every worker; without the variable this one must touch nothing."""
    monkeypatch.delenv(E.ENV_LOG_DIR, raising=False)
    monkeypatch.delenv(E.ENV_NORMS, raising=False)
    touched: list[Any] = []
    monkeypatch.setattr(E, "_runner_classes", lambda: touched.append(1))
    E.register()
    assert touched == []
    monkeypatch.setenv(E.ENV_NORMS, "1")
    with pytest.raises(RuntimeError, match=E.ENV_LOG_DIR):
        E.register()


def _engine_factory(**fields: Any) -> Any:
    """A factory over a throwaway engine spec, for the expert fields alone."""
    vllm = pytest.importorskip("vllm", reason="needs the vllm extra (linux)")
    import ullm.backends.vllm_factory
    import ullm.backends.vllmx

    return ullm.backends.vllm_factory.EngineFactory(
        engine_args=vllm.AsyncEngineArgs(model="facebook/opt-125m"),
        backend_cls=ullm.backends.vllmx.VLLM,
        **fields,
    )


def test_the_factory_ships_the_two_variables_to_the_engine_process() -> None:
    """The fields are how a run asks for the histogram; the plugin reads them in the spawned engine core."""
    factory = _engine_factory
    off = factory()
    off._apply_expert_log()
    assert E.ENV_LOG_DIR not in off.env_overrides and E.ENV_NORMS not in off.env_overrides
    on = factory(expert_log_dir="/tmp/hist", expert_norms=True)
    on._apply_expert_log()
    assert on.env_overrides[E.ENV_LOG_DIR] == "/tmp/hist" and on.env_overrides[E.ENV_NORMS] == "1"
    assert on.engine_args.enable_return_routed_experts is False, "the plugin binds the capturer itself"
    with pytest.raises(ValueError, match="expert_log_dir"):
        factory(expert_norms=True)._apply_expert_log()
    assert os.environ.get(E.ENV_LOG_DIR) is None, "applied to the overrides, not to this process"


def test_the_histogram_refuses_an_engine_whose_gate_returns_packed_routing() -> None:
    """XKB_XROUTER replaces each MoE layer's router logits with its routing,
    which the histogram would read as logits; the factory refuses the pair."""
    with pytest.raises(ValueError, match="XKB_XROUTER"):
        _engine_factory(expert_log_dir="/tmp/hist", env_overrides={"XKB_XROUTER": "1"})._apply_expert_log()
    counted = _engine_factory(expert_log_dir="/tmp/hist", env_overrides={"XKB_XROUTER": "0"})
    counted._apply_expert_log()
    assert counted.env_overrides[E.ENV_LOG_DIR] == "/tmp/hist"


def test_the_runner_wrapper_refuses_a_router_output_narrower_than_the_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine-side guard behind the factory's: a gate output with fewer
    columns than the layer has experts is not their logits."""
    torch = pytest.importorskip("torch")
    moe_runner = pytest.importorskip("vllm.model_executor.layers.fused_moe.runner.moe_runner")
    capturer = pytest.importorskip("vllm.model_executor.layers.fused_moe.routed_experts_capturer")

    class Runner:
        """A MoE runner reduced to the seam the wrapper binds."""

        layer_id = 3

        def _apply_quant_method(self, hidden_states: Any, router_logits: Any) -> Any:
            """Route as the real runner would; here, hand the logits back."""
            return router_logits

    def bind_nothing(model: Any, hist: Any) -> None:
        """vLLM's capturer binder, left out: this test reads the logits seam alone."""

    monkeypatch.setattr(moe_runner, "MoERunner", Runner)
    monkeypatch.setattr(capturer, "bind_routed_experts_capturer", bind_nothing)
    runner = Runner()
    model = types.SimpleNamespace(modules=lambda: [runner])
    E._bind_runners(model, E.ExpertHistogram(tmp_path / "hist.json", num_experts=8), 8)
    runner._apply_quant_method(None, torch.zeros(2, 9))  # eight experts and the shared-expert gate
    with pytest.raises(RuntimeError, match="not their logits"):
        runner._apply_quant_method(None, torch.zeros(2, 4))


def test_a_later_factory_takes_the_calibration_variables_out_of_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The factory owns the two variables in this process: the plugin reads
    them in the engine core spawned at each start, so a counts-only or a
    production factory started after a calibration one must not inherit
    its setting."""

    def start_env(factory: Any) -> None:
        """What _start does to the process environment, without an engine."""
        factory._apply_expert_log()
        factory._apply_env_overrides()

    calibration = _engine_factory(expert_log_dir="/tmp/hist", expert_norms=True)
    for key in (*calibration.env_overrides, E.ENV_LOG_DIR, E.ENV_NORMS):
        monkeypatch.delenv(key, raising=False)  # restored at teardown, whatever this test sets
    start_env(calibration)
    assert os.environ[E.ENV_LOG_DIR] == "/tmp/hist" and os.environ[E.ENV_NORMS] == "1"
    start_env(_engine_factory(expert_log_dir="/tmp/hist2"))
    assert os.environ[E.ENV_LOG_DIR] == "/tmp/hist2" and E.ENV_NORMS not in os.environ, "norms to counts-only"
    start_env(_engine_factory())
    assert E.ENV_LOG_DIR not in os.environ and E.ENV_NORMS not in os.environ, "calibration to production"
