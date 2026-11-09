"""Per-layer expert routing, measured: histograms and output norms out.

One seam, ``MoERunner._apply_quant_method``, wrapped per runner instance.

**Histogram** -- counts and gate-weight mass per (layer, expert), written to
``ULLM_EXPERT_LOG_DIR``. With ``ULLM_EXPERT_NORMS`` set, also the gate weight
times the expert's OUTPUT norm (and its square), which is what criteria in
the REAP family rank by; see ``stash_norms`` for units. The histogram is the
input of ``tools/checkpoints/expert_prune_plan.py``; nothing here changes
routing.

Why that seam and not the router: on this model the NvFp4 backend prefers
the MONOLITHIC TRT-LLM kernel, which fuses top-k into the expert kernel and
routes from the raw logits in CUDA -- the router's own selection method is
never called, and a wrapper there sees nothing (a control run under one
captured zero routings while playing normally). ``_apply_quant_method``
receives the logits after any gate implementation and before either routing
path, so the logits stashed there are what both paths route from. The
runner builds the logits from a COMBINED gate weight,
``[512 routed | 1 shared-expert gate]``, so only the leading ``num_experts``
columns are read.

Ids come from vLLM's own ``bind_routed_experts_capturer``: on the monolithic
path that is the kernel's routing-replay buffer, the one hook proven to fire
here. The kernel returns no ids with its weights, so mass is DERIVED --
softmax over the stashed logits gathered at the captured ids -- which is
exact for softmax-renormalise routing and labelled as derived in the output
either way, with the routing method recorded beside it.

Output norms need the per-expert outputs, which the kernel only hands back
on its unfinalised path -- and that path is dead here: vLLM guards it behind
tensor parallelism, and nothing in this build turns it on. So the kernel
entry point itself is wrapped and, in norm mode, called a SECOND time on the
same inputs with finalisation off. The production result is the first call,
untouched; the second exists only to be measured. That doubles the MoE
kernel time of a calibration run and nothing else.

Loaded as a ``vllm.general_plugins`` entry point, which vLLM runs in every
worker process -- the engine core is a spawned process, so a patch made from
the launcher would land in the wrong interpreter. Inert unless
``ULLM_EXPERT_LOG_DIR`` is set. The hook is installed at the end of the
model runner's KV-cache initialisation: after the profiling forward, whose
dummy routings would otherwise land in the counts, and before warm-up. It
binds vLLM's capturer itself, so ``enable_return_routed_experts`` stays off:
that flag makes vLLM build a per-request routing buffer anchored to a KV
group, and it refuses to coexist with a KV connector -- the host tier this
engine serves with. Aggregating across requests is all a histogram needs.

Padded rows stay out. A CUDA-graph batch is padded to its capture size and
the MoE routes the padded rows too, from whatever the input buffer last
held there; the v2 runner marks them in ``input_buffers.is_padding``
(``VLLM_MOE_SKIP_PADDING``, on by default), and every accumulator is
multiplied by that mask's complement, in-graph. The JSON carries how many
rows were seen and how many of them were padding.

Everything in the hot path is graph-safe: no ``.item()``, no allocation
sized from the data. A device-to-host sync during CUDA graph capture
invalidates the capture and takes the engine down.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, cast

ENV_LOG_DIR = "ULLM_EXPERT_LOG_DIR"
ENV_NORMS = "ULLM_EXPERT_NORMS"

# Writing means .tolist(), a device-to-host copy. Graph capture happens during
# warm-up, so the writer waits past it and then works on a slow cadence; the
# counts are cumulative, so a late or missed write loses nothing.
_WRITE_DELAY_S = 240.0
_WRITE_INTERVAL_S = 60.0
# The finalise self-check materialises [tokens, k, hidden] in fp32. Decode
# steps are small enough for that to cost nothing, and they run the same
# kernel path prefill does, so the check is confined to them.
_FINALIZE_CHECK_MAX_TOKENS = 64

#: The layer whose kernel call is in flight, set by the runner wrapper around
#: the original. The kernel wrapper has no other way to learn which layer it
#: serves: the kernel takes weights, not a module.
_CURRENT_LAYER: list[int | None] = [None]

NORM_UNITS = (
    "gate_norm: sum over the tokens routed to the expert of gate_weight * L2 norm of the "
    "UNWEIGHTED expert output, in residual-stream units. gate_norm_sq: sum of the square of "
    "that product, residual units squared. mass_sq: sum of squared gate weights, dimensionless. "
    "kernel_mass: sum of the kernel's own gate weights, dimensionless. Divide any table by "
    "counts_by_layer for the per-activation mean."
)


def _text_config(vllm_config: Any) -> Any:
    """The config the model reads: ``hf_text_config`` where there is one."""
    hf = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
    return getattr(hf, "text_config", None) or hf


def _num_experts_from(vllm_config: Any) -> int:
    """Expert count from config, since the hot path may not look at the data."""
    for obj in (_text_config(vllm_config), getattr(getattr(vllm_config, "model_config", None), "hf_config", None)):
        if obj is None:
            continue
        for name in ("num_experts", "n_routed_experts", "moe_num_experts", "num_local_experts"):
            value = getattr(obj, name, None)
            if isinstance(value, int) and value > 0:
                return value
    return 1024  # headroom over this model's 512; a wrong guess only wastes zeros


class ExpertHistogram:
    """Accumulates ``topk_ids`` and gate weights per layer, writes JSON periodically."""

    def __init__(self, out_path: Path, num_experts: int, padding: Any = None):
        """``num_experts`` fixes the counter width: capture() cannot size from the data.

        ``padding`` is the runner's device buffer marking a batch's padded
        rows (``input_buffers.is_padding``, True where a row is padding);
        every accumulator is multiplied by its complement, in-graph. None
        counts every row as live, and the JSON says so.
        """
        self.out_path = out_path
        self.num_experts = num_experts
        self._padding = padding
        #: the layer whose captures count the batch's rows, once per forward
        self._first_layer: int | None = None
        self._rows_seen: Any = None  # device int64 scalars, kept on device like the maxima
        self._rows_padded: Any = None
        self._counts: dict[int, Any] = {}
        self._mass: dict[int, Any] = {}
        self._mass_sq: dict[int, Any] = {}
        self._gate_norm: dict[int, Any] = {}
        self._gate_norm_sq: dict[int, Any] = {}
        self._kernel_mass: dict[int, Any] = {}
        self._pending: dict[int, Any] = {}  # this forward's routed logits, per layer
        self._pending_norms: dict[int, tuple[Any, Any]] = {}  # (kernel weights, gate*norm), per layer
        self._steps = 0
        self._norm_steps = 0
        self._norm_failures = 0
        self._derived_mass = False
        # Running maxima kept ON DEVICE: a self-check that synced would be a
        # self-check that cannot run during graph capture.
        self._finalize_err: Any = None  # max |our finalise - kernel finalise| / max |kernel finalise|
        self._weights_err: Any = None  # max |kernel gate weight - derived gate weight|
        self.routing_method: str = "unknown"

    def stash(self, layer_id: int, routed_logits: Any) -> None:
        """Keep this forward's logits (a view, no copy) so capture() can derive mass."""
        self._pending[layer_id] = routed_logits

    def stash_norms(self, layer_id: int, final_out: Any, rows: Any, weights: Any, permute_map: Any) -> None:
        """Turn one unfinalised kernel result into per-slot gate*norm; check its units.

        ``rows`` are the kernel's permuted, UNWEIGHTED expert outputs, one per
        (token, slot); ``permute_map`` says which row each slot got;
        ``weights`` are the gate weights the kernel applied. The norm is a
        true L2 norm (root of the sum of squares) taken in fp32 from the bf16
        row, of the expert output BEFORE gate weighting, so ``gate * norm`` is
        the magnitude of what that expert added to the residual stream. The
        square is formed later, at the accumulator, and nowhere else.

        The units are then PROVED rather than assumed on small batches:
        finalising the rows ourselves as the gate-weighted sum must reproduce
        the kernel's own finalised output. A constant-factor disagreement
        would mean a scale is folded into finalisation and the norms are off
        by it; the running maximum lands in the JSON.
        """
        import torch

        num_tokens, k = weights.shape
        idx = permute_map.reshape(num_tokens, k).to(torch.int64).clamp_min(0)
        row_norms = torch.linalg.vector_norm(rows, dim=-1, dtype=torch.float32)
        w = weights.detach().to(torch.float32)
        gate_norm = w * row_norms[idx]
        self._pending_norms[layer_id] = (w, gate_norm)
        if num_tokens <= _FINALIZE_CHECK_MAX_TOKENS:
            ours = (rows[idx].to(torch.float32) * w[..., None]).sum(dim=1)
            ref = final_out[:num_tokens].to(torch.float32)
            err = (ours - ref).abs().amax() / ref.abs().amax().clamp_min(1e-6)
            # Warm-up runs the kernel on uninitialised activations, whose
            # output is not finite; a check on that step says nothing about
            # the units and would poison the running maximum for good.
            err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
            if self._finalize_err is None:
                self._finalize_err = torch.zeros((), dtype=torch.float32, device=err.device)
            self._finalize_err.copy_(torch.maximum(self._finalize_err, err))

    def _accumulate(self, table: dict[int, Any], layer_id: int, ids: Any, values: Any) -> None:
        """scatter_add ``values`` (one per routing) into a float64 row per layer.

        float64 for precision, not units: a squared sum reaches magnitudes
        where fp32 silently drops every further small term.
        """
        import torch

        row = table.get(layer_id)
        if row is None:
            row = torch.zeros(self.num_experts, dtype=torch.float64, device=ids.device)
            table[layer_id] = row
        row.scatter_add_(0, ids, values.detach().flatten().to(torch.float64))

    def capture(self, layer_id: int, topk_ids: Any, weights: Any = None) -> None:
        """Called once per MoE layer per forward with the ids the kernel chose.

        Runs inside the model forward, which means during CUDA graph capture
        too -- so no device-to-host sync of any kind. With no ``weights``
        given (the monolithic kernel returns none with its ids), mass is
        derived from the logits stashed for this layer: gather at the ids,
        softmax over the k. Exact for softmax-renormalise routing; recorded
        as derived regardless.
        """
        try:
            import torch

            ids2d = topk_ids.detach().to(torch.int64)
            ids = ids2d.flatten()
            row = self._counts.get(layer_id)
            if row is None:
                row = torch.zeros(self.num_experts, dtype=torch.int64, device=ids.device)
                self._counts[layer_id] = row
            # A CUDA-graph batch is padded to its capture size and the MoE
            # routes the padded rows too, from whatever the input buffer last
            # held there. The runner marks them; a live row weighs 1, a padded
            # one 0, in every accumulator. All in-graph: the mask is a
            # persistent buffer the runner rewrites before each replay.
            live = None
            if self._padding is not None and ids2d.dim() == 2 and ids2d.shape[0] <= self._padding.shape[0]:
                live = (~self._padding[: ids2d.shape[0]]).to(torch.int64)
                if layer_id == self._first_layer:
                    self._count_rows(live)
            per_slot = torch.ones_like(ids) if live is None else live[:, None].expand(ids2d.shape).flatten()
            row.scatter_add_(0, ids, per_slot)
            if weights is None and ids2d.dim() == 2:
                logits = self._pending.get(layer_id)
                if logits is not None and logits.shape[0] == ids2d.shape[0]:
                    weights = torch.softmax(torch.gather(logits.float(), 1, ids2d), dim=1)
                    self._derived_mass = True
            if weights is not None and live is not None:
                weights = weights * live[:, None].to(weights.dtype)
            if weights is not None:
                # Contribution, not frequency. The top-k weights sum to 1 per
                # token, so summing them per expert gives that expert's share
                # of the layer's FFN output mass -- what a prune removes. An
                # expert called rarely at high weight matters more than one
                # called often at low weight; counts cannot tell them apart.
                self._accumulate(self._mass, layer_id, ids, weights)
                self._accumulate(self._mass_sq, layer_id, ids, weights * weights)
            pending = self._pending_norms.pop(layer_id, None)
            if pending is not None and pending[0].shape == ids2d.shape:
                kernel_w, gate_norm = pending
                if live is not None:
                    kernel_w = kernel_w * live[:, None].to(kernel_w.dtype)
                    gate_norm = gate_norm * live[:, None].to(gate_norm.dtype)
                self._accumulate(self._kernel_mass, layer_id, ids, kernel_w)
                self._accumulate(self._gate_norm, layer_id, ids, gate_norm)
                self._accumulate(self._gate_norm_sq, layer_id, ids, gate_norm * gate_norm)
                if weights is not None:
                    # Two sources for the same number -- the kernel's weights
                    # and our softmax -- so a slot misalignment between the
                    # replay ids and the kernel's weight order shows up here.
                    err = (kernel_w - weights).abs().amax()
                    err = torch.where(torch.isfinite(err), err, torch.zeros_like(err))
                    if self._weights_err is None:
                        self._weights_err = torch.zeros((), dtype=torch.float32, device=err.device)
                    self._weights_err.copy_(torch.maximum(self._weights_err, err))
                self._norm_steps += 1
            self._steps += 1
        except Exception:  # noqa: BLE001 - instrumentation must not kill a run
            pass

    def _count_rows(self, live: Any) -> None:
        """Tally a batch's rows and how many of them were padding, on device."""
        import torch

        if self._rows_seen is None:
            self._rows_seen = torch.zeros((), dtype=torch.int64, device=live.device)
            self._rows_padded = torch.zeros((), dtype=torch.int64, device=live.device)
        self._rows_seen.add_(live.numel())
        self._rows_padded.add_(live.numel() - live.sum())

    def reset(self) -> None:
        """Zero every accumulator in place, keeping the tensors the captured graphs reference.

        Called once warm-up is over: the hook has to be bound before graph
        capture (a replayed graph runs no Python, so a capture bound later
        would miss every graphed decode step), and the warm-up's dummy
        forwards route garbage that must not stay in the counts.
        """
        import torch

        # The rows were allocated inside the forward's inference mode, and an
        # inference tensor takes an in-place update only from inside it.
        with torch.inference_mode():
            for table in (
                self._counts,
                self._mass,
                self._mass_sq,
                self._gate_norm,
                self._gate_norm_sq,
                self._kernel_mass,
            ):
                for row in table.values():
                    row.zero_()
            for running_max in (self._finalize_err, self._weights_err, self._rows_seen, self._rows_padded):
                if running_max is not None:
                    running_max.zero_()
        self._steps = 0
        self._norm_steps = 0
        self._norm_failures = 0

    def start_writer(self) -> None:
        """Write from a daemon thread, off the forward and past graph capture."""

        def loop() -> None:
            """Sleep past warm-up, then write on a slow cadence forever."""
            import time

            time.sleep(_WRITE_DELAY_S)
            while True:
                self.write()
                time.sleep(_WRITE_INTERVAL_S)

        threading.Thread(target=loop, name="ullm-expert-log", daemon=True).start()
        # A run the backend gate stops dies about a minute after bind, before
        # the first scheduled write. The gate runs after warm-up, so this is
        # past graph capture and safe to sync.
        import atexit

        atexit.register(self.write)

    @staticmethod
    def _table(rows: dict[int, Any], digits: int) -> dict[str, dict[str, Any]]:
        """Nonzero entries only, per layer, as JSON-friendly strings."""
        return {
            str(layer): {str(i): round(float(m), digits) for i, m in enumerate(row.tolist()) if m}
            for layer, row in sorted(rows.items())
        }

    def write(self) -> None:
        """Dump counts and mass as JSON, atomically. Never raises."""
        try:
            payload: dict[str, Any] = {
                "captures": self._steps,
                "routing_method": self.routing_method,
                "mass_method": "derived: softmax over top-k logits (exact for softmax-renormalise)"
                if self._derived_mass
                else "kernel weights",
                "num_experts": self.num_experts,
                "padding": {
                    "masked": self._padding is not None,
                    "rows_seen": None if self._rows_seen is None else int(self._rows_seen),
                    "rows_padded": None if self._rows_padded is None else int(self._rows_padded),
                },
                "counts_by_layer": {
                    str(layer): {str(i): int(c) for i, c in enumerate(row.tolist()) if c}
                    for layer, row in sorted(self._counts.items())
                },
                "mass_by_layer": self._table(self._mass, 4),
                "mass_sq_by_layer": self._table(self._mass_sq, 6),
            }
            if self._norm_steps or self._norm_failures:
                payload["norm_units"] = NORM_UNITS
                payload["norm_captures"] = self._norm_steps
                payload["norm_checks"] = {
                    "finalize_max_rel_err": None if self._finalize_err is None else float(self._finalize_err),
                    "weights_max_abs_err": None if self._weights_err is None else float(self._weights_err),
                    "second_kernel_call_failures": self._norm_failures,
                }
                payload["kernel_mass_by_layer"] = self._table(self._kernel_mass, 4)
                payload["gate_norm_by_layer"] = self._table(self._gate_norm, 4)
                payload["gate_norm_sq_by_layer"] = self._table(self._gate_norm_sq, 4)
            tmp = self.out_path.with_name(self.out_path.name + ".tmp")
            tmp.write_text(json.dumps(payload))
            tmp.replace(self.out_path)
        except Exception:  # noqa: BLE001
            pass


def _bind_runners(model: Any, hist: ExpertHistogram, n: int) -> list[int]:
    """Wrap each runner's ``_apply_quant_method``; bind vLLM's id capture; return layer ids.

    The wrapper stashes the leading ``n`` logit columns -- a fresh tensor
    each forward, read and never written -- and marks the layer in flight
    for the kernel wrapper.

    Wrapped per instance rather than on the class: the bound original is
    captured in the closure, so nothing global is left patched if a second
    engine is built in the same process.
    """
    # From the module that DEFINES it: layer.py imports MoERunner without
    # re-exporting it, so importing from there reads a private detail.
    from vllm.model_executor.layers.fused_moe.routed_experts_capturer import bind_routed_experts_capturer
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

    bound: list[int] = []
    for module in model.modules():
        if not isinstance(module, MoERunner) or not hasattr(module, "_apply_quant_method"):
            continue
        layer_id = int(module.layer_id)

        def wrapped(
            *args: Any,
            _orig: Any = module._apply_quant_method,
            _lid: int = layer_id,
            **kw: Any,
        ) -> Any:
            """Stash the routed logit columns, then route as before."""
            logits = kw["router_logits"] if "router_logits" in kw else args[1]
            # A gate that returns something other than logits (the fork's
            # XKB_XROUTER packs weights and ids where they were) would make
            # every count derived below garbage; a shape is graph-safe to read.
            if logits.shape[-1] < n:
                raise RuntimeError(
                    f"expert_log: layer {_lid}'s router output has {logits.shape[-1]} columns, fewer than its "
                    f"{n} experts, so it is not their logits (XKB_XROUTER replaces them)"
                )
            hist.stash(_lid, logits[..., :n])
            _CURRENT_LAYER[0] = _lid
            try:
                return _orig(*args, **kw)
            finally:
                _CURRENT_LAYER[0] = None

        module._apply_quant_method = wrapped
        bound.append(layer_id)
    if not bound:
        raise RuntimeError("expert_log: no MoERunner with _apply_quant_method found")
    # vLLM's own binder: the router's capture_fn on modular kernels, the
    # routing-replay buffer on monolithic ones. Either way it calls
    # hist.capture(layer_id, topk_ids) with the ids the kernel chose.
    # Typed for vLLM's own capturer protocol; the histogram implements
    # the one method the binder calls (see capture()).
    bind_routed_experts_capturer(model, cast(Any, hist))
    try:
        method = next(m.moe_config.routing_method for m in model.modules() if isinstance(m, MoERunner))
        hist.routing_method = str(method)
    except Exception:  # noqa: BLE001 - a label, not a dependency
        pass
    return sorted(bound)


def _bind_kernel_norms(hist: ExpertHistogram) -> str:
    """Wrap the FlashInfer NvFp4 MoE entry point to measure expert outputs.

    vLLM's NvFp4 MoE backend calls this function by module attribute on every
    MoE forward, so replacing the attribute reaches it. The wrapper runs the real
    call first and returns exactly its result; then, when that call finalised
    (the only case this deployment has) and a layer is in flight, it runs the
    kernel again with finalisation off and hands the unfinalised result to the
    histogram. The second call gets no routing-replay buffer: ids are already
    captured from the first.

    Failures of the second call are counted, not raised -- a measurement must
    not take the run down -- and reported in the JSON, where a count of zero
    norm captures against a nonzero failure count says the kernel in this
    build does not honour the finalise switch.
    """
    import flashinfer.fused_moe as fi_moe

    orig = fi_moe.trtllm_fp4_block_scale_moe

    def wrapped(*args: Any, **kw: Any) -> Any:
        """The real call, then a measured one on the same inputs."""
        out = orig(*args, **kw)
        layer_id = _CURRENT_LAYER[0]
        if layer_id is None or not kw.get("do_finalize", True):
            return out
        try:
            again = dict(kw, do_finalize=False, routing_replay_out=None)
            res = orig(*args, **again)
            final = out[0] if isinstance(out, (tuple, list)) else out
            hist.stash_norms(layer_id, final, res[0], res[1], res[2])
        except Exception:  # noqa: BLE001 - instrumentation must not kill a run
            hist._norm_failures += 1
        return out

    fi_moe.trtllm_fp4_block_scale_moe = wrapped
    return f"{fi_moe.__name__}.trtllm_fp4_block_scale_moe"


def _padding_mask(runner: Any) -> Any:
    """The runner's padding mask, or None where the runner keeps none.

    The v2 runner's ``input_buffers.is_padding`` exists regardless of the
    flag, but is refreshed per batch only under ``VLLM_MOE_SKIP_PADDING``;
    graph capture fills it with True, so without the refresh it would mark
    every live row as padding once warm-up is over. A histogram asked for
    with the flag off is refused rather than written empty under a
    ``masked: true`` label. The v1 runner has no such buffer, and its rows
    count as they come, which the JSON says.
    """
    mask = getattr(getattr(runner, "input_buffers", None), "is_padding", None)
    if mask is None:
        return None
    import vllm.envs

    if not vllm.envs.VLLM_MOE_SKIP_PADDING:
        raise RuntimeError(
            f"{ENV_LOG_DIR} asks for the routing histogram, but VLLM_MOE_SKIP_PADDING is off and the runner "
            "then leaves its padding mask stale: every live row would count as padding. Set it to 1."
        )
    return mask


def _runner_classes() -> list[Any]:
    """The model runner classes this vLLM line can serve with.

    0.29 serves with the v1 runner unless ``VLLM_USE_V2_MODEL_RUNNER`` picks
    the v2 one; both define ``initialize_kv_cache``. Whichever is absent
    from this build is skipped.
    """
    classes: list[Any] = []
    for module_name in ("vllm.v1.worker.gpu_model_runner", "vllm.v1.worker.gpu.model_runner"):
        try:
            module = __import__(module_name, fromlist=["GPUModelRunner"])
        except Exception:  # noqa: BLE001 - a runner this build does not ship
            continue
        runner = getattr(module, "GPUModelRunner", None)
        if runner is not None and hasattr(runner, "initialize_kv_cache"):
            classes.append(runner)
    return classes


def register() -> None:
    """vLLM general plugin entry point."""
    log_dir = os.environ.get(ENV_LOG_DIR)
    norms = os.environ.get(ENV_NORMS, "") not in ("", "0", "false")
    if not log_dir:
        if norms:
            raise RuntimeError(f"{ENV_NORMS} is set but {ENV_LOG_DIR} is not: norms are written into the histogram")
        return

    def bind(runner: Any) -> None:
        """Bind the histogram to the loaded model, once per runner."""
        if getattr(runner, "_ullm_expert_histogram", None) is not None:
            return
        n = _num_experts_from(runner.vllm_config)
        # The writer never raises, so a missing directory would lose every
        # write silently; made here, where a failure still stops the engine.
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        padding = _padding_mask(runner)
        hist = ExpertHistogram(Path(log_dir) / f"routed_experts.{os.getpid()}.json", n, padding=padding)
        runner._ullm_expert_histogram = hist
        bound = _bind_runners(runner.model, hist, n)
        hist._first_layer = bound[0]
        hist.start_writer()
        print(
            f"[ullm.expert_log] routing histogram -> {hist.out_path} (experts={n}, routers={len(bound)}, "
            f"padded rows {'masked' if padding is not None else 'NOT masked: no is_padding buffer'})",
            flush=True,
        )
        if norms:
            where = _bind_kernel_norms(hist)
            print(f"[ullm.expert_log] expert output norms: second, unfinalised call of {where}", flush=True)

    for runner_cls in _runner_classes():

        def initialize_kv_cache(self: Any, *args: Any, _orig: Any = runner_cls.initialize_kv_cache, **kw: Any) -> Any:
            """vLLM's KV-cache init, then the hook: after profiling, before warm-up."""
            result = _orig(self, *args, **kw)
            bind(self)
            return result

        runner_cls.initialize_kv_cache = initialize_kv_cache

    from vllm.v1.worker.gpu_worker import Worker

    def compile_or_warm_up_model(self: Any, *args: Any, _orig: Any = Worker.compile_or_warm_up_model, **kw: Any) -> Any:
        """vLLM's warm-up, then the counts start from zero: what follows is served traffic."""
        result = _orig(self, *args, **kw)
        hist = getattr(self.model_runner, "_ullm_expert_histogram", None)
        if hist is not None:
            hist.reset()
            print("[ullm.expert_log] warm-up over, counts reset", flush=True)
        return result

    Worker.compile_or_warm_up_model = compile_or_warm_up_model
    print("[ullm.expert_log] router hook installed", flush=True)
