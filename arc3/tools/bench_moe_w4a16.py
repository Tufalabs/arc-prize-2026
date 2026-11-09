"""Marlin against FlashInfer's SM100 CuTe DSL W4A16 MoE, one Flash-Next MoE layer at a time.

Both read the same NVFP4 expert weights with bf16 activations. Times are per MoE call (routing sort,
both GEMMs, activation and the weighted sum), replayed from a CUDA graph as the engine's decode steps
are. The first W4A16 call per expert count is timed on its own: it compiles the CuTe DSL kernels.

    python tools/bench_moe_w4a16.py
"""

import time
import types
from collections.abc import Callable
from typing import Any, cast

import torch

HIDDEN = 2560
INTERMEDIATE = 640
TOP_K = 10
EXPERT_COUNTS = (205, 268, 108)
TOKEN_COUNTS = (216, 1080, 2160, 4096, 9272)
ITERS = 50

# A layer: (hidden states, top-k ids, top-k weights) -> output.
Layer = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _quantize(w: torch.Tensor, swizzled: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """(E, N, K) bf16 -> NVFP4 packed (E, N, K/2) and its block scales, swizzled as the kernel wants them."""
    from flashinfer.fp4_quantization import fp4_quantize

    e, rows, cols = w.shape
    q, sf = fp4_quantize(
        w.reshape(e * rows, cols),
        global_scale=torch.ones(1, device=w.device),
        sf_vec_size=16,
        is_sf_swizzled_layout=swizzled,
    )
    return q.view(e, rows, cols // 2), sf


def _graph_ms(fn: Callable[[], object]) -> float:
    """ms per call, replaying ``fn`` from a captured CUDA graph."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def marlin_layer(e: int, w13: torch.Tensor, w2: torch.Tensor) -> Layer:
    """A callable MoE layer on vLLM's Marlin path, from bf16 weights."""
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.scalar_type import scalar_types

    w13_q, w13_sf = _quantize(w13, swizzled=False)
    w2_q, w2_sf = _quantize(w2, swizzled=False)
    w13_sf = w13_sf.view(torch.float8_e4m3fn).view(e, 2 * INTERMEDIATE, HIDDEN // 16)
    w2_sf = w2_sf.view(torch.float8_e4m3fn).view(e, HIDDEN, INTERMEDIATE // 16)
    # The repack reads only these fields of the layer, and leaves its workspace on it.
    layer = cast(
        Any,
        types.SimpleNamespace(
            num_experts=e,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
        ),
    )
    ones = torch.ones(e, device="cuda", dtype=torch.float32)
    w13_m, w13_s, w13_g, w2_m, w2_s, w2_g = prepare_nvfp4_moe_layer_for_marlin(
        layer, w13_q, w13_sf, ones, w2_q, w2_sf, ones.clone(), is_act_and_mul=True
    )
    workspace = cast(torch.Tensor, layer.workspace)

    def run(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return fused_marlin_moe(
            hidden_states=x,
            w1=w13_m,
            w2=w2_m,
            bias1=None,
            bias2=None,
            w1_scale=w13_s,
            w2_scale=w2_s,
            topk_weights=weights,
            topk_ids=ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=e,
            global_scale1=w13_g,
            global_scale2=w2_g,
            workspace=workspace,
        )

    return run


def w4a16_layer(e: int, w13: torch.Tensor, w2: torch.Tensor) -> Layer:
    """A callable MoE layer on FlashInfer's CuTe DSL W4A16 path, from the same bf16 weights."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fused_moe.cute_dsl import CuteDslMoEWrapper

    w13_q, w13_sf = _quantize(w13, swizzled=True)
    w2_q, w2_sf = _quantize(w2, swizzled=True)
    w13_sf = convert_sf_to_mma_layout(w13_sf, m=2 * INTERMEDIATE, k=HIDDEN, num_groups=e, sf_vec_size=16)
    w2_sf = convert_sf_to_mma_layout(w2_sf, m=HIDDEN, k=INTERMEDIATE, num_groups=e, sf_vec_size=16)
    alpha = torch.ones(e, device="cuda", dtype=torch.float32)
    wrapper = cast(
        Any,
        CuteDslMoEWrapper(
            num_experts=e,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            use_cuda_graph=True,
            quant_mode="w4a16",
        ),
    )

    def run(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return cast(
            torch.Tensor, wrapper.run(x, None, ids, weights, w13_q, w13_sf, alpha, None, w2_q, w2_sf, alpha.clone())
        )

    return run


def main() -> None:
    """Each expert count, each token count: both kernels timed, W4A16 also autotuned."""
    from flashinfer.autotuner import autotune

    torch.manual_seed(0)
    print(
        f"device {torch.cuda.get_device_name()}  hidden {HIDDEN} intermediate {INTERMEDIATE} top_k {TOP_K}", flush=True
    )
    print(
        f"{'experts':>7} {'tokens':>6} {'marlin ms':>9} {'w4a16 ms':>8} {'w4a16 tuned':>11} {'speedup':>7}", flush=True
    )
    for e in EXPERT_COUNTS:
        w13 = torch.randn(e, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.bfloat16) / 10
        w2 = torch.randn(e, HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.bfloat16) / 10
        marlin = marlin_layer(e, w13, w2)
        w4a16 = w4a16_layer(e, w13, w2)
        del w13, w2
        for i, t in enumerate(TOKEN_COUNTS):
            x = torch.randn(t, HIDDEN, device="cuda", dtype=torch.bfloat16) / 10
            logits = torch.randn(t, e, device="cuda")
            weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).float()
            ids = ids.to(torch.int32)
            if i == 0:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                w4a16(x, ids, weights)
                torch.cuda.synchronize()
                print(f"  w4a16 first call at {e} experts (compiles): {time.perf_counter() - t0:.1f} s", flush=True)
            m_ms = _graph_ms(lambda: marlin(x, ids, weights))
            w_ms = _graph_ms(lambda: w4a16(x, ids, weights))
            with autotune(True):
                w4a16(x, ids, weights)
            torch.cuda.synchronize()
            wt_ms = _graph_ms(lambda: w4a16(x, ids, weights))
            print(
                f"{e:>7} {t:>6} {m_ms:>9.3f} {w_ms:>8.3f} {wt_ms:>11.3f} {m_ms / min(w_ms, wt_ms):>6.2f}x", flush=True
            )


if __name__ == "__main__":
    main()
