"""One real Flash-Next MoE layer through Marlin and the CuTe DSL W4A16 kernel, against an exact fp32 reference.

The layer's NVFP4 experts are read from CHECKPOINT and prepared by vLLM's own functions for each backend
(``prepare_nvfp4_moe_layer_for_marlin``; ``prepare_nvfp4_moe_layer_for_flashinfer_cutedsl`` and the MMA
view of its scales, the alphas the weights' global scales, as the oracle's W4A16 mode builds them). The
reference dequantizes the same weights exactly (E2M1 values, their FP8 block scales and the global scale)
and runs the layer in fp32 from the same bf16 activations. Reports each kernel's error against it.

    python tools/moe_layer_reference_check.py [LAYER]
"""

import json
import sys
import types
from typing import Any, cast

import torch
from safetensors import safe_open

CHECKPOINT = "/data/shared/research/checkpoints/Qwen3.8-Flash-Next-NVFP4-arc3-reap307-dials-joint205-u800"
HIDDEN, INTERMEDIATE, TOP_K = 2560, 640, 10
TOKEN_COUNTS = (216, 1080, 9272)
E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])


def load_layer(layer: int) -> dict[str, torch.Tensor]:
    """The layer's experts stacked: w13 (gate then up), w2, their FP8 block scales and global scales."""
    index = json.load(open(f"{CHECKPOINT}/model.safetensors.index.json"))["weight_map"]
    prefix = f"model.language_model.layers.{layer}.mlp.experts."
    experts = sorted({int(k[len(prefix) :].split(".")[0]) for k in index if k.startswith(prefix)})
    tensors: dict[str, torch.Tensor] = {}
    handles: dict[str, Any] = {}
    for name in [
        f"{prefix}{e}.{p}.{t}"
        for e in experts
        for p in ("gate_proj", "up_proj", "down_proj")
        for t in ("weight", "weight_scale", "weight_scale_2")
    ]:
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(f"{CHECKPOINT}/{shard}", framework="pt", device="cpu")
        tensors[name] = handles[shard].get_tensor(name)

    def stack(proj: str, kind: str) -> torch.Tensor:
        return torch.stack([tensors[f"{prefix}{e}.{proj}.{kind}"] for e in experts])

    gate_s2, up_s2 = stack("gate_proj", "weight_scale_2"), stack("up_proj", "weight_scale_2")
    if not torch.equal(gate_s2, up_s2):
        print(f"gate and up global scales differ in {int((gate_s2 != up_s2).sum())} experts; using gate's", flush=True)
    return {
        "w13": torch.cat([stack("gate_proj", "weight"), stack("up_proj", "weight")], dim=1).cuda(),
        "w13_scale": torch.cat([stack("gate_proj", "weight_scale"), stack("up_proj", "weight_scale")], dim=1).cuda(),
        "w13_scale_2": gate_s2.reshape(-1).float().cuda(),
        "w2": stack("down_proj", "weight").cuda(),
        "w2_scale": stack("down_proj", "weight_scale").cuda(),
        "w2_scale_2": stack("down_proj", "weight_scale_2").reshape(-1).float().cuda(),
    }


def dequantize(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor) -> torch.Tensor:
    """(E, N, K/2) packed E2M1 (element 2i in the low nibble), (E, N, K/16) FP8 scales, (E,) global -> fp32."""
    lut = E2M1.to(packed.device)
    low, high = lut[(packed & 0xF).long()], lut[(packed >> 4).long()]
    values = torch.stack([low, high], dim=-1).reshape(*packed.shape[:-1], packed.shape[-1] * 2)
    blocks = scale.float().repeat_interleave(16, dim=-1)
    return values * blocks * scale_2.view(-1, 1, 1)


def reference(
    x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """The layer in fp32 from the bf16 activations, expert by expert."""
    xf = x.float()
    out = torch.zeros_like(xf)
    experts = cast(list[int], ids.unique().tolist())
    for e in experts:
        hits = (ids == e).nonzero(as_tuple=True)
        rows, slots = hits[0], hits[1]
        h = xf[rows] @ w13[e].T
        gate, up = h[:, :INTERMEDIATE], h[:, INTERMEDIATE:]
        y = (torch.nn.functional.silu(gate) * up) @ w2[e].T
        out.index_add_(0, rows, y * weights[rows, slots, None])
    return out


def main() -> None:
    """Both kernels on LAYER at each token count, each against the reference and each other."""
    from flashinfer.fused_moe.cute_dsl import CuteDslMoEWrapper
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.flashinfer_fp4_moe import (
        nvfp4_swizzled_scale_to_cutedsl_mma_view,
        prepare_nvfp4_moe_layer_for_flashinfer_cutedsl,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.scalar_type import scalar_types

    layer_id = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    torch.manual_seed(0)
    w = load_layer(layer_id)
    experts = w["w13"].shape[0]
    w13_ref = dequantize(w["w13"], w["w13_scale"], w["w13_scale_2"])
    w2_ref = dequantize(w["w2"], w["w2_scale"], w["w2_scale_2"])
    print(
        f"layer {layer_id}: {experts} experts; dequantized weight RMS w13 {w13_ref.pow(2).mean().sqrt():.4g}",
        flush=True,
    )

    marlin_layer = cast(
        Any,
        types.SimpleNamespace(
            num_experts=experts,
            hidden_size=HIDDEN,
            intermediate_size_per_partition=INTERMEDIATE,
            params_dtype=torch.bfloat16,
        ),
    )
    m13, m13s, m13g, m2, m2s, m2g = prepare_nvfp4_moe_layer_for_marlin(
        marlin_layer,
        w["w13"].clone(),
        w["w13_scale"].clone(),
        w["w13_scale_2"].clone(),
        w["w2"].clone(),
        w["w2_scale"].clone(),
        w["w2_scale_2"].clone(),
        is_act_and_mul=True,
    )
    cute_layer = cast(
        Any,
        types.SimpleNamespace(
            activation=MoEActivation.SILU,
            moe_config=types.SimpleNamespace(moe_parallel_config=types.SimpleNamespace(enable_eplb=False)),
        ),
    )
    ones = torch.ones(experts, device="cuda")
    c13, c13s, c13g, _, c2, c2s, c2g, _ = prepare_nvfp4_moe_layer_for_flashinfer_cutedsl(
        cute_layer,
        w["w13"].clone(),
        w["w13_scale"].clone(),
        w["w13_scale_2"].clone(),
        ones,
        w["w2"].clone(),
        w["w2_scale"].clone(),
        w["w2_scale_2"].clone(),
        ones.clone(),
    )
    c13s, c2s = nvfp4_swizzled_scale_to_cutedsl_mma_view(c13s), nvfp4_swizzled_scale_to_cutedsl_mma_view(c2s)
    wrapper = cast(
        Any,
        CuteDslMoEWrapper(
            num_experts=experts,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            use_cuda_graph=True,
            quant_mode="w4a16",
        ),
    )

    print(
        f"{'tokens':>6} {'marlin rel err':>14} {'w4a16 rel err':>13} {'marlin max':>10} {'w4a16 max':>9} {'between rel':>11}",
        flush=True,
    )
    for t in TOKEN_COUNTS:
        # Activations at a decoder layer's MoE input scale: RMS-normed, so unit RMS.
        x = torch.randn(t, HIDDEN, device="cuda").to(torch.bfloat16)
        weights, ids = torch.topk(torch.softmax(torch.randn(t, experts, device="cuda"), -1), TOP_K, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).float()
        ids = ids.to(torch.int32)
        ref = reference(x, ids, weights, w13_ref, w2_ref)
        marlin = fused_marlin_moe(
            hidden_states=x,
            w1=m13,
            w2=m2,
            bias1=None,
            bias2=None,
            w1_scale=m13s,
            w2_scale=m2s,
            topk_weights=weights,
            topk_ids=ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=experts,
            global_scale1=m13g,
            global_scale2=m2g,
            workspace=marlin_layer.workspace,
        ).float()
        cute = cast(torch.Tensor, wrapper.run(x, None, ids, weights, c13, c13s, c13g, None, c2, c2s, c2g)).float()
        norm = cast(float, torch.linalg.vector_norm(ref).item())
        print(
            f"{t:>6} {(marlin - ref).norm() / norm:>14.3e} {(cute - ref).norm() / norm:>13.3e} "
            f"{(marlin - ref).abs().max():>10.3e} {(cute - ref).abs().max():>9.3e} {(marlin - cute).norm() / norm:>11.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
