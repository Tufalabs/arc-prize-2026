"""The head's routed experts in NVFP4, as the target's are quantized.

The target (``head.TARGET``) carries ModelOpt NVFP4 on its routed experts
only; its config's ignore list keeps attention, routers, shared experts,
hyper-connections, embedding, LM head and the whole ``mtp.*`` block in bf16.
The recipe, read off its weights (every expert tensor of layers 0, 5, 20 and
47 requantizes to the same bytes, up to the sign of zero, under ModelOpt
0.46's ``NVFP4QTensor`` with the stored global scale):

- FP4 E2M1 codes, two a byte (even element in the low nibble); one FP8 E4M3
  scale per 16-element block from the block's max, round to nearest with
  ties to even; one FP32 global scale ``weight_scale_2`` = amax / (6 x 448).
- gate and up share their global scale over blocks of 128 experts by index
  (the producer's four-way expert parallelism); down has one per expert.
- static input scales, amax / (6 x 448) of what the expert reads: one for
  all of a layer's gate/up, one per expert for down. Both deployment cells
  run the experts on Marlin (4-bit weights over bf16 activations), which
  reads none of them; they are carried because the config declares 4-bit
  activations, and vLLM's W4A4 path folds an unloaded one into garbage.

The quantizer below is ModelOpt 0.46's (``modelopt/torch/quantization/qtensor/nvfp4_tensor.py``:
``_cast_fp4``, ``get_weights_scaling_factor``, ``quantize``, the plain
``dequantize``), copied because the jobs' venv carries no ModelOpt.

    python -m drafter.nvfp4 HEAD_DIR OUT_DIR --calibrate NAME=CACHE:DATASET [...]

writes ``HEAD_DIR`` (a drafter directory, e.g. ``heads/v1-final``) with its
experts quantized, input scales calibrated over the given sets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Container, Optional

import torch

E2M1_MAX, E4M3_MAX = 6.0, 448.0
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5])
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
BLOCK = 16
GLOBAL_GROUP = 128
EXPERTS = "mtp.layers.0.mlp.experts."
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# The head's own Linear layers outside the decoder layer: the target has no
# counterpart, and it quantizes nothing but routed experts.
IGNORE_ADDED = ["mtp.fc_embedding", "mtp.fc_hidden"]
IGNORE_REMOVED = ["mtp.*", "model.mtp.*"]


def quantized_ignore(names: list[str], where: str) -> list[str]:
    """A quantization ignore list once the head's experts are quantized: the
    patterns that kept the whole head out go, its own Linear layers come in."""
    missing = [name for name in IGNORE_REMOVED if name not in names]
    if missing:
        raise ValueError(f"{where}: the ignore list lacks {missing}")
    return [name for name in names if name not in IGNORE_REMOVED] + IGNORE_ADDED


def _cast_fp4(weight: torch.Tensor) -> torch.Tensor:
    sign_bit = (weight < 0).to(torch.uint8)
    weight_abs = weight.abs_()
    bounds = E2M1_BOUNDS.to(weight.device)
    ord = torch.searchsorted(bounds, weight_abs, out_int32=True).to(torch.uint8)
    odd_bounds = bounds[[1, 3, 5]]
    equals_odd_bounds = torch.any(weight_abs.unsqueeze(-1) == odd_bounds, dim=-1).to(torch.uint8)
    return (sign_bit << 3) + ord + equals_odd_bounds


def global_scale(amax: torch.Tensor) -> torch.Tensor:
    return amax.float() / (E2M1_MAX * E4M3_MAX)


def quantize(weight: torch.Tensor, weights_scaling_factor_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``weight`` ``[..., out, in]`` -> packed codes ``[..., out, in/2]``
    uint8 and block scales ``[..., out, in/16]`` FP8, under the global scale
    (a scalar, or one per leading index)."""
    scale_2 = weights_scaling_factor_2.float().to(weight.device)
    scale_2 = scale_2.view(*scale_2.shape, *[1] * (weight.dim() - scale_2.dim()))
    per_block_amax = weight.view(*weight.shape[:-1], -1, BLOCK).abs().amax(dim=-1).float()
    per_block_scale = per_block_amax / (E2M1_MAX * scale_2)
    per_block_scale[per_block_scale == 0] = 1.0
    per_block_scale = per_block_scale.clamp(min=2**-9, max=E4M3_MAX).to(torch.float8_e4m3fn)
    blocks = weight.view(*weight.shape[:-1], -1, BLOCK)
    scaled = blocks / (per_block_scale.to(torch.float32) * scale_2).unsqueeze(-1)
    q = _cast_fp4(scaled.view(weight.shape))
    return (q[..., 1::2] << 4) | q[..., 0::2], per_block_scale


def dequantize(
    packed: torch.Tensor, block_scale: torch.Tensor, weights_scaling_factor_2: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    scale_2 = weights_scaling_factor_2.float().to(packed.device)
    scale_2 = scale_2.view(*scale_2.shape, *[1] * (packed.dim() - scale_2.dim()))
    codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.long, device=packed.device)
    codes[..., 1::2] = packed >> 4
    codes[..., 0::2] = packed & 0x0F
    values = E2M1_VALUES.to(packed.device)[codes]
    values = values.view(*values.shape[:-1], -1, BLOCK) * (block_scale.to(torch.float32) * scale_2).unsqueeze(-1)
    return values.view(codes.shape).to(dtype)


def weight_global_scales(gate_up: torch.Tensor, down: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Global scales ``[E]`` for the packed experts: gate and up share one
    over each block of ``GLOBAL_GROUP`` experts, down one per expert."""
    per_expert = gate_up.abs().amax(dim=(1, 2)).float()
    grouped = per_expert.view(-1, GLOBAL_GROUP).amax(dim=1).repeat_interleave(GLOBAL_GROUP)
    return global_scale(grouped), global_scale(down.abs().amax(dim=(1, 2)))


@torch.no_grad()
def round_trip(gate_up: torch.Tensor, down: torch.Tensor, chunk: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """The packed experts as Marlin serves them: quantized and back, in the
    experts' dtype."""
    scale_gate_up, scale_down = weight_global_scales(gate_up, down)
    out_gate_up, out_down = torch.empty_like(gate_up), torch.empty_like(down)
    for start in range(0, gate_up.shape[0], chunk):
        s = slice(start, start + chunk)
        for source, scale, out in ((gate_up, scale_gate_up, out_gate_up), (down, scale_down, out_down)):
            packed, block = quantize(source[s].float(), scale[s])
            out[s] = dequantize(packed, block, scale[s], source.dtype)
    return out_gate_up, out_down


@torch.no_grad()
def fake_quantize(head: Any) -> None:
    """``head``'s routed experts replaced in place by their NVFP4 round trip."""
    experts = head.layer.mlp.experts
    gate_up, down = round_trip(experts.gate_up_proj.data, experts.down_proj.data)
    experts.gate_up_proj.data.copy_(gate_up)
    experts.down_proj.data.copy_(down)


def quantized_tensors(
    gate_up: torch.Tensor, down: torch.Tensor, gate_up_input_amax: torch.Tensor, down_input_amax: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Per-expert checkpoint tensors under the target's names, from the
    packed experts ``[E, 2I, H]`` / ``[E, H, I]`` and the calibrated input
    amax (a scalar for gate/up, ``[E]`` for down)."""
    scale_gate_up, scale_down = weight_global_scales(gate_up, down)
    half = gate_up.shape[1] // 2
    input_gate_up = global_scale(gate_up_input_amax).cpu()
    input_down = global_scale(down_input_amax).cpu()
    tensors: dict[str, torch.Tensor] = {}
    for e in range(gate_up.shape[0]):
        parts = (
            ("gate_proj", gate_up[e, :half], scale_gate_up[e], input_gate_up),
            ("up_proj", gate_up[e, half:], scale_gate_up[e], input_gate_up),
            ("down_proj", down[e], scale_down[e], input_down[e]),
        )
        for name, weight, scale_2, input_scale in parts:
            packed, block = quantize(weight.float(), scale_2)
            prefix = f"{EXPERTS}{e}.{name}."
            tensors[prefix + "weight"] = packed.contiguous().cpu()
            tensors[prefix + "weight_scale"] = block.contiguous().cpu()
            # Each scalar its own storage: safetensors refuses shared memory.
            tensors[prefix + "weight_scale_2"] = scale_2.float().reshape(()).cpu().clone()
            tensors[prefix + "input_scale"] = input_scale.float().reshape(()).cpu().clone()
    return tensors


def is_quantized(names: Container[str]) -> bool:
    return f"{EXPERTS}0.gate_proj.weight" in names


def expert_tensor_names(experts: int) -> set[str]:
    suffixes = ("weight", "weight_scale", "weight_scale_2", "input_scale")
    return {f"{EXPERTS}{e}.{p}.{s}" for e in range(experts) for p in PROJECTIONS for s in suffixes}


def read_packed(get: Callable[[str], torch.Tensor], experts: int) -> dict[str, torch.Tensor]:
    """A quantized checkpoint's experts, dequantized to the packed bf16
    tensors the head holds (``EXPERTS + gate_up_proj`` / ``down_proj``)."""
    gate_up, down = [], []
    for e in range(experts):
        parts = {}
        for name in PROJECTIONS:
            prefix = f"{EXPERTS}{e}.{name}."
            parts[name] = dequantize(
                get(prefix + "weight"), get(prefix + "weight_scale"), get(prefix + "weight_scale_2"), torch.bfloat16
            )
        gate_up.append(torch.cat([parts["gate_proj"], parts["up_proj"]]))
        down.append(parts["down_proj"])
    return {EXPERTS + "gate_up_proj": torch.stack(gate_up), EXPERTS + "down_proj": torch.stack(down)}


class InputAmax:
    """Records, per expert, the largest |input| its gate/up and its down read,
    over every call of the head's routed experts while installed."""

    def __init__(self, experts_module: Any) -> None:
        self.module = experts_module
        n = experts_module.num_experts
        device = experts_module.gate_up_proj.device
        self.gate_up = torch.zeros(n, device=device)
        self.down = torch.zeros(n, device=device)
        self.tokens = torch.zeros(n, dtype=torch.long, device=device)
        self.original = experts_module.forward

    def __enter__(self) -> "InputAmax":
        self.module.forward = self._forward
        return self

    def __exit__(self, *exc: Any) -> None:
        del self.module.forward

    @torch.no_grad()
    def _forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        m = self.module
        for e in top_k_index.unique().tolist():
            tokens = (top_k_index == e).any(dim=-1)
            x = hidden_states[tokens]
            gate, up = torch.nn.functional.linear(x, m.gate_up_proj[e]).chunk(2, dim=-1)
            h = m.act_fn(gate) * up
            self.gate_up[e] = torch.maximum(self.gate_up[e], x.abs().amax().float())
            self.down[e] = torch.maximum(self.down[e], h.abs().amax().float())
            self.tokens[e] += x.shape[0]
        return self.original(hidden_states, top_k_index, top_k_weights)

    def amax(self) -> tuple[torch.Tensor, torch.Tensor, int]:
        """gate/up's one amax for all experts; down's per expert, an expert no
        token reached taking the largest of the others (ModelOpt's
        ``set_expert_quantizer_amax`` fallback); and how many were unreached."""
        reached = self.tokens > 0
        down = torch.where(reached, self.down, self.down[reached].max())
        return self.gate_up.max(), down, int((~reached).sum())


def export(head_dir: Path, out_dir: Path, sets: dict[str, list[Any]], steps: int = 3) -> None:
    from safetensors import safe_open

    from drafter.files import save_file
    from drafter.head import LabelHead, load_head
    from drafter.train import Trainer, evaluate

    started = time.monotonic()
    head = load_head(head_dir, device="cuda")
    label = LabelHead(device="cuda")
    args = argparse.Namespace(steps=steps, eager_attention=False, kv_fp8=True, checkpoint_steps=False, ce_weight=0.0)
    trainer = Trainer(head, label, args, optimize=False)
    experts = head.layer.mlp.experts
    with InputAmax(experts) as recorder:
        for name, chains in sets.items():
            evaluate(trainer, chains, "cuda")
            print(
                f"[nvfp4] calibrated on {name} ({len(chains)} chains) at {time.monotonic() - started:.0f}s", flush=True
            )
    gate_up_amax, down_amax, unreached = recorder.amax()
    print(
        f"[nvfp4] input amax: gate/up {gate_up_amax.item():.4g}; down median "
        f"{down_amax.median().item():.4g} max {down_amax.max().item():.4g}; {unreached} experts unreached",
        flush=True,
    )
    quantized = quantized_tensors(experts.gate_up_proj.data, experts.down_proj.data, gate_up_amax, down_amax)
    del trainer, head, label
    torch.cuda.empty_cache()

    out_dir.mkdir(parents=True, exist_ok=False)
    tensors = {}
    with safe_open(str(head_dir / "model.safetensors"), framework="pt") as f:
        for name in f.keys():
            if name in (EXPERTS + "gate_up_proj", EXPERTS + "down_proj"):
                continue
            tensors[name] = f.get_tensor(name)
    tensors.update(quantized)
    save_file(tensors, str(out_dir / "model.safetensors"), metadata={"format": "pt"})
    for path in head_dir.iterdir():
        if path.is_file() and path.name not in ("model.safetensors", "config.json"):
            shutil.copy2(path, out_dir / path.name)
    config = json.loads((head_dir / "config.json").read_text())
    quantization = config["quantization_config"]
    quantization["ignore"] = quantized_ignore(quantization["ignore"], f"{head_dir}/config.json")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # ModelOpt's own record of the same list, where the source carries one:
    # vLLM reads config.json first, other tools read this file.
    record_path = head_dir / "hf_quant_config.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        excluded = record["quantization"]["exclude_modules"]
        record["quantization"]["exclude_modules"] = quantized_ignore(excluded, str(record_path))
        (out_dir / "hf_quant_config.json").write_text(json.dumps(record, indent=4) + "\n")
    (out_dir / "nvfp4_provenance.json").write_text(
        json.dumps(
            {
                "source": str(head_dir),
                "calibration": {name: len(chains) for name, chains in sets.items()},
                "gate_up_input_amax": gate_up_amax.item(),
                "unreached_experts": unreached,
                "global_group": GLOBAL_GROUP,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[nvfp4] {out_dir}: {len(tensors)} tensors, {len(quantized)} of them quantized experts' at {time.monotonic() - started:.0f}s",
        flush=True,
    )


def main(argv: Optional[list[str]] = None) -> None:
    from drafter.train import load_chains

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("head_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--calibrate", action="append", required=True, help="NAME=CACHE_DIR:DATASET_DIR")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    torch.backends.cuda.matmul.allow_tf32 = False
    sets = {}
    for spec in args.calibrate:
        name, paths = spec.split("=", 1)
        cache, dataset = paths.split(":", 1)
        sets[name] = load_chains(Path(cache), Path(dataset))
    export(args.head_dir, args.out_dir, sets, steps=args.steps)


if __name__ == "__main__":
    main()
