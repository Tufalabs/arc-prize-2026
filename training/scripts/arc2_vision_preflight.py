"""Bounded BF16 vision FlashAttention diagnostics inside an existing GPU allocation.

The native probe uses vLLM's actual variable-length vision attention wrapper.
The encoder probe instantiates the pinned model's complete vision architecture
with deterministic diagnostic weights; the normal server smoke test separately
validates checkpoint loading. No decoder weights or training jobs are launched.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any


PACKAGES = (
    "vllm",
    "torch",
    "torchvision",
    "transformers",
    "flash-attn-4",
    "nvidia-cutlass-dsl",
    "nvidia-cutlass-dsl-libs-base",
    "nvidia-cutlass-dsl-libs-cu13",
    "nvidia-cuda-nvcc",
    "nvidia-cuda-runtime",
    "nvidia-cuda-cccl",
    "nvidia-cuda-crt",
    "nvidia-nvvm",
)


def runtime_metadata() -> dict[str, Any]:
    """Record relevant package/toolchain versions without exposing environment secrets."""
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cuda_home = os.environ.get("CUDA_HOME")
    nvcc = str(Path(cuda_home) / "bin/nvcc") if cuda_home else shutil.which("nvcc")
    result = None
    if nvcc and Path(nvcc).is_file():
        process = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, timeout=15, check=False
        )
        result = {
            "returncode": process.returncode,
            "output": process.stdout + process.stderr,
        }
    return {
        "versions": versions,
        "cuda_home": cuda_home,
        "nvcc": nvcc,
        "nvcc_version": result,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def native_vision_attention(config) -> dict[str, Any]:
    """Execute the serving vision wrapper with unequal-length BF16 image sequences."""
    import torch
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

    heads, dimension = config.num_heads, config.hidden_size // config.num_heads
    selected = get_flash_attn_version(head_size=dimension)
    if selected != 4:
        raise RuntimeError(
            f"B200 vision selected FlashAttention {selected}; FA4/CuTe execution is unverified"
        )
    lengths = (128, 192)
    qkv = torch.randn(
        (1, sum(lengths), 3, heads, dimension), device="cuda", dtype=torch.bfloat16
    )
    query, key, value = qkv.unbind(dim=2)
    query, key = query.contiguous(), key.contiguous()
    cumulative = torch.tensor(
        [0, lengths[0], sum(lengths)], device="cuda", dtype=torch.int32
    )
    output = vit_flash_attn_wrapper(
        query,
        key,
        value,
        batch_size=1,
        is_rocm_aiter=False,
        fa_version=selected,
        scale=dimension**-0.5,
        cu_seqlens=cumulative,
        max_seqlen=torch.tensor(max(lengths), dtype=torch.int32),
    )
    references = []
    cursor = 0
    for length in lengths:
        q, k, v = [
            item[0, cursor : cursor + length].transpose(0, 1).float()
            for item in (query, key, value)
        ]
        weights = (q @ k.transpose(-1, -2) * dimension**-0.5).softmax(dim=-1)
        references.append((weights @ v).transpose(0, 1))
        cursor += length
    reference = torch.cat(references).unsqueeze(0)
    difference = (output.float() - reference).abs().max().item()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all() or difference > 0.02:
        raise RuntimeError(
            f"BF16 vision attention failed unchanged 0.02 parity threshold: max_difference={difference}"
        )
    return {
        "backend": "FLASH_ATTN",
        "fa_version": selected,
        "dtype": str(output.dtype),
        "heads": heads,
        "head_dimension": dimension,
        "sequence_lengths": lengths,
        "max_abs_difference": difference,
        "threshold": 0.02,
        "value_stride": list(value.stride()),
    }


def vision_encoder(model: str, revision: str, config) -> dict[str, Any]:
    """Run the complete pinned HF vision architecture with observed FA4 calls."""
    import PIL.Image
    import torch
    import transformers
    from training.verify_fa4 import register_fa4_contiguous_inputs

    register_fa4_contiguous_inputs()
    processor = transformers.AutoProcessor.from_pretrained(
        model,
        revision=revision,
        local_files_only=True,
        trust_remote_code=True,
    )
    images = [
        PIL.Image.new("RGB", size, color)
        for size, color in (((128, 128), "red"), ((160, 128), "blue"))
    ]
    inputs = processor.image_processor(images=images, return_tensors="pt")
    calls = []
    interface = transformers.AttentionInterface()
    original = interface["flash_attention_4"]

    def observe(module, query, key, value, attention_mask, **kwargs):
        """Require the actual encoder to enter noncausal variable-length FA4."""
        if kwargs.get("is_causal", module.is_causal) is not False:
            raise RuntimeError("vision encoder requested causal attention")
        if "cu_seq_lens_q" not in kwargs or "cu_seq_lens_k" not in kwargs:
            raise RuntimeError(
                "vision encoder omitted variable-length sequence boundaries"
            )
        calls.append({"query_shape": list(query.shape), "dtype": str(query.dtype)})
        return original(module, query, key, value, attention_mask, **kwargs)

    transformers.AttentionInterface.register("flash_attention_4", observe)
    try:
        with torch.device("cuda"):
            encoder = (
                transformers.AutoModel.from_config(
                    config,
                    attn_implementation="flash_attention_4",
                    dtype=torch.bfloat16,
                )
                .eval()
                .requires_grad_(False)
            )
        inputs = {key: value.to("cuda") for key, value in inputs.items()}
        with torch.inference_mode():
            output = encoder(
                hidden_states=inputs["pixel_values"].to(torch.bfloat16),
                grid_thw=inputs["image_grid_thw"],
            )
        tensor = output.last_hidden_state
        torch.cuda.synchronize()
        if len(calls) != config.depth or not torch.isfinite(tensor).all():
            raise RuntimeError(
                "vision encoder did not execute finite FA4 output through every block"
            )
        return {
            "encoder_class": type(encoder).__name__,
            "weights": "seeded diagnostic initialization",
            "dtype": str(tensor.dtype),
            "output_shape": list(tensor.shape),
            "grid_thw": inputs["image_grid_thw"].tolist(),
            "fa4_calls": calls,
        }
    finally:
        transformers.AttentionInterface.register("flash_attention_4", original)


def run(model: str, revision: str, report: dict[str, Any]) -> None:
    """Require a B200 allocation and execute both serving and trainer vision paths."""
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError(
            "vision preflight must run inside the existing Slurm GPU allocation"
        )
    if report["runtime"]["versions"]["vllm"] != "0.25.1":
        raise RuntimeError("vision preflight requires the agreed vLLM 0.25.1")
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("vision preflight requires a visible Blackwell GPU")
    torch.manual_seed(1729)
    report["device"] = {
        "name": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "torch_cuda": torch.version.cuda,
    }
    report["stage"] = "cute_import"
    cute = importlib.import_module("cutlass.cute.experimental")
    report["cute_module"] = getattr(cute, "__file__", None)
    config = transformers.AutoConfig.from_pretrained(
        model, revision=revision, local_files_only=True, trust_remote_code=True
    )
    report["stage"] = "native_vision_attention"
    report["native_vision_attention"] = native_vision_attention(config.vision_config)
    report["stage"] = "vision_encoder"
    report["vision_encoder"] = vision_encoder(model, revision, config.vision_config)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    report["stage"] = "complete"


def main() -> int:
    """Write a durable structured success or failure record for the allocation owner."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--revision", default="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    report = {
        "model": args.model,
        "revision": args.revision,
        "stage": "metadata",
        "runtime": runtime_metadata(),
    }
    started = time.monotonic()
    try:
        if not args.metadata_only:
            run(args.model, args.revision, report)
        report["ok"] = True
    except Exception as error:
        report.update(
            ok=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
