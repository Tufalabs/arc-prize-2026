"""Select the drafter's MoE kernel independently of the target's kernel."""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Optional


def moe_backend(drafter: Path, override: Optional[str] = None) -> str:
    """BF16 MTP experts need Triton; the recipe's NVFP4 experts need Marlin.

    Base and exported heads retain the target's quantization config, with
    ``mtp.*`` excluded. Quantizing the head removes that exclusion.
    """
    if override is not None:
        return override
    config = json.loads((drafter / "config.json").read_text())
    quantization = config.get("quantization_config") or config.get("text_config", {}).get("quantization_config")
    if not quantization:
        return "triton"
    expert = "mtp.layers.0.mlp.experts.0.gate_proj"
    if any(fnmatch.fnmatchcase(name, pattern)
           for name in (expert, "model." + expert) for pattern in quantization.get("ignore", [])):
        return "triton"
    if str(quantization.get("quant_algo", "")).upper() == "NVFP4":
        return "marlin"
    raise ValueError(f"{drafter}: unsupported expert quantization; select --moe-backend explicitly")
