"""Assert TRL/PEFT wrapped exactly the intended LoRA modules at training setup (F3).

Today the projection names (`q_proj`, ...) not colliding with the Qwen3.5 vision
tower / MTP head module names (`qkv`, `proj`, `linear_fc1/2`) is luck, not a
contract. This check makes it an invariant: after ``SFTTrainer`` builds the
PeftModel and before ``trainer.train()``, every LoRA-wrapped module must live
under ``.language_model.`` (never ``.visual.`` / ``.mtp.``) and the wrapped
suffix set must equal the configured ``lora_target_modules``. A per-suffix
census is written to ``<output_dir>/lora_wrap_census.json`` -- counts are
recorded, never asserted (the hybrid 9B has q/k/v/o only on its 8
full-attention layers).
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

_CENSUS_FILENAME = "lora_wrap_census.json"
_FORBIDDEN_FRAGMENTS = (".visual.", ".mtp.")


def assert_lora_wrap(model, cfg) -> dict[str, int] | None:
    """Validate the LoRA wrap of ``model`` against ``cfg`` (an ``SFTRunConfig``).

    No-op when ``cfg.use_lora`` is false. On success writes the census file and
    returns the ``{suffix: count}`` mapping; raises ``RuntimeError`` naming the
    offending module path (or suffix sets) otherwise.
    """
    if not cfg.use_lora:
        return None

    # Deferred: peft is part of the heavy training stack (sft-train extra).
    from peft.tuners.lora import LoraLayer

    wrapped = [name for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    if not wrapped:
        raise RuntimeError(
            "use_lora is enabled but the trainer model contains no LoRA-wrapped modules; "
            "the adapter was silently not applied (check peft_config / target_modules)."
        )

    for name in wrapped:
        dotted = f".{name}."
        for fragment in _FORBIDDEN_FRAGMENTS:
            if fragment in dotted:
                raise RuntimeError(
                    f"LoRA wrapped a module outside the language model: {name!r} "
                    f"(matches {fragment.strip('.')!r}); only .language_model. paths may be adapted."
                )
        if ".language_model." not in dotted:
            raise RuntimeError(
                f"LoRA wrapped a module outside the language model: {name!r}; "
                "every adapted path must contain '.language_model.'."
            )

    counts = Counter(name.rsplit(".", 1)[-1] for name in wrapped)
    expected = set(cfg.lora_target_modules)
    actual = set(counts)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise RuntimeError(
            "LoRA-wrapped module suffixes do not match cfg.lora_target_modules: "
            f"missing={missing}, unexpected={unexpected} (wrapped={sorted(actual)})."
        )

    census = {"counts": dict(sorted(counts.items())), "total": sum(counts.values())}
    # Under torchrun every rank validates, but only rank 0 writes the census
    # (concurrent same-file writes over NFS can interleave).
    if int(os.environ.get("RANK", "0")) != 0:
        return dict(counts)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / _CENSUS_FILENAME).write_text(
        json.dumps(census, indent=2, sort_keys=True), encoding="utf-8"
    )
    return dict(counts)
