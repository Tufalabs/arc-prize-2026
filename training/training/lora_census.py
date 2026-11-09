"""LoRA module census: the served module that applies each module of an adapter.

vLLM 0.25.1 checks an adapter's modules by name suffix only when it loads it
(``LoRAModel.from_local_checkpoint``): a suffix the model has no LoRA layer for
fails the load, but a module with a known suffix whose path names no LoRA
layer of the served model (another tower, a layer the model lacks, a skipped
prefix such as MTP) is loaded and then never applied (``activate_adapter``
fills only the model's LoRA layers, by name). ``PEFTHelper`` also drops
``rank_pattern`` and ``alpha_pattern``: every module is served at
``lora_alpha / r``.

The serving gateway (``training/native_vllm_server.py``) takes this census
before the engine loads an adapter and refuses the load when any module would
be ignored or any pattern dropped; the engine's side of it comes from
``LoraCensusWorker``, a worker extension reached through ``collective_rpc``.
The load response carries the census, and the trainer
(``training/inference.py``) requires it to cover exactly the modules its
runtime export wrote.
"""

from __future__ import annotations

import json
from pathlib import Path

WORKER_EXTENSION = "training.lora_census.LoraCensusWorker"
CENSUS_RPC = "lora_served_modules"

_ADAPTER_CONFIG = "adapter_config.json"
_ADAPTER_WEIGHTS = "adapter_model.safetensors"
# The tensor-name suffixes of LoRA factors vLLM parses
# (vllm.lora.utils.parse_fine_tuned_lora_name), by factor.
_FACTORS = {
    ".lora_A.weight": "A",
    ".lora_B.weight": "B",
    ".lora_embedding_A": "A",
    ".lora_embedding_B": "B",
}
_DROPPED_CONFIG_FIELDS = ("rank_pattern", "alpha_pattern")


def read_adapter(path: Path) -> tuple[dict, list[str]]:
    """``adapter_config.json`` and the tensor names of the adapter's weights.

    Only the safetensors header is read. vLLM reads ``adapter_model.bin`` or
    ``.pt`` when the safetensors file is missing; the census does not, so such
    an adapter is refused.
    """
    try:
        config = json.loads((path / _ADAPTER_CONFIG).read_text(encoding="utf-8"))
        with open(path / _ADAPTER_WEIGHTS, "rb") as handle:
            size = int.from_bytes(handle.read(8), "little")
            header = json.loads(handle.read(size))
    except (OSError, ValueError) as error:
        raise ValueError(f"unreadable LoRA adapter at {path}: {error}") from error
    if not isinstance(config, dict) or not isinstance(header, dict):
        raise ValueError(f"malformed LoRA adapter at {path}")
    names = [name for name in header if name != "__metadata__"]
    if not names:
        raise ValueError(f"LoRA adapter at {path} has no tensors")
    return config, names


def refuse_dropped_fields(config: dict) -> None:
    """vLLM serves every module at ``lora_alpha / r``; per-module patterns are lost."""
    dropped = {key: config[key] for key in _DROPPED_CONFIG_FIELDS if config.get(key)}
    if dropped:
        raise ValueError(
            f"vLLM serves every LoRA module at r={config.get('r')!r}, "
            f"lora_alpha={config.get('lora_alpha')!r} and would drop {dropped}"
        )


def adapter_modules(tensor_names: list[str]) -> dict[str, list[str]]:
    """Each LoRA module of an adapter (a tensor name less its factor suffix) and its A and B tensors."""
    modules: dict[str, dict[str, str]] = {}
    for name in tensor_names:
        suffix = next((suffix for suffix in _FACTORS if name.endswith(suffix)), None)
        if suffix is None:
            raise ValueError(f"adapter tensor {name!r} is not a LoRA A or B factor")
        factors = modules.setdefault(name[: -len(suffix)], {})
        if _FACTORS[suffix] in factors:
            raise ValueError(f"adapter repeats the LoRA {_FACTORS[suffix]} factor of {name!r}")
        factors[_FACTORS[suffix]] = name
    incomplete = sorted(module for module, factors in modules.items() if len(factors) != 2)
    if incomplete:
        raise ValueError(
            f"{len(incomplete)} adapter LoRA module(s) lack an A or B factor: {incomplete[:5]}"
        )
    return {module: [factors["A"], factors["B"]] for module, factors in modules.items()}


def lora_census(tensor_names: list[str], served: dict) -> dict:
    """Map every module of an adapter onto the served LoRA layer that applies it.

    ``served`` is the engine's view (``served_modules``). A module is applied
    when the name vLLM files it under is one of the model's LoRA layers, or one
    of the modules a packed layer merges (``q_proj`` into ``qkv_proj``,
    ``in_proj_b`` into ``in_proj_ba``); anything else would be ignored and is
    refused. The census, per adapter module name (``q_proj``): the served
    layer name it is applied under (``applied``) and how many modules of that
    name the adapter has (``modules``).
    """
    modules = adapter_modules(tensor_names)
    mapped = dict(zip(tensor_names, served["mapped"], strict=True))
    layers = set(served["modules"])
    packed_into = {
        child: parent for parent, children in served["packed"].items() for child in children
    }
    applied: dict[str, set[str]] = {}
    counts: dict[str, int] = {}
    ignored = []
    for module, tensors in modules.items():
        names = {mapped[tensor] for tensor in tensors}
        name = names.pop() if len(names) == 1 else None
        layer = name if name in layers else packed_into.get(name)
        if layer is None:
            ignored.append(module)
            continue
        kind = module.rsplit(".", 1)[-1]
        applied.setdefault(kind, set()).add(layer.rsplit(".", 1)[-1])
        counts[kind] = counts.get(kind, 0) + 1
    if ignored:
        raise ValueError(
            f"the served model would not apply {len(ignored)} of the adapter's "
            f"{len(modules)} LoRA modules: {sorted(ignored)[:8]}"
        )
    return {
        "applied": {kind: "|".join(sorted(applied[kind])) for kind in sorted(applied)},
        "modules": {kind: counts[kind] for kind in sorted(counts)},
    }


def served_modules(manager, tensor_names: list[str]) -> dict:
    """The engine's side of the census, as vLLM 0.25.1 loads and applies an adapter.

    ``mapped``: per tensor, the module name vLLM files it under
    (``WorkerLoRAManager._load_adapter``: the model's unstacked
    ``hf_to_vllm_mapper`` and ``parse_fine_tuned_lora_name``), or None when
    ``LoRAModel.from_lora_tensors`` drops it (``lora_skip_prefixes``) or cannot
    parse it. ``modules``: the model's LoRA layers, which ``activate_adapter``
    fills by name. ``packed``: per packed layer, the module names
    ``_create_merged_loras_inplace`` merges into it.
    """
    from vllm.lora.utils import parse_fine_tuned_lora_name

    model = manager.model
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    if mapper is not None:
        # 0.25.1 unstacks (keeps q_proj, in_proj_b); the Flash fork renamed it.
        unstacked = getattr(mapper, "get_unstacked_mapper", None)
        mapper = (unstacked or mapper.get_rename_mapper)()
    skip = list(getattr(model, "lora_skip_prefixes", None) or [])
    mapped = []
    for name in tensor_names:
        if any(f".{prefix}" in name or name.startswith(prefix) for prefix in skip):
            mapped.append(None)
            continue
        try:
            mapped.append(parse_fine_tuned_lora_name(name, mapper)[0])
        except ValueError:
            mapped.append(None)
    return {
        "mapped": mapped,
        "modules": sorted(manager.modules),
        "packed": {
            parent: list(children) for parent, children in manager.packed_modules.items()
        },
    }


class LoraCensusWorker:
    """vLLM worker extension (``worker_extension_cls``) for LoRA engines: the
    gateway calls ``lora_served_modules`` by name through ``collective_rpc``,
    so no pickled callable crosses the process boundary."""

    def lora_served_modules(self, tensor_names: list[str]) -> dict:
        # LoRAModelManager registers itself on the model it wraps.
        return served_modules(self.get_model().lora_manager, list(tensor_names))
