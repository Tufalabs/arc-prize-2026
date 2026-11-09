from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import save_file


_VLLM_SUPPORTED_MAX_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)
_ADAPTER_WEIGHTS_FILENAME = "adapter_model.safetensors"
_RUNTIME_ADAPTER_SUMMARY_FILENAME = "runtime_adapter_summary.json"
_QWEN3_NAME_FRAGMENT = "Qwen3"
_QWEN3_TEXT_MODULE_PREFIX = "base_model.model.model."
_QWEN3_VLLM_LANGUAGE_MODEL_PREFIX = "base_model.model.model.language_model."
_QWEN3_LM_HEAD_PREFIX = "base_model.model.lm_head."
_QWEN3_REWRITABLE_COMPONENTS = ("layers.", "embed_tokens.", "norm.")

# The Qwen3.x language projections vLLM 0.25.1 applies LoRA to; add a name here
# when a new target module becomes serving-legal. Attention q/k/v and MLP
# gate/up are served through the packed qkv_proj / gate_up_proj layers.
# Exporting a module the served model does not apply is refused at load: the
# gateway's LoRA census (training/lora_census.py) maps every adapter module
# onto the layer that applies it.
_ALLOWED_LANGUAGE_PROJECTION_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    # Gated-DeltaNet token-mixer projections (48 of 64 Qwen3.6-27B layers).
    # Qwen3_5ForConditionalGeneration packs in_proj_qkv + in_proj_z into
    # in_proj_qkvz and in_proj_b + in_proj_a into in_proj_ba (one
    # MergedColumnParallelLinear LoRA layer each), so all five are applied.
    | {"in_proj_qkv", "in_proj_z", "out_proj", "in_proj_b", "in_proj_a"}
)

_EXPORT_MODE_PASSTHROUGH = "qwen3_multimodal_passthrough"
_EXPORT_MODE_REWRITE = "qwen3_vllm_namespace_rewrite"


@dataclass(slots=True)
class RuntimeAdapterExport:
    adapter_dir: Path
    source_adapter_dir: Path
    source_rank: int
    runtime_rank: int
    source_lora_alpha: float
    runtime_lora_alpha: float
    runtime_target_modules: list[str]
    converted_tensor_count: int
    export_mode: str = ""
    adapter_sha256: str = ""
    scaling_mode: str = "standard"
    base_model_name: str = ""
    delta_stats: dict[str, dict[str, float]] = field(default_factory=dict)


def _normalize_target_modules(modules: Iterable[Any]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for module in modules:
        name = str(module).strip()
        if not name or name in seen:
            continue
        normalized.append(name)
        seen.add(name)
    return normalized


def runtime_lora_target_modules(source_target_modules: list[str]) -> list[str]:
    normalized = _normalize_target_modules(source_target_modules)
    if not normalized:
        raise ValueError("Runtime LoRA export requires at least one target module.")
    return normalized


def runtime_lora_rank(*, source_rank: int, source_target_modules: list[str]) -> int:
    runtime_lora_target_modules(source_target_modules)
    rank = int(source_rank)
    if rank <= 0:
        raise ValueError(
            f"Runtime LoRA export requires a positive rank, got {source_rank}."
        )
    return rank


def runtime_lora_max_rank(*, source_rank: int, source_target_modules: list[str]) -> int:
    required_rank = runtime_lora_rank(
        source_rank=source_rank,
        source_target_modules=source_target_modules,
    )
    for supported_rank in _VLLM_SUPPORTED_MAX_LORA_RANKS:
        if supported_rank >= required_rank:
            return supported_rank
    raise ValueError(
        f"Runtime LoRA rank {required_rank} exceeds the supported vLLM max-lora-rank values "
        f"{list(_VLLM_SUPPORTED_MAX_LORA_RANKS)}."
    )


def read_runtime_export(adapter_dir: Path) -> tuple[list[str], int]:
    """The target modules and tensor count a runtime export wrote (its summary).

    The trainer checks the serving gateway's LoRA census against these: the
    engine must apply every exported module.
    """
    summary_path = Path(adapter_dir) / _RUNTIME_ADAPTER_SUMMARY_FILENAME
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        modules, tensors = summary["runtime_target_modules"], summary["converted_tensor_count"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"unreadable runtime LoRA export summary {summary_path}: {error!r}") from error
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(module, str) and module for module in modules)
        or type(tensors) is not int
        or tensors <= 0
    ):
        raise RuntimeError(f"invalid runtime LoRA export summary {summary_path}")
    return modules, tensors


def _load_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    return json.loads(config_path.read_text(encoding="utf-8"))


def _count_safetensor_tensors(adapter_dir: Path) -> int:
    weights_path = adapter_dir / _ADAPTER_WEIGHTS_FILENAME
    if not weights_path.exists():
        return 0
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        return len(list(handle.keys()))


def runtime_lora_has_effective_delta(adapter_dir: Path) -> bool:
    weights_path = adapter_dir / _ADAPTER_WEIGHTS_FILENAME
    if not weights_path.exists():
        return False

    saw_lora_b = False
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if key.endswith(".lora_B.weight"):
                saw_lora_b = True
                if bool(tensor.ne(0).any().item()):
                    return True

        if saw_lora_b:
            return False

        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if bool(tensor.ne(0).any().item()):
                return True
    return False


def _source_weights_path(adapter_dir: Path) -> Path:
    weights_path = adapter_dir / _ADAPTER_WEIGHTS_FILENAME
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Runtime LoRA export requires {_ADAPTER_WEIGHTS_FILENAME} in {adapter_dir}."
        )
    return weights_path


def _copy_adapter_config(*, source_adapter_dir: Path, output_dir: Path) -> None:
    shutil.copy2(
        source_adapter_dir / "adapter_config.json", output_dir / "adapter_config.json"
    )


def _is_qwen3_adapter(source_config: dict[str, Any]) -> bool:
    model_name = str(source_config.get("base_model_name_or_path", "")).strip()
    return _QWEN3_NAME_FRAGMENT in model_name


def _rewrite_qwen3_tensor_name(name: str) -> str:
    if name.startswith("base_model.model.lm_head."):
        return name
    if not name.startswith(_QWEN3_TEXT_MODULE_PREFIX):
        raise ValueError(
            "Qwen 3.x runtime LoRA export encountered an unsupported tensor name "
            f"{name!r}; expected it to start with {_QWEN3_TEXT_MODULE_PREFIX!r}."
        )

    suffix = name.removeprefix(_QWEN3_TEXT_MODULE_PREFIX)
    if suffix.startswith("language_model."):
        # Multimodal trainer (AutoModelForImageTextToText) already nests the
        # decoder as model.language_model.*, i.e. the tensor is ALREADY in the
        # vLLM runtime layout this rewrite targets -- pass it through.
        tail = suffix.removeprefix("language_model.")
        if not tail.startswith(_QWEN3_REWRITABLE_COMPONENTS):
            raise ValueError(
                "Qwen 3.x runtime LoRA export only supports LoRA tensors under "
                f"{_QWEN3_REWRITABLE_COMPONENTS!r}; got {name!r}."
            )
        return name
    if not suffix.startswith(_QWEN3_REWRITABLE_COMPONENTS):
        raise ValueError(
            "Qwen 3.x runtime LoRA export only supports LoRA tensors under "
            f"{_QWEN3_REWRITABLE_COMPONENTS!r}; got {name!r}."
        )
    return _QWEN3_VLLM_LANGUAGE_MODEL_PREFIX + suffix


def _load_safetensor_tensors(
    weights_path: Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    tensors: dict[str, Any] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as handle:
        # Preserve the source header metadata (PEFT writes {"format": "pt"}) so a
        # passthrough export is byte-reproducible against the raw checkpoint's
        # adapter, not just tensor-identical.
        metadata = dict(handle.metadata() or {})
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key)
    if "format" not in metadata:
        metadata["format"] = "pt"
    return tensors, metadata


def _convert_qwen3_tensors(
    source_tensors: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Route source LoRA tensors to a runtime layout by inspecting their key namespace.

    Two mutually-exclusive namespaces are supported (additive to the original
    text-class rewrite that the RL pipeline relies on):

    - **passthrough** -- every non-``lm_head`` key already lives under
      ``base_model.model.model.language_model.`` (the multimodal SFT namespace
      that PEFT emits for ``Qwen3_5ForConditionalGeneration``). Tensors are copied
      unchanged; each language-model suffix must start with ``layers.`` /
      ``embed_tokens.`` / ``norm.``. ``base_model.model.lm_head.`` keys pass
      through in either mode.
    - **rewrite** -- text-class keys under ``base_model.model.model.`` (never
      ``language_model``); the original RL contract, delegated to
      ``_rewrite_qwen3_tensor_name`` (which also rejects ``visual.*`` / ``mtp.*``
      / unknown suffixes by name).

    Routing is per key, so an adapter may mix both layouts (main's linear_attn
    coverage test exercises this); every key ends up in the vLLM
    ``language_model`` namespace either way. An all-passthrough adapter is
    returned unchanged so the export stays byte-reproducible against the raw
    checkpoint.
    """
    runtime_tensors: dict[str, Any] = {}
    rewrote = False
    for name, tensor in source_tensors.items():
        if name.startswith(_QWEN3_LM_HEAD_PREFIX):
            runtime_tensors[name] = tensor
            continue
        if name.startswith(_QWEN3_VLLM_LANGUAGE_MODEL_PREFIX):
            suffix = name.removeprefix(_QWEN3_VLLM_LANGUAGE_MODEL_PREFIX)
            if not suffix.startswith(_QWEN3_REWRITABLE_COMPONENTS):
                raise ValueError(
                    "Qwen 3.5 multimodal passthrough export only supports language-model "
                    f"LoRA tensors under {_QWEN3_REWRITABLE_COMPONENTS!r}; offending key {name!r}."
                )
            runtime_key = name
        else:
            runtime_key = _rewrite_qwen3_tensor_name(name)
            rewrote = True
        if runtime_key in runtime_tensors:
            raise ValueError(
                "Qwen 3.x runtime LoRA export produced duplicate runtime key "
                f"{runtime_key!r} (source key {name!r} collides after rewrite)."
            )
        runtime_tensors[runtime_key] = tensor
    if not rewrote:
        # Only passthrough tensors -- return the source mapping untouched.
        return dict(source_tensors), _EXPORT_MODE_PASSTHROUGH
    return runtime_tensors, _EXPORT_MODE_REWRITE


def _module_type_of(lora_key: str) -> str:
    """The projection name (e.g. ``q_proj``) a LoRA A/B tensor key belongs to."""
    base = lora_key
    for marker in (".lora_A.weight", ".lora_B.weight"):
        if base.endswith(marker):
            base = base[: -len(marker)]
            break
    return base.rsplit(".", 1)[-1]


def _validate_lora_tensor_structure(
    tensors: dict[str, Any],
    *,
    source_rank: int,
    source_target_modules: list[str],
) -> None:
    """Reject incomplete or config-inconsistent LoRA tensor payloads.

    A runtime adapter cannot apply a module unless both factors exist and use
    the rank declared in ``adapter_config.json``. Previously an interrupted or
    mismatched checkpoint could be rewritten and published successfully, only
    to fail (or omit a delta) when PPO later asked vLLM to load it.
    """
    pairs: dict[str, dict[str, str]] = {}
    for key in tensors:
        for factor in ("lora_A", "lora_B"):
            marker = f".{factor}.weight"
            if key.endswith(marker):
                pairs.setdefault(key[: -len(marker)], {})[factor] = key
                break
    if not pairs:
        raise ValueError("Runtime LoRA export found no LoRA A/B tensors.")

    configured_modules = set(source_target_modules)
    tensor_modules: set[str] = set()
    for base, pair in pairs.items():
        missing = [factor for factor in ("lora_A", "lora_B") if factor not in pair]
        if missing:
            raise ValueError(
                "Runtime LoRA export found an incomplete tensor pair for "
                f"{base!r}; missing {', '.join(missing)}."
            )
        module_type = _module_type_of(pair["lora_A"])
        tensor_modules.add(module_type)
        if module_type not in configured_modules:
            raise ValueError(
                "Runtime LoRA export tensor module is absent from adapter_config.json "
                f"target_modules: {module_type!r} at {base!r}."
            )
        a = tensors[pair["lora_A"]]
        b = tensors[pair["lora_B"]]
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(
                "Runtime LoRA export requires rank-2 A/B tensors for "
                f"{base!r}; got shapes {tuple(a.shape)} and {tuple(b.shape)}."
            )
        if a.shape[0] != source_rank or b.shape[1] != source_rank:
            raise ValueError(
                "Runtime LoRA export tensor rank does not match adapter_config.json "
                f"for {base!r}: configured r={source_rank}, "
                f"A shape={tuple(a.shape)}, B shape={tuple(b.shape)}."
            )

    missing_modules = sorted(configured_modules - tensor_modules)
    if missing_modules:
        raise ValueError(
            "Runtime LoRA export adapter_config.json target_modules have no "
            f"matching tensors: {missing_modules}."
        )


def _runtime_lora_scaling(*, rank: int, alpha: float, use_rslora: bool) -> float:
    if rank <= 0:
        return 0.0
    return (alpha / math.sqrt(rank)) if use_rslora else (alpha / rank)


def compute_runtime_lora_delta_stats(
    tensors: dict[str, Any], *, rank: int, alpha: float, use_rslora: bool
) -> dict[str, dict[str, float]]:
    """Per-module-type stats of the merged delta ``(alpha/r)*B@A`` (float32, CPU).

    Documents quantization fragility (this saga's trained deltas were a few bf16
    ulps) and fingerprints the checkpoint. Aggregated per projection *type* -- never
    per layer -- so the hybrid 9B (attention only on the 8 full-attention layers)
    validates by presence, not by per-layer count.
    """
    scaling = _runtime_lora_scaling(rank=rank, alpha=alpha, use_rslora=use_rslora)
    bases: dict[str, dict[str, str]] = {}
    for key in tensors:
        for tag in ("lora_A", "lora_B"):
            marker = f".{tag}.weight"
            if key.endswith(marker):
                bases.setdefault(key[: -len(marker)], {})[tag] = key

    agg: dict[str, dict[str, float]] = {}
    for base, pair in bases.items():
        if "lora_A" not in pair or "lora_B" not in pair:
            continue
        a = tensors[pair["lora_A"]].to(torch.float32)
        b = tensors[pair["lora_B"]].to(torch.float32)
        delta = (b @ a) * scaling
        abs_delta = delta.abs()
        module_type = base.rsplit(".", 1)[-1]
        stat = agg.setdefault(
            module_type,
            {
                "count": 0,
                "max_abs": 0.0,
                "sum_abs": 0.0,
                "numel": 0,
                "b_nonzero": 0,
                "b_numel": 0,
            },
        )
        stat["count"] += 1
        stat["max_abs"] = max(stat["max_abs"], float(abs_delta.max()))
        stat["sum_abs"] += float(abs_delta.sum())
        stat["numel"] += int(abs_delta.numel())
        stat["b_nonzero"] += int(b.ne(0).sum())
        stat["b_numel"] += int(b.numel())

    out: dict[str, dict[str, float]] = {}
    for module_type, stat in agg.items():
        out[module_type] = {
            "count": int(stat["count"]),
            "max_abs_delta": stat["max_abs"],
            "mean_abs_delta": (stat["sum_abs"] / stat["numel"])
            if stat["numel"]
            else 0.0,
            "frac_B_nonzero": (stat["b_nonzero"] / stat["b_numel"])
            if stat["b_numel"]
            else 0.0,
        }
    return out


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_peft_version(source_config: dict[str, Any]) -> str | None:
    recorded = source_config.get("peft_version")
    if recorded:
        return str(recorded)
    try:
        import peft

        return getattr(peft, "__version__", None)
    except Exception:  # pragma: no cover - peft always present in the training env
        return None


def _validate_vllm_load_invariants(
    *, source_config: dict[str, Any], source_target_modules: list[str], source_rank: int
) -> None:
    """Fail fast on adapter configs vLLM's ``PEFTHelper`` would reject at load.

    Turns "GPU job dies an hour after submit" into "export errors in seconds".
    """
    bias = source_config.get("bias", "none")
    if bias != "none":
        raise ValueError(
            f"Runtime LoRA export requires bias='none' for vLLM loading; got bias={bias!r}."
        )
    modules_to_save = source_config.get("modules_to_save")
    if modules_to_save:
        raise ValueError(
            "Runtime LoRA export cannot serve adapters with modules_to_save "
            f"(vLLM rejects it); got modules_to_save={modules_to_save!r}."
        )
    if source_config.get("use_dora"):
        raise ValueError(
            "Runtime LoRA export cannot serve DoRA adapters (use_dora=true); vLLM rejects it."
        )
    unknown = [
        m
        for m in source_target_modules
        if m not in _ALLOWED_LANGUAGE_PROJECTION_MODULES
    ]
    if unknown:
        raise ValueError(
            f"Runtime LoRA export target_modules {unknown} are not in the vLLM-servable "
            f"language-projection set {sorted(_ALLOWED_LANGUAGE_PROJECTION_MODULES)}."
        )
    # Raises naming the supported ladder when the rank overshoots the top rung.
    runtime_lora_max_rank(
        source_rank=source_rank, source_target_modules=source_target_modules
    )


def export_runtime_lora_adapter(
    *,
    source_adapter_dir: Path,
    output_dir: Path,
    allow_zero_delta: bool = True,
    validate_load_invariants: bool = False,
    compute_delta_stats: bool = False,
    compute_adapter_sha256: bool = False,
) -> RuntimeAdapterExport:
    """Write a serving-ready two-file runtime adapter dir from a raw checkpoint.

    Defaults preserve the RL contract (``allow_zero_delta=True``,
    ``validate_load_invariants=False``, ``compute_delta_stats=False``,
    ``compute_adapter_sha256=False``): the RL pipeline exports no-op init
    adapters and its own configs, so the new fail-fast checks and full-file
    checksum pass are opt-in. The SFT CLI (``sft.export_runtime``) enables all
    four.
    """
    source_config = _load_adapter_config(source_adapter_dir)
    source_target_modules = runtime_lora_target_modules(
        [str(item) for item in source_config.get("target_modules", [])]
    )
    source_rank = int(source_config.get("r", 0))
    source_lora_alpha = float(source_config.get("lora_alpha", 0.0))
    use_rslora = bool(source_config.get("use_rslora", False))
    runtime_rank = runtime_lora_rank(
        source_rank=source_rank,
        source_target_modules=source_target_modules,
    )
    runtime_lora_alpha = source_lora_alpha

    if validate_load_invariants:
        _validate_vllm_load_invariants(
            source_config=source_config,
            source_target_modules=source_target_modules,
            source_rank=source_rank,
        )

    if not allow_zero_delta and not runtime_lora_has_effective_delta(
        source_adapter_dir
    ):
        raise ValueError(
            f"Runtime LoRA export of {source_adapter_dir} has all-zero lora_B (no effective "
            "delta); pass allow_zero_delta=True (CLI: --allow-zero-delta) to export a no-op adapter."
        )

    if not _is_qwen3_adapter(source_config):
        raise ValueError(
            "Runtime LoRA export only supports Qwen 3.x adapters in this repo. "
            f"Got base_model_name_or_path={source_config.get('base_model_name_or_path')!r}."
        )

    source_tensors, source_metadata = _load_safetensor_tensors(
        _source_weights_path(source_adapter_dir)
    )
    _validate_lora_tensor_structure(
        source_tensors,
        source_rank=source_rank,
        source_target_modules=source_target_modules,
    )
    runtime_tensors, export_mode = _convert_qwen3_tensors(source_tensors)

    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_adapter_config(source_adapter_dir=source_adapter_dir, output_dir=output_dir)
    save_file(
        runtime_tensors,
        output_dir / _ADAPTER_WEIGHTS_FILENAME,
        metadata=source_metadata,
    )

    converted_tensor_count = _count_safetensor_tensors(output_dir)
    adapter_sha256 = (
        _file_sha256(output_dir / _ADAPTER_WEIGHTS_FILENAME)
        if compute_adapter_sha256
        else ""
    )
    scaling_mode = "rslora" if use_rslora else "standard"
    base_model_name = str(source_config.get("base_model_name_or_path", ""))
    delta_stats = (
        compute_runtime_lora_delta_stats(
            source_tensors,
            rank=source_rank,
            alpha=source_lora_alpha,
            use_rslora=use_rslora,
        )
        if compute_delta_stats
        else {}
    )

    (output_dir / _RUNTIME_ADAPTER_SUMMARY_FILENAME).write_text(
        json.dumps(
            {
                "source_adapter_dir": str(source_adapter_dir),
                "source_checkpoint": str(source_adapter_dir),
                "source_target_modules": source_target_modules,
                "runtime_target_modules": source_target_modules,
                "source_rank": source_rank,
                "runtime_rank": runtime_rank,
                "source_lora_alpha": source_lora_alpha,
                "runtime_lora_alpha": runtime_lora_alpha,
                "converted_tensor_count": converted_tensor_count,
                "export_mode": export_mode,
                "adapter_sha256": adapter_sha256,
                "peft_version": _resolve_peft_version(source_config),
                "base_model_name": base_model_name,
                "scaling_mode": scaling_mode,
                "delta_stats": delta_stats,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return RuntimeAdapterExport(
        adapter_dir=output_dir,
        source_adapter_dir=source_adapter_dir,
        source_rank=source_rank,
        runtime_rank=runtime_rank,
        source_lora_alpha=source_lora_alpha,
        runtime_lora_alpha=runtime_lora_alpha,
        runtime_target_modules=source_target_modules,
        converted_tensor_count=converted_tensor_count,
        export_mode=export_mode,
        adapter_sha256=adapter_sha256,
        scaling_mode=scaling_mode,
        base_model_name=base_model_name,
        delta_stats=delta_stats,
    )
