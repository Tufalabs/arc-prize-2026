"""Validate the cached-target configuration before starting any GPU process."""

import json
import logging
import math
from pathlib import Path

log = logging.getLogger(__name__)

# Placement settings removed with NVFP4 expert storage, which configs, attempt
# snapshots and checkpoint manifests written before the removal still carry,
# with the values each accepted. Experts are always BF16 and FSDP-sharded:
# NVFP4 storage computed bitwise the same, and the per-GPU replica
# ("replicated") required it. Any other value is refused.
RETIRED_PLACEMENT = {
    "expert_storage": ("bf16", "nvfp4"),
    "nvfp4_distribution": ("replicated", "fsdp"),
    "model_distribution": ("fsdp",),
}
_warned_retired = set()


def drop_retired_placement(payload, source):
    """``payload`` without its retired placement settings, each warned about
    once per process; a value the setting never accepted is refused."""
    for name, accepted in RETIRED_PLACEMENT.items():
        if name not in payload:
            continue
        if payload[name] not in accepted:
            raise ValueError(
                f"{source}: {name}={payload[name]!r} is not supported; the retired "
                f"setting accepts only {accepted} (experts are always BF16 and FSDP-sharded)"
            )
        if name not in _warned_retired:
            _warned_retired.add(name)
            log.warning("ignoring the retired setting %s in %s: it has no effect", name, source)
    return {key: value for key, value in payload.items() if key not in RETIRED_PLACEMENT}


def load_config(path):
    config = drop_retired_placement(json.loads(Path(path).read_text()), path)
    for key in ("teacher", "student", "data", "data_sha256", "cache", "probe", "wandb"):
        if not config.get(key):
            raise ValueError(f"Missing required configuration: {key}")
    for key in ("steps", "save_every", "chunk"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    for key in ("lr", "temperature", "ce_weight"):
        value = config.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
        if value < 0 or (key != "ce_weight" and value == 0):
            raise ValueError(f"Invalid {key}")
    config.setdefault("loss_weighting", "example")
    if config["loss_weighting"] not in ("token", "example"):
        raise ValueError("loss_weighting must be token or example")
    scheduler = config.setdefault("scheduler", {})
    if not isinstance(scheduler, dict):
        raise TypeError("scheduler must be an object")
    unknown_scheduler = set(scheduler) - {"account", "partition", "qos"}
    if unknown_scheduler or any(
        value is not None and (not isinstance(value, str) or not value)
        for value in scheduler.values()
    ):
        raise ValueError("invalid scheduler configuration")
    scheduler.setdefault("account", "research")
    scheduler.setdefault("partition", "dev")
    scheduler.setdefault("qos", "b200-pool")
    from .batching import normalize_batching

    batching = normalize_batching(config.get("batching"))
    if "sequence_packing" in config:
        raise ValueError("sequence_packing is no longer supported")
    accumulation = config.get("gradient_accumulation", 1)
    if type(accumulation) is not int or accumulation < 1:
        raise ValueError("gradient_accumulation must be a positive integer")
    if batching["policy"] == "length_within_update" and accumulation < 2:
        raise ValueError("length_within_update requires gradient_accumulation >= 2")
    total_data = config.get("data_microbatches")
    if total_data is not None and (type(total_data) is not int or total_data < 1):
        raise ValueError("data_microbatches must be positive")
    if type(config.get("allow_accumulation_change", False)) is not bool:
        raise ValueError("allow_accumulation_change must be boolean")
    # The execution stack: kernel_mode's preset, then any field the config
    # sets (training/flash/kernel_mode.py). The resolved fields are written
    # back, so the attempt snapshot and the worker see one complete stack.
    config.update(kernel_stack(config))
    from training.flash.expert_lora import normalize_lora

    config["expert_lora"] = normalize_lora(config.get("expert_lora"))
    config.setdefault("train_routers", True)
    if type(config["train_routers"]) is not bool:
        raise ValueError("train_routers must be boolean")
    if not config["train_routers"] and config["expert_lora"] is None:
        raise ValueError("training requires routers or expert LoRA")
    if config["expert_lora"] and config["expert_optimization"] == "none":
        raise ValueError("expert LoRA requires an optimized expert implementation")
    config.setdefault("world_size", 4)
    if type(config["world_size"]) is not int or config["world_size"] < 1:
        raise ValueError("world_size must be a positive integer")
    if config["world_size"] not in (2, 4):
        raise ValueError("The FSDP layout requires two or four ranks")
    from training.flash.offload import resolve_qsa_retention

    if (
        "retain_qsa_attention" in config
        and type(config["retain_qsa_attention"]) is not bool
    ):
        raise ValueError("retain_qsa_attention must be boolean")
    retain = resolve_qsa_retention(
        config.get("attention_backend", "sdpa"),
        config.get("reuse_qsa_selection", False),
        config.get("retain_qsa_attention"),
    )
    config["retain_qsa_attention"] = retain
    if type(retain) is not bool:
        raise ValueError("retain_qsa_attention must be boolean")
    if retain and (
        config.get("attention_backend", "sdpa") != "flex_qsa"
        or not config.get("reuse_qsa_selection", False)
    ):
        raise ValueError("retain_qsa_attention requires flex_qsa and selection reuse")
    reuse = config.get("reuse_qsa_selection", False)
    if type(reuse) is not bool:
        raise ValueError("reuse_qsa_selection must be boolean")
    if reuse and config.get("attention_backend", "sdpa") != "flex_qsa":
        raise ValueError("reuse_qsa_selection requires flex_qsa")
    every = config.get("offload_every", 1)
    if type(every) is not int or every < 0:
        raise ValueError("offload_every must be a nonnegative integer")
    if config.get("teacher_mode", "cached") != "cached":
        raise ValueError("Only cached teacher targets are supported")
    if config["steps"] > config.get("production_total_steps", config["steps"]):
        raise ValueError("Segment exceeds total update budget")
    if config["wandb"].get("mode", "online") == "online" and not config["wandb"].get(
        "entity"
    ):
        raise ValueError("Online W&B requires an explicit entity")
    if config.get("resume") and not Path(config["resume"]).is_file():
        raise ValueError("Resume checkpoint does not exist")
    if config.get("resume_sha256"):
        from .common import sha

        if (
            not config.get("resume")
            or sha(Path(config["resume"])) != config["resume_sha256"]
        ):
            raise ValueError("resume checkpoint hash changed")
    return config


def kernel_stack(config):
    """The resolved execution stack of a raw or loaded configuration.

    Distillation feeds one unpacked sequence per forward, so the reference
    preset keeps FLA's autotuner unguarded, as the existing results ran.
    """
    from training.flash.kernel_mode import KERNEL_FIELDS, resolve_kernel_stack

    mode = config.get("kernel_mode", "reference")
    try:
        stack = resolve_kernel_stack(
            mode, {name: config.get(name) for name in KERNEL_FIELDS}
        )
    except ValueError as error:
        raise ValueError(f"kernel stack: {error}") from error
    return {"kernel_mode": mode} | stack


# Kernel stack fields with their own worker flag; --kernel-options carries the rest.
WORKER_KERNEL_FLAGS = (
    "attention_backend",
    "qsa_kernel",
    "expert_optimization",
    "fuse_norm_tail",
)


def worker_kernel_stack(args):
    """The worker's stack: --kernel-mode's preset, --kernel-options, the flags."""
    from training.flash.kernel_mode import resolve_kernel_stack

    options = json.loads(args.kernel_options)
    if not isinstance(options, dict):
        raise ValueError("--kernel-options must be a JSON object")
    flags = {name: getattr(args, name) for name in WORKER_KERNEL_FLAGS}
    return resolve_kernel_stack(
        args.kernel_mode,
        options | {name: value for name, value in flags.items() if value is not None},
    )


def validate_inputs(config):
    from .common import Dataset, sha, validate_cache, validate_roles
    from .pruning_probe import validate_probe

    roles = validate_roles(config)
    if sha(config["data"]) != config["data_sha256"]:
        raise ValueError("Training dataset changed")
    # Never silently create targets or execute a teacher forward from this path.
    for cache in (config["cache"], config["probe"]["cache"]):
        if not (Path(cache) / "manifest.json").is_file():
            raise ValueError(f"Completed teacher cache required: {cache}")
    train = validate_cache(
        Path(config["cache"]), config["data"], Path(config["student"])
    )
    probe = validate_probe(config)
    validate_cache(Path(probe["cache"]), probe["data"], Path(config["student"]))
    rows = len(Dataset(config["data"]))
    from .batching import normalize_batching

    masked = normalize_batching(config.get("batching")).get("tail_policy") == "masked"
    world = config.get("world_size", 4)
    if rows < world or (rows % world and not masked):
        raise ValueError(
            "Training rows must be a nonzero multiple of world_size; "
            "no silent dropped examples"
        )
    return {
        "rows": rows,
        "student_experts": roles["experts"],
        "cache_rows": train["rows"],
    }
