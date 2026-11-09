"""SFT run configuration (JSON -> dataclass), mirroring the repo's config style."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sft.selection import DatasetSpecLike


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def _reject_non_finite_numbers(value: Any, *, path: str = "config") -> None:
    """Reject overflowed floats such as ``1e999`` throughout a config."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite JSON number at {path} is not allowed")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, path=f"{path}[{index}]")


@dataclass(slots=True)
class DatasetSpec:
    """One training/eval source: a run dir, filtered by game, tag, and score.

    `path` is a benchmark **run dir** (holds `benchmark.json` + `traces/`) when any
    filter is used; a bare TAAF trace root/folder is accepted only unfiltered (see
    sft/selection.py). `filter` is a generic predicate over any `game_run` field:
    `{"final_score": {">": 0}}` keeps non-zero-score rollouts; a bare scalar means
    equality (`{"state": "won"}`).
    """

    path: str
    games: list[str] = field(default_factory=list)
    exclude_games: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    exclude_tags: list[str] = field(default_factory=list)
    filter: dict[str, Any] = field(default_factory=dict)
    label: str | None = None
    # Named re-arc-3 dataset(s) ("train", "train,eval", or a list) whose
    # game_ids act as an additional include filter, resolved from the installed
    # re_arc package (or `re_arc_datasets_dir` when set) -- same knob as
    # ARC3-Inference's `dataset`.
    dataset: str | list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ValueError("dataset spec `path` must be a non-empty string")
        for name in ("games", "exclude_games", "tags", "exclude_tags"):
            values = getattr(self, name)
            if not isinstance(values, list) or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError(f"dataset spec `{name}` must be a list of non-empty strings")
        if not isinstance(self.filter, dict):
            raise ValueError("dataset spec `filter` must be an object")
        if self.label is not None and (
            not isinstance(self.label, str) or not self.label.strip()
        ):
            raise ValueError("dataset spec `label` must be a non-empty string or null")
        if self.dataset is not None:
            values = [self.dataset] if isinstance(self.dataset, str) else self.dataset
            if not isinstance(values, list) or not values or any(
                not isinstance(value, str) or not value.strip() for value in values
            ):
                raise ValueError("dataset spec `dataset` must name at least one dataset")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetSpec":
        if isinstance(payload, str):  # shorthand: a bare path string
            return cls(path=payload)
        if not isinstance(payload, dict):
            raise ValueError(f"dataset spec must be a path string or object, got {type(payload).__name__}")
        unknown = set(payload) - set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        if unknown:
            raise ValueError(f"Unknown dataset spec keys: {sorted(unknown)}")
        return cls(**payload)

    def to_selection_spec(self) -> DatasetSpecLike:
        return DatasetSpecLike(
            path=self.path,
            games=list(self.games),
            exclude_games=list(self.exclude_games),
            tags=list(self.tags),
            exclude_tags=list(self.exclude_tags),
            filter=dict(self.filter),
            label=self.label,
            dataset=self.dataset,
        )


@dataclass(slots=True)
class SFTRunConfig:
    # Data sources. Preferred: `datasets` (a list of filtered run dirs). Back-compat:
    # a bare `trace_root` (one unfiltered TAAF trace root). `eval_datasets` (optional)
    # is the held-out validation set, same schema. Tags are resolved via re-arc-3
    # `environment_files` at `re_arc_env_dir` (only needed when a tag filter is used).
    trace_root: str = "traces"
    datasets: list[DatasetSpec] = field(default_factory=list)
    eval_datasets: list[DatasetSpec] = field(default_factory=list)
    re_arc_env_dir: str | None = None
    # Optional override dir of re-arc-3 named dataset files
    # (re_arc/datasets/*.json) for DatasetSpec.dataset resolution. When unset,
    # names resolve from the installed re_arc package (`sft-rearc` extra),
    # so the game lists track re-arc-3 instead of a checked-in copy.
    re_arc_datasets_dir: str | None = None
    # Packed-dataset cache (durable dir, e.g. /data/.../dataset_cache). When
    # set, built datasets are saved there once (datasets.save_to_disk, keyed
    # by the selected trace dirs + mode) and every later build -- the train
    # pass of the same job and future jobs on the same selection -- loads the
    # pack instead of re-reading raw traces, which eagerly opens every
    # referenced media file (~1.4k tiny PNGs per trace; hostile to NFS).
    # See sft/dataset_cache.py. None -> build from raw traces every time.
    dataset_cache_dir: str | None = None
    # Cap the held-out validation set to this many rows (deterministic sample
    # applied AFTER the packed dataset loads, so it does not change the cache
    # key). Full-set eval at batch size 1 is hours per pass; a small fixed
    # sample keeps the eval-loss curve cheap. None -> use every selected row.
    eval_max_rows: int | None = None
    # Training mode: "auto" picks token_exact when every step has token ids and
    # no images, else retemplate. See sft/taaf_adapter.py. Input-token fidelity
    # is verified at capture time by the trace writer (ARC3-Inference,
    # input_token_reconstruction_status), not re-checked here.
    training_mode: str = "auto"

    # Model / processor (HF id or local path). Default is a small VLM so the
    # pipeline can smoke-run locally; swap for the real target checkpoint.
    model_name: str = "Qwen/Qwen3.5-9B"
    output_dir: str = "runs/sft-v0"
    # Run label. Sets the HF Trainer run_name (TensorBoard/W&B) and, via
    # slurm/submit_sft.sh, the Slurm job name + log filename. None -> defaults.
    run_name: str | None = None

    # GPUs for the training job (consumed by slurm/submit_sft_chain.sh, which
    # passes --gpus=b200:N; >1 launches DDP via torchrun in sft_train.sbatch).
    # The training code itself never reads this.
    gpus: int = 1

    # Optimization
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_length: int = 4096
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0

    # Validation: how often to run eval (in optimizer steps) when eval_datasets is
    # set. Ignored with no eval data.
    eval_steps: int = 50

    # Wall-clock checkpoint cadence (minutes). When set, checkpoint saves fire
    # every this-many minutes of training time instead of every eval_steps
    # optimizer steps (85 step-based checkpoints in a ~5 h run is NFS spam
    # nobody evaluates; the eval sidecar samples one checkpoint per 45-minute
    # rollout budget anyway). Eval-loss cadence is unaffected. None -> save on
    # the step-based schedule tied to eval_steps.
    checkpoint_interval_minutes: float | None = None

    # Toggles
    gradient_checkpointing: bool = True
    bf16: bool = True

    # LoRA (adapter-only). Set use_lora=false for full fine-tuning.
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )

    logging_steps: int = 1
    # TensorBoard event dir; None -> TrainingArguments default
    # (output_dir/runs). Point at a shared /data path for live viewing.
    logging_dir: str | None = None

    def __post_init__(self) -> None:
        if self.training_mode not in {"auto", "retemplate", "token_exact"}:
            raise ValueError(
                "training_mode must be one of ['auto', 'retemplate', 'token_exact'], "
                f"got {self.training_mode!r}"
            )
        for name in (
            "gpus",
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "gradient_accumulation_steps",
            "max_length",
            "eval_steps",
            "logging_steps",
            "lora_r",
            "lora_alpha",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        for name in ("learning_rate", "num_train_epochs"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay!r}")
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout!r}")
        if self.eval_max_rows is not None and (
            isinstance(self.eval_max_rows, bool)
            or not isinstance(self.eval_max_rows, int)
            or self.eval_max_rows <= 0
        ):
            raise ValueError(f"eval_max_rows must be a positive integer, got {self.eval_max_rows!r}")
        if self.checkpoint_interval_minutes is not None and (
            not isinstance(self.checkpoint_interval_minutes, (int, float))
            or isinstance(self.checkpoint_interval_minutes, bool)
            or self.checkpoint_interval_minutes <= 0
        ):
            raise ValueError(
                "checkpoint_interval_minutes must be positive, "
                f"got {self.checkpoint_interval_minutes!r}"
            )
        if not isinstance(self.lora_target_modules, list) or not self.lora_target_modules:
            raise ValueError("lora_target_modules must be a non-empty list")
        if any(
            not isinstance(module, str) or not module.strip()
            for module in self.lora_target_modules
        ):
            raise ValueError("lora_target_modules must contain non-empty strings")
        if len(set(self.lora_target_modules)) != len(self.lora_target_modules):
            raise ValueError("lora_target_modules must not contain duplicates")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SFTRunConfig":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"Unknown SFT config keys: {sorted(unknown)}")
        payload = dict(payload)
        for key in ("datasets", "eval_datasets"):
            if key in payload:
                if not isinstance(payload[key], list):
                    raise ValueError(f"{key} must be a list of dataset specs")
                payload[key] = [DatasetSpec.from_dict(item) for item in payload[key]]
        return cls(**payload)

    @classmethod
    def load(cls, path: str | Path) -> "SFTRunConfig":
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_json_constant,
        )
        _reject_non_finite_numbers(payload)
        if not isinstance(payload, dict):
            raise ValueError("SFT config must be a JSON object")
        return cls.from_dict(payload)

    def train_specs(self) -> list[DatasetSpecLike]:
        """Selection specs for training: `datasets`, or `trace_root` as one source."""
        specs = self.datasets or [DatasetSpec(path=self.trace_root)]
        return [spec.to_selection_spec() for spec in specs]

    def eval_specs(self) -> list[DatasetSpecLike]:
        return [spec.to_selection_spec() for spec in self.eval_datasets]
