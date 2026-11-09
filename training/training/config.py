from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


from training.context_contract import (
    ContextContract,
    resolve_context_contract as resolve_context_limits,
)
from training.game_sampling import GAME_SAMPLING_MODES
from training.flash.lora_targets import validate_target_modules
from training.flash_config import FlashSettings, ParityThresholds
from training.flash_drafting import (
    FLASH_DRAFT_TOKENS,
    FLASH_E307_DRAFTER,
    speculative_engine_config,
    speculative_identity,
)


DEFAULT_MALFORMED_CALL_PENALTY = 0.05


def _reject_non_finite_json_constant(value: str) -> None:
    raise ValueError(f"Configuration contains non-finite numeric value {value!r}.")


def _reject_non_finite_numbers(value: Any, *, path: str = "config") -> None:
    """Reject infinities produced by otherwise valid-looking JSON exponents."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Configuration contains non-finite numeric value at {path}.")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, path=f"{path}[{index}]")


def _validate_config_section(
    payload: object,
    *,
    section: str,
    allowed_fields: set[str],
    optional: bool = False,
) -> dict[str, Any]:
    """Return a JSON object after rejecting unknown fields."""
    if payload is None and optional:
        return {}
    if not isinstance(payload, dict):
        raise ValueError(f"{section} must be a JSON object.")
    unknown = sorted(str(key) for key in payload if key not in allowed_fields)
    if unknown:
        dotted_fields = ", ".join(f"{section}.{name}" for name in unknown)
        raise ValueError(f"Unknown configuration field(s): {dotted_fields}.")
    return payload


def _dataclass_config_fields(
    config_type: type[Any], *, exclude: set[str] | None = None
) -> set[str]:
    names = {item.name for item in fields(config_type)}
    return names - (exclude or set())


@dataclass(slots=True)
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LoraSettings":
        payload = _validate_config_section(
            payload,
            section="training.lora",
            allowed_fields=_dataclass_config_fields(cls),
        )
        return cls(
            r=int(payload.get("r", 16)),
            alpha=int(payload.get("alpha", 32)),
            dropout=float(payload.get("dropout", 0.05)),
            target_modules=[str(item) for item in payload.get("target_modules", [])],
        )


@dataclass(slots=True)
class FullWeightSettings:
    """Process topology and checkpoint retention for the FSDP2 backend."""

    device_ids: list[int] = field(default_factory=list)
    checkpoint_every_iterations: int = 20
    checkpoint_keep_last: int = 6

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FullWeightSettings":
        payload = _validate_config_section(
            payload,
            section="training.full_weight",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        device_ids = payload.get("device_ids", [])
        if not isinstance(device_ids, list) or any(
            type(device_id) is not int for device_id in device_ids
        ):
            raise ValueError(
                "training.full_weight.device_ids must be a JSON list of integers."
            )
        checkpoint_every_iterations = payload.get("checkpoint_every_iterations", 20)
        if type(checkpoint_every_iterations) is not int:
            raise ValueError(
                "training.full_weight.checkpoint_every_iterations must be an integer."
            )
        checkpoint_keep_last = payload.get("checkpoint_keep_last", 6)
        if type(checkpoint_keep_last) is not int:
            raise ValueError(
                "training.full_weight.checkpoint_keep_last must be an integer."
            )
        return cls(
            device_ids=list(device_ids),
            checkpoint_every_iterations=checkpoint_every_iterations,
            checkpoint_keep_last=checkpoint_keep_last,
        )


@dataclass(slots=True)
class SampleBudgetConfig:
    """Fits the trainer to the rollout clock: in ``auto`` mode the fraction of
    eligible samples trained per window is steered so that one FULL trainer
    iteration (window receipt -> adapter live on the servers, excluding
    opportunistic extra cycles) takes ``target_train_to_collection_ratio`` of
    one FULL rollout iteration (wall time between consecutive windows)."""

    mode: str = "auto"
    initial_fraction: float = 1.0
    min_fraction: float = 0.1
    max_fraction: float = 1.0
    target_train_to_collection_ratio: float = 0.95
    adjustment_rate: float = 0.25

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SampleBudgetConfig":
        payload = _validate_config_section(
            payload,
            section="training.sample_budget",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        mode = str(payload.get("mode", "auto") or "auto").strip().lower()
        if mode not in {"auto", "full"}:
            raise ValueError("training.sample_budget.mode must be 'auto' or 'full'.")
        min_fraction = min(1.0, max(0.0, float(payload.get("min_fraction", 0.1))))
        max_fraction = min(1.0, max(0.0, float(payload.get("max_fraction", 1.0))))
        if min_fraction > max_fraction:
            raise ValueError(
                "training.sample_budget.min_fraction must be less than or equal to max_fraction."
            )
        initial_fraction = min(
            max_fraction,
            max(min_fraction, float(payload.get("initial_fraction", 1.0))),
        )
        target_ratio = float(payload.get("target_train_to_collection_ratio", 0.95))
        if target_ratio <= 0.0:
            raise ValueError(
                "training.sample_budget.target_train_to_collection_ratio must be greater than 0."
            )
        adjustment_rate = min(
            1.0, max(0.0, float(payload.get("adjustment_rate", 0.25)))
        )
        return cls(
            mode=mode,
            initial_fraction=initial_fraction,
            min_fraction=min_fraction,
            max_fraction=max_fraction,
            target_train_to_collection_ratio=target_ratio,
            adjustment_rate=adjustment_rate,
        )


@dataclass(slots=True)
class EntropyControlConfig:
    """Adaptive entropy-coefficient controller (nats-band targeting).

    Fixed entropy coefficients failed in both directions on this task
    (0.0001 -> sharpening collapse; 0.001 -> late runaway; 0.003+ ->
    over-regularization), so the controller applies the field's convergent
    recipe (Skywork-OR1 / TRL use_adaptive_entropy / AER): a sign-based
    bounded integrator with a dead-band. Below ``target_low`` the coefficient
    steps up; above ``target_high`` it steps down toward ``coef_min``; inside
    the band it holds. Hard caps + the decay-above-band gate are the
    anti-windup/anti-domination guarantees: the bonus can only push while
    entropy is measurably low, and its strength is bounded by ``coef_max``.
    Targets are in nats of mean trained-span token entropy (the
    train/mean_entropy metric) -- calibrated per task from the entropy range
    where eval demonstrably improves, NOT a universal constant.
    """

    enabled: bool = False
    target_low: float = 0.30
    target_high: float = 0.40
    coef_step: float = 0.0002
    coef_min: float = 0.0
    coef_max: float = 0.002
    coef_init: float = 0.001

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "EntropyControlConfig":
        payload = _validate_config_section(
            payload,
            section="training.entropy_control",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        target_low = float(payload.get("target_low", 0.30))
        target_high = float(payload.get("target_high", 0.40))
        if not 0.0 < target_low < target_high:
            raise ValueError(
                "training.entropy_control requires 0 < target_low < target_high."
            )
        coef_min = max(0.0, float(payload.get("coef_min", 0.0)))
        coef_max = float(payload.get("coef_max", 0.002))
        if coef_max <= coef_min:
            raise ValueError("training.entropy_control.coef_max must exceed coef_min.")
        coef_step = float(payload.get("coef_step", 0.0002))
        if coef_step <= 0.0:
            raise ValueError("training.entropy_control.coef_step must be positive.")
        coef_init = min(coef_max, max(coef_min, float(payload.get("coef_init", 0.001))))
        return cls(
            enabled=bool(payload.get("enabled", False)),
            target_low=target_low,
            target_high=target_high,
            coef_step=coef_step,
            coef_min=coef_min,
            coef_max=coef_max,
            coef_init=coef_init,
        )


@dataclass(slots=True)
class SpeculativeDecodingConfig:
    """Flash rollout and eval servers drafting with e307's MTP head.

    Opt-in. Verification is exact rejection sampling, so every served token is
    a draw from the policy's own distribution and carries its logprob; only
    the speed depends on the drafter. The drafter and its settings are ullm's
    e307 deployment's (``FLASHNEXT_E307_DRAFTER``; the ARC-2 Duck's column
    serves REAP-307 with its own drafter instead), valid for e307 only and
    never trained: acceptance falls as the policy drifts from e307.
    A serving setting, outside the Flash learner recipe (a resume may change
    it); the servers attest it on ``/runtime``.
    """

    enabled: bool = False
    drafter: str = FLASH_E307_DRAFTER
    num_speculative_tokens: int = FLASH_DRAFT_TOKENS

    @classmethod
    def from_dict(cls, payload: Any) -> "SpeculativeDecodingConfig":
        section = "inference.speculative_decoding"
        payload = _validate_config_section(
            payload,
            section=section,
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        enabled = payload.get("enabled", False)
        if type(enabled) is not bool:
            raise ValueError(f"{section}.enabled must be true or false.")
        drafter = payload.get("drafter", FLASH_E307_DRAFTER)
        if (
            not isinstance(drafter, str)
            or not drafter.startswith("/")
            or drafter != drafter.strip()
        ):
            raise ValueError(f"{section}.drafter must be an absolute checkpoint path.")
        tokens = payload.get("num_speculative_tokens", FLASH_DRAFT_TOKENS)
        if type(tokens) is not int or not 1 <= tokens <= 8:
            raise ValueError(f"{section}.num_speculative_tokens must be an integer in 1..8.")
        return cls(enabled=enabled, drafter=drafter, num_speculative_tokens=tokens)

    def engine_config(self) -> dict | None:
        """The fork's ``--speculative-config`` (None: the engine does not draft)."""
        if not self.enabled:
            return None
        return speculative_engine_config(self.drafter, self.num_speculative_tokens)

    def identity(self) -> dict | None:
        """What each server must attest on ``/runtime`` (None: no drafting)."""
        if not self.enabled:
            return None
        return speculative_identity(self.drafter, self.num_speculative_tokens)


@dataclass(slots=True)
class RolloutAugmentationConfig:
    """Training-rollout data augmentation (opt-in; harness_support/augmentation.py).

    One augmentation per puzzle per training window, the same for every
    attempt in the window's group; evaluation windows never augment. ARC-2
    draws a dihedral transform (``dihedral``) and a permutation of the colors
    the puzzle uses (``colors``), either of which may be turned off. ARC-3
    permutes all 16 colors of every frame and never flips or rotates, which
    would change what its actions mean. Part of the checkpoint identity when
    enabled.
    """

    enabled: bool = False
    dihedral: bool = False
    colors: bool = True

    @classmethod
    def from_dict(cls, payload: Any, *, environment: str) -> "RolloutAugmentationConfig":
        section = "inference.augmentation"
        payload = _validate_config_section(
            payload,
            section=section,
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        values = {
            "enabled": payload.get("enabled", False),
            "dihedral": payload.get("dihedral", environment == "arc2"),
            "colors": payload.get("colors", True),
        }
        for name, value in values.items():
            if type(value) is not bool:
                raise ValueError(f"{section}.{name} must be true or false.")
        if environment == "arc3" and values["dihedral"]:
            raise ValueError(
                f"{section}.dihedral is ARC-2 only: flips and rotations change "
                "what ARC-3 actions mean."
            )
        if values["enabled"] and not (values["dihedral"] or values["colors"]):
            raise ValueError(f"{section} is enabled but neither dihedral nor colors is.")
        return cls(**values)

    def identity(self) -> dict[str, Any] | None:
        """What a checkpoint records (None: rollouts are not augmented)."""
        if not self.enabled:
            return None
        from harness_support.augmentation import AUGMENTATION_VERSION

        return {
            "version": AUGMENTATION_VERSION,
            "dihedral": self.dihedral,
            "colors": self.colors,
        }


@dataclass(slots=True)
class InferenceConfig:
    base_url: str
    model_id: str
    game: str
    games: list[str]
    max_runtime_minutes: float | None
    environment: str = "arc3"
    # ARC-2: one named puzzle set or several (their union); ``games`` selects
    # from it (empty: all of it).
    dataset: str | list[str] | None = None
    # Duck image observations, independent of its tools and text feedback.
    agent_multimodal_context: str = ""
    agent_max_episode_tokens: int | None = None
    # Admission cap per serving GPU, independent of the complete window size.
    max_concurrent_episodes: int | None = None
    # Optional wall-clock ramp for the per-rollout runtime of TRAIN windows:
    # a list of [hours_since_launch, minutes] control points with strictly
    # increasing hours, interpolated PIECEWISE-LINEARLY at collection time
    # (elapsed measured from the run_launched_at marker, so the ramp survives
    # relaunch legs and resumes). Beyond the last point the final value
    # holds; before the first point, and whenever the launch marker is
    # unreadable, max_runtime_minutes applies. Eval windows never ramp.
    max_runtime_minutes_schedule: list[tuple[float, float]] | None = None
    # One connect-only vLLM server per listed GPU id; rollout windows are
    # split across the servers and merged. Empty -> single server.
    server_gpus: list[str] = field(default_factory=list)
    # Explicit fleet endpoints; empty retains the single-node port derivation.
    base_urls: list[str] = field(default_factory=list)
    server_allocation: str = "shared"
    # Actual pre-launched server logs, ordered like the server URLs.
    server_log_paths: list[str] = field(default_factory=list)
    agent_temperature: float = 1.0
    agent_top_p: float = 1.0
    agent_top_k: int = 0
    # Path to the arc3-platypus venv whose python runs
    # scripts/platypus_rollout_worker.py (one benchmark worker per window;
    # the serving-edge gateway carries the training data). Required.
    platypus_venv: str = ""
    # Qwen3.8 chat-template reasoning effort ('xhigh' | 'medium' | 'low').
    # Empty = leave the template at its own default (xhigh, which injects
    # think-carefully instructions into the system prompt; 'medium' injects
    # nothing). Applies to train rollouts AND the official-eval worker.
    agent_reasoning_effort: str = ""
    agent_context_window: int | None = None
    # Per-generation output cap in tokens (LOCAL_AGENT_MAX_OUTPUT); 0
    # disables the cap so generations are bounded only by the context window
    # and the episode budget. No fallback default: every config states it
    # explicitly (it sets compaction spacing, so a silent default is a trap).
    agent_max_completion_tokens: int = -1
    # Optional over-trigger completion cap and independent served sequence ceiling.
    agent_max_summary_tokens: int | None = None
    agent_summary_forbidden_token_ids: list[int] = field(default_factory=list)
    agent_max_served_sequence_tokens: int | None = None
    # Reset prompt ceiling with a full reply budget. Summary compaction uses
    # the prompt budget; pruning uses a smaller target. A shorter remaining
    # episode budget gives summary resets the corresponding extra prompt room.
    agent_compact_to_tokens: int = 0
    # Compaction-trigger safety margin override (duck safety_margin_tokens).
    # None keeps the duck's own default. With 0, compaction fires when
    # prompt + max_completion exceeds the agent window.
    agent_safety_margin_tokens: int | None = None
    # Malformed-reply handling in the harness. ``agent_malformed_retries``
    # is the duck's re-sample budget (the ARC-2 duck defaults to 3, the ARC-3
    # duck and ullm's retry manager to 5; a missing value loads as 0);
    # ``agent_accept_malformed_output`` configures the duck's backend to
    # keep a reply with no final content as the model's turn, so it is
    # traced, reported as a format_break step, and trained on. Both ride
    # the worker spec into the shared Duck training configuration. Training wants
    # 0 + True: every draw the policy makes becomes a step.
    agent_malformed_retries: int | None = 0
    agent_accept_malformed_output: bool = True
    # ARC-2 only: a retained reply with no text and no tool call ends the
    # episode as a give-up (the Duck's ``terminate_on_empty_reply``), so the
    # standing answer is scored at once instead of the entry nudging the
    # model into another empty turn. Off keeps the nudge loop: the Duck's
    # analyzer entry nudges as after any reply without a tool call and the
    # episode goes on. Either way each empty reply is a trained step charged
    # as one malformed call (``training.malformed_call_penalty``, within
    # ``training.max_episode_penalty``).
    agent_terminate_on_empty_reply: bool = False
    # Flash servers only: MTP drafting with ullm's e307 drafter (opt-in).
    speculative_decoding: SpeculativeDecodingConfig = field(
        default_factory=SpeculativeDecodingConfig
    )
    # Training rollouts only (opt-in): per-window grid/frame augmentation.
    augmentation: RolloutAugmentationConfig = field(
        default_factory=RolloutAugmentationConfig
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InferenceConfig":
        payload = dict(payload)
        payload = _validate_config_section(
            payload,
            section="inference",
            allowed_fields=_dataclass_config_fields(cls, exclude={"game"}),
        )
        base_url = os.environ.get(
            "ARC3_INFERENCE_BASE_URL", str(payload["base_url"])
        ).strip()
        environment = str(payload.get("environment", "arc3")).strip().lower()
        if environment not in {"arc3", "arc2"}:
            raise ValueError("inference.environment must be 'arc3' or 'arc2'.")
        multimodal_context = payload.get("agent_multimodal_context", "")
        if not isinstance(multimodal_context, str) or multimodal_context not in {
            "", "current_grid", "level_start"
        }:
            raise ValueError("inference.agent_multimodal_context must be '', 'current_grid' or 'level_start'.")
        if environment != "arc3" and multimodal_context:
            raise ValueError("inference.agent_multimodal_context is an ARC-3 setting.")
        dataset = _arc2_dataset(payload.get("dataset"), section="inference")
        if environment == "arc3" and dataset is not None:
            raise ValueError("inference.dataset is supported only for ARC-2.")
        raw_games = payload.get("games")
        games: list[str]
        if isinstance(raw_games, list):
            games = [str(item).strip() for item in raw_games if str(item).strip()]
        elif raw_games is not None:
            text = str(raw_games).strip()
            games = [text] if text else []
        else:
            games = []
        if not games and environment == "arc3":
            games = ["pathfinder-0001"]
        if environment == "arc2" and dataset is None:
            raise ValueError("inference.dataset is required for ARC-2.")
        primary_game = games[0] if games else ""
        episode_tokens = payload.get("agent_max_episode_tokens")
        if episode_tokens is not None and (
            type(episode_tokens) is not int or episode_tokens <= 0
        ):
            raise ValueError(
                "inference.agent_max_episode_tokens must be positive or null."
            )
        raw_max_runtime_minutes = payload.get("max_runtime_minutes", 10.0)
        concurrency = payload.get("max_concurrent_episodes")
        if concurrency is not None and (
            type(concurrency) is not int or concurrency <= 0
        ):
            raise ValueError(
                "inference.max_concurrent_episodes must be positive or null."
            )
        raw_runtime_schedule = payload.get("max_runtime_minutes_schedule")
        runtime_schedule: list[tuple[float, float]] | None = None
        if raw_runtime_schedule not in (None, "", []):
            if not isinstance(raw_runtime_schedule, list):
                raise ValueError(
                    "inference.max_runtime_minutes_schedule must be a list of "
                    "[hours_since_launch, minutes] pairs."
                )
            runtime_schedule = []
            previous_hours: float | None = None
            for entry in raw_runtime_schedule:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise ValueError(
                        "inference.max_runtime_minutes_schedule entries must be "
                        "[hours_since_launch, minutes] pairs."
                    )
                hours = float(entry[0])
                minutes = float(entry[1])
                if hours < 0.0:
                    raise ValueError(
                        "inference.max_runtime_minutes_schedule hours must be >= 0."
                    )
                if minutes <= 0.0:
                    raise ValueError(
                        "inference.max_runtime_minutes_schedule minutes must be > 0."
                    )
                if previous_hours is not None and hours <= previous_hours:
                    raise ValueError(
                        "inference.max_runtime_minutes_schedule hours must be "
                        "strictly increasing."
                    )
                previous_hours = hours
                runtime_schedule.append((hours, minutes))
        server_gpus = _string_list(payload.get("server_gpus", []))
        server_allocation = payload.get("server_allocation", "shared")
        if server_allocation not in {"shared", "separate"}:
            raise ValueError("inference.server_allocation must be 'shared' or 'separate'")
        base_urls = payload.get("base_urls", [])
        if not isinstance(base_urls, list) or any(not isinstance(url, str) for url in base_urls):
            raise ValueError("inference.base_urls must be a list of URLs")
        if base_urls and len(base_urls) != len(server_gpus):
            raise ValueError("inference.base_urls requires one URL per server_gpus entry")
        if len(set(base_urls)) != len(base_urls):
            raise ValueError("inference.base_urls must be distinct")
        for url in base_urls:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.port:
                raise ValueError("inference.base_urls requires HTTP URLs with explicit ports")
        return cls(
            base_url=base_url,
            base_urls=list(base_urls),
            server_allocation=server_allocation,
            model_id=str(payload.get("model_id", "")),
            game=primary_game,
            games=games,
            environment=environment,
            dataset=dataset,
            agent_multimodal_context=multimodal_context,
            agent_max_episode_tokens=episode_tokens,
            max_concurrent_episodes=concurrency,
            max_runtime_minutes=(
                None
                if raw_max_runtime_minutes in (None, "")
                else float(raw_max_runtime_minutes)
            ),
            max_runtime_minutes_schedule=runtime_schedule,
            agent_context_window=None
            if payload.get("agent_context_window") in (None, "")
            else int(payload["agent_context_window"]),
            agent_summary_forbidden_token_ids=_summary_forbidden_token_ids(payload),
            agent_max_summary_tokens=None
            if payload.get("agent_max_summary_tokens") in (None, "")
            else int(payload["agent_max_summary_tokens"]),
            agent_max_served_sequence_tokens=None
            if payload.get("agent_max_served_sequence_tokens") in (None, "")
            else int(payload["agent_max_served_sequence_tokens"]),
            agent_safety_margin_tokens=None
            if payload.get("agent_safety_margin_tokens") in (None, "")
            else int(payload["agent_safety_margin_tokens"]),
            agent_compact_to_tokens=max(
                0, int(payload.get("agent_compact_to_tokens", 0))
            ),
            agent_max_completion_tokens=_require_completion_cap(payload),
            server_gpus=server_gpus,
            server_log_paths=_string_list(payload.get("server_log_paths", [])),
            agent_temperature=float(payload.get("agent_temperature", 1.0)),
            agent_top_p=float(payload.get("agent_top_p", 1.0)),
            agent_top_k=max(0, int(payload.get("agent_top_k", 0))),
            platypus_venv=str(payload.get("platypus_venv", "") or "").strip(),
            agent_reasoning_effort=_parse_agent_reasoning_effort(
                payload.get("agent_reasoning_effort")
            ),
            agent_malformed_retries=(
                0
                if payload.get("agent_malformed_retries") in (None, "")
                else max(0, int(payload["agent_malformed_retries"]))
            ),
            agent_accept_malformed_output=bool(
                payload.get("agent_accept_malformed_output", True)
            ),
            agent_terminate_on_empty_reply=bool(
                payload.get("agent_terminate_on_empty_reply", False)
            ),
            speculative_decoding=SpeculativeDecodingConfig.from_dict(
                payload.get("speculative_decoding")
            ),
            augmentation=RolloutAugmentationConfig.from_dict(
                payload.get("augmentation"), environment=environment
            ),
        )


def validate_training_sampling(inference: InferenceConfig) -> None:
    """Require the unmodified distribution whose probabilities PPO scores."""
    for name in ("agent_temperature", "agent_top_p"):
        if getattr(inference, name) != 1.0:
            raise ValueError(
                f"inference.{name} must be 1.0 for raw-logit PPO training."
            )
    if inference.agent_top_k not in (0, -1):
        raise ValueError(
            "inference.agent_top_k must be 0 or -1 (disabled) for training."
        )
    if inference.agent_malformed_retries != 0:
        raise ValueError(
            "inference.agent_malformed_retries must be 0: every draw is a step."
        )
    if inference.agent_accept_malformed_output is not True:
        raise ValueError(
            "inference.agent_accept_malformed_output must be true for training."
        )


def _summary_forbidden_token_ids(payload: dict) -> list[int]:
    value = payload.get("agent_summary_forbidden_token_ids", [])
    if not isinstance(value, list) or any(
        type(token) is not int or token < 0 for token in value
    ):
        raise ValueError(
            "agent_summary_forbidden_token_ids must be a list of nonnegative integers"
        )
    return list(value)


def require_server_endpoints(inference: InferenceConfig) -> None:
    """A separate allocation's endpoints are filled in by the job, never derived."""
    if inference.server_allocation == "separate" and not inference.base_urls:
        raise ValueError("Separate server allocations require inference.base_urls before training starts")


def max_served_sequence_tokens(inference: InferenceConfig) -> int | None:
    """Return the sequence ceiling without validating full-game budgets."""
    return (
        inference.agent_max_served_sequence_tokens
        if inference.agent_max_served_sequence_tokens is not None
        else inference.agent_context_window
    )


def resolve_context_contract(inference: InferenceConfig) -> ContextContract:
    """Resolve the full-game context budgets configured for inference."""
    return resolve_context_limits(
        inference.agent_context_window,
        inference.agent_max_completion_tokens,
        inference.agent_compact_to_tokens,
        inference.agent_safety_margin_tokens,
        inference.agent_max_summary_tokens,
        inference.agent_max_served_sequence_tokens,
        tuple(inference.agent_summary_forbidden_token_ids),
    )


def full_game_context_contract(inference: InferenceConfig) -> tuple[int, int, int]:
    """Return the legacy full-game trim thresholds."""
    context = resolve_context_contract(inference)
    return (
        context.compaction_trigger_tokens,
        context.reset_prompt_ceiling_tokens,
        context.safety_margin_tokens,
    )


def evaluation_inference_config(config: ExperimentConfig) -> InferenceConfig:
    """Resolve evaluation context once for both the collector and engine launcher."""
    from training.harness import get_trainer_support

    overrides = get_trainer_support(
        config.inference.environment
    ).evaluation_context_overrides(config.inference, config.official_eval)
    if config.official_eval.context_window_tokens is not None:
        overrides["agent_context_window"] = config.official_eval.context_window_tokens
    if config.official_eval.compact_to_tokens is not None:
        overrides["agent_compact_to_tokens"] = config.official_eval.compact_to_tokens
    return replace(config.inference, **overrides)


def _optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def _arc2_dataset(value: Any, *, section: str) -> str | list[str] | None:
    """Validate the separately versioned ARC-2 dataset selector: one name, or a
    list of distinct names whose union a section draws its games from."""
    from harness_support.arc2.dataset import DATASET_NAMES

    names = " or ".join(repr(name) for name in DATASET_NAMES)
    if isinstance(value, list):
        datasets = [_optional_text(item) for item in value]
        if (
            not datasets
            or any(name not in DATASET_NAMES for name in datasets)
            or len(set(datasets)) != len(datasets)
        ):
            raise ValueError(
                f"{section}.dataset must be {names}, or a list of distinct ones."
            )
        return datasets
    dataset = _optional_text(value)
    if dataset is not None and dataset not in DATASET_NAMES:
        raise ValueError(f"{section}.dataset must be {names}, or a list of distinct ones.")
    return dataset


# The Qwen3.8 chat template accepts exactly these (duck's enum also lists
# high/max, which the native template renderer rejects).
_AGENT_REASONING_EFFORTS = ("xhigh", "medium", "low")


def _parse_agent_reasoning_effort(value: Any) -> str:
    effort = str(value or "").strip().lower()
    if effort and effort not in _AGENT_REASONING_EFFORTS:
        raise ValueError(
            "inference.agent_reasoning_effort must be one of "
            f"{', '.join(_AGENT_REASONING_EFFORTS)} (or empty for the "
            "template default)."
        )
    return effort


def _require_completion_cap(payload: dict[str, Any]) -> int:
    """The reply cap has no fallback: it sets compaction spacing, so every
    config must state it (0 explicitly disables the cap)."""
    if "agent_max_completion_tokens" not in payload:
        raise ValueError(
            "inference.agent_max_completion_tokens is required: the reply cap "
            "sets compaction spacing and has no default. Set it explicitly "
            "(0 disables the cap)."
        )
    return max(0, int(payload["agent_max_completion_tokens"]))


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is not None:
        text = str(value).strip()
        return [text] if text else []
    return []


# Weight modes whose learner runs as torchrun ranks on training.<mode>.device_ids.
TORCHRUN_WEIGHT_MODES = ("full_weight", "flash")


@dataclass(slots=True)
class TrainingConfig:
    model_name: str
    output_dir: str
    device: str
    seed: int
    max_sequence_tokens: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    epochs_per_window: int
    learning_rate: float
    weight_decay: float
    # Asymmetric PPO ratio clipping (arXiv:2509.26114, issue #28): symmetric
    # epsilon lets the clip-high term dominate and systematically drains
    # policy entropy. clip_ratio_high=None disables the upper clip entirely.
    clip_ratio_low: float
    clip_ratio_high: float | None
    max_grad_norm: float
    disable_dropout: bool
    bf16: bool
    gradient_checkpointing: bool
    lora: LoraSettings
    advantage_estimator: str = "state_critic"
    weight_mode: str = "lora"
    full_weight: FullWeightSettings = field(default_factory=FullWeightSettings)
    flash: FlashSettings = field(default_factory=FlashSettings)
    parity_thresholds: dict[str, ParityThresholds] = field(default_factory=dict)
    # Pinned HF revision (commit hash / tag) for model_name. When set it is
    # passed everywhere the checkpoint is resolved: trainer tokenizer/model/
    # image-processor loads, vLLM --revision/--tokenizer-revision, and the
    # rollout agent's context-budget tokenizer. Empty = HEAD of main
    # (previous behavior).
    model_revision: str = ""
    attn_implementation: str = "sdpa"
    # Load the checkpoint through the image-text auto-class so the vision
    # tower survives and pixel inputs can be threaded through the trainer
    # forward (the tower stays frozen in both policy weight modes).
    multimodal: bool = False
    # Shard the policy model across these CUDA device indices (accelerate
    # device_map="balanced"). Empty -> single-device training on `device`,
    # which must name the first listed GPU when set.
    device_map_gpus: list[int] = field(default_factory=list)
    # Data-parallel gradient co-workers: one replica per inner list, each
    # layer-sharded across its own GPUs (training/ddp_worker.py). Rank 0 keeps
    # the only optimizer; every optimizer step trains a (1+len) x larger batch.
    worker_device_map_gpus: list[list[int]] = field(default_factory=list)
    # Initialize a FRESH run from an existing adapter/trainer state (ignored
    # when the run dir itself resumes). Used to seed a new topology run from a
    # previous run's learned policy and optimizer state.
    init_adapter_dir: str | None = None
    init_trainer_state_path: str | None = None
    # Merge per-step samples whose token ids verifiably nest into one packed
    # sequence per episode segment (training/episode_packing.py); spans keep
    # per-step advantages/logprobs and losses average per span.
    episode_packing: bool = False
    # With episode packing: weight each packed sequence by
    # steps_in_segment / steps_in_episode so every EPISODE contributes equally
    # to the update and every step contributes equally within its episode,
    # regardless of how the episode fragmented into packed segments.
    # After a full-coverage configured epoch, keep running extra accumulation
    # cycles over the same window (fresh passes, same PPO clip machinery) while
    # a conservatively timed cycle still fits before the next rollout window.
    extra_train_cycles_until_window: bool = False
    # Drift guard for the extra cycles: stop when a cycle's mean clip
    # fraction exceeds this bound (policy moved too far off the rollouts).
    extra_cycle_max_clip_fraction: float = 0.10
    # Window trust region: if the primary epoch's mean clip fraction exceeds
    # this bound, the whole window's optimizer update is REJECTED — trainable
    # params and optimizer state are restored to their pre-window snapshot,
    # the unchanged adapter is re-exported, and the effective learning rate
    # backs off for the next window (recovering on accepted windows). Healthy
    # runs sit at 0.01-0.07; the r13-r17 entropy blowups rode 0.3-0.6.
    # 0 disables the guard.
    max_window_clip_fraction: float = 0.25
    # Previous-policy KL anchor: k3-estimator KL(rollout policy || current)
    # over ALL completion tokens, added to the loss with this coefficient.
    # Bounds per-window policy movement independently of the PPO clip (which
    # stops bounding once ratios blow past it). 0 disables.
    previous_policy_kl_coef: float = 0.0
    warmup_steps: int = 0
    # Keep the newest 6 adapters and trainer states by default. Adapter
    # age/hard-cap/pinned rules still apply; the initial trainer state stays.
    # null keeps every adapter and trainer state without pruning or caps.
    # Integers must be >= 2 so the prior state survives publish/advance.
    # Flash costs ~0.9 GB/iteration (~32 GB/day at 40-minute iterations).
    # Storage only: outside checkpoint/recipe identity; may change on resume.
    checkpoint_keep_last: int | None = 6
    # Adapters whose iteration index is a multiple of this interval are
    # exempt from retention pruning — deliberate long-term snapshots for
    # post-hoc forensics. 0 disables pinning.
    pinned_adapter_interval: int = 0
    cleanup_rollout_artifacts: bool = False
    debug_rollout_examples_per_window: int = 0
    max_micro_batch_tokens: int = 0
    rollout_logprob_check_batches: int = 1
    sample_budget: SampleBudgetConfig = field(default_factory=SampleBudgetConfig)
    entropy_control: EntropyControlConfig = field(default_factory=EntropyControlConfig)
    # By default each rollout is centered against this window's same-game
    # mean ARC score. A hidden critic is opt-in.
    format_break_penalty: float = 0.0
    tool_error_penalty: float = 0.0
    # Terminal ARC-2 reward deduction per malformed call; zero disables it.
    # A malformed call is a malformed tool call, a retained reply the backend
    # could not parse, or an empty reply (no text and no tool call), one each
    # (docs/arc2_training.md).
    malformed_call_penalty: float = DEFAULT_MALFORMED_CALL_PENALTY
    # Ceiling on an ARC-2 episode's summed terminal penalty: the reward is
    # correctness - min(cap, malformed_call_penalty * malformed calls), so a
    # success with many penalties still beats no success. None (the default)
    # caps nothing. ARC-3 charges no reply penalties and refuses it.
    max_episode_penalty: float | None = None
    value_baseline: str = "group_mean"
    # The hidden critic always conditions on the policy's
    # detached final-prompt hidden state. It uses raw callback reward-to-go.
    state_critic_learning_rate: float = 1e-3
    state_critic_weight_decay: float = 0.0
    state_critic_hidden_dim: int = 64
    state_critic_game_embedding_dim: int = 16
    state_critic_game_embedding_buckets: int = 4096
    state_critic_initial_value: float | None = 0.0
    state_critic_target_scale: float = 100.0
    state_critic_target_normalization: str = "fixed"
    state_critic_normalization_alpha: float = 0.125
    state_critic_normalization_min_rollouts: int = 16
    state_critic_normalization_min_std: float = 1.0
    state_critic_epochs_per_window: int = 4
    state_critic_batch_size: int = 256
    state_critic_max_grad_norm: float = 10.0
    state_critic_policy_hidden_projection_dim: int = 512
    gae_gamma: float = 1.0
    # Lambda is per decision, not per token. One gives finite-horizon return-to-go.
    gae_lambda: float = 1.0
    entropy_coef: float = 0.0
    advantage_clip: float = 0.0
    # KL(current || base) anchor, k3 estimator on the trained spans against
    # the base model with adapters disabled (one extra forward per
    # micro-batch). Bounds slow, consistent drift that the previous-window
    # KL cannot see. 0 disables.
    reference_kl_coef: float = 0.0
    # Collapse watch on the leading indicators of the drift-to-degeneration
    # failure: the fraction of model turns the harness reported malformed
    # (format_break) and the actions executed per episode against their
    # trailing median. An alarm writes train/collapse_alarm=1 and a warning;
    # stop_on_collapse turns it into a fatal error (the run stays down until
    # someone looks, like a crashed rollout).
    collapse_format_break_fraction_alarm: float = 0.02
    collapse_actions_drop_fraction: float = 0.5
    collapse_watch_windows: int = 8
    stop_on_collapse: bool = False
    # PPO loss aggregation over the tokens of one sequence. "span_mean"
    # weights every env step equally, so a token in a short turn carries
    # more gradient than one in a long turn and a negative advantage is
    # diluted by length; "token_mean" weights every generated token across the window
    # equally (DAPO-style), removing that per-turn length bias;
    # "episode_mean" gives every episode the same total weight spread evenly
    # over its generated tokens, so a token in a 10-token episode carries ten
    # times the weight of a token in a 100-token episode.
    # "token_sum_fixed" keeps token_mean's equal token coefficients but divides
    # the window by episodes * loss_normalization_tokens instead of the sampled
    # token count, on the ordinary accumulation path (any schedule or budget).
    loss_aggregation: str = "token_mean"
    # The DPPO experiment is a complete, explicit recipe. PPO remains the
    # default; DPPO is selected explicitly by the new recipes.
    policy_loss: str = "ppo"
    score_centering_top_k: int | None = None
    tis_ratio_cap: float | None = None
    dppo_tv_threshold: float = 0.15
    dppo_importance_ratio_cap: float = 5.0
    # Fixed-denominator reference length; null unless loss_aggregation is
    # 'token_sum_fixed' or 'sequence_sum_fixed' (default 32768 for those).
    loss_normalization_tokens: int | None = None
    optimizer_minibatch_rollouts: int = 16
    rollout_schedule: str = "overlapped"

    def __post_init__(self) -> None:
        if self.policy_loss == "score_centered_tis" and self.tis_ratio_cap is None:
            self.tis_ratio_cap = 2.0
        if self.loss_normalization_tokens is None and self.loss_aggregation in (
            "token_sum_fixed",
            "sequence_sum_fixed",
        ):
            self.loss_normalization_tokens = 32768

    @property
    def uses_torchrun(self) -> bool:
        return self.weight_mode in TORCHRUN_WEIGHT_MODES

    @property
    def torchrun_device_ids(self) -> list[int]:
        """The torchrun ranks' GPUs (training.<mode>.device_ids); empty for LoRA."""
        return getattr(self, self.weight_mode).device_ids if self.uses_torchrun else []

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingConfig":
        payload = dict(payload)
        policy_loss = str(payload.get("policy_loss", "ppo")).strip().lower()
        if policy_loss not in ("ppo", "dppo_binary_tv", "score_centered_tis"):
            raise ValueError(
                "training.policy_loss must be 'ppo', 'dppo_binary_tv' or 'score_centered_tis'"
            )
        head_k = payload.get("score_centering_top_k")
        tis_cap = payload.get("tis_ratio_cap")
        if policy_loss == "score_centered_tis":
            if type(head_k) is not int or head_k < 1:
                raise ValueError(
                    "training.score_centering_top_k must be a positive integer"
                )
            tis_cap = 2.0 if tis_cap is None else tis_cap
            if (
                type(tis_cap) not in (int, float)
                or not math.isfinite(tis_cap)
                or tis_cap <= 1
            ):
                raise ValueError("training.tis_ratio_cap must be finite and > 1")
            tis_cap = float(tis_cap)
        elif head_k is not None or tis_cap is not None:
            raise ValueError(
                "training.score_centering_top_k and tis_ratio_cap require score_centered_tis"
            )
        if policy_loss == "dppo_binary_tv":
            baseline = str(payload.get("value_baseline", "group_mean")).strip().lower()
            if baseline not in ("group_mean", "hidden_critic"):
                raise ValueError("DPPO requires a group_mean or hidden_critic baseline")
            # Preserve the original synchronous experiment unless overlap is
            # explicitly requested. Scheduling does not change the DPPO loss;
            # both paths retain fixed-policy waves and the one-window lag cap.
            payload.setdefault("rollout_schedule", "synchronous")
            recipe = {
                "value_baseline": baseline,
                "loss_aggregation": (
                    "token_mean"
                    if baseline == "hidden_critic"
                    else "sequence_sum_fixed"
                ),
                "epochs_per_window": 1,
                "advantage_clip": 0.0,
                "clip_ratio_low": 0.0,
                "clip_ratio_high": None,
                "max_window_clip_fraction": 0.0,
                "previous_policy_kl_coef": 0.0,
                "reference_kl_coef": 0.0,
                "entropy_coef": 0.0,
                "extra_train_cycles_until_window": False,
                "gae_gamma": 1.0,
                "gae_lambda": 1.0,
            }
            for key, required in recipe.items():
                if key in payload and payload[key] != required:
                    raise ValueError(
                        f"DPPO recipe requires training.{key}={required!r}; "
                        f"got {payload[key]!r}"
                    )
                payload[key] = required
            budget = SampleBudgetConfig.from_dict(
                payload.get("sample_budget", {"mode": "full"})
            )
            if budget.mode != "full":
                raise ValueError(
                    "DPPO recipe requires training.sample_budget.mode='full'"
                )
            payload["sample_budget"] = asdict(budget)
            if EntropyControlConfig.from_dict(payload.get("entropy_control")).enabled:
                raise ValueError(
                    "DPPO recipe requires training.entropy_control.enabled=False"
                )
        tv_threshold = float(payload.get("dppo_tv_threshold", 0.15))
        ratio_cap = float(payload.get("dppo_importance_ratio_cap", 5.0))
        if not math.isfinite(tv_threshold) or not 0.0 < tv_threshold < 1.0:
            raise ValueError("training.dppo_tv_threshold must be finite and in (0, 1)")
        if not math.isfinite(ratio_cap) or ratio_cap < 1.0:
            raise ValueError(
                "training.dppo_importance_ratio_cap must be finite and >= 1"
            )
        positive_integers = {}
        for name, default in (
            ("loss_normalization_tokens", 32768),
            ("optimizer_minibatch_rollouts", 16),
        ):
            value = payload.get(name)
            value = default if value is None else value
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"training.{name} must be a positive integer")
            positive_integers[name] = value
        checkpoint_keep_last = payload.get("checkpoint_keep_last", 6)
        if checkpoint_keep_last is not None and (
            type(checkpoint_keep_last) is not int
            or checkpoint_keep_last < 2
        ):
            raise ValueError(
                "training.checkpoint_keep_last must be null or an integer >= 2; "
                "the prior trainer state must survive the publish/evaluate window "
                "before training_state.json advances."
            )
        rollout_schedule = str(payload.get("rollout_schedule", "overlapped"))
        if rollout_schedule not in ("overlapped", "synchronous"):
            raise ValueError(
                "training.rollout_schedule must be 'overlapped' or 'synchronous'"
            )
        advantage_estimator = (
            str(payload.get("advantage_estimator", "state_critic")).strip().lower()
        )
        if advantage_estimator not in {"state_critic", "grpo", "dr_grpo"}:
            raise ValueError(
                "training.advantage_estimator must be 'state_critic', 'grpo' or 'dr_grpo'."
            )
        if advantage_estimator in {"grpo", "dr_grpo"} and policy_loss not in (
            "ppo",
            "score_centered_tis",
        ):
            raise ValueError(
                "training.advantage_estimator='grpo' or 'dr_grpo' requires "
                "training.policy_loss='ppo' or 'score_centered_tis'."
            )
        weight_mode = str(payload.get("weight_mode", "lora") or "lora").strip().lower()
        if weight_mode not in {"lora", "full_weight", "flash"}:
            raise ValueError("training.weight_mode must be 'lora', 'full_weight' or 'flash'.")
        flash = FlashSettings.from_dict(payload.get("flash"))
        parity = payload.get("parity_thresholds", {})
        if not isinstance(parity, dict) or set(parity) - {"lora", "full_weight"}:
            raise ValueError("training.parity_thresholds accepts lora/full_weight; use training.flash.parity_thresholds for Flash")
        parity = {mode: ParityThresholds.from_dict(value) for mode, value in parity.items()}
        if flash.train_routers:
            if weight_mode != "flash":
                raise ValueError("training.flash.train_routers requires weight_mode='flash'")
            # The reference forward disables the adapters only; with trained
            # routers it would no longer be the base model.
            if float(payload.get("reference_kl_coef", 0.0) or 0.0) > 0.0:
                raise ValueError("training.flash.train_routers does not support reference_kl_coef > 0")
        if weight_mode == "flash":
            lora_payload = payload.get("lora", {})
            validate_target_modules(lora_payload.get("target_modules", []))
            for name, default in (("r", 16), ("alpha", 32)):
                value = lora_payload.get(name, default)
                if type(value) is not int or value <= 0:
                    raise ValueError(f"training.lora.{name} must be a positive integer for Flash")
            learning_rate = payload.get("learning_rate", 3e-6)
            if type(learning_rate) not in (int, float) or not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("Flash learning_rate must be finite and positive")
            if payload.get("init_adapter_dir") or payload.get("init_trainer_state_path"):
                raise ValueError("Flash warm starts must use a Flash run checkpoint (--run-dir)")
            if (payload.get("full_weight") or {}).get("device_ids"):
                raise ValueError("Flash uses training.flash.device_ids")
            if lora_payload.get("dropout", 0.05) != 0:
                raise ValueError("Flash dense LoRA requires training.lora.dropout=0")
            if not flash.checkpoint_root.strip():
                raise ValueError("training.flash.checkpoint_root is required")
            if not os.path.isabs(flash.checkpoint_root):
                raise ValueError(
                    "training.flash.checkpoint_root must be an absolute path; "
                    "the frozen launcher runs from a snapshot directory"
                )
            if (payload.get("device_map_gpus")
                    or payload.get("worker_device_map_gpus") or not payload.get("bf16", True)):
                raise ValueError("Flash requires BF16 torchrun training, without device_map/worker_device_map_gpus")
            if re.fullmatch(r"cuda(?::[0-9]+)?", str(payload.get("device", "cuda"))) is None:
                raise ValueError("Flash training requires a CUDA device")
        loss_aggregation = (
            str(payload.get("loss_aggregation", "token_mean") or "token_mean")
            .strip()
            .lower()
        )
        if loss_aggregation not in (
            "span_mean",
            "token_mean",
            "episode_mean",
            "sequence_sum_fixed",
            "token_sum_fixed",
        ):
            raise ValueError(
                "training.loss_aggregation must be 'span_mean', 'token_mean', "
                "'episode_mean', 'sequence_sum_fixed' or 'token_sum_fixed', "
                f"got {loss_aggregation!r}"
            )
        if advantage_estimator == "dr_grpo" and loss_aggregation != "token_sum_fixed":
            raise ValueError("dr_grpo requires loss_aggregation='token_sum_fixed'")
        if loss_aggregation == "token_sum_fixed" and (
            str(payload.get("value_baseline", "group_mean")).strip().lower()
            != "group_mean"
        ):
            raise ValueError("token_sum_fixed requires value_baseline='group_mean'")
        if loss_aggregation not in ("token_sum_fixed", "sequence_sum_fixed"):
            # The knob only scales the fixed-denominator losses. Setting it
            # elsewhere promises an effect it cannot have; it resolves to null
            # so full config dumps (eval worker, gradient replicas) reload.
            if payload.get("loss_normalization_tokens") is not None:
                raise ValueError(
                    "training.loss_normalization_tokens requires training."
                    "loss_aggregation='token_sum_fixed' or 'sequence_sum_fixed'"
                )
            positive_integers["loss_normalization_tokens"] = None
        sample_budget = SampleBudgetConfig.from_dict(payload.get("sample_budget"))
        if rollout_schedule == "synchronous" and (
            sample_budget.mode != "full"
            or payload.get("extra_train_cycles_until_window", False)
        ):
            raise ValueError(
                "Synchronous collection requires training.sample_budget.mode='full' "
                "and extra_train_cycles_until_window=False; there is no concurrent "
                "rollout clock to balance or idle time to fill"
            )
        if policy_loss == "dppo_binary_tv" or loss_aggregation == "sequence_sum_fixed":
            accumulation = payload.get("gradient_accumulation_steps", 1)
            if type(accumulation) is not int or accumulation != 1:
                raise ValueError(
                    "Rollout minibatching requires training.gradient_accumulation_steps=1; "
                    "optimizer_minibatch_rollouts controls accumulation"
                )
        payload = _validate_config_section(
            payload,
            section="training",
            allowed_fields=_dataclass_config_fields(cls),
        )
        if (
            payload.get("format_break_penalty", 0) != 0
            or payload.get("tool_error_penalty", 0) != 0
        ):
            raise ValueError(
                "Training rewards must be unmodified ARC scores; penalties must be zero"
            )
        output_dir = str(payload["output_dir"])
        malformed_call_penalty = payload.get(
            "malformed_call_penalty", DEFAULT_MALFORMED_CALL_PENALTY
        )
        if (
            type(malformed_call_penalty) not in (int, float)
            or not math.isfinite(malformed_call_penalty)
            or malformed_call_penalty < 0
        ):
            raise ValueError(
                "training.malformed_call_penalty must be finite and nonnegative"
            )
        max_episode_penalty = payload.get("max_episode_penalty")
        if max_episode_penalty is not None and (
            type(max_episode_penalty) not in (int, float)
            or not math.isfinite(max_episode_penalty)
            or max_episode_penalty <= 0
        ):
            raise ValueError(
                "training.max_episode_penalty must be null (no cap) or finite and positive"
            )
        value_baseline = (
            str(payload.get("value_baseline", "group_mean")).strip().lower()
        )
        if value_baseline not in ("hidden_critic", "group_mean"):
            raise ValueError(
                "training.value_baseline must be 'group_mean' or 'hidden_critic'"
            )
        advantage_clip = float(
            payload.get(
                "advantage_clip", 0.0 if value_baseline == "group_mean" else 5.0
            )
        )
        if value_baseline == "group_mean" and advantage_clip != 0.0:
            raise ValueError(
                "training.value_baseline='group_mean' requires advantage_clip=0"
            )
        if loss_aggregation == "sequence_sum_fixed":
            if value_baseline != "group_mean":
                raise ValueError(
                    "sequence_sum_fixed requires value_baseline='group_mean'"
                )
            if (
                SampleBudgetConfig.from_dict(payload.get("sample_budget")).mode
                != "full"
                or payload.get("epochs_per_window", 1) != 1
                or payload.get("extra_train_cycles_until_window", False)
            ):
                raise ValueError(
                    "sequence_sum_fixed requires a complete batch and exactly one pass"
                )
        if value_baseline == "hidden_critic" and loss_aggregation != "token_mean":
            raise ValueError(
                "training.value_baseline='hidden_critic' requires "
                "training.loss_aggregation='token_mean' because critic advantages "
                "are normalized with global completion-token weights"
            )
        target_normalization = str(
            payload.get("state_critic_target_normalization", "fixed")
        )
        normalization_alpha = float(
            payload.get("state_critic_normalization_alpha", 0.125)
        )
        normalization_min_rollouts = payload.get(
            "state_critic_normalization_min_rollouts", 16
        )
        normalization_min_std = float(
            payload.get("state_critic_normalization_min_std", 1.0)
        )
        from training.critic_normalization import validate_target_normalization

        validate_target_normalization(
            target_normalization,
            normalization_alpha,
            normalization_min_rollouts,
            normalization_min_std,
        )
        if target_normalization == "per_game" and value_baseline != "hidden_critic":
            raise ValueError(
                "per_game critic target normalization requires value_baseline='hidden_critic'"
            )
        gamma = float(payload.get("gae_gamma", 1.0))
        gae_lambda = float(payload.get("gae_lambda", 1.0))
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= gae_lambda <= 1.0:
            raise ValueError(
                "training.gae_gamma and training.gae_lambda must be in [0, 1]"
            )
        return cls(
            model_name=str(payload["model_name"]),
            advantage_estimator=advantage_estimator,
            malformed_call_penalty=float(malformed_call_penalty),
            max_episode_penalty=None
            if max_episode_penalty is None
            else float(max_episode_penalty),
            model_revision=str(payload.get("model_revision", "") or "").strip(),
            output_dir=output_dir,
            device=str(payload.get("device", "cuda")),
            seed=int(payload.get("seed", 7)),
            max_sequence_tokens=int(payload.get("max_sequence_tokens", 3072)),
            micro_batch_size=int(payload.get("micro_batch_size", 1)),
            max_micro_batch_tokens=max(
                0, int(payload.get("max_micro_batch_tokens", 0))
            ),
            sample_budget=sample_budget,
            gradient_accumulation_steps=int(
                payload.get("gradient_accumulation_steps", 1)
            ),
            epochs_per_window=int(payload.get("epochs_per_window", 1)),
            learning_rate=float(payload.get("learning_rate", 3e-6)),
            weight_decay=float(payload.get("weight_decay", 0.0)),
            clip_ratio_low=float(payload.get("clip_ratio_low", 0.2)),
            clip_ratio_high=(
                None
                if payload.get("clip_ratio_high") in (None, "")
                else float(payload.get("clip_ratio_high"))
            ),
            max_grad_norm=float(payload.get("max_grad_norm", 1.0)),
            disable_dropout=bool(payload.get("disable_dropout", True)),
            bf16=bool(payload.get("bf16", True)),
            multimodal=bool(payload.get("multimodal", False)),
            gradient_checkpointing=bool(payload.get("gradient_checkpointing", True)),
            lora=LoraSettings.from_dict(payload.get("lora", {})),
            weight_mode=weight_mode,
            full_weight=FullWeightSettings.from_dict(payload.get("full_weight")),
            flash=flash,
            parity_thresholds=parity,
            device_map_gpus=[
                int(gpu) for gpu in list(payload.get("device_map_gpus", []) or [])
            ],
            worker_device_map_gpus=[
                [int(gpu) for gpu in list(replica or [])]
                for replica in list(payload.get("worker_device_map_gpus", []) or [])
            ],
            init_adapter_dir=_optional_text(payload.get("init_adapter_dir")),
            init_trainer_state_path=_optional_text(
                payload.get("init_trainer_state_path")
            ),
            episode_packing=bool(payload.get("episode_packing", False)),
            extra_train_cycles_until_window=bool(
                payload.get("extra_train_cycles_until_window", False)
            ),
            extra_cycle_max_clip_fraction=max(
                0.0, float(payload.get("extra_cycle_max_clip_fraction", 0.10))
            ),
            max_window_clip_fraction=max(
                0.0, float(payload.get("max_window_clip_fraction", 0.25))
            ),
            previous_policy_kl_coef=max(
                0.0, float(payload.get("previous_policy_kl_coef", 0.0))
            ),
            attn_implementation=str(payload.get("attn_implementation", "sdpa")).strip(),
            warmup_steps=max(0, int(payload.get("warmup_steps", 0))),
            checkpoint_keep_last=checkpoint_keep_last,
            pinned_adapter_interval=max(
                0, int(payload.get("pinned_adapter_interval", 0))
            ),
            cleanup_rollout_artifacts=bool(
                payload.get("cleanup_rollout_artifacts", False)
            ),
            debug_rollout_examples_per_window=max(
                0, int(payload.get("debug_rollout_examples_per_window", 0))
            ),
            rollout_logprob_check_batches=max(
                1, int(payload.get("rollout_logprob_check_batches", 1))
            ),
            value_baseline=value_baseline,
            gae_gamma=gamma,
            gae_lambda=gae_lambda,
            state_critic_learning_rate=float(
                payload.get("state_critic_learning_rate", 1e-3)
            ),
            state_critic_weight_decay=float(
                payload.get("state_critic_weight_decay", 0.0)
            ),
            state_critic_hidden_dim=max(
                1, int(payload.get("state_critic_hidden_dim", 64))
            ),
            state_critic_game_embedding_dim=max(
                1, int(payload.get("state_critic_game_embedding_dim", 16))
            ),
            state_critic_game_embedding_buckets=max(
                1, int(payload.get("state_critic_game_embedding_buckets", 4096))
            ),
            state_critic_initial_value=(
                None
                if payload.get("state_critic_initial_value", 0.0) in (None, "")
                else max(0.0, float(payload.get("state_critic_initial_value", 0.0)))
            ),
            state_critic_target_scale=max(
                1e-8, float(payload.get("state_critic_target_scale", 100.0))
            ),
            state_critic_target_normalization=target_normalization,
            state_critic_normalization_alpha=normalization_alpha,
            state_critic_normalization_min_rollouts=normalization_min_rollouts,
            state_critic_normalization_min_std=normalization_min_std,
            state_critic_epochs_per_window=max(
                0, int(payload.get("state_critic_epochs_per_window", 4))
            ),
            state_critic_batch_size=max(
                1, int(payload.get("state_critic_batch_size", 256))
            ),
            state_critic_max_grad_norm=max(
                0.0, float(payload.get("state_critic_max_grad_norm", 10.0))
            ),
            state_critic_policy_hidden_projection_dim=max(
                1,
                int(payload.get("state_critic_policy_hidden_projection_dim", 512)),
            ),
            entropy_coef=max(0.0, float(payload.get("entropy_coef", 0.0))),
            entropy_control=EntropyControlConfig.from_dict(
                payload.get("entropy_control")
            ),
            reference_kl_coef=max(0.0, float(payload.get("reference_kl_coef", 0.0))),
            collapse_format_break_fraction_alarm=max(
                0.0, float(payload.get("collapse_format_break_fraction_alarm", 0.02))
            ),
            collapse_actions_drop_fraction=min(
                1.0, max(0.0, float(payload.get("collapse_actions_drop_fraction", 0.5)))
            ),
            collapse_watch_windows=max(
                1, int(payload.get("collapse_watch_windows", 8))
            ),
            stop_on_collapse=bool(payload.get("stop_on_collapse", False)),
            advantage_clip=max(0.0, advantage_clip),
            loss_aggregation=loss_aggregation,
            policy_loss=policy_loss,
            score_centering_top_k=head_k,
            tis_ratio_cap=tis_cap,
            dppo_tv_threshold=tv_threshold,
            dppo_importance_ratio_cap=ratio_cap,
            loss_normalization_tokens=positive_integers["loss_normalization_tokens"],
            optimizer_minibatch_rollouts=positive_integers[
                "optimizer_minibatch_rollouts"
            ],
            rollout_schedule=rollout_schedule,
        )


@dataclass(slots=True)
class MetricsConfig:
    log_per_game: bool = True

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MetricsConfig":
        payload = _validate_config_section(
            payload,
            section="metrics",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        return cls(log_per_game=bool(payload.get("log_per_game", True)))


# Window game scheduling is either "balanced"
# (the default: an equal per-game multiset every window) or the opt-in
# "nonzero_rate" curriculum (training.game_sampling).
@dataclass(slots=True)
class SamplingConfig:
    # Depth of the persistent per-game history buffers (returns / ARC scores /
    # levels completed) that feed the rollout/avg_* metrics in training.cli.
    return_history_size: int = 8
    # How each rollout window's game multiset is drawn: "balanced" cycles the
    # configured list into equal per-game counts; "nonzero_rate" draws
    # games_per_window DISTINCT games (weighted by trailing nonzero-return
    # rate above a floor) and runs each for parallel_rollouts/games_per_window
    # passes — windows stay uniform (worker equal-multiset contract), the
    # weighting acts across windows.
    game_sampling_mode: str = "balanced"
    # "nonzero_rate" un-normalized weight = floor + (1 - floor) * rate. Must
    # stay above zero so no configured game is ever excluded from sampling.
    game_sampling_weight_floor: float = 0.15
    # Distinct games per window: required in "nonzero_rate" mode; under
    # "balanced" 0 means every configured game appears in every window and
    # a positive value below the game count draws that many games uniformly
    # per window. Must divide parallel_rollouts and not exceed the count.
    game_sampling_games_per_window: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "SamplingConfig":
        payload = _validate_config_section(
            payload,
            section="sampling",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        mode = (
            str(payload.get("game_sampling_mode", "balanced") or "balanced")
            .strip()
            .lower()
        )
        if mode not in GAME_SAMPLING_MODES:
            raise ValueError(
                "sampling.game_sampling_mode must be one of "
                f"{sorted(GAME_SAMPLING_MODES)}; got {mode!r}."
            )
        weight_floor = float(payload.get("game_sampling_weight_floor", 0.15))
        if not 0.0 < weight_floor <= 1.0:
            raise ValueError(
                "sampling.game_sampling_weight_floor must be in (0, 1] so "
                "every configured game keeps a nonzero sampling weight; got "
                f"{weight_floor}."
            )
        games_per_window = int(payload.get("game_sampling_games_per_window", 0))
        if mode == "nonzero_rate" and games_per_window < 1:
            raise ValueError(
                "sampling.game_sampling_games_per_window must be >= 1 for "
                f"nonzero_rate sampling; got {games_per_window}."
            )
        return cls(
            return_history_size=max(1, int(payload.get("return_history_size", 8))),
            game_sampling_mode=mode,
            game_sampling_weight_floor=weight_floor,
            game_sampling_games_per_window=games_per_window,
        )


@dataclass(slots=True)
class MemoryProfilerReportConfig:
    enabled: bool = True
    format: str = "html"

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryProfilerReportConfig":
        payload = _validate_config_section(
            payload,
            section="memory_profiler.report",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        return cls(
            enabled=bool(payload.get("enabled", True)),
            format=str(payload.get("format", "html") or "html"),
        )


@dataclass(slots=True)
class MemoryProfilerConfig:
    enabled: bool = False
    ranks: str | list[int] = field(default_factory=lambda: [0])
    sync: bool = False
    log_every_n_steps: int = 10
    max_logged_steps: int | None = None
    reset_peak_stats_per_region: bool = False
    log_gpu_utilization: bool = False
    gpu_utilization_sample_interval_seconds: float = 1.0
    output_subdir: str = "memory"
    report: MemoryProfilerReportConfig = field(
        default_factory=MemoryProfilerReportConfig
    )

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "MemoryProfilerConfig":
        payload = _validate_config_section(
            payload,
            section="memory_profiler",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        ranks_payload = payload.get("ranks", [0])
        if isinstance(ranks_payload, str):
            ranks: str | list[int] = (
                "all"
                if ranks_payload.strip().lower() == "all"
                else [int(ranks_payload)]
            )
        elif isinstance(ranks_payload, int):
            ranks = [int(ranks_payload)]
        else:
            ranks = [int(rank) for rank in list(ranks_payload or [0])]
        return cls(
            enabled=bool(payload.get("enabled", False)),
            ranks=ranks,
            sync=bool(payload.get("sync", False)),
            log_every_n_steps=max(1, int(payload.get("log_every_n_steps", 10))),
            max_logged_steps=(
                None
                if payload.get("max_logged_steps") in (None, "")
                else max(0, int(payload.get("max_logged_steps")))
            ),
            reset_peak_stats_per_region=bool(
                payload.get("reset_peak_stats_per_region", False)
            ),
            log_gpu_utilization=bool(payload.get("log_gpu_utilization", False)),
            gpu_utilization_sample_interval_seconds=max(
                0.0,
                float(payload.get("gpu_utilization_sample_interval_seconds", 1.0)),
            ),
            output_subdir=str(payload.get("output_subdir", "memory") or "memory"),
            report=MemoryProfilerReportConfig.from_dict(payload.get("report")),
        )


@dataclass(slots=True)
class OfficialEvalConfig:
    """Continuous evaluation on dedicated vLLM server(s).

    When enabled, a separate worker process owns `server_gpus` (one vLLM
    server per GPU; scheduled rollouts split round-robin across them), runs
    every game `rollouts_per_game` times in parallel for `cycle_minutes` per
    cycle with the latest published adapter, and writes eval/* metrics at the
    trainer iteration the adapter came from. The trainer itself runs no eval
    windows.
    """

    enabled: bool = False
    games: list[str] = field(default_factory=list)
    dataset: str | list[str] | None = None
    server_gpus: list[str] = field(default_factory=list)
    # Actual pre-launched server logs, ordered like the server URLs.
    server_log_paths: list[str] = field(default_factory=list)
    # Explicit pre-launched eval vLLM server URLs, one per server_gpus entry
    # (declared like inference.base_url). Empty derives them from the
    # training serving contract instead: eval servers continue the
    # deterministic port sequence after the training servers (see
    # training.official_eval._eval_server_base_urls).
    base_urls: list[str] = field(default_factory=list)
    # Rollouts per game per cycle (decision 13: 2 halves the per-cycle noise
    # floor; copies share the game's fixed layout seed but decode stochastically).
    rollouts_per_game: int = 1
    cycle_minutes: float = 45.0
    # Eval-only context window (server max-model-len + agent budget). The
    # training 32k limit is a packed-sequence constraint that eval does not
    # share; None inherits the inference/training context.
    context_window_tokens: int | None = None
    # Eval can use a smaller context than training, requiring its own trim
    # target. None preserves the training/inference target.
    compact_to_tokens: int | None = None
    # Additional time marks (minutes) at which per-cycle stats are computed
    # from the SAME rollouts, e.g. [10.0] logs eval_10min/avg_arc_score --
    # progress under the training time budget without extra rollouts.
    stats_minutes: list[float] = field(default_factory=list)
    # Eval-only sampling params (deployment-style decoding, cuts cycle noise);
    # training rollouts keep inference.agent_* untouched.
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "OfficialEvalConfig":
        payload = _validate_config_section(
            payload,
            section="official_eval",
            allowed_fields=_dataclass_config_fields(cls),
            optional=True,
        )
        raw_context = payload.get("context_window_tokens")
        raw_compact = payload.get("compact_to_tokens")
        raw_marks = payload.get("stats_minutes", [])
        if raw_marks in (None, ""):
            raw_marks = []
        stats_minutes = sorted(
            {
                float(mark)
                for mark in (raw_marks if isinstance(raw_marks, list) else [raw_marks])
                if float(mark) > 0.0
            }
        )
        return cls(
            enabled=bool(payload.get("enabled", False)),
            games=_string_list(payload.get("games", [])),
            dataset=_arc2_dataset(payload.get("dataset"), section="official_eval"),
            server_gpus=_string_list(payload.get("server_gpus", [])),
            server_log_paths=_string_list(payload.get("server_log_paths", [])),
            base_urls=_string_list(payload.get("base_urls", [])),
            rollouts_per_game=max(1, int(payload.get("rollouts_per_game", 1))),
            cycle_minutes=float(payload.get("cycle_minutes", 45.0)),
            context_window_tokens=(
                None if raw_context in (None, "") else max(1, int(raw_context))
            ),
            compact_to_tokens=(None if raw_compact in (None, "") else int(raw_compact)),
            stats_minutes=stats_minutes,
            temperature=float(payload.get("temperature", 0.6)),
            top_p=float(payload.get("top_p", 0.95)),
            top_k=int(payload.get("top_k", 20)),
        )


@dataclass(slots=True)
class ExperimentConfig:
    num_iterations: int
    parallel_rollouts: int
    rollout_buffer_size: int
    inference: InferenceConfig
    training: TrainingConfig
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    official_eval: OfficialEvalConfig = field(default_factory=OfficialEvalConfig)
    memory_profiler: MemoryProfilerConfig = field(default_factory=MemoryProfilerConfig)

    def to_dict(self) -> dict[str, Any]:
        """Serialize a strict-loader-compatible effective configuration."""
        payload = asdict(self)
        # `game` is derived from the resolved `games` list for runtime
        # convenience; it is deliberately not part of the input schema.
        payload["inference"].pop("game", None)
        return payload

    @classmethod
    def load(cls, path: str | Path, *, gpu_claims: bool = True) -> "ExperimentConfig":
        """Load and validate a config. ``gpu_claims=False`` leaves the GPU-claim
        check (validate_gpu_claims) to a caller that renumbers the claims first:
        training.job_shape for a heterogeneous Flash job."""
        config_path = Path(path)
        payload = json.loads(
            config_path.read_text(encoding="utf-8"),
            parse_constant=_reject_non_finite_json_constant,
        )
        _reject_non_finite_numbers(payload)
        payload = _validate_config_section(
            payload,
            section="config",
            allowed_fields=_dataclass_config_fields(cls) | {"viewer"},
        )
        if "viewer" in payload:
            _validate_config_section(
                payload["viewer"],
                section="viewer",
                allowed_fields={"port"},
            )
        rollout_buffer_size = int(payload.get("rollout_buffer_size", 1))
        if rollout_buffer_size != 1:
            raise ValueError(
                "rollout_buffer_size is fixed at 1. RL training always uses one-window parallel rollout/training overlap."
            )
        inference_payload = payload["inference"]
        config = cls(
            num_iterations=int(payload.get("num_iterations", 1)),
            parallel_rollouts=int(payload.get("parallel_rollouts", 16)),
            rollout_buffer_size=1,
            inference=InferenceConfig.from_dict(inference_payload),
            training=TrainingConfig.from_dict(payload["training"]),
            metrics=MetricsConfig.from_dict(payload.get("metrics")),
            sampling=SamplingConfig.from_dict(payload.get("sampling")),
            official_eval=OfficialEvalConfig.from_dict(payload.get("official_eval")),
            memory_profiler=MemoryProfilerConfig.from_dict(
                payload.get("memory_profiler")
            ),
        )
        if not config.inference.platypus_venv:
            raise ValueError(
                "inference.platypus_venv must point at the arc3-platypus venv; "
                "rollout collection runs one platypus benchmark worker per window."
            )
        validate_training_sampling(config.inference)
        from training.harness import get_trainer_support

        get_trainer_support(config.inference.environment).validate_config(config)
        if config.training.advantage_estimator in {"grpo", "dr_grpo"}:
            per_window = config.sampling.game_sampling_games_per_window
            if per_window <= 0 or config.parallel_rollouts % per_window:
                raise ValueError(
                    "GRPO requires parallel_rollouts divisible by positive sampling.game_sampling_games_per_window."
                )
            if config.parallel_rollouts // per_window < 2:
                raise ValueError("GRPO requires at least two episodes per puzzle.")
            if config.sampling.game_sampling_mode != "balanced":
                raise ValueError(
                    "GRPO requires uniformly sampled distinct puzzles (balanced mode)."
                )
        if (
            config.inference.speculative_decoding.enabled
            and config.training.weight_mode != "flash"
        ):
            raise ValueError(
                "inference.speculative_decoding requires training.weight_mode='flash': "
                "its drafter is ullm's, for the Flash e307 checkpoint."
            )
        if config.training.weight_mode == "flash" and config.official_eval.enabled:
            # Flash publishes over the training servers' NCCL group only; a
            # dedicated eval fleet would need a group of its own.
            raise ValueError(
                "official_eval is not supported with training.weight_mode='flash': "
                "policies reach servers only through the learner's NCCL broadcast"
            )
        if config.training.weight_mode == "lora":
            if config.training.full_weight.device_ids:
                raise ValueError(
                    "training.full_weight.device_ids is only valid when "
                    "training.weight_mode='full_weight'."
                )
        elif config.training.weight_mode == "full_weight":
            full_weight_devices = config.training.full_weight.device_ids
            if not full_weight_devices:
                raise ValueError(
                    "training.full_weight.device_ids must be non-empty for "
                    "training.weight_mode='full_weight'."
                )
            if any(device_id < 0 for device_id in full_weight_devices):
                raise ValueError(
                    "training.full_weight.device_ids must contain non-negative GPU ids."
                )
            if len(set(full_weight_devices)) != len(full_weight_devices):
                raise ValueError(
                    "training.full_weight.device_ids must not repeat GPU ids."
                )
            if config.training.full_weight.checkpoint_every_iterations <= 0:
                raise ValueError(
                    "training.full_weight.checkpoint_every_iterations must be positive."
                )
            if config.training.full_weight.checkpoint_keep_last <= 0:
                raise ValueError(
                    "training.full_weight.checkpoint_keep_last must be positive."
                )
            if re.fullmatch(r"[0-9a-fA-F]{40}", config.training.model_revision) is None:
                raise ValueError(
                    "Full-weight training requires training.model_revision to be "
                    "an immutable 40-character commit SHA."
                )
            if (
                config.training.device_map_gpus
                or config.training.worker_device_map_gpus
            ):
                raise ValueError(
                    "training.device_map_gpus and training.worker_device_map_gpus "
                    "are LoRA-only; use training.full_weight.device_ids for "
                    "full-weight training."
                )
            if (
                config.training.init_adapter_dir
                or config.training.init_trainer_state_path
            ):
                raise ValueError(
                    "training.init_adapter_dir and training.init_trainer_state_path "
                    "are LoRA-only and must be unset for full-weight training."
                )
            if not config.training.bf16:
                raise ValueError(
                    "Full-weight training currently requires training.bf16=true."
                )
            if config.training.reference_kl_coef > 0.0:
                raise ValueError(
                    "training.reference_kl_coef currently uses disabled LoRA adapters "
                    "as the frozen reference and must be 0 for full-weight training."
                )
            training_device = str(config.training.device).strip()
            explicit_cuda_index = training_device.removeprefix("cuda:")
            if training_device != "cuda" and not (
                training_device.startswith("cuda:") and explicit_cuda_index.isdigit()
            ):
                raise ValueError(
                    "Full-weight training requires training.device='cuda' or "
                    "'cuda:<non-negative integer>'."
                )
        if config.training.device_map_gpus:
            expected_device = f"cuda:{config.training.device_map_gpus[0]}"
            if str(config.training.device).strip() != expected_device:
                raise ValueError(
                    "training.device must name the first entry of "
                    f"training.device_map_gpus ({expected_device}); got "
                    f"{config.training.device!r}."
                )
        for replica_index, replica in enumerate(config.training.worker_device_map_gpus):
            if not replica:
                raise ValueError(
                    "training.worker_device_map_gpus entries must be non-empty "
                    f"GPU lists; replica {replica_index} is empty."
                )
        if (
            config.training.worker_device_map_gpus
            and not config.training.device_map_gpus
        ):
            raise ValueError(
                "training.worker_device_map_gpus requires training.device_map_gpus "
                "for the rank-0 replica."
            )
        if (
            config.training.init_trainer_state_path
            and not config.training.init_adapter_dir
        ):
            raise ValueError(
                "training.init_trainer_state_path requires training.init_adapter_dir."
            )
        configured_games = [
            game for game in (config.inference.games or []) if str(game).strip()
        ]
        if (
            config.training.advantage_estimator == "state_critic"
            and config.training.value_baseline == "group_mean"
        ):
            games_per_window = config.sampling.game_sampling_games_per_window
            group_count = (
                min(games_per_window, len(configured_games))
                if games_per_window > 0
                else len(configured_games)
            )
            if config.parallel_rollouts < 2 * group_count:
                raise ValueError(
                    "Training without a critic requires at least two rollouts per game"
                )
        if config.sampling.game_sampling_mode == "nonzero_rate":
            per_window = config.sampling.game_sampling_games_per_window
            if per_window > len(configured_games):
                raise ValueError(
                    f"sampling.game_sampling_games_per_window={per_window} "
                    f"exceeds the {len(configured_games)} configured games."
                )
            if config.parallel_rollouts % per_window:
                raise ValueError(
                    "parallel_rollouts must be a multiple of "
                    "sampling.game_sampling_games_per_window so weighted "
                    "windows stay uniform (worker equal-multiset contract); "
                    f"got {config.parallel_rollouts} slots for "
                    f"{per_window} games per window."
                )
        elif 0 < config.sampling.game_sampling_games_per_window < len(configured_games):
            # Balanced mode with a per-window subset: the window holds
            # games_per_window distinct games, each an equal number of times.
            per_window = config.sampling.game_sampling_games_per_window
            if config.parallel_rollouts % per_window:
                raise ValueError(
                    "parallel_rollouts must be a multiple of "
                    "sampling.game_sampling_games_per_window so each window "
                    "holds an equal number of runs per game (worker "
                    f"equal-multiset contract); got {config.parallel_rollouts} "
                    f"slots for {per_window} games per window."
                )
        elif len(configured_games) > 1 and config.parallel_rollouts % len(
            configured_games
        ):
            # The remainder repeats random games once more; each repeat must
            # land on a different server so every child window holds unique
            # games (training.inference.split_slots_into_balanced_chunks).
            server_count = max(1, len(config.inference.server_gpus))
            copies = -(-config.parallel_rollouts // len(configured_games))
            if copies > server_count:
                raise ValueError(
                    "parallel_rollouts must be a multiple of the number of "
                    "inference.games unless the repeated games can be spread "
                    "over distinct servers (one copy per server): got "
                    f"{config.parallel_rollouts} slots for {len(configured_games)} "
                    f"games on {server_count} server(s), which needs up to "
                    f"{copies} copies of one game on one server."
                )
        # Loading historically checks only the learner ceiling here. Full-game
        # reply/compaction validation remains at rollout collection time.
        sequence_ceiling = max_served_sequence_tokens(config.inference)
        if (
            sequence_ceiling is not None
            and config.training.max_sequence_tokens < sequence_ceiling
        ):
            ceiling_label = (
                "inference.agent_context_window"
                if config.inference.agent_max_served_sequence_tokens is None
                else "inference.agent_max_served_sequence_tokens"
            )
            raise ValueError(
                "training.max_sequence_tokens="
                f"{config.training.max_sequence_tokens} is below "
                f"{ceiling_label}={sequence_ceiling}: "
                "a rollout segment may legally grow to the served ceiling, so a "
                "smaller training budget guarantees a fatal packing error "
                "mid-run. Use one max context for both sides."
            )
        if config.official_eval.enabled:
            if not config.official_eval.games and config.official_eval.dataset is None:
                raise ValueError(
                    "official_eval.games must be non-empty when official_eval is enabled."
                )
            if not config.official_eval.server_gpus:
                raise ValueError(
                    "official_eval.server_gpus must be non-empty when official_eval is enabled."
                )
            if config.official_eval.cycle_minutes <= 0.0:
                raise ValueError("official_eval.cycle_minutes must be positive.")
        if gpu_claims:
            validate_gpu_claims(config)
        return config


def _gpu_claim(field: str, raw_ids: list[Any], *, non_negative: bool) -> tuple[str, set[int]]:
    """Parse one config field's GPU ids, rejecting non-integers and repeats."""
    ids: list[int] = []
    for raw in raw_ids:
        try:
            ids.append(int(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} entries must be integer GPU ids; got {raw!r}.") from exc
    if non_negative and any(gpu < 0 for gpu in ids):
        raise ValueError(f"{field} must contain non-negative GPU ids.")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{field} must not repeat GPU ids.")
    return field, set(ids)


def validate_gpu_claims(config: "ExperimentConfig") -> None:
    """Every process that owns GPUs claims a disjoint set of one absolute index space.

    Rollout servers, the trainer (its device map, LoRA worker replicas, or the
    torchrun ranks of full-weight and Flash training) and official evaluation
    are listed unconditionally, so a claim is checked whether or not the
    others are configured.
    """
    mode = config.training.weight_mode
    torchrun = config.training.uses_torchrun
    server_gpus = config.inference.server_gpus
    if config.inference.server_allocation == "separate":
        # These indices belong to other allocations. Validate each entry but
        # permit the same local index on multiple nodes and on the learner.
        for gpu in server_gpus:
            _gpu_claim("inference.server_gpus", [gpu], non_negative=True)
        server_gpus = []
    elif mode == "full_weight" and not server_gpus:
        server_gpus = ["0"]  # the launcher's default rollout GPU
    claims = [
        _gpu_claim("inference.server_gpus", server_gpus, non_negative=torchrun),
        _gpu_claim("training.device_map_gpus", config.training.device_map_gpus, non_negative=False),
        *(
            _gpu_claim(f"training.worker_device_map_gpus[{index}]", replica, non_negative=False)
            for index, replica in enumerate(config.training.worker_device_map_gpus)
        ),
    ]
    if torchrun:
        claims.append(_gpu_claim(f"training.{mode}.device_ids", config.training.torchrun_device_ids, non_negative=True))
    if config.official_eval.enabled:
        claims.append(_gpu_claim("official_eval.server_gpus", config.official_eval.server_gpus, non_negative=torchrun))
    owner: dict[int, str] = {}
    for name, ids in claims:
        overlap = sorted(gpu for gpu in ids if gpu in owner)
        if overlap:
            raise ValueError(
                f"{owner[overlap[0]]} and {name} must claim disjoint GPUs; "
                f"GPUs {overlap} are claimed twice."
            )
        owner.update(dict.fromkeys(ids, name))
