"""Select harness adapters without importing either harness in the trainer."""

from importlib import import_module
from typing import Any, Protocol


class HarnessSupport(Protocol):
    """The task-specific surface consumed by the shared rollout worker."""

    def build_benchmark(self, spec: dict[str, Any]) -> Any:
        """Construct tasks using the selected harness's public API."""
        ...

    def enrich_results(
        self,
        spec: dict,
        runs: list[dict],
        originals: list[dict],  # noqa: F841 - used by harness implementations
    ) -> None:
        """Translate outcomes into the shared gateway result contract."""
        ...

    def progress_snapshot(self, game_run: Any, *, start_level: int = 0) -> dict:
        """Read live task progress for request attribution."""
        ...


class TrainerHarnessSupport(Protocol):
    """The task-specific surface consumed by the shared trainer process.

    Implementations are pure Python (no harness package imports). They only
    translate between one harness's conventions and the shared contract;
    gateway verification, packing, algorithms and checkpointing stay shared.
    """

    MEDIA: dict[str, bool]
    REQUIRES_BEHAVIOR_POLICY_SNAPSHOT: bool
    DEADLINE_TERMINAL_STATES: frozenset[str]

    def validate_config(self, config: Any) -> None:
        """Reject recipe settings the harness cannot honor."""
        ...

    def evaluation_context_overrides(
        self, inference: Any, evaluation: Any
    ) -> dict[str, int]:
        """Derive context-dependent harness limits for an evaluation window."""
        ...

    def resolve_datasets(self, config: Any) -> dict[str, Any] | None:
        """Pin the task set before scheduling; None when games are explicit."""
        ...

    def training_identity(
        self, config: Any, manifest: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Harness facts a checkpoint is bound to, without the media identity."""
        ...

    def validate_run_result(self, run: Any) -> None:
        """Independently check a reported outcome; raise ValueError on defects."""
        ...

    def rollout_identity(self, rollouts: Any) -> dict[str, Any] | None:
        """Harness settings a window's rollouts report that the checkpoint
        identity pins (training/rollout_identity.py); None when none are."""
        ...

    def run_to_trajectory(self, window: Any, run: Any, **kwargs: Any) -> Any:
        """Translate a verified run and its captures into a RolloutTrajectory."""
        ...


ENVIRONMENTS = ("arc3", "arc2")


def _require_environment(environment: str) -> None:
    if environment not in ENVIRONMENTS:
        raise ValueError(f"Unsupported harness: {environment!r}")


def get_harness(environment: str) -> HarnessSupport:
    """Load only the selected adapter in its independent worker environment."""
    _require_environment(environment)
    return import_module(f"harness_support.{environment}.worker")


def get_trainer_support(environment: str) -> TrainerHarnessSupport:
    """Load the selected harness's trainer-side translations."""
    _require_environment(environment)
    return import_module(f"harness_support.{environment}.trainer")


def harness_engine(training) -> str:
    """The duck engine preset a rollout worker builds its agent with.

    Flash trains the deployed Qwen3.8-Flash-Next checkpoint, so its rollouts
    use the harness preset that model shipped with (no served opening text);
    every other recipe keeps the 27B preset.
    """
    return "qwen3.8-flash-next" if getattr(training, "weight_mode", "") == "flash" else "qwen3.8-27b"
