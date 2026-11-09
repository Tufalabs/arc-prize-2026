"""Trainer task metadata over Platypus's shared local/remote vLLM backend."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import time
from typing import Any, TYPE_CHECKING

from platypus import LLMAgentSolver
from ullm.backends.proxy import ProxyBackend

if TYPE_CHECKING:
    from ullm.backends.engine_proxy import EngineProxy, EngineProxyFactory


def qwen_engine_factory(proxy: EngineProxy, *, engine: str) -> EngineProxyFactory:
    """Render requests around the trainer's engine transport as the harness
    renders them for ``engine`` (a Duck engine preset, ullm's
    ``deployment.ENGINES``): its parsers and template options are ullm's
    ``backend_kwargs`` for that engine, never another engine's."""
    from ullm.backends.deployment import backend_kwargs
    from ullm.backends.engine_proxy import EngineProxyFactory

    return EngineProxyFactory(engine=proxy, backend_kwargs=backend_kwargs(engine))


def training_sampling(sampling: dict[str, Any]):
    """State the trainer's sampling distribution without inherited processors."""
    from ullm.backend import GenericSamplingParameters

    return GenericSamplingParameters(
        max_tokens=None,
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        extra_params={
            "top_k": int(sampling.get("top_k") or -1),
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "min_p": 0.0,
            "min_tokens": 0,
            "ignore_eos": False,
            "skip_special_tokens": False,
            "logprobs": 0,
        },
    )


def training_backend_factory(*, source_factory: Any):
    """Enable training guards on the shipped remote-engine backend."""
    from ullm.backends.engine_proxy import EngineProxy, EngineProxyFactory
    from ullm.backends.vllmx import VLLM

    if not isinstance(source_factory, EngineProxyFactory) or not isinstance(
        source_factory.engine, EngineProxy
    ):
        raise TypeError("training requires the shipped remote engine backend")
    if source_factory.started:
        raise RuntimeError("training must configure the backend before startup")
    backend = VLLM(engine=source_factory.engine, **source_factory.backend_kwargs)
    expected = {
        "accept_malformed_output": False,
        "require_explicit_sampling": False,
        "verify_engine_output": False,
        "token_count_evidence": False,
        "replay_on_link_loss": True,
    }
    for name, value in expected.items():
        if getattr(backend, name) is not value:
            raise ValueError(
                f"shipped backend {name} changed before training configuration"
            )
    return replace(
        source_factory,
        backend_kwargs={
            **source_factory.backend_kwargs,
            "accept_malformed_output": True,
            "require_explicit_sampling": True,
            "verify_engine_output": True,
            "token_count_evidence": True,
            "replay_on_link_loss": False,
        },
    )


# The agent fields that decide what the engine serves ahead of the model's
# reply (ullm ``response_injection``), as the harness resolves them.
PROMPT_INJECTION_FIELDS = (
    "inject_text_on_first_prompt",
    "inject_text_on_later_prompts",
    "prompt_injection_schedule",
)


def prompt_injection_settings(agent: Any) -> dict[str, str]:
    """The built agent's prompt injection, for every run's agent settings."""
    settings = {name: getattr(agent, name) for name in PROMPT_INJECTION_FIELDS}
    if any(not isinstance(value, str) for value in settings.values()):
        raise TypeError(f"the harness's prompt injection is not text: {settings!r}")
    return settings


def validate_training_backend(agent: Any) -> None:
    """Require the shared single-attempt engine client before host startup."""
    from ullm.backends.engine_proxy import EngineProxy, EngineProxyFactory
    from ullm.backends.vllmx import VLLM

    factory = agent.llm_config.backend_factory
    if not isinstance(factory, EngineProxyFactory) or not isinstance(
        factory.engine, EngineProxy
    ):
        raise TypeError("training requires the shared remote engine backend")
    if factory.started:
        raise RuntimeError(
            "training transport must be configured before backend startup"
        )
    backend = VLLM(engine=factory.engine, **factory.backend_kwargs)
    if backend.classify_tool_calls != getattr(agent, "classify_tool_calls", False):
        raise ValueError(
            "training requires matching backend and agent classify_tool_calls"
        )
    if (
        getattr(agent, "preserve_failed_draws", False)
        and not backend.accept_malformed_output
    ):
        raise ValueError(
            "training preserve_failed_draws requires backend accept_malformed_output"
        )
    if (
        not backend.accept_malformed_output
        or not backend.require_explicit_sampling
        or not backend.verify_engine_output
        or not backend.token_count_evidence
        or backend.replay_on_link_loss
    ):
        raise ValueError(
            "training requires verified single-attempt generation and retained malformed replies"
        )
    for name, params in agent.llm_config.sampling_profiles.items():
        if params.stop_sequences is not None or "stop" in (params.extra_params or {}):
            raise ValueError(
                f"training does not support stop sequences in profile {name!r}"
            )


@dataclass
class TrainingLLMAgentSolver(LLMAgentSolver):
    """Attach live task feedback to the local host's shared vLLM backend."""

    _progress_games: dict[str, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    environment: str = "arc3"

    def _request_context(
        self, run_id: str, previous_request_id: str | None
    ) -> dict[str, Any]:
        """Bind observed score progress to the preceding engine generation."""
        from training.harness import get_harness

        game = self._progress_games[run_id]
        started = game.game_run.started_at_monotonic
        if started is None:
            raise RuntimeError("training progress requires the live game start clock")
        snapshot = get_harness(self.environment).progress_snapshot(
            game.game_run, start_level=int(getattr(game, "start_level", 0))
        )
        if "after_nonce" in snapshot:
            raise RuntimeError(
                "task progress must not assign its own generation identity"
            )
        if (
            previous_request_id is not None
            and previous_request_id.rpartition("#")[0] != run_id
        ):
            raise RuntimeError("task progress refers to another game's generation")
        snapshot["after_nonce"] = (
            previous_request_id.rpartition("#")[2] if previous_request_id else None
        )
        snapshot["data"]["decision_elapsed_seconds"] = time.monotonic() - started
        return {"arc3_progress": snapshot}

    def configure_game_backend(self, game: Any, host_backend: Any, client: Any) -> None:
        """Attach task progress through the harness's public host-side hook."""
        from ullm.backends.engine_proxy import EngineProxy
        from ullm.backends.vllmx import VLLM

        if not isinstance(client, ProxyBackend):
            raise TypeError("training requires a per-game proxy backend")
        if (
            not isinstance(host_backend, VLLM)
            or not isinstance(host_backend.engine, EngineProxy)
            or game.game_run is None
        ):
            raise TypeError(
                "training progress requires a remote engine host and live game run"
            )
        self._progress_games[game.game_run.run_id] = game
        host_backend.request_context_provider = self._request_context
        client.replay_on_link_loss = False


def configure_training_agent(
    agent: Any,
    *,
    backend_factory: Any,
    sampling: Any,
    traces: bool = True,
    max_concurrent: int | None = None,
) -> Any:
    """Apply shared experiment settings through public harness configuration.

    The prompt injection (``inject_text_on_first_prompt``,
    ``inject_text_on_later_prompts``, ``prompt_injection_schedule``) is left
    as the harness preset resolved it for the engine: training serves what
    the harness serves, and the workers report the resolved values with
    every run.
    """
    import copy
    from dataclasses import replace
    from serving_contract import TrainingContextObserver

    if sampling.seed is not None:
        raise ValueError("training sampling must leave per-call seeds to the game RNG")
    if sampling.temperature is None or sampling.top_p is None:
        raise ValueError("training sampling requires explicit temperature and top_p")
    if sampling.response_injection is not None:
        raise ValueError("training response injection must be scheduled by the agent")
    if sampling.send_back_reasoning not in (None, "all"):
        raise ValueError("training requires complete reasoning history")
    if agent.llm_config.replay_dir is not None or agent.zero_recurrent:
        raise ValueError("training requires live requests with recurrent history")
    if max_concurrent is not None and max_concurrent <= 0:
        raise ValueError("training max_concurrent must be positive")
    stated = copy.deepcopy(sampling)
    stated.send_back_reasoning = "all"
    agent.llm_config = replace(
        agent.llm_config,
        backend_factory=backend_factory,
        sampling_profiles={**agent.llm_config.sampling_profiles, agent.profile: stated},
        traces=traces,
        exact_history=True,
    )
    agent.run_config = replace(
        agent.run_config,
        min_request_gap_s=0.0,
        max_concurrent=agent.run_config.max_concurrent
        if max_concurrent is None
        else max_concurrent,
    )
    agent.token_native_context = True
    agent.compact_before_request_only = True
    agent.require_exact_context = True
    # The reply cap is what a summary reset's ceiling is widened against when
    # the request runs under a smaller remaining episode budget.
    agent.context_observer = TrainingContextObserver(
        reply_cap_tokens=int(agent.reply_cap_tokens)
    )
    return agent
