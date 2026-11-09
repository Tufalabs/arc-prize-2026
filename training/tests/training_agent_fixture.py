"""Compose the shipped Duck and shared training backend in contract tests."""

from training_backend import configure_training_agent
from duck.agent import DuckAgent
from serving_contract import TrainingContextObserver
from ullm.backends.engine_proxy import EngineProxy
from ullm.backend import GenericSamplingParameters
from training_backend import qwen_engine_factory, training_backend_factory


def agent_for_training(
    cls,
    *,
    base_url="http://127.0.0.1:1",
    model="policy",
    **kwargs,
):
    """Return an unused training agent without opening the remote connection."""
    traces = kwargs.pop("traces", True)
    reply_cap = kwargs.pop("max_completion_tokens", 4096)
    run_config = kwargs.pop("run_config", None)
    max_concurrent = kwargs.pop("max_concurrent", None)
    if hasattr(cls, "derive_max_pixels"):
        kwargs.setdefault("derive_max_pixels", False)
    from duck import presets

    agent = cls.build(
        backend_factory=qwen_engine_factory(
            EngineProxy(base_url=base_url, model=model),
            engine=kwargs.get("engine", presets.DEFAULT_ENGINE),
        ),
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        reply_cap_tokens=reply_cap,
        **kwargs,
    )
    if run_config is not None:
        agent.run_config = run_config
    if max_concurrent is not None:
        agent.run_config.max_concurrent = max_concurrent
    return configure_training_agent(
        agent,
        backend_factory=training_backend_factory(
            source_factory=agent.llm_config.backend_factory
        ),
        sampling=GenericSamplingParameters(
            temperature=1.0,
            top_p=1.0,
            extra_params={
                "top_k": -1,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "repetition_penalty": 1.0,
                "min_p": 0.0,
                "min_tokens": 0,
                "ignore_eos": False,
                "skip_special_tokens": False,
                "logprobs": 0,
            },
        ),
        traces=traces,
    )


def context_agent(**kwargs):
    """Use the ordinary Duck with public context controls and a training observer."""
    kwargs.setdefault("reply_cap_tokens", 512)
    return DuckAgent(
        file_tools="none",
        compact_before_request_only=True,
        require_exact_context=True,
        context_observer=TrainingContextObserver(),
        **kwargs,
    )
