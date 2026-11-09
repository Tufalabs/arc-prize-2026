"""The training deployment states its malformed-reply policy in the agent
configuration: zero re-samples and a backend that keeps an empty reply as
the model's turn. Both travel with the agent, never with the environment."""

import ullm.backends.deployment as deployment
import ullm.backends.oai
import ullm.factory
from ullm.factory import PrebuiltBackend

from duck import presets
from duck.agent import DuckAgent


def _agent(*, accept_malformed: bool = False, **kwargs: object) -> DuckAgent:
    # No server behind this URL: the recurrent-state blank (on by default)
    # would ask it which vLLM it runs at construction, so it stays off here.
    agent = DuckAgent.build(
        engine="qwen3.8-27b",
        environment="slurm-b200",
        derive_max_model_len=False,
        derive_max_num_seqs=False,
        derive_max_pixels=False,
        backend_factory=ullm.factory.PrebuiltBackend(
            backend=ullm.backends.oai.OpenAIServer(
                base_url="http://127.0.0.1:9/v1", model="policy", **deployment.client_kwargs(presets.DEFAULT_ENGINE)
            )
        ),
        zero_recurrent=False,
        **kwargs,  # type: ignore[arg-type]
    )
    if accept_malformed:
        factory = agent.llm_config.backend_factory
        assert isinstance(factory, PrebuiltBackend)
        factory.backend.accept_malformed_output = True
    return agent


def _accepts_malformed(agent: DuckAgent) -> bool:
    factory = agent.llm_config.backend_factory
    assert isinstance(factory, PrebuiltBackend)
    return bool(factory.backend.accept_malformed_output)


def test_deployment_defaults_keep_the_retry_protocol() -> None:
    agent = _agent()
    assert agent.malformed_retries == 5
    assert _accepts_malformed(agent) is False


def test_training_states_zero_retries_and_accepted_replies() -> None:
    agent = _agent(malformed_retries=0, accept_malformed=True)
    assert agent.malformed_retries == 0
    assert _accepts_malformed(agent) is True
