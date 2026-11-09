"""I9/U5: backend extensions run on the host after run attribution is wired."""

from dataclasses import dataclass, field
from typing import Any

import taaf.game_examples
import ullm.backend
import ullm.backends.proxy
import ullm.backends.stub

from duck.agent import DuckAgent
from platypus import LLMAgentSolver


@dataclass
class ConfiguredSolver(LLMAgentSolver):
    """A consumer that configures the supplied client through the public hook."""

    seen: list[Any] = field(default_factory=lambda: list[Any]())

    def configure_game_backend(
        self, game: Any, host_backend: ullm.backend.Backend, client: ullm.backends.proxy.ProxyBackend
    ) -> None:
        self.seen.append((game, host_backend, client.run_id))
        client.replay_on_link_loss = False


def test_public_hook_configures_the_attributed_client() -> None:
    game = taaf.game_examples.ExampleGame()
    game.start_game()
    backend = ullm.backends.stub.StubBackend()
    solver = ConfiguredSolver(
        agent=DuckAgent(
            reply_cap_tokens=None,
            compaction_strategy="prune",
        )
    )
    solver._socket_path = "/unused/test.sock"
    solver._host_backend = backend
    client = solver._game_llm_backend(game)
    assert isinstance(client, ullm.backends.proxy.ProxyBackend)
    assert game.game_run is not None
    assert solver.seen == [(game, backend, game.game_run.run_id)]
    assert not client.replay_on_link_loss
