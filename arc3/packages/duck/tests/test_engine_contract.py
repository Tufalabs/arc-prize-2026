"""Supplied engine opt-outs and live window validation fail before games start."""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import ullm.backend
import ullm.backends.stub
import ullm.factory

from duck.agent import DuckAgent
from platypus import LLMAgentSolver, LLMConfig
from platypus.solver import AgentSolver

FLAGS = ("derive_max_model_len", "derive_max_num_seqs", "derive_max_pixels")


@pytest.mark.parametrize("flag", FLAGS)
def test_each_derivation_requires_its_own_supplied_engine_opt_out(flag: str) -> None:
    factory = ullm.factory.PrebuiltBackend(backend=ullm.backends.stub.StubBackend())
    choices: dict[str, Any] = {name: name == flag for name in FLAGS}
    with pytest.raises(ValueError, match=rf"{flag}=True.*engine isn't built here"):
        DuckAgent.build(backend_factory=factory, **choices)
    choices[flag] = False
    agent = DuckAgent.build(backend_factory=factory, **choices)
    assert agent.llm_config.backend_factory is factory


@dataclass
class ConnectedBackend(ullm.backends.stub.StubBackend):
    engine: Any = field(default_factory=lambda: SimpleNamespace(model_config=None))
    window: int = 65536
    closed: bool = False
    probes: int = 0

    async def probe_continuation(self) -> None:
        self.probes += 1
        self.engine.model_config = SimpleNamespace(max_model_len=self.window)

    def supports_token_continuation(self) -> bool:
        return self.engine.model_config is not None

    async def close(self) -> None:
        self.closed = True
        await super().close()


@pytest.mark.parametrize("window", [16384, 32768, 65536])
@pytest.mark.parametrize("exact_history", [False, True])
async def test_live_window_is_checked_after_connection_and_failure_closes_backend(
    monkeypatch: pytest.MonkeyPatch, window: int, exact_history: bool
) -> None:
    reached: list[bool] = []
    closed: list[ConnectedBackend] = []
    original_close = ConnectedBackend.close

    async def close(backend: ConnectedBackend) -> None:
        closed.append(backend)
        await original_close(backend)

    monkeypatch.setattr(ConnectedBackend, "close", close)

    async def games(self: AgentSolver, games: Any) -> None:
        reached.append(True)

    monkeypatch.setattr(AgentSolver, "_run_games", games)
    opt_outs: dict[str, Any] = {flag: False for flag in FLAGS}
    agent = DuckAgent(
        compaction_strategy="prune",  # the window contract; a summary's is test_summary_fits_engine_window
        reply_cap_tokens=4096,
        context_window_tokens=32768,
        compact_to_tokens=8192,
        llm_config=LLMConfig(
            backend_factory=ullm.factory.PrebuiltBackend(backend=ConnectedBackend(window=window), close_backend=True),
            exact_history=exact_history,
        ),
        **opt_outs,
    )
    solver = LLMAgentSolver(agent=agent)
    if window < 32768:
        with pytest.raises(ValueError, match="exceeds the connected engine's max_model_len"):
            await solver._run_games([])
        assert reached == []
    else:
        await solver._run_games([])
        assert reached == [True]
    assert closed and all(backend.closed for backend in closed)
    assert all(backend.probes == 1 for backend in closed)
    assert solver._host_backend is None


def test_live_engine_must_report_its_window() -> None:
    agent = DuckAgent()
    with pytest.raises(ValueError, match="must report a positive max_model_len"):
        agent._validate_host_backend(ConnectedBackend())
    agent._validate_host_backend(ullm.backends.stub.StubBackend())


async def test_shared_engine_must_fit_the_widest_sweep_arm() -> None:
    from sweeper.agent import LLMSweepAgent
    from sweeper.knobs import Choice, Design, Field, Knob

    design = Design(arm={"window": Knob(draw=Choice((32768, 65536)), apply=Field("inner.context_window_tokens"))})
    sweep = LLMSweepAgent(
        inner=DuckAgent(
            compaction_strategy="prune", reply_cap_tokens=4096, context_window_tokens=32768, compact_to_tokens=8192
        ),
        design=design,
        arm_seeds=tuple(range(12)),
        run_seed=0,
        announce=False,
    )
    assert {design.arm_values_for(seed)["window"] for seed in sweep.arm_seeds} == {32768, 65536}
    backend = ConnectedBackend(window=65536)
    await backend.probe_continuation()
    sweep._validate_host_backend(backend)
    backend.window = 32768
    await backend.probe_continuation()
    with pytest.raises(ValueError, match="context_window_tokens=65536"):
        sweep._validate_host_backend(backend)
