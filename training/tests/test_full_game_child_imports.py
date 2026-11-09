"""The real per-game payload must unpickle without host inference libraries."""

import os
import pickle
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("duck.agent")
pytest.importorskip("vllm")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from training_agent_fixture import context_agent  # noqa: E402

from serving_contract import (  # noqa: E402
    CompactionTokenCount,
    ContractLLMAgentSolver,
)
from platypus.subproc import _ChildPayload  # noqa: E402


def test_full_game_payload_passes_real_child_import_guard(tmp_path):
    agent = context_agent(multimodal_context="", token_native_context=True)
    from ullm.backends.vllmx import VLLM
    from ullm.backends.engine_proxy import EngineProxy

    solver = ContractLLMAgentSolver(agent=agent)
    solver._host_backend = VLLM(
        engine=EngineProxy(base_url="http://unused", model="policy")
    )
    solver._socket_path = str(tmp_path / "host.sock")
    solver._trace_collector = None
    client = solver._game_llm_backend(
        SimpleNamespace(game_run=SimpleNamespace(run_id="test"))
    )
    assert client.replay_on_link_loss is False
    child = solver._child_agent(agent)
    assert child.llm_config.backend_factory is None
    payload = _ChildPayload(
        agent_blob=pickle.dumps(child),
        game_info=None,
        initial_state=None,
        number_of_levels=None,
        base_actions_per_level=None,
        rng_seed=42,
        memory_limit_bytes=0,
        llm_backend=client,
    )
    blob = tmp_path / "payload.pkl"
    blob.write_bytes(pickle.dumps((payload, CompactionTokenCount(3, [1, 2, 3]))))
    program = textwrap.dedent("""
        import asyncio, importlib.abc, pickle, sys
        from pathlib import Path
        class NoHostImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if (fullname == 'vllm' or fullname.startswith('vllm.')
                    or fullname in {'ullm.backends.engine_proxy', 'ullm.backends.vllmx',
                                    'ullm.backends.vllm_factory'}):
                    raise AssertionError('game child attempted host import: ' + fullname)
        sys.meta_path.insert(0, NoHostImports())
        import platypus.subproc as subproc
        from platypus.solver import AgentSolver
        payload, count = pickle.loads(Path(sys.argv[1]).read_bytes())
        assert type(payload.llm_backend).__name__ == 'ProxyBackend'
        assert payload.llm_backend.replay_on_link_loss is False
        assert int(count) == 3 and count.prompt_token_ids == (1, 2, 3)
        # Run the actual child preamble, including its second agent unpickle
        # and vLLM exclusion assertion. Stop before game/diagnostic setup.
        class PreamblePassed(Exception): pass
        def stop_before_game_setup(*args): raise PreamblePassed()
        AgentSolver._diag_logger = staticmethod(stop_before_game_setup)
        try:
            asyncio.run(subproc._child_main(None, None, payload))
        except PreamblePassed:
            pass
        else:
            raise AssertionError('child preamble was not exercised')
        assert 'vllm' not in sys.modules
        assert 'ullm.backends.engine_proxy' not in sys.modules
        print('child preamble and proxy/count unpickle passed')
    """)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "scripts"), str(ROOT), *[item for item in sys.path if item]]
    )
    env["MPLCONFIGDIR"] = str(tmp_path / "matplotlib")
    result = subprocess.run(
        [sys.executable, "-c", program, str(blob)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "child preamble and proxy/count unpickle passed" in result.stdout
