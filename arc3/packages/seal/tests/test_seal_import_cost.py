"""What a game child pays to hold the seal.

Every game runs in a spawned child that unpickles the agent, which runs
this module's imports — and rebuilds whatever the agent's fields hold —
in that child's own heap. So a top-level import here, or a host-side
object left on the agent, is a per-game memory cost, paid by every game
of a submission at once and never freed. vllm with the torch under it is
most of what there was to pay, for an engine the child never calls:
inference goes through a proxy client to the host process.

``platypus.subproc`` asserts the child never has vllm, so a regression
fails every subprocess run rather than quietly costing a run its memory.
These tests are the cheaper half of the same guard — they name the whole
heavy set, and they fail on the import alone without needing a game.

Each probe runs in a fresh interpreter, since the test session has these
modules imported already.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from pathlib import Path

import pytest

# What a game child must never carry: vllm with the torch under it, for
# an engine only the host runs; the plotting and vision chain, for
# rendering no move performs. The chain arrives through taaf, which loads
# its submodules on first use — asserted from this side too, because the
# cost lands here and the pin is ours to move.
FORBIDDEN = ("vllm", "torch", "matplotlib", "scipy", "arc_agi", "imageio", "PIL")


def _top_level_modules(statement: str) -> set[str]:
    """Module names live in a fresh interpreter after ``statement`` — the
    full names plus their top-level roots, so a caller can ask either way."""
    probe = f"import sys; {statement}; print(' '.join(sorted(sys.modules)))"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    names = completed.stdout.split()
    return {*names, *(name.split(".")[0] for name in names)}


def test_importing_the_agent_carries_no_engine() -> None:
    loaded = _top_level_modules("import seal")
    carried = sorted(loaded.intersection(FORBIDDEN))
    assert not carried, (
        f"importing seal pulled {carried} into every game child. Move the import inside the "
        "function that needs it (see seal.agent._installed_vllm)."
    )


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_the_deferred_imports_still_resolve() -> None:
    # The counterpart: made lazy, not dropped. The engine probe serves the
    # zero_recurrent and server-build checks, so all this asks of it is that
    # the deferred import lands on the real thing.
    from seal.agent import _installed_vllm

    assert _installed_vllm().__version__


@pytest.mark.filterwarnings("ignore:Can't initialize NVML")
@pytest.mark.filterwarnings("ignore:CUDA initialization")
def test_the_agent_a_child_receives_carries_no_engine_spec(tmp_path: Path) -> None:
    """The path that actually costs the memory: not `import seal`, but
    unpickling the agent. A configured seal's ``llm_config.backend_factory``
    holds vllm objects, so a child handed the agent whole imports vllm to
    rebuild them however lazy this module's own imports are. The solver
    drops the spec on the way in, and this replays that from a blob in a
    fresh interpreter, which is what the child does.
    """
    from platypus import LLMAgentSolver
    from platypus.llm import LLMAgent
    from seal import SealAgent

    agent = SealAgent.build()
    solver = LLMAgentSolver(agent=agent)
    child_agent = solver._child_agent(agent)  # pyright: ignore[reportPrivateUsage]
    assert isinstance(child_agent, LLMAgent)
    # The template keeps its spec: the host starts the engine from it.
    assert agent.llm_config.backend_factory is not None
    assert child_agent.llm_config.backend_factory is None

    blob = tmp_path / "agent.pkl"
    blob.write_bytes(pickle.dumps(child_agent))
    probe = f"import pickle, sys; pickle.loads(open({str(blob)!r}, 'rb').read()); print(' '.join(sorted(sys.modules)))"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    loaded = {name.split(".")[0] for name in completed.stdout.split()}
    carried = sorted(loaded.intersection(FORBIDDEN))
    assert not carried, f"unpickling the child's agent pulled {carried} into every game child"


def test_an_arm_without_images_never_loads_the_renderer() -> None:
    # The legend renderer pulls in PIL and the bundled font; it is imported
    # inside `_palette_elements`, which only an image-bearing prompt calls, so
    # a prompt built with the image off must leave both unloaded.
    statement = (
        "from seal import SealAgent; "
        "agent = SealAgent(zero_recurrent=False, multimodal_context=''); "
        "agent._build_user_prompt_element(None)"
    )
    loaded = _top_level_modules(statement)
    assert "seal.palette_legend" not in loaded
    assert "PIL" not in loaded
