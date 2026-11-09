"""The ``status_hook`` seam: the phases a host cannot see from outside a
call (request rendering, generation, output parsing, queueing for a kernel
pool slot) naming themselves, and never travelling with the object that
reported them."""

from __future__ import annotations

import contextlib
import copy
import pickle
from typing import Any, Generator

import ullm.backend
import ullm.backends.stub
import ullm.conv
import ullm.manager
import ullm.tools.python


def _recording_hook(events: list[tuple[str, str]]) -> Any:
    """A hook logging entry and exit, so a test can assert one scope closed
    before another opened rather than only that both fired."""

    @contextlib.contextmanager
    def hook(label: str) -> Generator[None, None, None]:
        events.append(("enter", label))
        try:
            yield
        finally:
            events.append(("exit", label))

    return hook


async def test_prepare_generation_and_parse_each_name_themselves() -> None:
    # A caller awaiting respond() cannot see where inside it the time goes:
    # rendering the prompt and parsing the output are local CPU work, and
    # only the middle is the model. Each phase names itself — and the
    # pre/postprocessing scopes must CLOSE around the generation wait, so
    # host-side CPU time never counts as engine time.
    events: list[tuple[str, str]] = []
    backend = ullm.backends.stub.StubBackend.from_script("hello")
    backend.status_hook = _recording_hook(events)
    conversation = ullm.conv.Conversation(
        system_prompt=ullm.conv.SystemPrompt(content=ullm.conv.StringContent(text="sys"))
    )
    conversation.elements.append(ullm.conv.UserPrompt(content=ullm.conv.StringContent(text="go")))
    response_manager = ullm.manager.ResponseManager(
        backend=backend,
        conversation=conversation,
        sampling_params=ullm.backend.GenericSamplingParameters(temperature=0.0, max_tokens=16),
    )
    await ullm.manager.TurnManager(response_manager=response_manager, conversation=conversation).run()

    assert events == [
        ("enter", "preprocessing LLM response"),
        ("exit", "preprocessing LLM response"),
        ("enter", "waiting for LLM response"),
        ("exit", "waiting for LLM response"),
        ("enter", "postprocessing LLM response"),
        ("exit", "postprocessing LLM response"),
    ]


async def test_the_kernel_queue_is_reported_apart_from_the_execution() -> None:
    # Only the waiter knows which pool slot it is queued for; a host counting
    # the semaphores sees a total, not which game is stuck on what. The
    # queue scope must also CLOSE before the run scope opens — otherwise
    # every executing kernel is also counted as a queued one.
    events: list[tuple[str, str]] = []
    tool = ullm.tools.python.PythonTool(status_hook=_recording_hook(events))
    try:
        tool.arguments = {"code": "print(6 * 7)"}
        await tool.run()
        assert not tool.is_error and "42" in tool.result.to_string()
    finally:
        await tool.clear_state()

    run_opens = events.index(("enter", "running kernel"))
    assert events[run_opens:] == [("enter", "running kernel"), ("exit", "running kernel")]
    assert events[:run_opens], "the pool slot was taken without ever being reported as a wait"
    assert all(label == "waiting for kernel slot" for _, label in events[:run_opens])
    assert ("exit", "waiting for kernel slot") == events[run_opens - 1]


def test_a_wired_hook_never_travels_with_the_tool() -> None:
    # The hook closes over the process that wired it (a control pipe, a
    # registry) and is typically a lambda, which does not pickle at all —
    # so an agent holding a wired tool would stop satisfying A1.
    tool = ullm.tools.python.PythonTool(status_hook=lambda label: contextlib.nullcontext())
    assert pickle.loads(pickle.dumps(tool)).status_hook is None
    assert copy.deepcopy(tool).status_hook is None


def test_a_wired_hook_never_travels_with_the_backend() -> None:
    backend = ullm.backends.stub.StubBackend.from_script("hi")
    backend.status_hook = lambda label: contextlib.nullcontext()
    assert pickle.loads(pickle.dumps(backend)).status_hook is None
