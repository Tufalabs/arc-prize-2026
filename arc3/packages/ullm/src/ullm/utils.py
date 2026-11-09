"""Utility classes for ullm."""

import asyncio
import atexit
import os
import signal
import subprocess
import sys
import threading
import urllib.error
import urllib.request
import weakref
from dataclasses import dataclass, field
from typing import IO, Any, Literal, Optional

import ullm.backend
import ullm.conv
import ullm.manager
import ullm.support as uls
import ullm.tool

# Global registry of active servers, so we can clean them up on exit
_active_servers: list["VLLMServer"] = []


def _cleanup_all_servers() -> None:
    """Terminate all active servers on program exit."""
    for server in list(_active_servers):
        server.stop()


atexit.register(_cleanup_all_servers)


def make_turn_manager(
    backend: ullm.backend.Backend,
    system_prompt: Optional[str] = "You are a helpful assistant.",
    tools: Optional[list[ullm.tool.Tool]] = None,
    # The full SystemPrompt set: this wrapper narrowing it (it once dropped
    # "max") silently rejected levels the conversation itself accepts.
    reasoning_effort: Optional[Literal["low", "medium", "high", "max", "xhigh"]] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    extra_params: Optional[dict[str, Any]] = None,
    max_tool_rounds: Optional[int] = None,
) -> ullm.manager.TurnManager:
    """Create a TurnManager with sensible defaults. Convenience wrapper that builds
    the Conversation, SamplingParameters, ResponseManager, and TurnManager stack
    from simple Python types. Access the conversation via the returned TurnManager's
    .conversation attribute."""
    # Build system prompt, omitting None fields so constructor defaults apply
    sp_kwargs = dict[str, Any]()
    if system_prompt is not None:
        sp_kwargs["content"] = ullm.conv.StringContent(text=system_prompt)
    if reasoning_effort is not None:
        sp_kwargs["reasoning_effort"] = reasoning_effort
    if tools is not None:
        sp_kwargs["tools"] = tools

    # Build sampling parameters, omitting None fields
    sampling_kwargs = dict[str, Any]()
    if temperature is not None:
        sampling_kwargs["temperature"] = temperature
    if max_tokens is not None:
        sampling_kwargs["max_tokens"] = max_tokens
    if top_p is not None:
        sampling_kwargs["top_p"] = top_p
    if seed is not None:
        sampling_kwargs["seed"] = seed
    if extra_params is not None:
        sampling_kwargs["extra_params"] = extra_params

    conversation = ullm.conv.Conversation(system_prompt=ullm.conv.SystemPrompt(**sp_kwargs))
    sampling_params = ullm.backend.GenericSamplingParameters(**sampling_kwargs)

    # Assemble the manager stack
    response_manager = ullm.manager.ResponseManager(backend=backend, sampling_params=sampling_params)
    tm_kwargs = dict[str, Any](response_manager=response_manager, conversation=conversation)
    if max_tool_rounds is not None:
        tm_kwargs["max_tool_rounds"] = max_tool_rounds
    turn_manager = ullm.manager.TurnManager(**tm_kwargs)
    return turn_manager


@dataclass
class VLLMServer(uls.BaseClass):
    """Manages a vllm serve process. Call start() to launch, or use start_all() for parallel startup.
    The server is stopped when stop() is called, the object is garbage collected, or the program exits."""

    model: str = field()  # Model name to serve (e.g. "Qwen/Qwen2-0.5B-Instruct")
    port: int = field(default=8000)  # Port to serve on
    extra_args: list[str] = field(default_factory=list)  # Additional args to pass to vllm serve
    # A last-resort bound: a cold compile of a large model runs into the tens
    # of minutes, and a bound that fires on one turns a slow start into a
    # failed run.
    startup_timeout: float = field(default=3600.0)  # Seconds to wait for server to become healthy
    tee_to_parent: bool = field(default=False)  # Also write subprocess stdout/stderr lines to parent stdout/stderr
    _process: Optional[subprocess.Popen] = field(init=False, default=None, repr=False)  # type: ignore
    _finalizer: Optional[weakref.finalize] = field(init=False, default=None, repr=False)
    _stdout_log: list[str] = field(init=False, default_factory=list, repr=False)  # captured stdout lines
    _stderr_log: list[str] = field(init=False, default_factory=list, repr=False)  # captured stderr lines

    def launch(self) -> None:
        """Spawn the vllm serve subprocess. Returns immediately; call wait_healthy() to block until ready."""
        if self._process is not None:
            raise RuntimeError("Server already running")

        # Launch vllm serve in its own process group so we can kill it cleanly
        cmd = ["vllm", "serve", self.model, "--port", str(self.port)] + self.extra_args
        self._process = subprocess.Popen(
            cmd,
            preexec_fn=os.setsid,  # new process group
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start background threads to capture stdout/stderr
        self._stdout_log.clear()
        self._stderr_log.clear()
        out_tee = sys.stdout if self.tee_to_parent else None
        err_tee = sys.stderr if self.tee_to_parent else None
        threading.Thread(
            target=_read_stream,
            args=(self._process.stdout, self._stdout_log),
            kwargs={"tee": out_tee, "prefix": f"[vllm {self.port} stdout] "},
            daemon=True,
        ).start()
        threading.Thread(
            target=_read_stream,
            args=(self._process.stderr, self._stderr_log),
            kwargs={"tee": err_tee, "prefix": f"[vllm {self.port} stderr] "},
            daemon=True,
        ).start()

        # Register cleanup via weak reference (handles garbage collection)
        self._finalizer = weakref.finalize(self, _terminate_process_group, self._process.pid)

        # Register in global set for atexit cleanup
        _active_servers.append(self)

    async def wait_healthy(self) -> None:
        """Poll the health endpoint until the server is ready. Raises on timeout or process death."""
        if self._process is None:
            raise RuntimeError("Server not launched; call launch() first")

        health_url = f"http://localhost:{self.port}/health"
        elapsed = 0.0
        while elapsed < self.startup_timeout:
            # Check if process died
            if self._process.poll() is not None:
                returncode = self._process.returncode
                self._process = None
                if self in _active_servers:
                    _active_servers.remove(self)
                # Give reader threads a moment to flush remaining output
                import time

                time.sleep(0.5)
                raise RuntimeError(
                    f"vllm serve ({self.model}) exited with code {returncode}\n"
                    f"stdout:\n{chr(10).join(self._stdout_log)}\n"
                    f"stderr:\n{chr(10).join(self._stderr_log)}"
                )
            # Check health endpoint
            try:
                with urllib.request.urlopen(health_url, timeout=2) as resp:
                    if resp.status == 200:
                        return  # server is ready
            except (urllib.error.URLError, ConnectionError, OSError):
                pass
            # Yield control to the event loop
            await asyncio.sleep(1)
            elapsed += 1.0

        # Timed out; kill and raise
        self.stop()
        raise TimeoutError(
            f"vllm serve ({self.model}) did not become healthy within {self.startup_timeout}s\n"
            f"stdout:\n{chr(10).join(self._stdout_log)}\n"
            f"stderr:\n{chr(10).join(self._stderr_log)}"
        )

    async def start(self) -> None:
        """Launch the subprocess and wait until healthy. Convenience for launch() + wait_healthy()."""
        self.launch()
        await self.wait_healthy()

    def stop(self) -> None:
        """Stop the vllm serve process and its children."""
        if self._process is not None:
            _terminate_process_group(self._process.pid)
            self._process = None
        if self._finalizer is not None:
            self._finalizer.detach()
            self._finalizer = None
        if self in _active_servers:
            _active_servers.remove(self)

    @property
    def base_url(self) -> str:
        """The OpenAI-compatible API base URL for this server."""
        return f"http://localhost:{self.port}/v1"

    @property
    def running(self) -> bool:
        """Whether the server process is currently alive."""
        return self._process is not None and self._process.poll() is None

    @staticmethod
    async def start_all(*servers: "VLLMServer") -> None:
        """Start multiple servers in parallel."""
        await asyncio.gather(*(s.start() for s in servers))


@dataclass
class StreamingVisualizer(uls.Processor):
    """Runs a processor while displaying streaming output from a TurnManager.

    The processor is the task being run (any Processor). peek_into is the target
    for intermediate output display — either a TurnManager directly, or a
    ProcessorMulti whose children are TurnManagers (peeks the first running one).
    Defaults to processor when not provided.

    Display modes:
      - "none": no display, only updates intermediate_string for external clients
      - "jupyter": clears and reprints full output each update (for Jupyter notebooks)
      - "print_incremental": prints only new characters as they appear (for terminals)
    """

    processor: uls.Processor = field(kw_only=True)  # the processor to run and wait on
    peek_into: Optional[uls.Processor] = field(
        kw_only=True, default=None
    )  # TurnManager or ProcessorMulti; defaults to processor
    update_interval: float = field(default=0.2)  # seconds between peek updates
    display_mode: Literal["none", "jupyter", "print_incremental"] = field(default="print_incremental")
    intermediate_string: str = field(default="", init=False, repr=False)  # current display text, always updated
    _printed_len: int = field(default=0, init=False, repr=False)  # characters already printed (for print_incremental)

    def __post_init__(self) -> None:
        if self.peek_into is None:
            self.peek_into = self.processor
        super().__post_init__()

    def _active_turn_manager(self) -> Optional[ullm.manager.TurnManager]:
        """Resolve peek_into to the currently active TurnManager."""
        if isinstance(self.peek_into, ullm.manager.TurnManager):
            return self.peek_into
        if isinstance(self.peek_into, uls.ProcessorMulti):
            # Return the first running TurnManager child
            for child in self.peek_into.children:
                if isinstance(child, ullm.manager.TurnManager) and child.state == "running":
                    return child
            # Fallback: last TurnManager child (for final display)
            for child in reversed(self.peek_into.children):
                if isinstance(child, ullm.manager.TurnManager):
                    return child
        return None

    async def _run(self) -> None:
        """Run the processor with streaming visualization."""
        # Print the user prompt
        if self.display_mode != "none":
            tm = self._active_turn_manager()
            if tm is not None and tm.conversation.elements:
                last_element = tm.conversation.elements[-1]
                if isinstance(last_element, ullm.conv.UserPrompt):
                    print(f"\n> {last_element.content.to_string()}\n", flush=True)

        # Run processor and peek loop concurrently
        async with asyncio.TaskGroup() as tg:
            tg.create_task(self.processor.run())
            tg.create_task(self._display_loop())

        # Final update after the processor completes
        self._update(final=True)
        if self.display_mode == "print_incremental":
            print(flush=True)  # trailing newline

    def _rearm(self) -> None:
        """Reset display state and rearm the processor."""
        self._printed_len = 0
        self.intermediate_string = ""
        self.processor.rearm()

    async def _display_loop(self) -> None:
        """Periodically peek the active turn manager and update display."""
        try:
            while self.processor.state == "running":
                tm = self._active_turn_manager()
                if tm is not None and tm.state == "running":
                    tm.peek()
                self._update(final=False)
                await asyncio.sleep(self.update_interval)
        finally:
            # Ensure the final render happens even if we're cancelled mid-sleep
            # (e.g. TaskGroup sibling errored, KeyboardInterrupt). Without this,
            # the last partial stays on screen and the final output is lost.
            self._update(final=True)

    def _update(self, final: bool) -> None:
        """Update intermediate_string and display based on mode."""
        self.intermediate_string = self._get_display_text(final)

        if self.display_mode == "jupyter":
            from IPython.display import Markdown, clear_output, display  # type: ignore

            clear_output(wait=True)
            display(Markdown(self.intermediate_string.replace("\n", "\n\n")))
        elif self.display_mode == "print_incremental":
            if len(self.intermediate_string) > self._printed_len:
                new_text = self.intermediate_string[self._printed_len :]
                sys.stdout.write(new_text)
                sys.stdout.flush()
                self._printed_len = len(self.intermediate_string)

    def _get_display_text(self, final: bool) -> str:
        """Build the displayable text from the active turn manager's conversation."""
        tm = self._active_turn_manager()
        if tm is None:
            return ""

        conversation = tm.conversation
        if not conversation.elements:
            return ""

        last = conversation.elements[-1]
        if not isinstance(last, ullm.conv.ResponseTurn):
            return ""

        parts = list[str]()

        # Show completed elements in the last turn (LLM responses, tool results, etc.)
        completed_text = last.to_string()
        if completed_text:
            parts.append(completed_text)

        # Intermediate output (from peeking) — only if not final
        if not final:
            rm = tm.response_manager
            if rm._intermediate_output is not None:
                intermediate_text = rm._intermediate_output.to_string()
                if intermediate_text:
                    parts.append(intermediate_text)

        return "\n".join(parts)


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract the contents of the last \\boxed{} in the text, handling nested braces.
    Returns None if no \\boxed{} is found."""
    # Find the last occurrence of \boxed{
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    # Walk forward from the opening brace, tracking nesting depth
    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        return None  # unbalanced braces
    return text[start : i - 1]


def _read_stream(stream: Optional[IO[bytes]], log: list[str], tee: Optional[IO[str]] = None, prefix: str = "") -> None:
    """Read lines from a subprocess stream into a log list. Runs in a background thread.
    If ``tee`` is set, each line is also written (with ``prefix``) to that text stream."""
    if stream is None:
        return
    for line in stream:
        decoded = line.decode(errors="replace").rstrip("\n")
        log.append(decoded)
        if tee is not None:
            try:
                tee.write(prefix + decoded + "\n")
                tee.flush()
            except Exception:
                pass  # don't let a closed/broken parent stream kill the reader thread
    stream.close()


def _terminate_process_group(pid: int) -> None:
    """Terminate an entire process group by PID. Safe to call if already dead."""
    try:
        os.killpg(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
