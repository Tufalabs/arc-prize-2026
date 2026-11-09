"""Tool definitions for ULLM. Tools combine definition (metadata for the LLM) and
handler (execution logic) in a single class that extends Processor."""

# TODO: be more strutural about making sure tool states are cleared

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import ullm.conv
import ullm.support as uls
import ullm.tool_helpers

# jsonschema is imported lazily inside the two call sites below: importing it
# costs ~0.8s (it pulls rfc3987-syntax/lark through its format checker when
# those are installed), which every process that touches ullm would otherwise
# pay at startup — including every spawned game subprocess.


@dataclass
class Tool(uls.Processor):
    """Base class for tools. Combines LLM-facing metadata (name, description, parameters)
    with execution logic (_execute). Extends Processor for state machine lifecycle.

    Subclasses must implement _execute() and may override clear_state() for resource cleanup.

    Lifecycle per call: set arguments -> run() -> read result -> rearm()
    State management: TurnManager calls clear_state() based on clear_state_on setting."""

    # Definition (metadata for the LLM)
    name: str = ""
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())  # JSON Schema

    # State management
    # TODO: clear_state semantics are inconsistent across tool subclasses —
    # for some (e.g. DummyTool) it's a between-call/turn reset, for others
    # (e.g. PythonTool) it's the FINAL teardown. The 'run'/'turn' options
    # only make sense under the reset interpretation; under the teardown
    # interpretation they would destroy the tool after one use. Likely fix:
    # split into a separate `reset_state_on` knob (between-call recycling)
    # and reserve `clear_state` for final teardown across the board.
    clear_state_on: Literal["run", "turn", "never"] = "turn"

    # Execution state (set before run, available after)
    arguments: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    result: ullm.conv.Content = field(default_factory=ullm.conv.EmptyContent, init=False)
    is_error: bool = field(default=False, init=False)

    # Warnings accumulated during the last run, drained by the caller (e.g.
    # TurnManager) after `run()` completes. Each entry is a dict with at least
    # a 'type' field, matching Backend.log_warning's contract. Tools that want
    # to surface a warning should append here; they don't need a Backend ref.
    _pending_warnings: list[dict[str, Any]] = field(
        default_factory=lambda: list[dict[str, Any]](), init=False, repr=False
    )

    # The task inside which ``run()`` is executing, recorded so ``shutdown()``
    # can cancel a call that is still in flight. Only ever set by ``_run``, and
    # only meaningful while the state is "running".
    _running_task: "Optional[asyncio.Task[Any]]" = field(default=None, init=False, repr=False)

    # Permit argument-name/JSON repairs and tool-specific escape heuristics.
    # False leaves the decoded model arguments unchanged before validation.
    auto_correct_arguments: bool = True

    def _repaired_text(self, value: Any) -> Any:
        """Apply the tool's escape-repair policy to newly written text."""
        if self.auto_correct_arguments:
            return ullm.tool_helpers.heuristic_fix_double_escapes(value)
        return value

    def pop_pending_warnings(self) -> list[dict[str, Any]]:
        """Return and clear the pending-warnings buffer."""
        out = self._pending_warnings
        self._pending_warnings = list[dict[str, Any]]()
        return out

    def _check_constraints(self) -> None:
        """The declared parameter schema must itself be a valid JSON Schema."""
        import jsonschema.validators

        jsonschema.validators.validator_for(self.parameters).check_schema(self.parameters)
        super()._check_constraints()

    def _maybe_unwrap_raw_json_object(self) -> None:
        """If arguments is `{"_raw": "<json object string>"}`, replace it with
        the parsed object. Handles cases where the backend's JSON-parse
        fallback wrapped the model's output as `_raw`, but the model's output
        is *itself* a valid JSON object — e.g., the model double-stringified
        its arguments, so `tc.arguments = '"{\\"path\\":\\"x\\",\\"content\\":\\"y\\"}"'`.
        Without this, the schema-validation would fail because the model's
        actual intended arguments are trapped inside the _raw string.

        Only fires when there's exactly one argument named `_raw` whose value
        parses as a JSON *object* (dict). Strings, numbers, arrays, etc. are
        left alone — the single-arg remap path handles those."""
        if len(self.arguments) != 1 or "_raw" not in self.arguments:
            return
        raw = self.arguments["_raw"]
        if not isinstance(raw, str):
            return
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return
        if isinstance(parsed, dict):
            self.arguments = parsed

    def _maybe_remap_single_arg(self) -> None:
        """If the schema has exactly one required property and the model
        supplied exactly one argument, route it to the required property
        regardless of the name the model used. This accommodates small models
        that invent key names (e.g. `_raw`, `input`, or the tool's own name)
        instead of the declared parameter, and the JSON-parse fallback in
        backends which wraps bad JSON as `{"_raw": raw_text}`.

        Single-arg calls with the right name pass through untouched. Multi-arg
        calls and tools with multiple required params are unaffected."""
        required = self.parameters.get("required", [])
        if len(required) == 1 and len(self.arguments) == 1:
            expected_name = required[0]
            actual_name = next(iter(self.arguments))
            if actual_name != expected_name:
                self.arguments = {expected_name: self.arguments[actual_name]}

    def argument_error(self, arguments: dict[str, Any]) -> Optional[str]:
        """Validate a call without running its body or altering sampled arguments."""
        import jsonschema

        try:
            jsonschema.validate(instance=arguments, schema=self.parameters)
        except jsonschema.ValidationError as exc:
            return f"Invalid arguments for tool '{self.name}': {exc.message}"
        return None

    async def _run(self) -> None:
        """Run one call: optionally repair arguments, validate their schema,
        then execute.
        """
        self._running_task = asyncio.current_task()
        self.is_error = False
        if self.auto_correct_arguments:
            self._maybe_unwrap_raw_json_object()
            self._maybe_remap_single_arg()
        elif "_raw" in self.arguments and "_raw" not in self.parameters.get("properties", {}):
            # A backend's unparsed-JSON fallback must not execute an all-optional
            # tool just because its schema permits unknown properties.
            self.is_error = True
            self.result = ullm.conv.StringContent(
                text=f"Invalid arguments for tool '{self.name}': expected a JSON object matching the tool schema."
            )
            return
        # Validate arguments against JSON Schema before entering the body.
        invalid = self.argument_error(self.arguments)
        if invalid is not None:
            self.is_error = True
            self.result = ullm.conv.StringContent(text=invalid)
            return
        self.result = await self._execute(self.arguments)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Execute the tool with the given arguments. Subclasses must implement this."""
        raise NotImplementedError("Subclasses must implement _execute()")

    async def clear_state(self) -> None:
        """Release persistent resources (kill subprocess, end subagent, etc.)."""
        assert not self.state == "running", "Cannot clear state while tool is running"
        await self._clear_state()

    def __getstate__(self) -> dict[str, Any]:
        """State for pickle and deepcopy, minus the live task handle.

        A running task is not state: a copy of a tool is never the copy that
        is executing, and an ``asyncio.Task`` is not serialisable in any
        case. Subclasses that override this must preserve the same rule.
        """
        return {**self.__dict__, "_running_task": None}

    async def shutdown(self) -> None:
        """Final teardown, safe to call with a call still in flight.

        ``clear_state`` refuses to run against a live call, for good reason: a
        tool that is mid-execution owns resources it is still using, and
        freeing them underneath itself is a race. But a caller tearing down
        after its work is over — an agent whose game ended while a cell was
        still executing, say — has no way to reach that call and no reason to
        keep it. Left alone it leaks whatever ``_clear_state`` would have
        freed, which for a subprocess-backed tool is a live process, a
        temporary directory and a concurrency slot.

        So: cancel the in-flight call, wait for it to finish unwinding — at
        which point ``Processor.run`` has moved the state to "cancelled" —
        and only then clear. Nothing is forced and no assertion is bypassed;
        the state the assertion asks for is reached properly.
        """
        task = self._running_task
        if self.state == "running" and task is not None and not task.done():
            task.cancel()
            # The cancellation belongs to that task, not to ours: shield the
            # wait so a caller who is themselves being cancelled still sees
            # the teardown through.
            await asyncio.wait({task})
        await self.clear_state()

    async def _clear_state(self) -> None:
        """Nothing to reset; stateful subclasses override this."""
        pass


@dataclass
class DummyTool(Tool):
    """Computes 3*x + 2*y + N, where N is the number of calls since the last state clear."""

    name: str = "dummy_compute"
    description: str = "Computes 3*x + 2*y + N, where N is the number of calls since last state clear."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "x": {"type": "integer", "description": "First input"},
                "y": {"type": "integer", "description": "Second input"},
            },
            "required": ["x", "y"],
        }
    )
    _call_count: int = field(default=0, init=False, repr=False)

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """The example tool's body: an arithmetic rule with one special case the
        model is meant to discover by calling it.
        """
        x = arguments["x"]
        y = arguments["y"]
        if x == -5 and y == 3:
            # For testing: LLM's job is to infer this
            return ullm.conv.StringContent(
                text="x=-5 and y=3 is not allowed, you have to figure that one out by yourself! But you can try other values of x and y..."
            )
        result = 3 * x + 2 * y + self._call_count
        self._call_count += 1
        return ullm.conv.StringContent(text=str(result))

    async def _clear_state(self) -> None:
        """Forget how many times this tool has been called."""
        self._call_count = 0
