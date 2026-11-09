"""DuckPythonTool: the duck's python tool as a self-contained ullm Tool.

Wraps ullm's ``PythonTool`` so TurnManager can drive it directly: the
model's snippet is syntax-checked host-side, the kernel is reset before
the call (process-per-call semantics), the code runs as
``_duck_run('<code>')`` under the duck prelude, and the kernel's
``__DUCK_PAYLOAD__`` JSON line is re-rendered into the original duck's
tool-result format (stdout/error/result selection, ``planned_actions``
compaction, per-field truncation with a note).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional, cast

import ullm.conv
import ullm.tools.python

from duck import prompts
from duck.kernel_code import PRELUDE, SESSION_LISTING_CODE, SESSION_LISTING_MARKER

_PAYLOAD_MARKER = "__DUCK_PAYLOAD__"


class SessionInspectionError(RuntimeError):
    """The kernel did not produce its optional compaction inventory."""


def _duck_tool_parameters() -> dict[str, Any]:
    """The duck's ``python`` tool schema (one ephemeral ``code`` string)."""
    return {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": prompts.PYTHON_TOOL_CODE_PARAM_DESCRIPTION},
        },
        "required": ["code"],
    }


@dataclass
class DuckPythonTool(ullm.tools.python.PythonTool):
    """The duck's ``python`` tool. ``tool_output_tokens`` mirrors the
    original's response-budget knob (x4 = chars); ullm-level output
    truncation is disabled (``max_output_chars`` huge) because the whole
    kernel output is one payload line that must survive parsing."""

    name: str = "python"
    description: str = prompts.PYTHON_TOOL_DESCRIPTION
    parameters: dict[str, Any] = field(default_factory=_duck_tool_parameters)
    initial_code: Optional[str] = PRELUDE
    max_output_chars: int = 50_000_000
    tool_output_tokens: int = 1024
    # True = the original's process-per-call sandbox: module state, globals
    # and tmpdir are dropped before every snippet. False keeps them, and the
    # AGENT owns the reset cadence instead (DuckAgent.persistence).
    reset_before_each_call: bool = True

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """One ephemeral snippet: reset kernel, run under ``_duck_run``,
        render the payload in the original's format. ``is_error`` mirrors
        the rendered ``error`` field (what the model is told)."""
        code = str(arguments.get("code", "")).rstrip()
        if not code:
            self.is_error = True
            return ullm.conv.StringContent(
                text=json.dumps({"error": "python requires a non-empty `code` string."}, indent=2)
            )
        try:
            compile(code, "<python_tool>", "exec")
        except SyntaxError as exc:
            self.is_error = True
            return ullm.conv.StringContent(text=json.dumps({"error": f"Python syntax error: {exc}"}, indent=2))

        # Fresh kernel per call (~50 ms): module state and globals go, not
        # just the namespace. (_reset_impl: Tool.state is "running" here,
        # but no kernel operation is in flight.) Skipped when the agent
        # keeps state for longer than one snippet; the reset then happens
        # at its boundary. The working directory's CONTENTS survive every
        # reset — see DuckAgent._reset_repl for why.
        if self.reset_before_each_call:
            self._reset_impl(keep_tmpdir_contents=True)
        # ullm's kernel timeout is a wall-clock deadline over the whole
        # call, host-call time included (the original sandbox's deadline
        # semantics) — a snippet looping on the selected action function
        # cannot outlive it.
        persist = not self.reset_before_each_call
        record_start = len(self._execution_records)
        content = await super()._execute({"code": f"_duck_run({code!r}, persist={persist!r})"})
        kernel_errored = self.is_error
        sandbox_result = self._parse_kernel_payload(content.to_string(), tool_errored=kernel_errored)

        raw_results = cast("list[Any]", sandbox_result.get("action_results") or [])
        action_results: list[dict[str, Any]] = [item for item in raw_results if isinstance(item, dict)]
        payload: dict[str, Any] = {"tool": "python"}
        rendered_stdout = str(sandbox_result.get("stdout", "") or "")
        rendered_error = str(sandbox_result.get("error", "") or "")
        if rendered_error:
            # _duck_run renders model exceptions into its own successful JSON
            # response, so the shared PythonTool cannot see this exhaustion.
            rendered_error = self._restart_if_memory_exhausted(rendered_error)
        if rendered_error:
            payload["error"] = rendered_error
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
        else:
            payload["returncode"] = 0
            if rendered_stdout:
                payload["stdout"] = rendered_stdout
            elif sandbox_result.get("result") is not None:
                payload["result"] = sandbox_result.get("result")
            elif action_results:
                if len(action_results) == 1:
                    payload["result"] = action_results[-1]
                else:
                    payload["result"] = {
                        "action_calls": len(action_results),
                        "last_action_result": action_results[-1],
                    }
        self.is_error = bool(rendered_error)
        # PythonTool records whether its worker RPC failed.  Duck user-code
        # exceptions are caught inside _duck_run and returned through a
        # successful RPC, so align those records with the final ToolResult the
        # model receives.  Preflight failures above never entered the kernel
        # and intentionally have no kernel-exec record.
        for record in self._execution_records[record_start:]:
            record["is_error"] = self.is_error
        return ullm.conv.StringContent(
            text=self._render_tool_payload(payload, truncate_fields=("stdout", "error", "result"))
        )

    def working_files(self) -> list[tuple[str, int]]:
        """The kernel's working directory, as ``(name, bytes)`` pairs sorted
        by name; a subdirectory is listed by its name with a trailing ``/``
        and a size of -1. Read host-side: the directory is this tool's. The
        import system's ``__pycache__`` is not the model's and is left out."""
        tmpdir = self._tmpdir
        if tmpdir is None or not os.path.isdir(tmpdir):
            return []
        entries: list[tuple[str, int]] = []
        for name in sorted(os.listdir(tmpdir)):
            if name == "__pycache__":
                continue
            path = os.path.join(tmpdir, name)
            if os.path.isdir(path):
                entries.append((f"{name}/", -1))
            elif os.path.isfile(path):
                entries.append((name, os.path.getsize(path)))
        return entries

    async def session_objects(self) -> list[tuple[str, str, bool]]:
        """What the model bound in its persistent ``python`` session, as
        ``(name, description, is_function)`` in binding order; empty when no
        session outlives a call or none has started.

        A host call between rounds, while the kernel is idle: it is not a tool
        call and reaches no transcript, so its execution record is dropped
        and the tool's error flag is left as it was."""
        if self.reset_before_each_call or self.is_fresh:
            return []
        was_error = self.is_error
        records = len(self._execution_records)
        try:
            content = await ullm.tools.python.PythonTool._execute(self, {"code": SESSION_LISTING_CODE})
        finally:
            del self._execution_records[records:]
            self.is_error = was_error
        for line in reversed(content.to_string().splitlines()):
            if line.startswith(SESSION_LISTING_MARKER):
                parsed = cast("list[list[Any]]", json.loads(line[len(SESSION_LISTING_MARKER) :]))
                return [(str(name), str(text), bool(is_function)) for name, text, is_function in parsed]
        raise SessionInspectionError(f"the session listing returned no listing: {content.to_string()[:500]!r}")

    def _parse_kernel_payload(self, output_text: str, *, tool_errored: bool) -> dict[str, Any]:
        """Recover the ``__DUCK_PAYLOAD__`` JSON line from kernel output; map
        kernel-level failures (timeout, crash) onto the error payload shape."""
        for line in reversed(output_text.splitlines()):
            if line.startswith(_PAYLOAD_MARKER):
                try:
                    parsed: Any = json.loads(line[len(_PAYLOAD_MARKER) :])
                    if isinstance(parsed, dict):
                        return {str(key): value for key, value in cast("dict[Any, Any]", parsed).items()}
                except json.JSONDecodeError:
                    break
        # No payload line: kernel-level failure (timeout, crash, interrupt).
        text = output_text.strip() or "Python tool produced no output."
        if tool_errored and ("timeout" in text.lower() or "timed out" in text.lower()):
            text = f"Tool timed out after {int(self.timeout)}s"
        return {"error": text, "stdout": "", "action_results": []}

    def _trim_tool_text(self, text: str) -> tuple[str, bool]:
        """The original's tool-output cap: hard truncation at
        tool_output_tokens*4 chars with an omitted-count marker."""
        max_chars = max(256, self.tool_output_tokens * 4)
        if len(text) <= max_chars:
            return text, False
        omitted = len(text) - max_chars
        return f"{text[:max_chars]}\n... [truncated {omitted} chars]", True

    def _summarize_planned_actions(self, value: Any) -> Any:
        """The original's payload compaction: any ``planned_actions`` list
        collapses to its count before rendering."""
        if isinstance(value, dict):
            mapping = cast("dict[Any, Any]", value)
            compacted: dict[Any, Any] = {key: self._summarize_planned_actions(item) for key, item in mapping.items()}
            planned_actions = compacted.pop("planned_actions", None)
            if isinstance(planned_actions, list):
                compacted["planned_action_count"] = len(cast("list[Any]", planned_actions))
            return compacted
        if isinstance(value, list):
            return [self._summarize_planned_actions(item) for item in cast("list[Any]", value)]
        return value

    def _render_tool_payload(self, payload: dict[str, Any], *, truncate_fields: tuple[str, ...] = ()) -> str:
        """Render the tool-result JSON the model sees, with per-field
        truncation and a truncation note (the original's format)."""
        result: dict[str, Any] = self._summarize_planned_actions(dict(payload))
        truncated = False
        for field_name in truncate_fields:
            value = result.get(field_name)
            if isinstance(value, str):
                result[field_name], field_truncated = self._trim_tool_text(value)
                truncated = truncated or field_truncated
        if truncated:
            result["truncated"] = True
            result["truncation_note"] = (
                f"Tool output was cut off to stay within the ~{self.tool_output_tokens}-token response budget."
            )
        return json.dumps(result, indent=2)
