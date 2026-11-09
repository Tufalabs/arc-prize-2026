"""The second productive path: importing what the model wrote.

The file tools are half of the surface. The other half is that a file in the
working directory is a MODULE — the kernel runs there, so a helper the model
wrote is one ``import`` away, and the analysis it spent a turn building is
reusable instead of retyped.

Three things have to hold for that to be true rather than nearly true: the
import guard has to permit it, an edited file has to be what the next import
sees, and neither may cost the sandbox its shape — a workspace file must not
be able to answer an import the allowlist meant for the standard library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from duck.python_tool import DuckPythonTool


def _state(*, file_tools: bool) -> dict[str, Any]:
    """A minimal runtime-state payload: these rows only care about imports."""
    return {
        "frame_api": "ergonomic",
        "file_tools": file_tools,
        "timeline": [],
        "valid_actions": [],
        "last_action_result": {},
    }


class _Kernel:
    """A persistent duck kernel and the directory its files live in."""

    def __init__(self, *, file_tools: bool = True) -> None:
        self.tool = DuckPythonTool(timeout=30.0, reset_before_each_call=False)

        # No row here executes an action; the acting host call is present
        # only because the prelude requires one to exist.
        def _action(normalized: list[dict[str, Any]]) -> dict[str, Any]:
            return {"ack": {"queued": len(normalized)}}

        self.tool.host_functions = {
            "get_state": lambda: _state(file_tools=file_tools),
            "action": _action,
        }

    @property
    def directory(self) -> Path:
        """Where a file has to be written to be importable."""
        assert self.tool._tmpdir is not None  # pyright: ignore[reportPrivateUsage] — the test writes where it runs
        return Path(self.tool._tmpdir)  # pyright: ignore[reportPrivateUsage] — same

    async def run(self, code: str) -> str:
        """One snippet, as the model's tool call would run it."""
        content = await self.tool._execute({"code": code})
        return content.to_string()

    async def close(self) -> None:
        """Drop the kernel and its directory."""
        await self.tool.clear_state()


async def test_a_file_the_model_wrote_is_importable() -> None:
    """The whole point of the surface: work written to a file comes back as a
    module instead of being retyped into the next snippet."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        out = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in out, out
    finally:
        await kernel.close()


async def test_an_edited_file_is_what_the_next_import_sees() -> None:
    """The failure this closes is silent: without eviction the model edits a
    file, imports it again, gets the cached module and reasons at length about
    code it has already replaced."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        first = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in first, first

        kernel.directory.joinpath("helper.py").write_text("AREA = 100\n")
        second = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "100" in second, second
    finally:
        await kernel.close()


async def test_files_outlive_the_kernel_they_were_written_from() -> None:
    """A reset drops definitions, not work. Every persistence mode resets the
    kernel somewhere, and one that also wiped the directory would delete the
    model's files without ever saying so."""
    kernel = _Kernel()
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        await kernel.run("carried = 1\nimport helper\n")
        kernel.tool.reset()
        out = await kernel.run("print(carried)\n")
        assert "NameError" in out, f"the reset must drop kernel state; got {out[:300]}"
        out = await kernel.run("import helper\nprint(helper.AREA)\n")
        assert "64" in out, out
    finally:
        await kernel.close()


async def test_the_import_is_a_permission_and_stays_refused_without_it() -> None:
    """File access is granted, not assumed: with the tools off the widening is
    not merely unused, it is not there."""
    kernel = _Kernel(file_tools=False)
    try:
        kernel.directory.joinpath("helper.py").write_text("AREA = 64\n")
        out = await kernel.run("import helper\n")
        assert "Module 'helper' is not allowed in the sandbox." in out, out
    finally:
        await kernel.close()


async def test_a_workspace_file_cannot_answer_for_a_standard_module() -> None:
    """The allowlist decides BEFORE the import runs, so a workspace file named
    after a permitted module would be reached under that module's permission.
    The working directory is appended to the import path, never inserted,
    which is what keeps the real module in front of it."""
    kernel = _Kernel()
    try:
        # `fractions` is on the allowlist and, unlike `json`, is not already
        # imported by the prelude — so the shadow would actually be reached.
        kernel.directory.joinpath("fractions.py").write_text("SHADOW = True\n")
        out = await kernel.run("import fractions\nprint(hasattr(fractions, 'SHADOW'))\n")
        assert "False" in out, out
    finally:
        await kernel.close()
