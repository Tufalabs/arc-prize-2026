"""File tools: read, write, and edit files within an allowed set.

All file tools share a common base class (FileTool) that enforces an
allowlist of permitted file paths. The tools operate on a single file
at a time, identified by the 'path' argument."""

import os
from dataclasses import dataclass, field
from typing import Any

import ullm.conv
import ullm.tool


def _reason(error: OSError) -> str:
    """Why an OS call failed, without the path it failed on.

    ``str(error)`` renders the filename, which is the resolved on-disk path —
    host-specific, absent from the caller's world, and different on every run.
    ``strerror`` is the same diagnosis with none of that.
    """
    return error.strerror or type(error).__name__


@dataclass
class FileTool(ullm.tool.Tool):
    """Base class for file tools. Enforces an allowlist of permitted file paths.

    Subclasses implement _execute_file() for their specific operation.
    The base _execute() validates the path against allowed_files before delegating."""

    allowed_files: list[str] = field(default_factory=lambda: list[str]())  # exact-match paths as the model sees them
    path_mapping: dict[str, str] = field(default_factory=lambda: dict[str, str]())  # model path -> real path on disk
    allowed_dir: str | None = None  # if set, also permit any file resolving inside this dir (no ../ escape)
    allowed_suffixes: tuple[str, ...] | None = (
        None  # restrict allowed_dir matches to these suffixes, e.g. ('.py',); None = any
    )
    # Auto-scope to another tool's working directory (e.g. a PythonTool's
    # tmpdir): the directory is read off the linked tool at RESOLVE time, so
    # copies/restores that give the tool a new tmpdir need no re-pointing —
    # pickle the linked tool and the file tools in one graph and the link
    # stays object-identical. Mutually exclusive with ``allowed_dir``;
    # relative model paths (and ``allowed_files`` entries without an explicit
    # ``path_mapping``) resolve inside the linked dir.
    linked_tool: Any | None = None  # duck-typed: anything with a `_tmpdir` str attribute

    def _check_constraints(self) -> None:
        """A link and an explicit directory are mutually exclusive — the link IS
        the directory scope, so setting both leaves which one wins ambiguous.
        """
        assert self.linked_tool is None or self.allowed_dir is None, (
            "linked_tool and allowed_dir are mutually exclusive; the link IS the directory scope"
        )
        super()._check_constraints()

    def _scope_dir(self) -> str | None:
        """The directory scope in effect: the linked tool's current working
        dir, or the static ``allowed_dir``."""
        if self.linked_tool is not None:
            linked_dir = self.linked_tool._tmpdir
            assert linked_dir is not None, "linked tool has no working directory (already cleared?)"
            return linked_dir
        return self.allowed_dir

    def _resolve_path(self, path: str) -> str | None:
        """Resolve a model-visible path to a real on-disk path, or None if it
        is not permitted. An explicit ``allowed_files`` entry (optionally
        remapped via ``path_mapping``) always wins; otherwise, if a directory
        scope is in effect (``linked_tool`` or ``allowed_dir``), any path
        resolving inside it with an allowed suffix is permitted — this lets
        the model create and edit its own helper files next to the main one."""
        # Exact allowlist, with optional remap to the real on-disk path.
        # Under a linked tool, unmapped allowlist entries resolve inside the
        # linked dir (that is the point of the link: no manual re-pointing).
        if path in self.allowed_files:
            if path in self.path_mapping:
                return self.path_mapping[path]
            if self.linked_tool is not None:
                return os.path.join(self._scope_dir() or "", path)
            return path
        # Directory scope: a file inside it (reject ../ escapes and the dir
        # itself) carrying one of the allowed suffixes.
        scope = self._scope_dir()
        if scope is not None:
            base = os.path.realpath(scope)
            real = os.path.realpath(os.path.join(scope, path))
            within = real.startswith(base + os.sep)
            suffix_ok = self.allowed_suffixes is None or real.endswith(self.allowed_suffixes)
            if within and suffix_ok:
                return real
        return None

    def _denied_message(self, path: str) -> str:
        """Explain to the model why ``path`` was rejected and what it may use."""
        msg = f"File '{path}' is not allowed. Allowed files: {self.allowed_files}."
        if self._scope_dir() is not None:
            kinds = ", ".join(self.allowed_suffixes) if self.allowed_suffixes else "any"
            msg += f" You may also read/write {kinds} files in the working directory."
        return msg

    async def _execute(self, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Resolve the requested path against this tool's scope and, if it is
        allowed, hand it to the subclass.
        """
        path = arguments["path"]
        real_path = self._resolve_path(path)
        if real_path is None:
            self.is_error = True
            return ullm.conv.StringContent(text=self._denied_message(path))
        return await self._execute_file(real_path, arguments)

    async def _execute_file(self, real_path: str, arguments: dict[str, Any]) -> ullm.conv.Content:
        # `real_path` is the resolved on-disk path and may contain absolute /
        # host-specific components — DO NOT embed it in user-facing messages.
        # For any text the caller (or LLM) will see, use ``arguments["path"]``,
        # which is the path as the caller provided it.
        """Act on the resolved path. Subclasses implement; ``real_path`` is
        host-specific and must never reach text the model reads.
        """
        raise NotImplementedError("Subclasses must implement _execute_file()")


@dataclass
class ReadFileTool(FileTool):
    """Read the contents of a file."""

    name: str = "read_file"
    description: str = "Read the contents of a file."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        }
    )

    async def _execute_file(self, real_path: str, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Return the file's contents."""
        shown_path = arguments["path"]
        try:
            with open(real_path) as f:
                content = f.read()
        except FileNotFoundError:
            self.is_error = True
            return ullm.conv.StringContent(text=f"File '{shown_path}' does not exist.")
        except OSError as e:
            self.is_error = True
            return ullm.conv.StringContent(text=f"Error reading '{shown_path}': {_reason(e)}.")
        return ullm.conv.StringContent(text=content)


@dataclass
class WriteFileTool(FileTool):
    """Write content to a file, replacing any existing content."""

    name: str = "write_file"
    description: str = "Write content to a file, replacing any existing content."
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        }
    )

    async def _execute_file(self, real_path: str, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Write the file under the configured argument-repair policy."""
        shown_path = arguments["path"]
        content = self._repaired_text(arguments["content"])
        try:
            with open(real_path, "w") as f:
                f.write(content)
        except OSError as e:
            self.is_error = True
            return ullm.conv.StringContent(text=f"Error writing '{shown_path}': {_reason(e)}.")
        return ullm.conv.StringContent(text=f"Successfully wrote {len(content)} characters to '{shown_path}'.")


@dataclass
class EditFileTool(FileTool):
    """Replace an exact string in a file with a new string."""

    name: str = "edit_file"
    description: str = (
        "Edit a file by replacing an exact substring with new content. "
        "The old_string must appear exactly once in the file."
    )
    parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to edit"},
                "old_string": {
                    "type": "string",
                    "description": "Exact string to find and replace (must be unique in the file)",
                },
                "new_string": {"type": "string", "description": "Replacement string"},
            },
            "required": ["path", "old_string", "new_string"],
        }
    )

    async def _execute_file(self, real_path: str, arguments: dict[str, Any]) -> ullm.conv.Content:
        """Replace one verbatim, unique occurrence. ``old_string`` is matched as
        given — unescaping it would corrupt a file that really contains those
        characters.
        """
        shown_path = arguments["path"]
        # Search text always matches verbatim; only new text permits repair.
        old_string = arguments["old_string"]
        new_string = self._repaired_text(arguments["new_string"])

        # Read current content
        try:
            with open(real_path) as f:
                content = f.read()
        except FileNotFoundError:
            self.is_error = True
            return ullm.conv.StringContent(text=f"File '{shown_path}' does not exist.")
        except OSError as e:
            self.is_error = True
            return ullm.conv.StringContent(text=f"Error reading '{shown_path}': {_reason(e)}.")

        # Validate uniqueness
        count = content.count(old_string)
        if count == 0:
            self.is_error = True
            return ullm.conv.StringContent(text=f"old_string not found in '{shown_path}'.")
        # TODO: add hints?
        if count > 1:
            self.is_error = True
            return ullm.conv.StringContent(text=f"old_string appears {count} times in '{shown_path}'; must be unique.")

        # Apply replacement and write
        new_content = content.replace(old_string, new_string, 1)
        try:
            with open(real_path, "w") as f:
                f.write(new_content)
        except OSError as e:
            self.is_error = True
            return ullm.conv.StringContent(text=f"Error writing '{shown_path}': {_reason(e)}.")
        return ullm.conv.StringContent(
            text=f"Successfully edited '{shown_path}': replaced {len(old_string)} characters with {len(new_string)} characters."
        )
