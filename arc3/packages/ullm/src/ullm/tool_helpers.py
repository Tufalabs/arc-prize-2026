"""Shared helpers for tool implementations.

Centralizes cross-tool logic (like fixing up malformed escape sequences in
tool arguments) so individual `Tool` subclasses can import what they need
without reinventing it. Kept separate from `ullm.tool` to keep that module
focused on the base class.
"""

import ast
import codecs
from typing import Any


def heuristic_fix_double_escapes(value: Any) -> Any:
    """Heuristically undo double-escaped strings in tool arguments.

    Small models sometimes emit JSON string values with doubled escape
    sequences (`\\\\n` in the JSON source, which decodes to `\\n` — backslash
    + n — instead of a newline). For callers that can't verify the result via
    execution, this returns the unescaped form only when the signal is
    unambiguous: the value contains literal `\\n` / `\\t` / `\\r` AND no real
    newlines/tabs/carriage returns. Non-string values pass through unchanged.
    """
    if not isinstance(value, str):
        return value
    has_literal_escape = ("\\n" in value) or ("\\t" in value) or ("\\r" in value)
    has_real_whitespace = ("\n" in value) or ("\t" in value) or ("\r" in value)
    if has_literal_escape and not has_real_whitespace:
        try:
            return codecs.decode(value, "unicode_escape")
        except Exception:
            return value
    return value


def fix_double_escapes_if_python_valid(code: str) -> str:
    """Stronger variant for Python source specifically.

    If `code` fails to parse via `ast.parse` but the `unicode_escape`-decoded
    version parses cleanly, return the decoded version. Otherwise return
    `code` unchanged. Eliminates the false-positive risk of the string
    heuristic by requiring that the "fix" demonstrably produces valid Python.
    Non-string inputs pass through unchanged.
    """
    if not isinstance(code, str):
        return code
    try:
        ast.parse(code)
        return code  # already valid; don't touch
    except SyntaxError:
        pass
    # Only attempt the fix if there's something to unescape.
    if "\\n" not in code and "\\t" not in code and "\\r" not in code:
        return code
    try:
        decoded = codecs.decode(code, "unicode_escape")
    except Exception:
        return code
    try:
        ast.parse(decoded)
        return decoded
    except SyntaxError:
        return code
