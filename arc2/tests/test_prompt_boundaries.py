"""Implicitly-concatenated prompt literals must not swallow the boundary
between them.

Python joins adjacent string literals silently, so a missing trailing
`\\n` or space runs two bullets together in text the model reads. Both
instances upstream found were invisible at the call site and both were
faithfully inherited from the system being ported, which is how they
survived review — and a reference recorded against the malformed text
replays it green forever, so capture/replay cannot be the check.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

# Modules whose string literals reach a model: prompt templates and the
# kernel surfaces rendered into tool results.
PROMPT_MODULES = [
    "packages/duck/src/duck/prompts.py",
    "packages/duck/src/duck/kernel_code.py",
    "packages/duck/src/duck/sandbox.py",
]

REPO = pathlib.Path(__file__).resolve().parent.parent

# Instances the committed references still record, so fixing one costs a
# reference regeneration and belongs in a batch rather than alone. Registered
# by a distinctive fragment of the left-hand piece rather than by line number,
# and asserted to still be PRESENT — so whichever way it changes (fixed, or
# drifting), this file has to be updated deliberately.
FENCE_HELD: dict[str, tuple[str, ...]] = {}

# A right-hand piece that starts a bullet, a numbered item, or a new
# sentence — the shapes that read as broken when run onto the previous one.
_STARTS_NEW_UNIT = re.compile(r"^(-\s|\*\s|\d+\.\s|[A-Z][a-z]+[ ,.])")
# Pieces of one implicit concatenation, tolerating escapes and f-prefixes.
_LITERAL = re.compile(r'(?:[frbu]{0,2})"((?:[^"\\]|\\.)*)"', re.IGNORECASE)

# Known-good joins: the left piece ends a line the right piece continues
# deliberately, so the concatenation is the point.
_ALLOWED_ENDINGS = ("\\n", " ", "\\t", "(", "[", "{")


def _swallowed_boundaries(path: pathlib.Path) -> list[tuple[int, str, str]]:
    """Every implicit concatenation in ``path`` whose left piece ends mid-word
    while the right piece starts a new bullet or sentence."""
    source = path.read_text()
    tree = ast.parse(source)
    found: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        segment = ast.get_source_segment(source, node)
        if segment is None:
            continue
        pieces = _LITERAL.findall(segment)
        for i in range(len(pieces) - 1):
            left, right = pieces[i], pieces[i + 1]
            if not left or not right or left.endswith(_ALLOWED_ENDINGS):
                continue
            if _STARTS_NEW_UNIT.match(right):
                found.append((node.lineno + i, left[-60:], right[:60]))
    return found


@pytest.mark.parametrize("relative", PROMPT_MODULES)
def test_no_swallowed_prompt_boundary(relative: str) -> None:
    """No adjacent prompt literals run together without a separator, except
    the fence-held instances, which must still be exactly where they were
    registered."""
    hits = _swallowed_boundaries(REPO / relative)
    registered = FENCE_HELD.get(relative, ())

    unregistered = [h for h in hits if not any(fragment in h[1] for fragment in registered)]
    report = "\n".join(f"  {relative}:{line}\n    ...{left!r}\n    +  {right!r}" for line, left, right in unregistered)
    assert not unregistered, f"prompt literals concatenate without a separator:\n{report}"

    for fragment in registered:
        assert any(fragment in left for _, left, _ in hits), (
            f"{relative}: registered boundary {fragment!r} is gone. If it was fixed "
            f"(with a reference regeneration), drop it from FENCE_HELD."
        )


def test_detector_catches_a_planted_boundary(tmp_path: pathlib.Path) -> None:
    """The scan finds the defect it exists for, and tolerates a correct
    join — otherwise a green suite above says nothing."""
    planted = tmp_path / "planted.py"
    planted.write_text('BAD = (\n    "- first bullet."\n    "- second bullet.\\n"\n)\n')
    assert len(_swallowed_boundaries(planted)) == 1

    fixed = tmp_path / "fixed.py"
    fixed.write_text('GOOD = (\n    "- first bullet.\\n"\n    "- second bullet.\\n"\n)\n')
    assert _swallowed_boundaries(fixed) == []
