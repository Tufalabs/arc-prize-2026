"""Re-baseline the ``viewer_html`` slice of every committed reference.

Not tests of behaviour — the sanctioned way to REWRITE the viewer pages
pinned inside the committed references after a change that touches nothing
but viewer rendering (a chunk class, a page-layout edit). Opt-in via
``PLATYPUS_REFRESH_VIEWER_HTML=1``; free — each case replays its recorded
traffic, and the rewrite is gated on every surface except ``viewer_html``
reproducing bit-exact:

    PLATYPUS_REFRESH_VIEWER_HTML=1 make test TEST_ARGS='-k refresh_viewer_html'
"""

from __future__ import annotations

from pathlib import Path

import pytest

import test_duck_reference
import test_prompted_flip_reference
from reference_support import ReferenceCase, refresh_case, require_viewer_refresh

#: Every case whose reference pins viewer_html. A case missing from this list
#: has no free path to a re-render, so a viewer change leaves it stale and the
#: only way back is a paid regeneration.
CASES = [test_prompted_flip_reference.CASE, test_duck_reference.CASE]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_refresh_viewer_html(case: ReferenceCase, tmp_path: Path) -> None:
    require_viewer_refresh()
    await refresh_case(case, tmp_path)
