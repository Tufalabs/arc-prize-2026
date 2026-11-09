"""Re-baseline the ``viewer_html`` slice of every committed reference (I13).

Not tests of behaviour — the sanctioned way to REWRITE the viewer pages
pinned inside the committed references after a change that touches nothing
but viewer rendering (a chunk class, a page-layout edit). Opt-in via
``PLATYPUS_REFRESH_VIEWER_HTML=1``; free — each case replays its recorded
traffic, and the rewrite is gated on every surface except ``viewer_html``
reproducing bit-exact. The true regeneration flow (live backend) remains
``regenerate_case`` — see ``platypus.reference.refresh_viewer_html`` for
why both exist.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import test_duck_nondefault_reference
import test_duck_pathfinder_reference
import test_duck_scripted_file_suite
import test_duck_scripted_numeric
import test_duck_scripted_reference
import test_prompted_random_reference
import test_seal_scripted_animation
import test_seal_scripted_counter
import test_seal_scripted_file_suite
import test_seal_scripted_frame_delta
import test_seal_scripted_irregular_reset
import test_seal_scripted_matching
import test_seal_scripted_optional
import test_seal_scripted_reference
import test_seal_scripted_render_colours
import test_seal_scripted_search
import test_seal_scripted_surface
import test_seal_scripted_world_model
from reference_support import ReferenceCase, refresh_case, require_viewer_refresh

CASES = [
    test_duck_pathfinder_reference.CASE,
    test_duck_nondefault_reference.CASE,
    test_duck_scripted_reference.CORNERS,
    test_duck_scripted_reference.EVICTION,
    test_duck_scripted_reference.HOSTILE,
    test_duck_scripted_reference.SUMMARIZE,
    test_duck_scripted_file_suite.FILE_SUITE,
    test_duck_scripted_numeric.NUMERIC,
    test_prompted_random_reference.CASE,
    test_seal_scripted_reference.CORNERS,
    test_seal_scripted_reference.EVICTION,
    test_seal_scripted_reference.SUMMARIZE,
    test_seal_scripted_reference.SUMMARIZE_WORLD_MODEL,
    test_seal_scripted_reference.HOSTILE,
    test_seal_scripted_file_suite.FILE_SUITE,
    test_seal_scripted_surface.SURFACE,
    test_seal_scripted_world_model.CASE,
    test_seal_scripted_world_model.KEYBOARD,
    test_seal_scripted_search.CASE,
    test_seal_scripted_frame_delta.CASE,
    test_seal_scripted_render_colours.CASE,
    test_seal_scripted_irregular_reset.IRREGULAR,
    test_seal_scripted_animation.CASE,
    test_seal_scripted_animation.SETTLED_CASE,
    test_seal_scripted_animation.NONE_CASE,
    test_seal_scripted_animation.ON_REQUEST_CASE,
    test_seal_scripted_animation.UNMARKED_CASE,
    test_seal_scripted_animation.WORLD_MODEL_CASE,
    test_seal_scripted_animation.OPTIONAL_CASE,
    test_seal_scripted_optional.CASE,
    test_seal_scripted_counter.UNDO,
    test_seal_scripted_counter.PLAN,
    test_seal_scripted_matching.OFFERED,
    test_seal_scripted_matching.NOT_OFFERED,
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
async def test_refresh_viewer_html(case: ReferenceCase, tmp_path: Path) -> None:
    """Rewrite this case's committed viewer_html from a replay render."""
    require_viewer_refresh()
    await refresh_case(case, tmp_path)
