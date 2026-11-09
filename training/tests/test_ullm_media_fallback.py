"""The isolated Flash runtime's older ullm gets image media from the platypus copy."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from training.source_snapshot import ULLM_ENGINE_PROXY_SOURCE, ULLM_MEDIA_SOURCE, snapshot_paths
from training.ullm_media import ensure_ullm_image_serving, ensure_ullm_media

REPO = Path(__file__).resolve().parents[1]

# The copies live in the private platypus submodule, which CI cannot clone.
pytestmark = pytest.mark.skipif(
    not ((REPO / ULLM_MEDIA_SOURCE).is_file() and (REPO / ULLM_ENGINE_PROXY_SOURCE).is_file()),
    reason="Requires the private platypus submodule",
)


def test_an_older_ullm_gets_the_platypus_media_module(monkeypatch):
    old_ullm = types.ModuleType("ullm")  # a package without media, as pinned in the runtime
    monkeypatch.setitem(sys.modules, "ullm", old_ullm)
    monkeypatch.delitem(sys.modules, "ullm.media", raising=False)
    media = ensure_ullm_media()
    assert media.__file__ == str(REPO / ULLM_MEDIA_SOURCE)
    assert sys.modules["ullm.media"] is media and old_ullm.media is media
    assert media._position_algorithm("Qwen4ExpForConditionalGeneration") == "qwen3-vl-image-mrope-v1"


def test_frozen_snapshots_carry_the_media_module():
    paths = snapshot_paths(REPO)
    assert ULLM_MEDIA_SOURCE in paths and ULLM_ENGINE_PROXY_SOURCE in paths


def test_an_older_engine_proxy_is_replaced_for_image_serving(monkeypatch):
    """The Flash runtime's proxy refuses media prompts; image serving loads
    the platypus copy, which accepts and proves them."""
    # The installed proxy and all it imports, not only the package: a checkout
    # of the submodule without the harness installed has the first alone.
    pytest.importorskip("ullm.backends.engine_proxy", reason="Requires the installed harness", exc_type=ImportError)
    import ullm.backends

    old_proxy = types.ModuleType("ullm.backends.engine_proxy")
    old_proxy.EngineProxy = type("EngineProxy", (), {})  # no prepare_media
    monkeypatch.setitem(sys.modules, "ullm.backends.engine_proxy", old_proxy)
    monkeypatch.setattr(ullm.backends, "engine_proxy", old_proxy, raising=False)
    proxy = ensure_ullm_image_serving()
    assert proxy.__file__ == str(REPO / ULLM_ENGINE_PROXY_SOURCE)
    assert sys.modules["ullm.backends.engine_proxy"] is proxy and ullm.backends.engine_proxy is proxy
    assert hasattr(proxy.EngineProxy, "prepare_media")
    # A media-capable proxy is kept as installed.
    assert ensure_ullm_image_serving() is proxy
