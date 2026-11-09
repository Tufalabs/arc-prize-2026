"""``ullm.media`` for runtimes whose pinned ullm predates it.

The isolated Flash runtime bundles ullm at an older platypus revision (its
``PLATYPUS`` pin in ``scripts/flash_runtime.py``), before image media existed.
Image replay needs one module from it, which imports only the standard
library at load time: this loads the platypus submodule's copy (the one the
ARC-2 worker's ullm installs, and which frozen snapshots carry) as
``ullm.media`` when the installed ullm has none. Wherever ullm ships its own,
that one is used. Image serving also needs the engine proxy that accepts and
proves media (``ensure_ullm_image_serving``).
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

from training.source_snapshot import ULLM_ENGINE_PROXY_SOURCE, ULLM_MEDIA_SOURCE


def _load_platypus_copy(name: str, relative: Path) -> ModuleType:
    """Load the platypus submodule's copy of ``relative`` as ``name``."""
    roots = [os.environ.get("ARC3_FROZEN_REPO_ROOT"), os.environ.get("ARC3_FLASH_REPO_ROOT"), str(Path(__file__).resolve().parents[1])]
    for root in filter(None, roots):
        source = Path(root) / relative
        if source.is_file():
            spec = importlib.util.spec_from_file_location(name, source)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            parent, _, child = name.rpartition(".")
            setattr(importlib.import_module(parent), child, module)
            return module
    raise ModuleNotFoundError(f"{name}: the installed ullm has none and no platypus copy was found")


def ensure_ullm_media() -> ModuleType:
    try:
        return importlib.import_module("ullm.media")
    except ModuleNotFoundError:
        pass
    return _load_platypus_copy("ullm.media", ULLM_MEDIA_SOURCE)


def ensure_ullm_image_serving() -> ModuleType:
    """``ullm.media`` plus an engine proxy whose server accepts image prompts.

    The Flash runtime's proxy refuses any prompt carrying ``media``; the
    platypus copy validates it and returns the engine's media evidence. Call
    before anything imports ``ullm.backends.engine_proxy``'s functions."""
    ensure_ullm_media()
    proxy = importlib.import_module("ullm.backends.engine_proxy")
    if hasattr(proxy.EngineProxy, "prepare_media"):
        return proxy
    return _load_platypus_copy("ullm.backends.engine_proxy", ULLM_ENGINE_PROXY_SOURCE)
