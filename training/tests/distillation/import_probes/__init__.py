"""Child-process import probes.

Each module here is run by ``tests/distillation/test_flash_imports.py`` as
``python -m tests.distillation.import_probes.<name>`` from the repository root,
so the imports it exercises happen in a fresh interpreter with the named
packages made unavailable. Keeping the probes as modules (not string literals)
lets linters and refactors see them.
"""

import importlib.abc
import sys


def block_modules(*roots, prefixes=()):
    """Make imports of the given top-level packages (or name prefixes) fail."""

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in roots or fullname.startswith(prefixes):
                raise ModuleNotFoundError(fullname)

    sys.meta_path.insert(0, Blocked())


def assert_never_imported(*roots):
    loaded = sorted(name for name in sys.modules if name.split(".")[0] in roots)
    assert not loaded, f"blocked packages were imported: {loaded}"
