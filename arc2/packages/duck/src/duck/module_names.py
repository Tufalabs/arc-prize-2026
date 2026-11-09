"""Which module names a file of the model's cannot take.

The working directory is importable, so a file there is a module under its
own name. A name Python already has — a standard-library module, or an
installed package — keeps meaning that module: the kernel's start-up imports
the harness from the same directory, and a file standing in for ``inspect``
there breaks it before the model's code has run. So the model's file loses
that name everywhere, and ``create_file`` says so rather than writing a file
that no import would reach.

Light on purpose: the kernel's import hook, the search's loader and the file
tools all ask, and the file tools load in every game process.
"""

from __future__ import annotations

import importlib.machinery
import os
import sys


def taken(name: str, directory: str) -> bool:
    """Whether ``import name`` reaches a module other than a file in
    ``directory``: one built into Python, the standard library's, or one on
    ``sys.path`` outside the directory (an installed package)."""
    if name in sys.builtin_module_names or name in sys.stdlib_module_names:
        return True
    here = os.path.abspath(directory)
    elsewhere = [entry for entry in sys.path if entry and os.path.abspath(entry) != here]
    return importlib.machinery.PathFinder.find_spec(name, elsewhere) is not None
