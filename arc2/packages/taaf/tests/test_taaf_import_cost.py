# type: ignore
"""What ``import taaf`` costs a process that only wants part of it.

The plotting chain (matplotlib, scipy) belongs to ``taaf.diagnostics``, tens
of MB of private heap in every process that imports it. A solver hosting
each game in a child process pays that per child, so the package loads
submodules on first use and these tests pin that it stays that way — an
eager import added at the top of ``taaf/__init__.py`` or ``taaf.grid``
reads as harmless and silently restores the cost.

Each probe runs in a fresh interpreter: this one has the whole chain
imported already.
"""

import subprocess
import sys

# Heavy enough to be worth a test, and none of them needed to play a game.
HEAVY = ("matplotlib", "scipy")


def _top_level_modules(statement: str) -> set[str]:
    """Top-level module names live in a fresh interpreter after ``statement``."""
    probe = f"import sys; {statement}; print(' '.join(sorted({{name.split('.')[0] for name in sys.modules}})))"
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    return set(completed.stdout.split())


def test_importing_the_package_loads_no_submodule():
    loaded = _top_level_modules("import taaf")
    assert not loaded.intersection(HEAVY), f"import taaf pulled {sorted(loaded.intersection(HEAVY))}"


def test_importing_game_pulls_no_matplotlib():
    # taaf.game owns the task types and needs taaf.grid; the colormap built
    # from the grid's palette is the one thing there that wants matplotlib.
    loaded = _top_level_modules("import taaf.game")
    assert not loaded.intersection(HEAVY), f"import taaf.game pulled {sorted(loaded.intersection(HEAVY))}"
    assert "numpy" in loaded


def test_diagnostics_still_brings_its_own_chain():
    # The counterpart: nothing was made lazy by removing a dependency.
    loaded = _top_level_modules("import taaf.diagnostics")
    assert {"matplotlib", "scipy"}.issubset(loaded)


def test_submodules_resolve_as_attributes_without_being_imported_first():
    # The compatibility promise of the lazy __init__: `taaf.benchmark` works
    # on a bare `import taaf`, imported on first touch.
    probe = "import taaf; assert taaf.benchmark.Benchmark; assert 'benchmark' in dir(taaf)"
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_an_unknown_attribute_still_raises_attribute_error():
    import taaf

    try:
        _ = taaf.not_a_submodule
    except AttributeError:
        return
    raise AssertionError("taaf.not_a_submodule should raise AttributeError")


def test_the_annotations_resolve_at_runtime_without_matplotlib():
    # Runtime consumers of annotations (typeguard, schema and doc tooling)
    # evaluate them with typing.get_type_hints; a name that exists only for
    # the type checker turns each of those into a NameError.
    probe = (
        "import sys, typing, taaf.grid; "
        "typing.get_type_hints(taaf.grid.Grid.draw); typing.get_type_hints(taaf.grid); "
        "typing.get_type_hints(taaf.grid._arc_cmap); "
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_arc_cmap_is_discoverable_before_it_is_built():
    # dir() and star import are the public surface; both must show
    # ARC_CMAP whether or not anything has touched it yet.
    probe = (
        "import taaf.grid; assert 'ARC_CMAP' in dir(taaf.grid); "
        "assert 'ARC_CMAP' not in vars(taaf.grid); "
        "from taaf.grid import *; assert ARC_CMAP.N == 10; "
        "assert 'ARC_CMAP' in vars(taaf.grid)"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_arc_cmap_is_the_palette():
    # The colormap is built on access rather than at import; it must still
    # be the same ten colours, in order.
    import matplotlib.colors

    import taaf.grid

    assert taaf.grid.ARC_CMAP.N == taaf.grid.N_COLORS
    assert taaf.grid.ARC_CMAP(0)[:3] == matplotlib.colors.to_rgb(taaf.grid.ARC_COLORS[0])
    assert taaf.grid.ARC_CMAP(9)[:3] == matplotlib.colors.to_rgb(taaf.grid.ARC_COLORS[9])
    # Bound once: a second access is the same object, not a rebuild.
    assert taaf.grid.ARC_CMAP is taaf.grid.ARC_CMAP
