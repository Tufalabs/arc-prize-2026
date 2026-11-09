"""The bundled fonts survive a wheel build.

The duck renders the colour legend the model is shown (`duck.palette`), so
it carries its own font rather than reading the machine's: a substituted
DejaVu build does not degrade the image, it produces a *different* one.
Fonts are DATA, though, and setuptools drops data from a wheel unless it is
named — while a workspace install is editable and reads the source tree, so
nothing local ever notices. The failure appears where it costs most: a
wheel, raising `FileNotFoundError` from inside a render.

Builds the real wheel rather than inspecting the declaration, because the
declaration is what was wrong; a test that read `package-data` would have
agreed with the bug. It builds from a copy of the tracked files because the
working tree carries build state that masks the failure — see the comment
below. Reading the bytes out of the archive stands in for a full
install-and-render: the missing bytes ARE the failure, and a throwaway venv
would cost minutes to prove the same thing.

The obligation is the package's, not this one file's: it holds wherever an
agent here draws an image the model sees.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Where the renderer looks, as the wheel spells it.
FONT_DIR = "duck/fonts"
REQUIRED = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LICENSE_DEJAVU")


def _export_tracked(destination: Path) -> None:
    """The files git tracks, copied out — a fresh clone that still sees
    uncommitted edits."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True)
    for name in listed.stdout.decode().split("\0"):
        if not name:
            continue
        source = REPO_ROOT / name
        if not source.is_file():
            continue  # a submodule or a path removed since it was indexed
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def test_the_built_wheel_carries_its_fonts(tmp_path: Path) -> None:
    """A wheel missing these renders nothing — and the licence has to travel
    with the fonts, which is what the Bitstream Vera terms require."""
    # Build from a copy of the TRACKED files, not the working tree itself.
    # Any build leaves a `src/duck.egg-info` whose SOURCES.txt setuptools
    # reads as the file list on the NEXT build, so once a wheel has carried
    # the fonts, later wheels keep carrying them whatever the declaration
    # says — and a test run in place would agree with the last build instead
    # of with the packaging.
    export = tmp_path / "export"
    _export_tracked(export)

    out = tmp_path / "dist"
    built = subprocess.run(
        ["uv", "build", "--package", "duck", "--wheel", "--out-dir", str(out)],
        cwd=export,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr[-2000:]

    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, [w.name for w in wheels]
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        for asset in REQUIRED:
            member = f"{FONT_DIR}/{asset}"
            assert member in names, f"{wheels[0].name} is missing {member}"
            # Present but empty would pass a name check and fail a render.
            assert len(archive.read(member)) > 0, f"{member} is empty"
