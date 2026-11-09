"""The bundled fonts survive a wheel build, for every package that draws.

An agent that renders an image the model sees carries its own font, because
the committed references pin an image by its bytes and a substituted font
produces a different one. Fonts are DATA, though, and setuptools drops data
from a wheel unless it is named — while a workspace install is editable and
reads the source tree, so nothing local ever notices. The failure only
appears where it costs most: a wheel on Kaggle, raising `FileNotFoundError`
from inside a render.

Builds the real wheel rather than inspecting the declaration, because the
declaration is what was wrong; a test that read `package-data` would have
agreed with the bug. It builds from a copy of the tracked files because
the working tree carries build state that masks the failure -- see the
comment below. Reading the bytes out of the archive stands in for a full
install-and-render: the missing bytes ARE the failure, and a throwaway
venv would cost minutes to prove the same thing.
"""

from __future__ import annotations

import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Package to the font files its renderer needs. A mapping rather than one
#: constant: any agent that draws an image the model is shown carries the
#: same obligation, and this is where it registers.
FONT_PACKAGES = {
    "duck": "duck/fonts",
    "seal": "seal/fonts",
}
REQUIRED = ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "LICENSE_DEJAVU")


@pytest.mark.parametrize("package", sorted(FONT_PACKAGES))
def test_the_built_wheel_carries_its_fonts(package: str, tmp_path: Path) -> None:
    """A wheel missing these renders nothing — and the licence has to travel
    with the fonts, which is what the Bitstream Vera terms require."""
    # Build from a copy of the TRACKED files, not the working tree itself.
    # Any build leaves a `src/<pkg>.egg-info` whose SOURCES.txt setuptools
    # reads as the file list on the NEXT build -- so once a wheel has carried
    # the fonts, later wheels keep carrying them whatever the declaration
    # says, and a test run in place would agree with the last build instead of
    # with the packaging. Copying what git tracks reproduces a fresh clone
    # while still seeing uncommitted edits.
    export = tmp_path / "export"
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, check=True)
    for name in listed.stdout.decode().split("\0"):
        if not name:
            continue
        source = REPO_ROOT / name
        if not source.is_file():
            continue  # a submodule or a path removed since it was indexed
        target = export / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    out = tmp_path / "dist"
    built = subprocess.run(
        ["uv", "build", "--package", package, "--wheel", "--out-dir", str(out)],
        cwd=export,
        capture_output=True,
        text=True,
    )
    assert built.returncode == 0, built.stderr[-2000:]

    wheels = sorted(out.glob("*.whl"))
    assert len(wheels) == 1, [w.name for w in wheels]
    with zipfile.ZipFile(wheels[0]) as archive:
        names = set(archive.namelist())
        prefix = FONT_PACKAGES[package]
        for asset in REQUIRED:
            member = f"{prefix}/{asset}"
            assert member in names, f"{wheels[0].name} is missing {member}"
            # Present but empty would pass a name check and fail a render.
            assert len(archive.read(member)) > 0, f"{member} is empty"
