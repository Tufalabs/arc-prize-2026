"""Write ``arc2-site-0290.ipynb``: the Kaggle notebook whose output an ARC2
kernel installs the engine from -- the wheelhouse, and the record of installing it
on the kernel's image, for replay instead of pip.

A fork of ``jeroencottaar/arc3-utils-0290``: its cells come first, verbatim
(``ORIGINAL``), and download the closure of ``ullm.kaggle.KAGGLE_VLLM_REQUIREMENTS``
into ``wheels/``. ARC2's kernel installs nothing before the engine (TAAF's ARC2
notebook: its competition mount is data only), then ``pip install --no-index
--find-links <wheels>``. This notebook runs that install on the same image,
diffs the installed distributions before and after it, and packs
every file of what pip added into one zstd archive, with a manifest of what
pip removed and of the state it started from. pip still resolves
everything; a kernel whose state matches the manifest's starting state
replays the result, any other installs from ``wheels/`` with pip.

The kernel's reader is ``ullm.kaggle_site``; its snapshot and unpack
functions are embedded here from that module's source, so the record's
digests and the kernel's are computed by one text, and the timed unpack is
the kernel's own.

    python tools/kaggle/arc2-site-0290/build_notebook.py   # writes the .ipynb beside it

The record starts from the image as an ARC2 kernel boots. ullm's own
wheelhouse records its install after the ARC3 competition's ``arc-agi``
install (``arc-agi``, ``arcengine``, pillow 12.2.0), which an ARC2 kernel never
runs, so an ARC2 kernel refuses that record (``duck.presets.KAGGLE_WHEELS``).
"""

import ast
import json
import pathlib

import ullm.kaggle_site

HERE = pathlib.Path(__file__).resolve().parent
# The pinned ullm's, which is the one the kernel runs.
SITE_MODULE = pathlib.Path(ullm.kaggle_site.__file__)
EMBEDDED = ("canonical", "site_dirs", "snapshot", "state_digest", "_decompressor", "_unpack")


def embedded_source() -> str:
    """``EMBEDDED``, verbatim from ``ullm.kaggle_site``."""
    source = SITE_MODULE.read_text()
    functions: dict[str, str] = {}
    for node in ast.parse(source).body:
        segment = ast.get_source_segment(source, node) if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.FunctionDef) and segment is not None:
            functions[node.name] = segment
    return "\n\n\n".join(functions[name] for name in EMBEDDED)


# jeroencottaar/arc3-utils-0290's cells, verbatim: the wheelhouse download.
ORIGINAL = [
    ("code", "!mkdir -p /kaggle/working/wheels\n!nvidia-smi"),
    (
        "code",
        "!pip download \\\n    --dest /kaggle/working/wheels \\\n    --python-version 3.12.12 \\\n    --only-binary=:all: \\\n    --extra-index-url https://download.pytorch.org/whl/cu130 \\\n    --extra-index-url https://pypi.nvidia.com \\\n    --extra-index-url https://flashinfer.ai/whl \\\n    --extra-index-url https://flashinfer.ai/whl/cu130 \\\n    vllm==0.29.0 \\\n    transformers==5.16.1 \\\n    pillow==11.3.0 \\\n    flashinfer-python==0.6.18 \\\n    flashinfer-cubin==0.6.18 \\\n    flashinfer-jit-cache==0.6.18+cu130 \\\n    nvidia-cutlass-dsl==4.6.2 \\\n    nvidia-cuda-nvcc==13.0.88 \\\n    nvidia-cuda-crt==13.0.88 \\\n    nvidia-nvvm==13.0.88",
    ),
    ("markdown", ""),
    ("code", "!ls -la /kaggle/working/wheels | grep -i flashinfer"),
]

# The one change to the original: its download is held to the versions of
# jeroencottaar/arc3-utils-0290's wheelhouse (constraints.txt), so a rebuild
# gives the environment our kernels have run on rather than whatever was
# released since; pip still resolves the closure. Unconstrained, a rebuild on
# 2026-10-01 moved 51 of 198 wheels (builder v4).
CONSTRAINTS = (HERE / "constraints.txt").read_text()
CONSTRAINTS_FILE = "/kaggle/working/constraints.txt"
CONSTRAINT_ARGUMENT = "    --only-binary=:all: \\\n"


def constrained(source: str) -> str:
    """The original download cell, constrained to ``CONSTRAINTS``."""
    if CONSTRAINT_ARGUMENT not in source:
        raise ValueError("the original download cell no longer has the line the constraint follows")
    return source.replace(CONSTRAINT_ARGUMENT, CONSTRAINT_ARGUMENT + f"    --constraint {CONSTRAINTS_FILE} \\\n", 1)


WRITE_CONSTRAINTS = f"""\
# The versions the download is held to (build_notebook.py, CONSTRAINTS).
import pathlib

pathlib.Path({CONSTRAINTS_FILE!r}).write_text({CONSTRAINTS!r})
"""

INTRO = """\
# ARC2 wheelhouse, with the record of its install

The engine's wheelhouse, and a record of what installing it does to this image, so
a kernel can replay the install instead of running pip (~200 s on the RTX). A fork
of `jeroencottaar/arc3-utils-0290`: its cells come first, unchanged, and download the
wheels into `wheels/`.

1. Nothing installed first: ARC2's TAAF notebook installs no runtime (its competition
   mount is data only), so the record starts from the image as it boots.
2. A snapshot of the installed distributions: the state a kernel must match to replay.
3. The kernel's own install from `wheels/`, unchanged (`ullm.kaggle_site.pip_install`).
4. What pip added (every file in its `RECORD`, `.pyc` and scripts included) is packed
   into `site_delta.tar.zst` (`.tar.gz` on an image without zstd); what it removed,
   and the starting state, go in `site_delta.json`.
5. The archive is unpacked once into a scratch root by the kernel's own code, timed.

The snapshot and unpack functions are `ullm.kaggle_site`'s, embedded verbatim.
"""

SETUP = """\
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import site
import subprocess
import sys
import time
from typing import Any

OUT = pathlib.Path("/kaggle/working")
WHEELHOUSE_REF = "jeroencottaar/arc2-site-0290"  # this notebook: wheels/ beside the record
# ullm.kaggle.KAGGLE_VLLM_REQUIREMENTS at arc3-platypus main 6f20219e. The manifest
# records them, and a kernel replays only for the same list.
REQUIREMENTS = [
    "vllm==0.29.0",
    "transformers==5.16.1",
    "pillow==11.3.0",
    "flashinfer-python==0.6.18",
    "flashinfer-cubin==0.6.18",
    "flashinfer-jit-cache==0.6.18+cu130",
    "nvidia-cutlass-dsl==4.6.2",
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-nvvm==13.0.88",
]
TIMINGS = {}

print(sys.version)
print(platform.platform(), "| cpus", os.cpu_count())
print({k: v for k, v in os.environ.items() if any(s in k for s in ("DOCKER", "IMAGE", "KERNEL_RUN_TYPE"))})
print(subprocess.run(["df", "-h", "/kaggle/working", "/tmp"], capture_output=True, text=True).stdout)
print({tool: shutil.which(tool) for tool in ("zstd", "pigz", "gzip", "tar")})
try:
    import zstandard
    print("python zstandard", zstandard.__version__)
except ImportError:
    zstandard = None
    print("no python zstandard")
print(subprocess.run("find /kaggle/input /kaggle/usr/lib -maxdepth 3 | head -60", shell=True,
                     capture_output=True, text=True).stdout)


WHEELS = OUT / "wheels"  # the original's download
WHEEL_FILES = sorted((p.name, p.stat().st_size) for p in WHEELS.iterdir() if p.is_file())
print(WHEELS, len(WHEEL_FILES), "files", round(sum(s for _, s in WHEEL_FILES) / 2**30, 2), "GiB")
"""

SNAPSHOT_RUN = """\
start = time.monotonic()
BEFORE = snapshot()
BEFORE_SHA, BEFORE_LINES = state_digest(BEFORE)
TIMINGS["snapshot_s"] = round(time.monotonic() - start, 1)
print(len(BEFORE), "distributions before; state", BEFORE_SHA[:16], TIMINGS)
"""

INSTALL = """\
# The kernel's install without a record (ullm.kaggle_site.pip_install), unchanged.
PIP_COMMAND = [sys.executable, "-m", "pip", "install", "--no-input", "--no-index", "--find-links", str(WHEELS), *REQUIREMENTS]
start = time.monotonic()
result = subprocess.run(PIP_COMMAND, capture_output=True, text=True)
TIMINGS["pip_install_s"] = round(time.monotonic() - start, 1)
(OUT / "pip_install.log").write_text(result.stdout + "\\n--- stderr ---\\n" + result.stderr)
print(result.stdout[-3000:])
print(result.stderr[-2000:])
result.check_returncode()
print(TIMINGS)
"""

DIFF = """\
AFTER = snapshot()
AFTER_SHA, _ = state_digest(AFTER)
added = {p: d for p, d in AFTER.items() if p not in BEFORE}
removed = {p: d for p, d in BEFORE.items() if p not in AFTER}
same_path_changed = [p for p in AFTER if p in BEFORE and AFTER[p]["version"] != BEFORE[p]["version"]]
assert not same_path_changed, same_path_changed  # pip writes a new dist-info for a new version

FILES = sorted({f for d in added.values() for f in d["files"]})
missing = [f for f in FILES if not os.path.lexists(f)]
FILES = [f for f in FILES if os.path.lexists(f)]
UNCOMPRESSED = sum(os.lstat(f).st_size for f in FILES)
print(len(added), "added,", len(removed), "removed;", len(FILES), "files,", round(UNCOMPRESSED / 2**30, 2), "GiB;",
      len(missing), "RECORD entries absent")
for p, d in sorted(added.items(), key=lambda x: x[1]["name"]):
    print("  +", d["name"], d["version"], p)
for p, d in sorted(removed.items(), key=lambda x: x[1]["name"]):
    print("  -", d["name"], d["version"], p)
"""

ARCHIVE = """\
listing = OUT / "site_delta.files"
listing.write_text("\\n".join(f.lstrip("/") for f in FILES) + "\\n")
start = time.monotonic()
if shutil.which("zstd"):
    ARCHIVE, COMPRESSOR = OUT / "site_delta.tar.zst", "zstd -T0 -3"
    subprocess.run(f"tar -C / --no-recursion -cf - -T {listing} | zstd -T0 -3 -q -o {ARCHIVE}",
                   shell=True, check=True, executable="/bin/bash")
elif zstandard is not None:
    import tarfile

    ARCHIVE, COMPRESSOR = OUT / "site_delta.tar.zst", "python zstandard, level 3, all threads"
    with open(ARCHIVE, "wb") as raw, zstandard.ZstdCompressor(level=3, threads=-1).stream_writer(raw) as zw:
        with tarfile.open(fileobj=zw, mode="w|") as tar:
            for f in FILES:
                tar.add(f, arcname=f.lstrip("/"), recursive=False)
else:
    ARCHIVE, COMPRESSOR = OUT / "site_delta.tar.gz", "pigz -3" if shutil.which("pigz") else "gzip -3"
    subprocess.run(f"set -o pipefail; tar -C / --no-recursion -cf - -T {listing} | {COMPRESSOR} > {ARCHIVE}",
                   shell=True, check=True, executable="/bin/bash")
TIMINGS["archive_s"] = round(time.monotonic() - start, 1)
ARCHIVE_BYTES = ARCHIVE.stat().st_size
digest = hashlib.sha256()
with open(ARCHIVE, "rb") as fh:
    for block in iter(lambda: fh.read(1 << 24), b""):
        digest.update(block)
ARCHIVE_SHA = digest.hexdigest()
listing.unlink()

# Unpack it once as a kernel does (the embedded _unpack), into a scratch root: the
# replay's cost on this image, and a check that every file comes back.
scratch = pathlib.Path("/tmp/site_unpack_test")
shutil.rmtree(scratch, ignore_errors=True)
scratch.mkdir()
start = time.monotonic()
_unpack(ARCHIVE, root=scratch)
TIMINGS["unpack_s"] = round(time.monotonic() - start, 1)
TIMINGS["unpack_reader"] = _decompressor(str(ARCHIVE))
members = sum(len(files) + sum(os.path.islink(os.path.join(d, x)) for x in dirs) for d, dirs, files in os.walk(scratch))
shutil.rmtree(scratch)
assert members == len(FILES), (members, len(FILES))
print(COMPRESSOR, round(ARCHIVE_BYTES / 2**30, 2), "GiB from", round(UNCOMPRESSED / 2**30, 2), "GiB;", members,
      "members;", TIMINGS)
"""

MANIFEST = """\
manifest = {
    "format": 1,
    "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "wheelhouse": WHEELHOUSE_REF,
    "wheelhouse_files": WHEEL_FILES,
    "constraints_sha256": hashlib.sha256(pathlib.Path("/kaggle/working/constraints.txt").read_bytes()).hexdigest(),
    "requirements": REQUIREMENTS,
    "pip_command": [str(c) for c in PIP_COMMAND[1:7]] + ["$WHEELS"] + REQUIREMENTS,
    "python": sys.version,
    "site_dirs": site_dirs(),
    "platform": platform.platform(),
    "image_env": {k: v for k, v in os.environ.items() if any(s in k for s in ("DOCKER", "IMAGE"))},
    "before": {"sha256": BEFORE_SHA, "distributions": BEFORE_LINES},
    "after_sha256": AFTER_SHA,
    "added": [{"name": d["name"], "version": d["version"], "dist_info": p} for p, d in sorted(added.items())],
    "removed": [
        {"name": d["name"], "version": d["version"], "dist_info": p, "files": d["files"]}
        for p, d in sorted(removed.items())
    ],
    "archive": {
        "file": ARCHIVE.name,
        "compressor": COMPRESSOR,
        "bytes": ARCHIVE_BYTES,
        "sha256": ARCHIVE_SHA,
        "members": members,
        "uncompressed_bytes": UNCOMPRESSED,
        "record_entries_absent": missing,
    },
    "timings_s": TIMINGS,
}
(OUT / "site_delta.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps({k: v for k, v in manifest.items() if k not in ("before", "removed", "wheelhouse_files")}, indent=1)[:4000])
"""


CELLS: list[dict[str, object]] = []


def cell(kind: str, source: str) -> dict[str, object]:
    """A notebook cell of ``kind`` holding ``source``, numbered in order."""
    base: dict[str, object] = {
        "cell_type": kind,
        "id": f"cell-{len(CELLS)}",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    CELLS.append(base)
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


notebook = {
    "cells": [
        cell(*ORIGINAL[0]),
        cell("code", WRITE_CONSTRAINTS),
        cell(ORIGINAL[1][0], constrained(ORIGINAL[1][1])),
        *(cell(kind, source) for kind, source in ORIGINAL[2:]),
        cell("markdown", INTRO),
        cell("code", SETUP),
        cell(
            "code",
            "# ullm.kaggle_site's own functions, verbatim (build_notebook.py embeds them).\n"
            + embedded_source()
            + "\n\n\n"
            + SNAPSHOT_RUN,
        ),
        cell("code", INSTALL),
        cell("code", DIFF),
        cell("code", ARCHIVE),
        cell("code", MANIFEST),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
(HERE / "arc2-site-0290.ipynb").write_text(json.dumps(notebook, indent=1) + "\n")
print("wrote", HERE / "arc2-site-0290.ipynb")
