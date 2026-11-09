"""Replay a recorded wheelhouse install on a Kaggle kernel instead of running pip.

The engine's install (``ullm.kaggle``) is ``pip install --no-index`` from a
wheelhouse, ~200 s of a kernel's boot on the RTX box. The notebook that
downloads the wheelhouse (``tools/kaggle/arc3-site-0290``) then runs that same
install on the kernel's own image, after the competition's ``arc-agi`` install
as TAAF's notebook runs it, and records the result beside ``wheels/``: every
file pip added, packed into one zstd archive, and a manifest of the
distributions it removed and of the installed state it started from. pip
resolves everything there; a kernel only replays what pip decided.

A kernel replays when the record matches it: the same requirements, the same
wheelhouse files, the same Python, and the same installed distributions as the
record started from. It then removes what pip removed, as pip does (the files
each distribution's ``RECORD`` lists, each source file's cached bytecode, the
directories left empty), and unpacks what pip added. A mismatch, a missing
decompressor or a record that cannot be read runs pip instead, before anything
is touched. A replay that fails, or leaves a state other than the one the
record ended in, fails the install: by then files are removed, and pip cannot
repair a half-unpacked distribution.

``canonical``, ``site_dirs``, ``snapshot`` and ``state_digest`` are the builder's own
functions (the notebook embeds their source), so a kernel's digest and the
record's are computed alike. Nothing here imports beyond the standard
library: it runs before the engine is installed.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import re
import shlex
import shutil
import site
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any

MANIFEST = "site_delta.json"
FORMAT = 1


def canonical(name: str) -> str:
    """A distribution name as pip compares it."""
    return re.sub(r"[-_.]+", "-", name).lower()


def site_dirs() -> list[str]:
    """The interpreter's own site directories: where the image's packages
    are and pip installs. Not the rest of sys.path, which on a kernel holds
    the source bundle's roots (TAAF's ``.pth``) and in a notebook its own."""
    dirs = list(site.getsitepackages())
    if site.ENABLE_USER_SITE:
        dirs.append(site.getusersitepackages())
    return [d for d in dict.fromkeys(dirs) if os.path.isdir(d)]


def snapshot() -> dict[str, dict[str, Any]]:
    """Every installed distribution: canonical name, version, its dist-info
    path, and the absolute files its RECORD lists."""
    importlib.invalidate_caches()
    dists: dict[str, dict[str, Any]] = {}
    for dist in md.distributions(path=site_dirs()):
        info = pathlib.Path(dist._path)  # type: ignore[attr-defined]  # the .dist-info / .egg-info directory
        files: list[str] = []
        record = info / "RECORD"
        if record.is_file():
            for row in record.read_text(errors="replace").splitlines():
                rel = row.rsplit(",", 2)[0] if row.count(",") >= 2 else row
                if rel:
                    files.append(os.path.normpath(os.path.join(info.parent, rel)))
        dists[str(info)] = {"name": canonical(dist.metadata["Name"]), "version": dist.version, "files": files}
    return dists


def state_digest(dists: dict[str, dict[str, Any]]) -> tuple[str, list[str]]:
    """A digest of every distribution's name, version and dist-info path, and its lines."""
    lines = sorted(f"{d['name']}=={d['version']}@{path}" for path, d in dists.items())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest(), lines


def _wheel_listing(wheels: pathlib.Path) -> list[list[object]]:
    """The wheelhouse's files, by name and size."""
    return [[p.name, p.stat().st_size] for p in sorted(wheels.iterdir()) if p.is_file()]


def refusal(manifest: dict[str, Any], *, wheels: pathlib.Path, requirements: Sequence[str]) -> str | None:
    """Why this kernel cannot replay ``manifest``, or None when it can."""
    if manifest.get("format") != FORMAT:
        return f"record format {manifest.get('format')!r}, this reader takes {FORMAT}"
    if manifest["requirements"] != list(requirements):
        return "the record installed a different requirement list"
    if manifest["python"] != sys.version:
        return f"the record ran Python {manifest['python'].split()[0]}, this kernel {sys.version.split()[0]}"
    if manifest["wheelhouse_files"] != _wheel_listing(wheels):
        return "the attached wheelhouse is not the one the record installed from"
    current, lines = state_digest(snapshot())
    if current != manifest["before"]["sha256"]:
        here, recorded = set(lines), set(manifest["before"]["distributions"])
        return (
            "the installed distributions are not the ones the record started from: "
            f"here only {sorted(here - recorded)[:8]}, recorded only {sorted(recorded - here)[:8]}"
        )
    return None


def _remove(files: Sequence[str]) -> int:
    """Remove a distribution as pip does: its RECORD's files, the cached
    bytecode of its source files, then the directories left empty."""
    removed = 0
    parents: set[str] = set()
    for path in files:
        candidates = [path]
        if path.endswith(".py"):
            candidates.append(importlib.util.cache_from_source(path))
        for candidate in candidates:
            if os.path.lexists(candidate) and not os.path.isdir(candidate):
                os.unlink(candidate)
                removed += 1
                parents.add(os.path.dirname(candidate))
    for directory in sorted(parents, key=len, reverse=True):
        while directory not in ("/", "") and os.path.isdir(directory) and not os.listdir(directory):
            os.rmdir(directory)
            directory = os.path.dirname(directory)
    return removed


def _decompressor(archive: str) -> str | None:
    """The command or module that reads ``archive``: the record packs zstd
    where the image has it and gzip (pigz) where it has not; None when this
    image can read neither."""
    if archive.endswith(".tar.zst"):
        if shutil.which("zstd"):
            return "zstd -dc --no-progress"
        try:
            import zstandard  # type: ignore[import-not-found]  # noqa: F401
        except ImportError:
            return None
        return "zstandard"
    if archive.endswith(".tar.gz"):
        return "pigz -dc" if shutil.which("pigz") else "gzip -dc" if shutil.which("gzip") else "tarfile"
    return None


def _unpack(archive: pathlib.Path, root: str | pathlib.Path = "/") -> None:
    """Unpack the record's archive at ``root``: the root it was packed from,
    or a scratch root to time it."""
    reader = _decompressor(str(archive))
    if reader is None:
        raise RuntimeError(f"no reader for {archive} on this image")
    if reader not in ("zstandard", "tarfile"):
        subprocess.run(
            [
                "/bin/bash",
                "-o",
                "pipefail",
                "-c",
                f"{reader} {shlex.quote(str(archive))} | tar -x -C {shlex.quote(str(root))} -f -",
            ],
            check=True,
        )
        return
    import tarfile

    with open(archive, "rb") as raw:
        if reader == "tarfile":
            with tarfile.open(fileobj=raw, mode="r|gz") as tar:
                tar.extractall(root, filter="fully_trusted")
            return
        import zstandard  # type: ignore[import-not-found]

        with zstandard.ZstdDecompressor().stream_reader(raw) as stream:
            if not shutil.which("tar"):
                with tarfile.open(fileobj=stream, mode="r|") as tar:
                    tar.extractall(root, filter="fully_trusted")
                return
            # tar writes the files, the module decompresses beside it: Python's
            # own extraction of ~145k files took 121 s on a 4-core session.
            extract = subprocess.Popen(["tar", "-x", "-C", str(root), "-f", "-"], stdin=subprocess.PIPE)
            assert extract.stdin is not None
            try:
                shutil.copyfileobj(stream, extract.stdin, 16 << 20)
            finally:
                extract.stdin.close()
            if extract.wait() != 0:
                raise subprocess.CalledProcessError(extract.returncode, extract.args)


def pip_install(wheels: pathlib.Path, requirements: Sequence[str]) -> None:
    """The install a kernel runs without a record (``ullm.kaggle``)."""
    command = [sys.executable, "-m", "pip", "install", "--no-input", "--no-index", "--find-links", str(wheels)]
    subprocess.run([*command, *requirements], check=True)


def install(wheels_ref: str, requirements: Sequence[str]) -> str:
    """Install ``requirements`` from wheelhouse ``wheels_ref``: replay the
    record beside its wheels when it matches this kernel, else pip. Returns
    which ran, and writes a report beside the run's outputs; a replay that
    fails raises once the report is written."""
    import ullm.kaggle

    start = time.monotonic()
    wheels = ullm.kaggle.resolve_kaggle_wheel_dir(wheels_ref)
    report: dict[str, object] = {
        "wheels": wheels_ref,
        "mount": str(wheels.parent),
        # What the kernel actually ran on: whether the pin (TAAF R2.65) held.
        "image": os.environ.get("KAGGLE_DOCKER_IMAGE"),
    }
    record: pathlib.Path | None = None
    manifest: dict[str, Any] | None = None
    reason: str | None
    try:
        if not (wheels.parent / MANIFEST).is_file():
            reason = f"no record beside {wheels}"
        else:
            record = wheels.parent
            loaded: dict[str, Any] = json.loads((record / MANIFEST).read_text())
            manifest = loaded
            reason = refusal(loaded, wheels=wheels, requirements=requirements)
            if reason is None and _decompressor(loaded["archive"]["file"]) is None:
                reason = "this image cannot read the record's archive"
            archive = record / loaded["archive"]["file"]
            if reason is None and archive.stat().st_size != loaded["archive"]["bytes"]:
                reason = "the record's archive is not the size its manifest gives"
    except Exception as exc:  # a record that cannot be read is no reason to fail the kernel
        reason, manifest = f"reading the record failed: {exc!r}", None
    report["checked_s"] = round(time.monotonic() - start, 1)

    if reason is not None or record is None or manifest is None:
        print(f"[site] installing with pip: {reason}", flush=True)
        pip_install(wheels, requirements)
        report.update(path="pip", reason=reason, total_s=round(time.monotonic() - start, 1))
        _write_report(report)
        return "pip"

    try:
        replay_start = time.monotonic()
        removed = sum(_remove(entry["files"]) for entry in manifest["removed"])
        _unpack(record / manifest["archive"]["file"])
        after, _ = state_digest(snapshot())
        report.update(
            removed_files=removed,
            unpacked_files=manifest["archive"]["members"],
            replay_s=round(time.monotonic() - replay_start, 1),
        )
        if after != manifest["after_sha256"]:
            raise RuntimeError("the replayed state is not the one the record ended in")
    except Exception as exc:
        # Files are already removed and pip cannot repair a half-unpacked
        # distribution: the kernel ends here, the cause in its report.
        report.update(path="replay failed", reason=repr(exc), total_s=round(time.monotonic() - start, 1))
        _write_report(report)
        raise
    report.update(path="replay", total_s=round(time.monotonic() - start, 1))
    print(
        f"[site] replayed {wheels_ref}: {report['unpacked_files']} files unpacked, {removed} removed, "
        f"{report['total_s']} s",
        flush=True,
    )
    _write_report(report)
    return "replay"


def _write_report(report: dict[str, object]) -> None:
    """Write the install's report beside the run's outputs, if the directory takes it."""
    out = pathlib.Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR", "/kaggle/working"))
    try:
        (out / "site_install.json").write_text(json.dumps(report, indent=1))
    except OSError:
        pass


def main(argv: Sequence[str] | None = None) -> None:
    """``python -m ullm.kaggle_site WHEELS_REF REQUIREMENT...``"""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        raise SystemExit("usage: python -m ullm.kaggle_site WHEELS_REF REQUIREMENT...")
    install(args[0], args[1:])


if __name__ == "__main__":
    main()
