"""The vLLM fork's Kaggle overlay: built on the launcher, applied offline.

Kaggle installs stock vLLM from a wheelhouse ``--no-index``; this module
carries the fork's difference from that wheel across as a single payload
file and replaces the installed files with it before the engine starts.

The delta is taken against the precompiled base wheel the fork was built
on, not against a git revision. Two thirds of the installed tree
(vendored ``third_party``, ``vllm_flash_attn/cute``, the generated
``_version.py``) has no counterpart in the fork's git tree at all, so a
name list from ``git diff`` cannot see a change there — and a bumped
vendored dependency is exactly the change that would slip through.

Stdlib only, and no other ``ullm`` import: the apply half runs as a
Kaggle setup command, on a kernel where nothing of ours is configured
yet and importing the engine would cost minutes.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from typing import Any, cast

#: The nightly/commit wheel index vLLM's own precompiled build reads.
VLLM_WHEEL_INDEX = "https://wheels.vllm.ai"
#: CUDA major version -> hosted wheel variant, as vLLM's setup.py maps it.
CUDA_WHEEL_VARIANTS = {12: "cu129", 13: "cu130"}
#: The payload file name, and the manifest member inside it. The payload is
#: a base64-encoded gzipped tar, and the encoding is load-bearing: Kaggle's
#: dataset ingestion unpacks an archive it recognizes, recursively — a
#: plain .tar came back as a directory, which puts every name inside it in
#: front of Kaggle's filename validation. 561 of the engine's fused-MoE
#: config files carry "=" and "," and are rejected there.
OVERLAY_PAYLOAD_NAME = "vllm_fork_overlay.b64"
MANIFEST_MEMBER = "overlay_manifest.json"
#: Bumped when the manifest's shape changes; apply refuses what it cannot read.
MANIFEST_SCHEMA = 1


@dataclasses.dataclass(frozen=True)
class InstalledVllm:
    """Where the launcher's vLLM lives and which two commits made it.

    ``base_commit`` is the upstream revision whose precompiled wheel the
    build unpacked, read from uv's own record of the build variables, so
    it is what the venv was actually built with rather than what a
    pyproject says today. ``fork_commit`` is the revision pip resolved.
    """

    site_packages: pathlib.Path
    version: str  # e.g. 0.27.2.dev18+g968051f96
    fork_commit: str
    base_commit: str


def read_installed_vllm() -> InstalledVllm:
    """Describe the installed vLLM, or explain which record is missing."""
    distribution = importlib.metadata.distribution("vllm")
    # PEP 610 records the VCS revision pip/uv resolved.
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        raise RuntimeError(
            "the installed vllm has no direct_url.json: it came from an index, not from the "
            "arc-prize-2026-vllm pin, so there is no fork delta to ship (see vllm_fork.md)"
        )
    direct_url = json.loads(direct_url_text)
    fork_commit = str(direct_url.get("vcs_info", {}).get("commit_id") or "")
    if not fork_commit:
        raise RuntimeError(f"the installed vllm's direct_url.json records no commit: {direct_url}")
    # uv records extra-build-variables with the installed distribution, which
    # is the only place the base commit survives the build.
    build_text = distribution.read_text("uv_build.json")
    if not build_text:
        raise RuntimeError(
            "the installed vllm has no uv_build.json: it was not built through uv's "
            "extra-build-variables, so the precompiled base commit is unknown"
        )
    build_variables = json.loads(build_text).get("extra_build_variables", {})
    base_commit = str(build_variables.get("VLLM_PRECOMPILED_WHEEL_COMMIT") or "")
    if len(base_commit) != 40:
        raise RuntimeError(
            f"VLLM_PRECOMPILED_WHEEL_COMMIT is not a full revision ({base_commit!r}); the wheel this "
            "was built on cannot be identified, so neither can the delta against it"
        )
    site_packages = pathlib.Path(str(distribution.locate_file("vllm"))).resolve().parent
    return InstalledVllm(
        site_packages=site_packages,
        version=distribution.version,
        fork_commit=fork_commit,
        base_commit=base_commit,
    )


def detect_cuda_variant() -> str | None:
    """The wheel variant for this machine's CUDA, or ``None`` to take the
    index's default. Ported from vLLM's ``detect_system_cuda_variant``."""
    explicit = os.environ.get("VLLM_PRECOMPILED_WHEEL_VARIANT")
    if explicit:
        return explicit
    main_cuda = os.environ.get("VLLM_MAIN_CUDA_VERSION")
    if main_cuda:
        return "cu" + main_cuda.replace(".", "")[:3]
    cuda_version: str | None = None
    try:
        import torch  # noqa: PLC0415

        cuda_version = torch.version.cuda
    except Exception:
        cuda_version = None
    if not cuda_version:
        try:
            probe = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", probe.stdout)
            cuda_version = match.group(1) if match else None
        except Exception:
            cuda_version = None
    if not cuda_version:
        return None
    major = int(cuda_version.split(".")[0])
    return CUDA_WHEEL_VARIANTS.get(major, CUDA_WHEEL_VARIANTS[max(CUDA_WHEEL_VARIANTS)])


def _wheel_metadata(base_commit: str, variant: str | None) -> tuple[list[dict[str, Any]], str]:
    """The index's wheel list for one commit and variant, plus its base URL."""
    variant_dir = f"{variant}/" if variant is not None else ""
    repo_url = f"{VLLM_WHEEL_INDEX}/{base_commit}/{variant_dir}vllm/"
    with urllib.request.urlopen(repo_url + "metadata.json") as response:
        wheels = json.loads(response.read().decode("utf-8"))
    return wheels, repo_url


def base_wheel_url(base_commit: str, *, variant: str | None = None) -> str:
    """The precompiled wheel URL for ``base_commit`` on this platform.

    Selection follows vLLM's own build: the requested variant, falling
    back to the index default, then the first listed vllm wheel whose
    platform tag carries this machine's architecture.
    """
    variant = detect_cuda_variant() if variant is None else variant
    try:
        wheels, repo_url = _wheel_metadata(base_commit, variant)
    except Exception:
        wheels, repo_url = _wheel_metadata(base_commit, None)
    architecture = platform.machine()
    for wheel in wheels:
        if wheel.get("package_name") == "vllm" and architecture in str(wheel.get("platform_tag", "")):
            return urllib.parse.urljoin(repo_url, str(wheel["path"]))
    raise RuntimeError(f"no vllm wheel for {architecture} at {repo_url}")


def _wheel_cache_dir() -> pathlib.Path:
    """Where downloaded base wheels are kept between deploys."""
    override = os.environ.get("ULLM_VLLM_WHEEL_CACHE")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".cache" / "ullm" / "vllm-base-wheels"


def fetch_base_wheel(base_commit: str, *, cache_dir: pathlib.Path | None = None) -> pathlib.Path:
    """The base wheel for ``base_commit``, downloading it once per commit."""
    cache_dir = _wheel_cache_dir() if cache_dir is None else cache_dir
    url = base_wheel_url(base_commit)
    destination = cache_dir / base_commit / urllib.parse.unquote(url.rsplit("/", maxsplit=1)[-1])
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Downloaded beside the target and renamed, so an interrupted deploy
    # never leaves a half wheel that the next one would trust.
    partial = destination.with_name(destination.name + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)
    return destination


def _digest(path: pathlib.Path) -> str:
    """The sha256 of a file, streamed — the base wheel is a 300 MB one."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _wheel_package_digests(wheel_path: pathlib.Path) -> dict[str, str]:
    """Every ``vllm/`` member of a wheel, as ``relative path -> sha256``."""
    digests: dict[str, str] = {}
    with zipfile.ZipFile(wheel_path) as archive:
        for name in archive.namelist():
            if name.endswith("/") or not name.startswith("vllm/"):
                continue
            digests[name] = hashlib.sha256(archive.read(name)).hexdigest()
    return digests


def _installed_package_digests(site_packages: pathlib.Path) -> dict[str, str]:
    """Every file of the installed ``vllm`` package, as ``relative path -> sha256``.

    ``__pycache__`` is skipped: bytecode is rebuilt from whatever source
    ends up in place, so it is noise in a comparison and a hazard in a
    payload.
    """
    digests: dict[str, str] = {}
    for path in sorted((site_packages / "vllm").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        digests[path.relative_to(site_packages).as_posix()] = _digest(path)
    return digests


def build_overlay(dest_payload: pathlib.Path, *, cache_dir: pathlib.Path | None = None) -> pathlib.Path:
    """Write the launcher's fork-versus-base delta to ``dest_payload``.

    The archive holds the changed and added files at their site-packages
    relative paths, plus a manifest recording, for every path it touches,
    the digest the base wheel has (what Kaggle must be holding before the
    overlay lands) and the digest the fork has (what it must hold after).
    """
    installed = read_installed_vllm()
    wheel_path = fetch_base_wheel(installed.base_commit, cache_dir=cache_dir)
    base = _wheel_package_digests(wheel_path)
    current = _installed_package_digests(installed.site_packages)
    replace = {path: digest for path, digest in current.items() if path in base and base[path] != digest}
    add = {path: digest for path, digest in current.items() if path not in base}
    remove = sorted(path for path in base if path not in current)
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "base_commit": installed.base_commit,
        "base_wheel": wheel_path.name,
        "base_wheel_sha256": _digest(wheel_path),
        "base_version": _wheel_version(wheel_path),
        "fork_commit": installed.fork_commit,
        "fork_version": installed.version,
        "replace": {path: {"before": base[path], "after": digest} for path, digest in sorted(replace.items())},
        "add": {path: {"after": digest} for path, digest in sorted(add.items())},
        "remove": remove,
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        info = tarfile.TarInfo(MANIFEST_MEMBER)
        info.size = len(manifest_bytes)
        archive.addfile(info, io.BytesIO(manifest_bytes))
        for path in sorted([*replace, *add]):
            archive.add(installed.site_packages / path, arcname=path)
    dest_payload.parent.mkdir(parents=True, exist_ok=True)
    # Written beside the target and renamed: a deploy interrupted mid-write
    # must not leave a payload the next one would read.
    partial = dest_payload.with_name(dest_payload.name + ".partial")
    partial.write_bytes(base64.b64encode(buffer.getvalue()))
    partial.replace(dest_payload)
    return dest_payload


def _wheel_version(wheel_path: pathlib.Path) -> str:
    """The version a wheel declares, taken from its file name."""
    return wheel_path.name.split("-")[1]


def open_payload(payload_path: pathlib.Path) -> tarfile.TarFile:
    """The archive inside an overlay payload, decoded in memory."""
    if not payload_path.is_file():
        raise RuntimeError(
            f"no vllm overlay payload at {payload_path}. Kaggle unpacks an archive it recognizes, "
            "so the payload is base64-encoded to arrive as one file; a directory here means it "
            "was uploaded in a form Kaggle unpacked."
        )
    raw = base64.b64decode(payload_path.read_bytes(), validate=True)
    return tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz")


def read_manifest(payload_path: pathlib.Path) -> dict[str, Any]:
    """The manifest inside an overlay payload."""
    with open_payload(payload_path) as archive:
        member = archive.extractfile(MANIFEST_MEMBER)
        if member is None:
            raise RuntimeError(f"{payload_path} carries no {MANIFEST_MEMBER}")
        manifest = json.loads(member.read().decode("utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise RuntimeError(f"overlay manifest schema {manifest.get('schema')!r}, expected {MANIFEST_SCHEMA}")
    return manifest


def apply_overlay(payload_path: pathlib.Path, *, site_packages: pathlib.Path | None = None) -> str:
    """Replace the installed vLLM's files with the fork's, in place.

    Refuses unless every file it is about to overwrite still holds the
    base wheel's bytes: the fork's Python paired with binaries from
    another build is worse than not overlaying at all, and a stock engine
    accepts the fork's sampling parameter as an unknown extra and serves
    the request unchanged, so the failure it prevents is silent.

    Returns a one-line summary. Applying twice is not an error — a
    payload already in place is reported as such.
    """
    site_packages = pathlib.Path(sysconfig.get_paths()["purelib"]) if site_packages is None else site_packages
    manifest = read_manifest(payload_path)
    replace = _digest_section(manifest, "replace")
    add = _digest_section(manifest, "add")
    remove = [str(path) for path in cast(list[Any], manifest["remove"])]
    wanted = {path: entry["after"] for path, entry in [*replace.items(), *add.items()]}

    if _matches(site_packages, wanted) and not any((site_packages / path).exists() for path in remove):
        return f"vllm overlay already applied: {manifest['fork_version']}"

    installed_version = installed_version_at(site_packages)
    if installed_version != manifest["base_version"]:
        raise RuntimeError(
            f"installed vllm is {installed_version}, but this overlay was built against "
            f"{manifest['base_version']} (commit {manifest['base_commit']}). Overlaying the fork's "
            "Python onto another build's binaries pairs mismatched halves of an engine."
        )
    _verify_pre_images(site_packages, replace, add, remove)

    with open_payload(payload_path) as archive:
        members = [member for member in archive.getmembers() if member.name != MANIFEST_MEMBER]
        archive.extractall(site_packages, members=members, filter="data")
    for path in remove:
        (site_packages / path).unlink()
    # Bytecode cached against the stock sources would otherwise be checked
    # only by mtime and size, both of which a copy can reproduce.
    for cache_dir in (site_packages / "vllm").rglob("__pycache__"):
        shutil.rmtree(cache_dir, ignore_errors=True)

    mismatched = sorted(path for path, digest in wanted.items() if _file_digest(site_packages / path) != digest)
    if mismatched:
        raise RuntimeError(f"vllm overlay wrote {len(mismatched)} file(s) that do not match the manifest: {mismatched}")
    return (
        f"vllm overlay applied: {manifest['base_version']} -> {manifest['fork_version']} "
        f"({len(replace)} replaced, {len(add)} added, {len(remove)} removed)"
    )


def installed_version_at(site_packages: pathlib.Path) -> str:
    """The vLLM version installed in ``site_packages``.

    Read from that tree rather than from the running interpreter: the
    overlay's question is what it is about to overwrite, which need not be
    what this process imports.
    """
    context = importlib.metadata.DistributionFinder.Context(path=[str(site_packages)])
    for distribution in importlib.metadata.Distribution.discover(context=context):
        if (distribution.metadata["Name"] or "").lower() == "vllm":
            return distribution.version
    raise RuntimeError(f"no installed vllm distribution under {site_packages}")


def _file_digest(path: pathlib.Path) -> str | None:
    """The sha256 of ``path``, or ``None`` when it does not exist."""
    return _digest(path) if path.is_file() else None


def _matches(site_packages: pathlib.Path, wanted: dict[str, str]) -> bool:
    """Whether every wanted path already holds its post-overlay digest."""
    return all(_file_digest(site_packages / path) == digest for path, digest in wanted.items())


def _digest_section(manifest: dict[str, Any], name: str) -> dict[str, dict[str, str]]:
    """One ``path -> {before, after}`` block of a manifest, as plain strings."""
    section = cast(dict[str, dict[str, str]], manifest[name])
    return {str(path): {key: str(value) for key, value in entry.items()} for path, entry in section.items()}


def _verify_pre_images(
    site_packages: pathlib.Path,
    replace: dict[str, dict[str, str]],
    add: dict[str, dict[str, str]],
    remove: list[str],
) -> None:
    """Refuse an overlay whose target is not the base wheel it was built on."""
    problems: list[str] = []
    for path, entry in sorted(replace.items()):
        before = entry["before"]
        found = _file_digest(site_packages / path)
        if found is None:
            problems.append(f"{path}: missing, expected the base wheel's copy")
        elif found != before:
            problems.append(f"{path}: sha256 {found}, expected {before}")
    for path in sorted(add):
        if (site_packages / path).exists():
            problems.append(f"{path}: present, expected absent")
    for path in remove:
        if not (site_packages / path).exists():
            problems.append(f"{path}: absent, expected the base wheel's copy")
    if problems:
        listed = "\n  ".join(problems[:20])
        raise RuntimeError(
            "the installed vllm is not the base wheel this overlay was built against, so the "
            "overlay is refused; the wheelhouse and the fork's precompiled base must be the same "
            f"build. Mismatches:\n  {listed}"
        )


def apply_from_bundle() -> None:
    """Apply the overlay shipped in a TAAF Kaggle bundle, and say so.

    The entry point the Kaggle setup command calls; it locates the payload
    through the bundle path TAAF exports to setup commands.
    """
    bundle = os.environ.get("TAAF_KAGGLE_BUNDLE_DIR", "")
    if not bundle:
        raise RuntimeError("TAAF_KAGGLE_BUNDLE_DIR is unset; this runs as a TAAF Kaggle setup command")
    print(apply_overlay(pathlib.Path(bundle) / "payload" / OVERLAY_PAYLOAD_NAME), flush=True)


FORK_PIN_PATTERN = re.compile(r"arc-prize-2026-vllm(?:\.git)?@([0-9a-f]{40})")


def pinned_fork_commit(start: pathlib.Path | None = None) -> str | None:
    """The fork revision the workspace's root pyproject pins, or None when
    no workspace root is in reach (ullm installed from somewhere else).

    The pin is the ``vllm @ git+…arc-prize-2026-vllm@<sha>`` override in
    ``[tool.uv]``; found by walking up from this module, which the
    workspace installs in place.
    """
    here = (start or pathlib.Path(__file__)).resolve()
    for directory in (here, *here.parents):
        candidate = directory / "pyproject.toml"
        if not candidate.is_file():
            continue
        match = FORK_PIN_PATTERN.search(candidate.read_text(encoding="utf-8"))
        if match:
            return match.group(1)
    return None


def build_for_deploy(dest_dir: pathlib.Path | None = None, *, workspace: pathlib.Path | None = None) -> pathlib.Path:
    """Build the overlay into a deploy-time location and return its path.

    Called on the launcher while a Kaggle deployment is being assembled,
    so it writes somewhere stable rather than into the repo. The overlay
    is whatever vllm the launcher venv holds, so a venv left behind by a
    pin bump would ship the old engine without a word: when the
    workspace's pin is in reach, the installed revision has to match it.
    ``workspace`` is where the pin is looked for — the caller names it,
    since ullm installed as a package rather than a workspace member has
    no root of its own above it; with none given the search starts here.
    """
    pinned = pinned_fork_commit(workspace)
    if pinned is not None:
        installed = read_installed_vllm()
        if installed.fork_commit != pinned:
            raise RuntimeError(
                f"the launcher venv's vllm is {installed.version} ({installed.fork_commit[:10]}) but the "
                f"workspace pins {pinned[:10]}: `uv sync` before deploying, or Kaggle ships the old engine"
            )
    if dest_dir is None:
        dest_dir = pathlib.Path(tempfile.gettempdir()) / f"ullm-vllm-overlay-{os.getuid()}"
    return build_overlay(dest_dir / OVERLAY_PAYLOAD_NAME)
