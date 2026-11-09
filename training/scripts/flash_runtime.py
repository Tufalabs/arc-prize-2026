"""Bundle, verify and attest the isolated Flash runtime (vLLM 0.29 fork + Transformers 5.16.1).

Verbatim from experiment/ar25-flash-router-backprop (scripts/routing_runtime.py); the
pinned revisions below are the ones the router experiments validated."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import pathlib
import subprocess
import sys
import sysconfig
import tarfile

BASE_VLLM = "29da4c8996db9a677e441fc1fa6e4755752dba70"
VLLM = "67208a493bb4dea543091939d2c60cf0ec0c1596"
COMPILED_BASE = "98dff2a81d747d1dba01a47f939f48c3526d4206"
PLATYPUS = "051f19ea3a0b62c5244cdcb5e4e16bcc7a1d4864"
FRAMEWORK = "4685cda91f3380e761cc6fc68c21b787e5521b6c"
BASE_ARCHIVE_SHA256 = "26c5fd32c9b4fab7c055fe3895d39119a97bea5b96a531bbfcc02b0e588a11c1"


def digest(data: bytes) -> str:
    """Return the content digest used by the overlay manifest."""
    return hashlib.sha256(data).hexdigest()


def git(repo: str, *args: str) -> bytes:
    """Read a pinned object without changing the supplied checkout."""
    return subprocess.check_output(["git", "-C", repo, *args])


def add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    """Add a deterministic regular-file member to a bundle."""
    member = tarfile.TarInfo(name)
    member.size = len(data)
    member.mode = 0o644
    archive.addfile(member, io.BytesIO(data))


def build(args: argparse.Namespace) -> None:
    """Create a small offline source bundle from three exact git revisions."""
    changes = (
        git(args.vllm_source, "diff", "--name-status", BASE_VLLM, VLLM)
        .decode()
        .splitlines()
    )
    replacements: dict[str, dict[str, str | None]] = {}
    files: dict[str, bytes] = {}
    for change in changes:
        status, name = change.split("\t")
        if name.startswith("tests/"):
            continue
        if (
            not name.startswith("vllm/")
            or not name.endswith(".py")
            or status not in {"M", "A"}
        ):
            raise RuntimeError(
                f"The fork delta is not a Python-only addition/update: {change}"
            )
        old = (
            git(args.vllm_source, "show", f"{BASE_VLLM}:{name}")
            if status == "M"
            else None
        )
        new = git(args.vllm_source, "show", f"{VLLM}:{name}")
        files[f"overlay/{name}"] = new
        replacements[name] = {
            "before": digest(old) if old is not None else None,
            "after": digest(new),
        }
    sources = (
        (
            args.platypus_source,
            PLATYPUS,
            "platypus",
            ["packages/platypus", "packages/duck", "packages/ullm"],
        ),
        (
            args.framework_source,
            FRAMEWORK,
            "framework",
            ["src", "pyproject.toml", "README.md"],
        ),
    )
    for repo, revision, prefix, paths in sources:
        data = git(repo, "archive", revision, *paths)
        with tarfile.open(fileobj=io.BytesIO(data)) as source:
            for member in source.getmembers():
                if not member.isfile():
                    continue
                body = source.extractfile(member)
                assert body is not None
                files[f"source/{prefix}/{member.name}"] = body.read()
    manifest = {
        "schema": 1,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
        "base_vllm": BASE_VLLM,
        "vllm": VLLM,
        "compiled_base": COMPILED_BASE,
        "platypus": PLATYPUS,
        "framework": FRAMEWORK,
        "replacements": replacements,
        "files": {name: digest(data) for name, data in sorted(files.items())},
    }
    target = pathlib.Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(target, "w:gz") as output:
        add_bytes(
            output,
            "manifest.json",
            json.dumps(manifest, sort_keys=True, indent=2).encode(),
        )
        for name, data in sorted(files.items()):
            add_bytes(output, name, data)
    print(
        json.dumps(
            {
                "bundle": str(target.resolve()),
                "sha256": digest(target.read_bytes()),
                "files": len(files),
            }
        )
    )


def apply(args: argparse.Namespace) -> None:
    """Verify the installed base and every payload byte before applying changes."""
    runtime = pathlib.Path(args.runtime).resolve()
    site = pathlib.Path(sysconfig.get_paths()["purelib"]).resolve()
    if not site.is_relative_to(runtime):
        raise RuntimeError(
            f"Use the isolated runtime interpreter; {site} is not inside {runtime}"
        )
    with tarfile.open(args.bundle) as archive:
        bodies: dict[str, bytes] = {}
        for member in archive.getmembers():
            path = pathlib.PurePosixPath(member.name)
            if not member.isfile() or path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"Unsafe overlay member: {member.name}")
            body = archive.extractfile(member)
            assert body is not None
            bodies[member.name] = body.read()
    manifest = json.loads(bodies.pop("manifest.json"))
    for key, expected in {
        "base_vllm": BASE_VLLM,
        "vllm": VLLM,
        "compiled_base": COMPILED_BASE,
        "platypus": PLATYPUS,
        "framework": FRAMEWORK,
        "base_archive_sha256": BASE_ARCHIVE_SHA256,
    }.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"Unexpected runtime pin {key}: {manifest.get(key)}")
    if manifest["files"] != {
        name: digest(data) for name, data in sorted(bodies.items())
    }:
        raise RuntimeError("Overlay payload hashes do not match its manifest")
    distribution = importlib.metadata.distribution("vllm")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    build_data = json.loads(distribution.read_text("uv_build.json") or "{}")
    if direct_url.get("vcs_info", {}).get("commit_id") != BASE_VLLM:
        raise RuntimeError("Installed vLLM is not the attested archive's fork revision")
    if (
        build_data.get("extra_build_variables", {}).get("VLLM_PRECOMPILED_WHEEL_COMMIT")
        != COMPILED_BASE
    ):
        raise RuntimeError("Installed native vLLM wheel has a different base commit")
    for name, hashes in manifest["replacements"].items():
        target = site / name
        actual = digest(target.read_bytes()) if target.exists() else None
        if actual != hashes["before"]:
            raise RuntimeError(f"Installed file differs from the pinned base: {name}")
    # All checks finish before replacing any runtime sources.
    for name, data in bodies.items():
        target = (
            site / name.removeprefix("overlay/")
            if name.startswith("overlay/")
            else runtime / name
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    receipt = dict(manifest)
    receipt["overlay_sha256"] = digest(pathlib.Path(args.bundle).read_bytes())
    receipt["base_distribution_version"] = distribution.version
    receipt["native_extensions_modified"] = False
    (runtime / "routing_runtime_overlay.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"Applied {len(manifest['replacements'])} verified Python files; native vLLM payload unchanged",
        file=sys.stderr,
    )


def attest(args: argparse.Namespace) -> None:
    """Record exact runtime/source identities beside the experiment outputs."""
    import torch

    runtime = pathlib.Path(sys.prefix)
    overlay = runtime / "routing_runtime_overlay.json"
    if not overlay.is_file():
        raise RuntimeError(f"No verified Flash overlay receipt: {overlay}")
    data = json.loads(overlay.read_text())
    site = pathlib.Path(sysconfig.get_paths()["purelib"])
    for name, hashes in data["replacements"].items():
        if digest((site / name).read_bytes()) != hashes["after"]:
            raise RuntimeError(f"Runtime source changed after provisioning: {name}")
    source_prefixes = [
        "source/platypus/packages/duck/src/",
        "source/platypus/packages/ullm/src/",
        "source/platypus/packages/platypus/src/",
        "source/framework/src/",
    ]
    for name, expected in data["files"].items():
        prefix = next(
            (prefix for prefix in source_prefixes if name.startswith(prefix)), None
        )
        if prefix is not None:
            installed = site / name.removeprefix(prefix)
            if not installed.is_file() or digest(installed.read_bytes()) != expected:
                raise RuntimeError(
                    f"Installed harness source differs from its pinned bundle: {name}"
                )
    names = [
        "vllm",
        "torch",
        "transformers",
        "flashinfer-python",
        "duck",
        "ullm",
        "platypus",
        "tufa-arc-agi-framework",
        "arc-agi-3-local",
    ]
    data["versions"] = {name: importlib.metadata.version(name) for name in names}
    data["python"] = sys.version
    data["executable"] = sys.executable
    data["torch_cuda"] = torch.version.cuda
    root = pathlib.Path(args.repo).resolve()
    data["trainer_commit"] = git(str(root), "rev-parse", "HEAD").decode().strip()
    data["trainer_status"] = (
        git(str(root), "status", "--porcelain").decode().splitlines()
    )
    paths = (
        git(
            str(root),
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "training",
            "scripts",
            "configs",
        )
        .decode()
        .splitlines()
    )
    data["trainer_source_sha256"] = {
        name: digest((root / name).read_bytes())
        for name in paths
        if (root / name).is_file()
    }
    pathlib.Path(args.output).write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    """Dispatch source bundling, isolated application, or run attestation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    bundle = commands.add_parser("build")
    for name in ("vllm-source", "platypus-source", "framework-source", "output"):
        bundle.add_argument(f"--{name}", required=True)
    bundle.set_defaults(func=build)
    patch = commands.add_parser("apply")
    patch.add_argument("--runtime", required=True)
    patch.add_argument("--bundle", required=True)
    patch.set_defaults(func=apply)
    receipt = commands.add_parser("attest")
    receipt.add_argument("--repo", required=True)
    receipt.add_argument("--output", required=True)
    receipt.set_defaults(func=attest)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
