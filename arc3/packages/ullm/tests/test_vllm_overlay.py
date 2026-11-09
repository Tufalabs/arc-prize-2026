"""The vLLM fork's Kaggle overlay: what it ships, and what it refuses.

Covers ``ullm.vllm_overlay``'s apply half against synthetic trees — the
build half needs the fork installed and the base wheel fetched, which the
deploy path does for real. The refusals are the point: a stock engine
accepts the fork's sampling parameter as an unknown extra and serves the
request unchanged, so an overlay that silently fails to land costs a
submission its whole design without a single error.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
import tarfile
from collections.abc import Mapping

import pytest
import ullm.vllm_overlay as vllm_overlay


def _write(path: pathlib.Path, text: str) -> str:
    """Write a file, creating parents, and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _site_packages(tmp_path: pathlib.Path, version: str = "0.27.1") -> pathlib.Path:
    """A minimal installed-vllm tree: two files and a dist-info."""
    site_packages = tmp_path / "site-packages"
    _write(site_packages / "vllm" / "sampling_params.py", "class SamplingParams: ...\n")
    _write(site_packages / "vllm" / "_version.py", "__version__ = '0.27.1'\n")
    _write(site_packages / f"vllm-{version}.dist-info" / "METADATA", f"Name: vllm\nVersion: {version}\n")
    return site_packages


def _overlay(
    tmp_path: pathlib.Path,
    *,
    replace: dict[str, tuple[str, str]] | None = None,
    add: dict[str, str] | None = None,
    remove: list[str] | None = None,
    base_version: str = "0.27.1",
) -> pathlib.Path:
    """An overlay payload whose members carry the post-overlay content.

    ``replace`` maps a path to its (before-content, after-content); the
    manifest records digests of both, as ``build_overlay`` does.
    """
    replace = replace or {}
    add = add or {}
    manifest = {
        "schema": vllm_overlay.MANIFEST_SCHEMA,
        "base_commit": "0" * 40,
        "base_wheel": "vllm-0.27.1-cp38-abi3-manylinux_2_28_x86_64.whl",
        "base_wheel_sha256": "unused",
        "base_version": base_version,
        "fork_commit": "1" * 40,
        "fork_version": "0.27.2.dev18+gforked",
        "replace": {
            path: {
                "before": hashlib.sha256(before.encode()).hexdigest(),
                "after": hashlib.sha256(after.encode()).hexdigest(),
            }
            for path, (before, after) in replace.items()
        },
        "add": {path: {"after": hashlib.sha256(after.encode()).hexdigest()} for path, after in add.items()},
        "remove": remove or [],
    }
    staging = tmp_path / "staging"
    for path, (_, after) in replace.items():
        _write(staging / path, after)
    for path, after in add.items():
        _write(staging / path, after)
    return _encode(tmp_path / vllm_overlay.OVERLAY_PAYLOAD_NAME, manifest, staging, [*replace, *add])


def _encode(
    payload_path: pathlib.Path,
    manifest: Mapping[str, object],
    staging: pathlib.Path,
    members: list[str],
) -> pathlib.Path:
    """Write a payload in the shipped encoding: base64 over a gzipped tar."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        blob = (json.dumps(manifest) + "\n").encode()
        info = tarfile.TarInfo(vllm_overlay.MANIFEST_MEMBER)
        info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))
        for path in members:
            archive.add(staging / path, arcname=path)
    payload_path.write_bytes(base64.b64encode(buffer.getvalue()))
    return payload_path


def test_overlay_replaces_adds_and_removes_in_place(tmp_path: pathlib.Path) -> None:
    """The engine is replaced where it stands: no sys.path entry, no second
    copy. Games run in spawned children and the engine spawns its own
    workers, so a path-order trick would have to hold across all of them."""
    site_packages = _site_packages(tmp_path)
    tar_path = _overlay(
        tmp_path,
        replace={"vllm/sampling_params.py": ("class SamplingParams: ...\n", "class SamplingParams: forked\n")},
        add={"vllm/v1/worker/mamba_utils.py": "# new in the fork\n"},
        remove=["vllm/_version.py"],
    )

    summary = vllm_overlay.apply_overlay(tar_path, site_packages=site_packages)

    assert (site_packages / "vllm" / "sampling_params.py").read_text() == "class SamplingParams: forked\n"
    assert (site_packages / "vllm" / "v1" / "worker" / "mamba_utils.py").read_text() == "# new in the fork\n"
    assert not (site_packages / "vllm" / "_version.py").exists()
    assert "1 replaced, 1 added, 1 removed" in summary


def test_payload_does_not_look_like_an_archive(tmp_path: pathlib.Path) -> None:
    """Kaggle's dataset ingestion unpacks an archive it recognizes, and does
    it recursively: a plain .tar came back as a directory, which puts every
    name inside it in front of Kaggle's filename validation — where 561 of
    the engine's fused-MoE config names, carrying "=" and ",", are rejected.
    So the payload must be neither an archive by name nor by content."""
    payload = _overlay(
        tmp_path,
        replace={"vllm/sampling_params.py": ("class SamplingParams: ...\n", "class SamplingParams: forked\n")},
    )

    assert payload.suffix == ".b64"
    raw = payload.read_bytes()
    assert raw[:2] != b"\x1f\x8b"  # gzip
    assert b"ustar" not in raw[:512]  # tar
    assert base64.b64decode(raw, validate=True)[:2] == b"\x1f\x8b"


def test_overlay_reapplied_is_reported_not_refused(tmp_path: pathlib.Path) -> None:
    """A kernel cell re-run must not fail: with the payload already in
    place there is nothing to do, and the pre-image check would otherwise
    read the fork's own bytes as a foreign build."""
    site_packages = _site_packages(tmp_path)
    tar_path = _overlay(
        tmp_path,
        replace={"vllm/sampling_params.py": ("class SamplingParams: ...\n", "class SamplingParams: forked\n")},
    )

    vllm_overlay.apply_overlay(tar_path, site_packages=site_packages)
    assert "already applied" in vllm_overlay.apply_overlay(tar_path, site_packages=site_packages)


def test_overlay_refuses_a_tree_that_is_not_its_base_wheel(tmp_path: pathlib.Path) -> None:
    """The fork's Python on another build's binaries pairs mismatched
    halves of an engine — worse than not overlaying at all. Every file the
    overlay is about to overwrite has to still hold the base wheel's bytes."""
    site_packages = _site_packages(tmp_path)
    (site_packages / "vllm" / "sampling_params.py").write_text("# a different build\n")
    tar_path = _overlay(
        tmp_path,
        replace={"vllm/sampling_params.py": ("class SamplingParams: ...\n", "class SamplingParams: forked\n")},
    )

    with pytest.raises(RuntimeError, match="not the base wheel this overlay was built against"):
        vllm_overlay.apply_overlay(tar_path, site_packages=site_packages)

    assert (site_packages / "vllm" / "sampling_params.py").read_text() == "# a different build\n"


def test_overlay_refuses_a_wheelhouse_on_another_version(tmp_path: pathlib.Path) -> None:
    """The version check names the mismatch before the per-file digests do,
    which is the difference between "the wheelhouse moved" and "one file
    was patched"."""
    site_packages = _site_packages(tmp_path, version="0.25.1")
    tar_path = _overlay(
        tmp_path,
        replace={"vllm/sampling_params.py": ("class SamplingParams: ...\n", "class SamplingParams: forked\n")},
    )

    with pytest.raises(RuntimeError, match="installed vllm is 0.25.1"):
        vllm_overlay.apply_overlay(tar_path, site_packages=site_packages)


def test_overlay_refuses_an_unreadable_manifest_schema(tmp_path: pathlib.Path) -> None:
    """A payload from a future build is refused rather than half-read."""
    manifest = vllm_overlay.read_manifest(_overlay(tmp_path))
    manifest["schema"] = vllm_overlay.MANIFEST_SCHEMA + 1
    rewritten = _encode(tmp_path / "future.b64", manifest, tmp_path / "staging", [])

    with pytest.raises(RuntimeError, match="overlay manifest schema"):
        vllm_overlay.apply_overlay(rewritten, site_packages=_site_packages(tmp_path))


def test_installed_version_reads_the_target_tree_not_the_interpreter(tmp_path: pathlib.Path) -> None:
    """The question is what the overlay is about to overwrite, which is not
    what this process happens to import."""
    assert vllm_overlay.installed_version_at(_site_packages(tmp_path, version="0.27.1")) == "0.27.1"

    with pytest.raises(RuntimeError, match="no installed vllm distribution"):
        vllm_overlay.installed_version_at(tmp_path / "empty")


def test_apply_from_bundle_needs_the_taaf_bundle_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The setup-command entry point says what it is missing rather than
    resolving a payload path by guesswork."""
    monkeypatch.delenv("TAAF_KAGGLE_BUNDLE_DIR", raising=False)
    with pytest.raises(RuntimeError, match="TAAF_KAGGLE_BUNDLE_DIR is unset"):
        vllm_overlay.apply_from_bundle()


def test_cuda_variant_follows_the_engine_build_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """The base wheel must be the variant the fork's own build resolved, so
    the mapping is vLLM's, not ours."""
    monkeypatch.setenv("VLLM_PRECOMPILED_WHEEL_VARIANT", "cu126")
    assert vllm_overlay.detect_cuda_variant() == "cu126"

    monkeypatch.delenv("VLLM_PRECOMPILED_WHEEL_VARIANT")
    monkeypatch.setenv("VLLM_MAIN_CUDA_VERSION", "13.0")
    assert vllm_overlay.detect_cuda_variant() == "cu130"


def test_the_pin_is_read_from_the_nearest_pyproject_and_gates_the_deploy_build(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A venv left behind by a pin bump would ship the old engine; the deploy
    build refuses when the installed revision is not the pinned one."""
    import ullm.vllm_overlay as ov

    root = tmp_path / "ws"
    (root / "packages" / "ullm" / "src" / "ullm").mkdir(parents=True)
    pinned = "a" * 40
    (root / "pyproject.toml").write_text(
        f'[tool.uv]\noverride-dependencies = ["vllm @ git+https://github.com/Tufalabs/arc-prize-2026-vllm@{pinned}"]\n'
    )
    module_file = root / "packages" / "ullm" / "src" / "ullm" / "vllm_overlay.py"
    module_file.write_text("")
    assert ov.pinned_fork_commit(module_file) == pinned
    assert ov.pinned_fork_commit(tmp_path / "elsewhere") is None

    def pinned_stub(start: pathlib.Path | None = None) -> str:
        return pinned

    monkeypatch.setattr(ov, "pinned_fork_commit", pinned_stub)
    monkeypatch.setattr(
        ov,
        "read_installed_vllm",
        lambda: ov.InstalledVllm(
            site_packages=tmp_path, version="0.27.2.dev1+gbbbbbbbbb", fork_commit="b" * 40, base_commit="c" * 40
        ),
    )
    with pytest.raises(RuntimeError, match="uv sync"):
        ov.build_for_deploy(tmp_path / "out")
