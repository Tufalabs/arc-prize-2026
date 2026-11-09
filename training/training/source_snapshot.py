from __future__ import annotations

import hashlib
import json
import os
import shutil
import site
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


SNAPSHOT_ARCHIVE_FILENAME = "source.tar"
SNAPSHOT_METADATA_FILENAME = "source_snapshot.json"
SNAPSHOT_READY_MARKER = ".arc3_source_ready"
FROZEN_CONFIG_FILENAME = "frozen_config.json"
RUNTIME_FINGERPRINT_FILENAME = "runtime_fingerprint.json"

# The frozen source lives in the run directory as ONE tar archive: a single
# streamed write onto shared storage instead of one metadata round-trip per
# file (a 580-file copytree took nine minutes on a congested NFS and wedged
# two trainers in RPC waits). At launch the archive is unpacked once into a
# node-local workspace and training executes from there.

_IGNORED_DIR_NAMES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".venvs",
    "__pycache__",
    "build",
    "dist",
    # Harness code is installed non-editably into separately fingerprinted
    # environments. Only the ARC-2 dataset archives are needed by the trainer.
    "platypus",
    "arc2-platypus",
    "runs",
    "runs_old",
}
_IGNORED_SUFFIXES = {
    ".egg-info",
    ".pyc",
    ".pyo",
}


@dataclass(frozen=True, slots=True)
class FrozenRunPaths:
    live_repo_root: Path
    run_root: Path
    source_archive_path: Path
    source_root: Path
    frozen_config_path: Path
    snapshot_metadata_path: Path
    runtime_fingerprint_path: Path


def _resolve_repo_path(path: str | Path, *, repo_root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (repo_root / candidate).resolve()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
    except ValueError:
        return False
    return True


def _ignored_names(
    directory: str,
    names: list[str],
    *,
    ignored_roots: Iterable[Path] = (),
) -> set[str]:
    ignored: set[str] = set()
    resolved_ignored_roots = tuple(root.resolve(strict=False) for root in ignored_roots)
    for name in names:
        candidate = Path(directory) / name
        if any(
            _is_relative_to(candidate, ignored_root)
            for ignored_root in resolved_ignored_roots
        ):
            ignored.add(name)
            continue
        try:
            mode = candidate.lstat().st_mode
        except OSError:
            ignored.add(name)
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)):
            ignored.add(name)
            continue
        if name in _IGNORED_DIR_NAMES:
            ignored.add(name)
            continue
        if any(name.endswith(suffix) for suffix in _IGNORED_SUFFIXES):
            ignored.add(name)
    return ignored


def _git_output(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        return f"<unavailable: {exc}>"


def _snapshot_git_state(repo_root: Path) -> dict[str, Any]:
    return {
        "root_commit": _git_output(repo_root, ["rev-parse", "HEAD"]),
        "root_branch": _git_output(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"]),
        "root_status": _git_output(repo_root, ["status", "--short"]),
        "harness_gitlinks": _git_output(
            repo_root, ["ls-tree", "HEAD", "--", "platypus", "arc2-platypus"]
        ),
    }


def _is_git_worktree(repo_root: Path) -> bool:
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _git_snapshot_paths(live_repo_root: Path) -> list[Path]:
    """Tracked plus untracked-but-not-ignored files, as they are in the worktree."""
    output = subprocess.check_output(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--deduplicate",
        ],
        cwd=live_repo_root,
    )
    paths: list[Path] = []
    for raw in output.split(b"\0"):
        if not raw:
            continue
        relative = Path(raw.decode("utf-8", errors="surrogateescape"))
        if relative.is_absolute() or ".." in relative.parts:
            continue
        try:
            (live_repo_root / relative).lstat()
        except OSError:
            continue
        paths.append(relative)
    return paths


def _walk_snapshot_paths(live_repo_root: Path) -> list[Path]:
    """Every regular file/symlink under a plain (non-git) tree."""
    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(live_repo_root):
        base = Path(directory)
        for name in list(dirnames):
            if (base / name).is_symlink():
                dirnames.remove(name)
                filenames.append(name)
        for name in filenames:
            paths.append((base / name).relative_to(live_repo_root))
    return paths


ULLM_MEDIA_SOURCE = Path("platypus/packages/ullm/src/ullm/media.py")
ULLM_ENGINE_PROXY_SOURCE = Path("platypus/packages/ullm/src/ullm/backends/engine_proxy.py")


def snapshot_paths(
    live_repo_root: Path, *, ignored_roots: Iterable[Path] = ()
) -> list[Path]:
    candidates = (
        _git_snapshot_paths(live_repo_root)
        if _is_git_worktree(live_repo_root)
        else _walk_snapshot_paths(live_repo_root)
    )
    kept: list[Path] = []
    for relative in candidates:
        parts = relative.parts
        ignored = False
        for depth in range(len(parts)):
            names = _ignored_names(
                str(live_repo_root / Path(*parts[:depth]))
                if depth
                else str(live_repo_root),
                [parts[depth]],
                ignored_roots=ignored_roots,
            )
            if names:
                ignored = True
                break
        if not ignored:
            kept.append(relative)
    for dataset in ("arc2-train", "arc2-eval"):
        relative = Path("arc2-platypus/packages/taaf/src/taaf/data") / f"{dataset}.npz"
        if (live_repo_root / relative).is_file():
            kept.append(relative)
    # ullm's image-media module and the engine proxy that serves it, for
    # runtimes whose pinned ullm predates them (the isolated Flash runtime;
    # training.ullm_media).
    for source in (ULLM_MEDIA_SOURCE, ULLM_ENGINE_PROXY_SOURCE):
        if (live_repo_root / source).is_file():
            kept.append(source)
    return sorted(set(kept), key=lambda path: path.as_posix())


def _write_archive(
    *,
    live_repo_root: Path,
    paths: list[Path],
    archive_path: Path,
) -> tuple[str, int]:
    """Stream ``tar -c`` of ``paths`` into ``archive_path`` (atomic), returning (sha256, bytes)."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    list_fd, list_name = tempfile.mkstemp(prefix=".source-files-", suffix=".lst")
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=f".{SNAPSHOT_ARCHIVE_FILENAME}.tmp-", dir=str(archive_path.parent)
    )
    digest = hashlib.sha256()
    size = 0
    try:
        with os.fdopen(list_fd, "wb") as handle:
            for path in paths:
                handle.write(path.as_posix().encode("utf-8", errors="surrogateescape"))
                handle.write(b"\0")
        pack = subprocess.Popen(
            ["tar", "--null", "-C", str(live_repo_root), "-T", list_name, "-cf", "-"],
            stdout=subprocess.PIPE,
        )
        assert pack.stdout is not None
        with os.fdopen(tmp_fd, "wb") as out:
            while chunk := pack.stdout.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
                size += len(chunk)
        pack.stdout.close()
        if pack.wait() != 0:
            raise subprocess.CalledProcessError(pack.returncode, pack.args)
        os.replace(tmp_name, archive_path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    finally:
        Path(list_name).unlink(missing_ok=True)
    return digest.hexdigest(), size


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def create_source_snapshot(
    *,
    live_repo_root: Path,
    run_root: Path,
    force: bool = False,
) -> Path:
    """Write ``<run_root>/source.tar`` (+ metadata) from the live checkout; returns the archive path."""
    live_repo_root = live_repo_root.resolve()
    run_root = run_root.resolve()
    archive_path = run_root / SNAPSHOT_ARCHIVE_FILENAME
    metadata_path = run_root / SNAPSHOT_METADATA_FILENAME

    if archive_path.exists() and not force:
        if metadata_path.exists():
            return archive_path
        raise RuntimeError(
            f"{archive_path} already exists but {metadata_path.name} is missing. "
            "Refusing to guess whether this run has a complete frozen source snapshot."
        )

    run_root.mkdir(parents=True, exist_ok=True)
    paths = snapshot_paths(live_repo_root, ignored_roots=(run_root,))
    sha256, size = _write_archive(
        live_repo_root=live_repo_root,
        paths=paths,
        archive_path=archive_path,
    )
    metadata = {
        "schema_version": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "live_repo_root": str(live_repo_root),
        "source_archive": str(archive_path),
        "archive_sha256": sha256,
        "archive_bytes": size,
        "file_count": len(paths),
        "git": _snapshot_git_state(live_repo_root),
        "ignored_dir_names": sorted(_IGNORED_DIR_NAMES),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return archive_path


def default_source_workspace() -> Path:
    """Node-local directory the frozen source is unpacked into.

    ``ARC3_SOURCE_WORKSPACE`` wins; otherwise the job's node-local cache root
    (``ARC3_RL_CACHE_ROOT``, exported by the sbatch); otherwise the system
    temp dir (local development).
    """
    explicit = str(os.environ.get("ARC3_SOURCE_WORKSPACE", "") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    cache_root = str(os.environ.get("ARC3_RL_CACHE_ROOT", "") or "").strip()
    if cache_root:
        return Path(cache_root).expanduser() / "source"
    return Path(tempfile.gettempdir()) / f"arc3-source-{os.getuid()}"


def materialize_source_snapshot(
    *,
    archive_path: Path,
    workspace_root: Path,
    expected_sha256: str | None = None,
) -> Path:
    """Unpack ``archive_path`` under ``workspace_root`` (once per content hash) and return the tree."""
    archive_path = archive_path.resolve()
    sha256 = expected_sha256 or _sha256_of_file(archive_path)
    workspace_root = workspace_root.expanduser()
    workspace_root.mkdir(parents=True, exist_ok=True)
    target = workspace_root / f"{archive_path.parent.name}-{sha256[:12]}"
    marker = target / SNAPSHOT_READY_MARKER
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256:
        return target
    if target.exists():
        shutil.rmtree(target)
    tmp_root = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=str(workspace_root))
    )
    try:
        subprocess.run(
            ["tar", "-C", str(tmp_root), "-xf", str(archive_path)],
            check=True,
        )
        (tmp_root / SNAPSHOT_READY_MARKER).write_text(sha256 + "\n", encoding="utf-8")
        os.replace(tmp_root, target)
    except Exception:
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise
    return target


def _site_paths_from_python(python_bin: Path) -> list[str]:
    script = (
        "import json, site, sys; "
        "paths=[]; "
        "getsite=getattr(site, 'getsitepackages', None); "
        "paths.extend(getsite() if getsite else []); "
        "user=getattr(site, 'getusersitepackages', lambda: '')(); "
        "paths.append(user) if user else None; "
        "print(json.dumps({'prefix': sys.prefix, 'base_prefix': sys.base_prefix, 'paths': paths}))"
    )
    try:
        output = subprocess.check_output(
            [str(python_bin), "-c", script],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        payload = json.loads(output)
    except Exception:
        return []
    paths = payload.get("paths")
    if not isinstance(paths, list):
        return []
    return [str(item) for item in paths if str(item).strip()]


def _site_paths_for_fingerprint(python_bin: Path) -> list[Path]:
    site_paths = [Path(path) for path in _site_paths_from_python(python_bin)]
    if not site_paths:
        site_paths = [Path(path) for path in site.getsitepackages()]
    return [path for path in site_paths if path.exists()]


def _read_text_if_present(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _metadata_header_value(metadata_text: str, header: str) -> str:
    prefix = f"{header.lower()}:"
    for line in metadata_text.splitlines():
        if line.lower().startswith(prefix):
            return line.split(":", 1)[1].strip()
    return ""


def _distribution_payload(dist_info: Path) -> dict[str, Any]:
    metadata_text = _read_text_if_present(dist_info / "METADATA")
    direct_url_text = _read_text_if_present(dist_info / "direct_url.json")
    direct_url: dict[str, Any] = {}
    if direct_url_text:
        try:
            parsed = json.loads(direct_url_text)
            if isinstance(parsed, dict):
                direct_url = parsed
        except json.JSONDecodeError:
            direct_url = {
                "raw_sha256": hashlib.sha256(
                    direct_url_text.encode("utf-8")
                ).hexdigest()
            }
    editable = bool(
        isinstance(direct_url.get("dir_info"), dict)
        and direct_url["dir_info"].get("editable")
    )
    payload: dict[str, Any] = {
        "dist_info": dist_info.name,
        "name": _metadata_header_value(metadata_text, "Name") or dist_info.name,
        "version": _metadata_header_value(metadata_text, "Version"),
        "editable": editable,
    }
    if editable:
        payload["direct_url"] = direct_url
        return payload

    files: dict[str, str] = {}
    for name in (
        "METADATA",
        "direct_url.json",
        "entry_points.txt",
        "WHEEL",
        "top_level.txt",
    ):
        candidate = dist_info / name
        if candidate.exists():
            files[name] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    payload["files"] = files
    return payload


def python_environment_fingerprint(python_bin: str | Path) -> dict[str, Any]:
    python_path = Path(python_bin).expanduser()
    digest = hashlib.sha256()
    site_paths = _site_paths_for_fingerprint(python_path)
    pth_count = 0
    distribution_count = 0
    for site_path in sorted(site_paths):
        for pth_file in sorted(site_path.glob("*.pth")):
            digest.update(b"pth\0")
            digest.update(pth_file.name.encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update(pth_file.read_bytes())
            except OSError:
                digest.update(b"<unreadable>")
            digest.update(b"\0")
            pth_count += 1
        for dist_info in sorted(site_path.glob("*.dist-info")):
            if not dist_info.is_dir():
                continue
            digest.update(b"dist\0")
            digest.update(
                json.dumps(
                    _distribution_payload(dist_info),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\0")
            distribution_count += 1
    stable_digest = digest.hexdigest()
    return {
        "fingerprint_kind": "stable-python-env-v2",
        "python": str(python_path),
        "python_resolved": str(python_path.resolve(strict=False)),
        "distribution_count": distribution_count,
        "pth_file_count": pth_count,
        "metadata_sha256": stable_digest,
    }


def write_runtime_fingerprint(
    *,
    destination: Path,
    training_python: Path,
    inference_python: Path | None,
    server_python: Path | None = None,
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "training": python_environment_fingerprint(training_python),
    }
    if inference_python is not None:
        payload["inference"] = python_environment_fingerprint(inference_python)
    if server_python is not None and server_python != training_python:
        payload["server"] = python_environment_fingerprint(server_python)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def verify_runtime_fingerprint(path: str | Path) -> None:
    fingerprint_path = Path(path)
    if not fingerprint_path.exists():
        return
    payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"{fingerprint_path}: expected runtime fingerprint JSON object"
        )

    checks = {
        "training": Path(sys.executable),
    }
    inference_payload = payload.get("inference")
    if isinstance(inference_payload, dict) and inference_payload.get("python"):
        checks["inference"] = Path(str(inference_payload["python"]))
    server_payload = payload.get("server")
    if isinstance(server_payload, dict) and server_payload.get("python"):
        checks["server"] = Path(str(server_payload["python"]))

    for section, python_bin in checks.items():
        expected = payload.get(section)
        if not isinstance(expected, dict):
            continue
        if expected.get("fingerprint_kind") != "stable-python-env-v2":
            raise RuntimeError(
                "Unsupported runtime Python environment fingerprint format: "
                f"{section}.fingerprint_kind={expected.get('fingerprint_kind')!r}."
            )
        current = python_environment_fingerprint(python_bin)
        keys = (
            "python_resolved",
            "distribution_count",
            "pth_file_count",
            "metadata_sha256",
        )
        for key in keys:
            if current.get(key) != expected.get(key):
                raise RuntimeError(
                    "Runtime Python environment changed after the run started: "
                    f"{section}.{key} expected {expected.get(key)!r}, got {current.get(key)!r}. "
                    "Refusing to continue with a contaminated run."
                )


def prepare_frozen_config(
    *,
    config_path: str | Path,
    live_repo_root: Path,
    source_root: Path,
    run_root: Path,
    force: bool = False,
) -> Path:
    live_repo_root = live_repo_root.resolve()
    source_root = source_root.resolve()
    run_root = run_root.resolve()
    frozen_config_path = run_root / FROZEN_CONFIG_FILENAME
    if frozen_config_path.exists() and not force:
        return frozen_config_path

    source_config_path = _resolve_repo_path(config_path, repo_root=live_repo_root)

    if _is_relative_to(source_config_path, live_repo_root):
        relative_config = source_config_path.resolve().relative_to(live_repo_root)
        snapshot_config_path = source_root / relative_config
    else:
        snapshot_config_path = run_root / "input_config.external.json"
        shutil.copy2(source_config_path, snapshot_config_path)

    payload = json.loads(snapshot_config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{snapshot_config_path}: expected a JSON object")

    training_payload = payload.setdefault("training", {})
    if not isinstance(training_payload, dict):
        raise ValueError(f"{snapshot_config_path}: training must be a JSON object")
    training_seed_override = str(os.environ.get("ARC3_TRAINING_SEED", "")).strip()
    if training_seed_override:
        try:
            training_payload["seed"] = int(training_seed_override)
        except ValueError as exc:
            raise ValueError("ARC3_TRAINING_SEED must be an integer when set.") from exc
    inference_payload = payload.setdefault("inference", {})
    if not isinstance(inference_payload, dict):
        raise ValueError(f"{snapshot_config_path}: inference must be a JSON object")
    platypus_venv = str(inference_payload.get("platypus_venv", "") or "").strip()
    if platypus_venv and not Path(platypus_venv).is_absolute():
        inference_payload["platypus_venv"] = str(live_repo_root / platypus_venv)

    frozen_config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frozen_config_path


def prepare_frozen_run(
    *,
    config_path: str | Path,
    live_repo_root: Path,
    run_root: Path,
    training_python: Path,
    force_snapshot: bool = False,
    source_workspace: Path | None = None,
) -> FrozenRunPaths:
    source_archive_path = create_source_snapshot(
        live_repo_root=live_repo_root,
        run_root=run_root,
        force=force_snapshot,
    )
    snapshot_metadata_path = run_root / SNAPSHOT_METADATA_FILENAME
    metadata = json.loads(snapshot_metadata_path.read_text(encoding="utf-8"))
    source_root = materialize_source_snapshot(
        archive_path=source_archive_path,
        workspace_root=source_workspace or default_source_workspace(),
        expected_sha256=str(metadata.get("archive_sha256") or "") or None,
    )
    frozen_config_path = prepare_frozen_config(
        config_path=config_path,
        live_repo_root=live_repo_root,
        source_root=source_root,
        run_root=run_root,
        force=force_snapshot,
    )
    payload = json.loads(frozen_config_path.read_text(encoding="utf-8"))
    platypus_venv_raw = (
        payload.get("inference", {}).get("platypus_venv")
        if isinstance(payload.get("inference"), dict)
        else None
    )
    inference_python = (
        Path(str(platypus_venv_raw)).expanduser() / "bin" / "python"
        if platypus_venv_raw
        else None
    )
    runtime_fingerprint_path = run_root / RUNTIME_FINGERPRINT_FILENAME
    if force_snapshot or not runtime_fingerprint_path.exists():
        write_runtime_fingerprint(
            destination=runtime_fingerprint_path,
            training_python=training_python,
            inference_python=inference_python,
            server_python=(
                Path(os.environ["PLATYPUS_VENV"]) / "bin" / "python"
                if os.environ.get("PLATYPUS_VENV")
                else None
            ),
        )
    return FrozenRunPaths(
        live_repo_root=live_repo_root,
        run_root=run_root,
        source_archive_path=source_archive_path,
        source_root=source_root,
        frozen_config_path=frozen_config_path,
        snapshot_metadata_path=snapshot_metadata_path,
        runtime_fingerprint_path=runtime_fingerprint_path,
    )
