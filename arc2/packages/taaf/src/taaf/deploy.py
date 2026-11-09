"""Deployment ABCs. ``DeploymentTarget`` is the
polymorphic seam, ``DeploymentHandle`` the post-launch interface.
Concrete targets: ``taaf.deploy_inline``,
``taaf.deploy_slurm``, ``taaf.deploy_kaggle``.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.metadata
import json
import os
import re
import shutil
import socket
import subprocess
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import taaf.support

if TYPE_CHECKING:
    import taaf.benchmark  # noqa: TC004 - runtime use behind a cycle-breaking guard


DEPLOY_META_FILENAME = "deploy_meta.json"
DEPLOY_META_SCHEMA_VERSION = 1


@dataclass
class DeploymentHandle(ABC):
    """Handle to a deployed benchmark, returned by
    ``DeploymentTarget.deploy()``. To rediscover a handle whose original
    Python process exited, use :meth:`attach` against the job directory.

    Fields:

    - ``job_dir``: where the run's artifacts live.
    """

    job_dir: Path

    @abstractmethod
    def wait(self) -> taaf.benchmark.Benchmark:
        """Block until the run finishes, return the populated
        ``Benchmark``. Idempotent. Implementations reload from
        ``benchmark.json`` (Slurm reattach) or return the in-memory
        instance (Inline).
        """
        ...

    @abstractmethod
    def stop(self) -> None:
        """Request graceful shutdown. Must route through
        ``Benchmark.run``'s teardown so artifacts still land in
        ``job_dir``. Idempotent. Returns immediately — chain with
        ``wait()`` to block until the worker is actually gone. Stopped
        runs may take up to ~10 minutes to wind down via the
        soft-deadline path.
        """
        ...

    @property
    @abstractmethod
    def is_done(self) -> bool:
        """True iff the run has reached a terminal state. Non-blocking
        and cheap. Transient "don't know" states (e.g. Slurm bookkeeping
        lag) return False so callers keep polling.
        """
        ...

    @classmethod
    def attach(cls, job_dir: Path) -> DeploymentHandle:
        """Reconstruct a handle from ``job_dir/deploy_meta.json``.
        Dispatches to the recorded ``handle_class``'s ``_attach()``.
        Raises ``FileNotFoundError`` when no metadata is present.
        """
        meta_path = job_dir / DEPLOY_META_FILENAME
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"no {DEPLOY_META_FILENAME} in {job_dir} — deploy() may have crashed "
                "before writing metadata, or this is not a TAAF job directory."
            )
        with open(meta_path) as f:
            meta = cast(dict[str, Any], json.load(f))
        handle_class_path = cast(str, meta["handle_class"])
        handle_cls = _resolve_class(handle_class_path)
        if not issubclass(handle_cls, DeploymentHandle):
            raise TypeError(f"{handle_class_path} is not a DeploymentHandle subclass")
        return handle_cls._attach(job_dir, meta)

    @classmethod
    @abstractmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> DeploymentHandle:
        """Reconstruct a concrete handle from disk state. Inline loads
        the benchmark snapshot eagerly; Slurm leaves it lazy."""


@dataclass
class DeploymentTarget(ABC):
    """Polymorphic seam for deployment environments. Subclasses
    implement ``deploy(benchmark)`` and are expected to (modulo the
    exemptions a given target documents for itself):

    - validate ``benchmark.job_dir`` and create it if missing;
    - snapshot tufalabs source code into ``job_dir/src/`` via
      :func:`snapshot_editable_sources`;
    - write ``job_dir/deploy_meta.json`` via :func:`write_deploy_meta`;
    - tee stdout / stderr into ``job_dir``;
    - print a preamble (label / solver / git) via :func:`format_preamble`;
    - derive ``soft_end_time = hard_kill − 10min`` and pass it to
      ``Benchmark.run``;
    - return a ``DeploymentHandle`` once launched.
    """

    @abstractmethod
    async def deploy(self, benchmark: taaf.benchmark.Benchmark) -> DeploymentHandle:
        """Package ``benchmark`` for this environment and launch it.

        ``async`` so notebook callers can ``await bm.deploy(target)``
        directly. Inline awaits ``Benchmark.run`` in-process; Slurm /
        Kaggle return immediately while the run continues elsewhere.
        Either way the returned handle's ``wait()`` blocks until done.
        """
        ...


# --- Metadata write / class resolution ------------------------------------


def check_job_dir_unused(job_dir: Path) -> None:
    """Refuse to deploy when ``job_dir/deploy_meta.json`` already exists —
    either a previous run's artifacts live there or another deploy would
    race us on every output file. Raises ``FileExistsError`` so the user
    removes the old dir or picks a fresh one.
    """
    meta_path = job_dir / DEPLOY_META_FILENAME
    if meta_path.exists():
        raise FileExistsError(
            f"deploy: {meta_path} already exists — a prior deploy targeted "
            f"this directory. To re-deploy, remove or rename it first "
            f"(e.g. `rm -rf {job_dir}` or `mv {job_dir} {job_dir}.bak`), "
            f"or pick a different job_dir."
        )


def write_deploy_meta(
    job_dir: Path,
    target: DeploymentTarget,
    handle_class: type[DeploymentHandle],
    benchmark: taaf.benchmark.Benchmark,
    job_id: str | None,
    step_name: str | None = None,
    step_id: str | None = None,
) -> None:
    """Write ``job_dir/deploy_meta.json`` describing this
    deployment. Carries enough for ``DeploymentHandle.attach`` to rebuild
    a handle (``handle_class``, ``job_id``) plus human-readable fields
    (label, target_config, started_at, host, pid).
    """
    meta: dict[str, Any] = {
        "schema_version": DEPLOY_META_SCHEMA_VERSION,
        "target_class": _qualname(type(target)),
        "target_config": _target_config_dict(target),
        "handle_class": _qualname(handle_class),
        "job_id": job_id,
        **({"step_name": step_name} if step_name is not None else {}),
        **({"step_id": step_id} if step_id is not None else {}),
        "benchmark_label": benchmark.label,
        "started_at": datetime.now().isoformat(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    taaf.support.atomic_json_dump(meta, job_dir / DEPLOY_META_FILENAME)


def _qualname(cls: type) -> str:
    """The dotted path that :func:`_resolve_class` can read back."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_class(dotted: str) -> type:
    """Import and return the class named by a dotted path."""
    module_name, _, class_name = dotted.rpartition(".")
    if not module_name:
        raise ValueError(f"not a dotted class path: {dotted!r}")
    module = importlib.import_module(module_name)
    obj = getattr(module, class_name)
    if not isinstance(obj, type):
        raise TypeError(f"{dotted!r} resolved to non-class {obj!r}")
    return obj


def _target_config_dict(target: DeploymentTarget) -> dict[str, Any]:
    """Convert a target dataclass to a JSON-safe dict. ``default=str``
    handles ``Path`` and any other non-JSON-native field a target carries.
    """
    raw = dataclasses.asdict(target) if dataclasses.is_dataclass(target) else {}
    return cast(dict[str, Any], json.loads(json.dumps(raw, default=str)))


# --- Shared helpers used by concrete targets ------------------------------


def snapshot_editable_sources(
    dest_dir: Path,
    extra_repos: list[Path] | None = None,
    exclude_repos: Iterable[str] | None = None,
    include_noneditable_tufa: bool = True,
) -> list[Path]:
    """Copy every editable repo in the launcher's venv (plus any
    ``extra_repos``) into ``dest_dir/<repo>/``. "Editable" is decided
    per PEP 610 — see :func:`_discover_editable_repos`. By construction
    the worker venv gets every repo the launcher had editable, so
    ``.run()`` and ``.deploy()`` see the same package set.

    Each per-repo snapshot contains ``pyproject.toml``, ``README.md``
    (if present), ``uv.lock`` (if present), common project support
    files/directories such as ``Makefile``, ``configs/``, and
    ``scripts/``, and the importable package source tree(s).

    Non-editable distributions installed from a ``github.com/Tufalabs``
    VCS URL are also bundled, but only their installed package tree(s)
    (no ``pyproject.toml`` / ``Makefile``) — there is no checkout to
    snapshot, so this is "just the package" at the pinned commit. Such a
    bundle can't serve as a Slurm ``main_project`` (no ``pyproject`` to
    ``uv sync``); it's enough for the Kaggle worker's ``sys.path``.
    Standard PyPI deps (no Tufa VCS URL) are left to the target's own
    install.

    Fails loudly on a per-repo error: catching here on the launcher
    surfaces the problem with full context, instead of letting the job
    queue and crash later when ``uv pip install`` can't find the
    bundled source.

    ``exclude_repos`` drops repos whose directory name matches (case-
    insensitively), for a bundle that must not carry everything the
    workspace holds.

    ``include_noneditable_tufa=False`` drops the installed-package
    bundles entirely. A target whose worker resolves its own
    dependencies has no use for them — Slurm ``uv sync``s from the
    bundled ``main_project``'s lock and never reads them — and an
    installed tree can be large (the vLLM fork's runs to hundreds of MB).

    Returns the list of per-repo destination dirs that were written.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    excluded = {str(name).strip().lower() for name in (exclude_repos or ())}
    seen: set[Path] = set()
    repos: list[Path] = []
    for r in [*_discover_editable_repos(), *(extra_repos or [])]:
        rr = r.resolve()
        if rr not in seen:
            seen.add(rr)
            repos.append(rr)
    written: list[Path] = []
    bundled_names: set[str] = set()
    for repo_root in repos:
        if repo_root.name.lower() in excluded:
            continue
        try:
            written.append(_snapshot_one_repo(repo_root, dest_dir / repo_root.name))
            bundled_names.add(repo_root.name)
        except Exception as e:
            raise RuntimeError(f"deploy: failed to snapshot {repo_root} into {dest_dir / repo_root.name}: {e}") from e
    # Non-editable Tufa repos have no checkout, so copy the installed
    # package tree (the pinned commit). A same-named editable repo wins.
    noneditable = _discover_noneditable_tufa_packages() if include_noneditable_tufa else []
    for repo_name, pkg_dirs in noneditable:
        if repo_name in bundled_names or repo_name.lower() in excluded:
            continue
        dest = dest_dir / repo_name
        try:
            written.append(_snapshot_installed_packages(pkg_dirs, dest))
            bundled_names.add(repo_name)
        except Exception as e:
            raise RuntimeError(f"deploy: failed to snapshot installed Tufa package {repo_name} into {dest}: {e}") from e
    return written


def snapshot_editable_sources_archive(
    dest_tar: Path,
    extra_repos: list[Path] | None = None,
    exclude_repos: Iterable[str] | None = None,
    include_noneditable_tufa: bool = True,
) -> Path:
    """Like :func:`snapshot_editable_sources`, but write the snapshot as a
    single uncompressed tar at ``dest_tar`` instead of a directory of ~1k
    small files.

    On clusters whose shared storage is NFS (e.g. Kander), creating the
    snapshot's many small files directly on shared storage costs ~100ms
    each — minutes on the submit critical path. Build the tree on local
    temp (fast disk) and write one sequential tar to shared storage
    instead. The worker extracts it before ``uv sync`` (see deploy_slurm).
    Top-level tar members are the per-repo dirs, so ``tar -C <dir> -xf
    <dest_tar>`` reproduces exactly what :func:`snapshot_editable_sources`
    lays out under ``<dir>``. Written to a sibling ``*.tmp`` and atomically
    renamed so a reader never sees a partial archive.
    """
    import tarfile
    import tempfile

    dest_tar.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="taaf-src-snap-"))
    try:
        snapshot_editable_sources(
            staging,
            extra_repos=extra_repos,
            exclude_repos=exclude_repos,
            include_noneditable_tufa=include_noneditable_tufa,
        )
        tmp_tar = dest_tar.with_name(dest_tar.name + ".tmp")
        with tarfile.open(tmp_tar, "w") as tf:
            for child in sorted(staging.iterdir()):
                tf.add(child, arcname=child.name)
        tmp_tar.replace(dest_tar)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest_tar


def format_git_status() -> str:
    """Per-tufalabs-repo git overview captured on the launcher
    (short SHA, branch, clean-or-DIRTY, last-commit subject). Header
    line included so the text reads on its own when persisted to
    ``job_dir/git_status.txt``. Editable repos show clean/DIRTY from their
    checkout; non-editable Tufa installs show their pinned ``vcs_info``
    commit (immutable, so "pinned").
    """
    lines: list[str] = ["git status:"]
    subject_width = 60
    for repo in _discover_editable_repos():
        commit, branch, subject, clean = _git_info(repo)
        marker = "clean" if clean else "DIRTY"
        if len(subject) > subject_width:
            subject = subject[: subject_width - 1] + "…"
        lines.append(f"  {repo.name:<32} {commit:<10}  {marker:<5}  {branch:<24}  {subject}")
    for repo_name, commit, rev in _noneditable_tufa_git_lines():
        lines.append(f"  {repo_name:<32} {commit:<10}  {'pinned':<5}  {rev:<24}  (non-editable)")
    return "\n".join(lines)


def write_git_status(job_dir: Path) -> None:
    """Persist :func:`format_git_status` output to
    ``job_dir/git_status.txt``. Called on the launcher (deploy targets,
    or inline ``Benchmark.run()`` when ``job_dir`` is set) so the
    durable file carries the real git state — worker venvs install
    from source snapshots, which deliberately exclude ``.git``."""
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "git_status.txt").write_text(format_git_status())


def format_preamble(benchmark: taaf.benchmark.Benchmark) -> str:
    """Preamble printed at the start of every deployed run.
    Label / solver / n_passes / n_games, plus a per-tufalabs-repo git
    line (short SHA, branch, clean-or-DIRTY, subject).
    """
    return "\n".join(
        [
            f"benchmark.label : {benchmark.label}",
            f"benchmark.solver: {benchmark.solver!r}",
            f"benchmark.passes: {benchmark.n_passes}",
            f"benchmark.games : {len(benchmark.games)}",
            format_git_status(),
        ]
    )


def _editable_source_paths() -> set[Path]:
    """The directories this venv has installed editable (PEP 610)."""
    paths: set[Path] = set()
    for dist in importlib.metadata.distributions():
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        try:
            data = cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            continue
        url = cast(str, data.get("url") or "")
        if cast(dict[str, Any], data.get("dir_info") or {}).get("editable") and url.startswith("file://"):
            paths.add(Path(url[len("file://") :]).resolve())
    return paths


def _discover_editable_repos() -> list[Path]:
    """Return the repo root of every editable distribution in the
    launcher's venv. PEP 610: editable installs ship a
    ``*.dist-info/direct_url.json`` with
    ``{"dir_info": {"editable": true}, "url": "file:///..."}``. We walk
    up from each recorded URL to the directory containing
    ``pyproject.toml``. Monorepos shipping several packages from one
    tree collapse to a single entry.
    """
    repos: list[Path] = []
    seen: set[Path] = set()
    for dist in importlib.metadata.distributions():
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        try:
            data = cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            continue
        dir_info = cast(dict[str, Any], data.get("dir_info") or {})
        if not dir_info.get("editable"):
            continue
        url = cast(str, data.get("url") or "")
        if not url.startswith("file://"):
            continue
        src_path = Path(url[len("file://") :]).resolve()
        try:
            repo = _find_repo_root(src_path)
        except RuntimeError:
            # direct_url.url is outside any pyproject — skip.
            continue
        repo = repo.resolve()
        if repo in seen:
            continue
        seen.add(repo)
        repos.append(repo)
    return repos


def _noneditable_tufa_dists() -> list[tuple[str, importlib.metadata.Distribution, dict[str, Any]]]:
    """Non-editable distributions installed from a ``github.com/Tufalabs``
    VCS URL, as ``(repo_name, distribution, parsed direct_url.json)``,
    de-duped by repo name (first wins). Shared by the source-snapshot and
    git-status paths.
    """
    out: list[tuple[str, importlib.metadata.Distribution, dict[str, Any]]] = []
    seen: set[str] = set()
    for dist in importlib.metadata.distributions():
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        try:
            data = cast(dict[str, Any], json.loads(text))
        except json.JSONDecodeError:
            continue
        if cast(dict[str, Any], data.get("dir_info") or {}).get("editable"):
            continue
        if not data.get("vcs_info"):
            continue
        url = cast(str, data.get("url") or "")
        if not _is_tufa_url(url):
            continue
        repo_name = _repo_name_from_url(url)
        if repo_name in seen:
            continue
        seen.add(repo_name)
        out.append((repo_name, dist, data))
    return out


def _discover_noneditable_tufa_packages() -> list[tuple[str, list[Path]]]:
    """Non-editable Tufa distributions with importable packages, as
    ``(repo_name, [installed top-level package dirs])``.

    These can't be ``pip``-installed on an offline/Kaggle worker and have
    no checkout to snapshot, so the bundler copies the installed package
    tree — which is the exact commit the launcher venv resolved. Editable
    Tufa repos are handled by :func:`_discover_editable_repos`; standard
    PyPI deps (no Tufa VCS URL) are left to the target's own install.
    """
    out: list[tuple[str, list[Path]]] = []
    for repo_name, dist, _data in _noneditable_tufa_dists():
        pkg_dirs = _installed_top_level_packages(dist)
        if pkg_dirs:
            out.append((repo_name, pkg_dirs))
    return out


def _noneditable_tufa_git_lines() -> list[tuple[str, str, str]]:
    """``(repo_name, short_commit, requested_revision)`` for non-editable
    Tufa repos, from their PEP 610 ``vcs_info`` — there's no checkout to
    inspect, but the pinned commit is recorded at install time."""
    lines: list[tuple[str, str, str]] = []
    for repo_name, _dist, data in _noneditable_tufa_dists():
        vcs = cast(dict[str, Any], data.get("vcs_info") or {})
        commit = str(vcs.get("commit_id") or "?")[:10]
        rev = str(vcs.get("requested_revision") or "?")
        lines.append((repo_name, commit, rev))
    return lines


def _is_tufa_url(url: str) -> bool:
    """True for a ``github.com/Tufalabs`` repository URL (any scheme/case)."""
    return re.search(r"github\.com[:/]+tufalabs/", url, re.IGNORECASE) is not None


def _repo_name_from_url(url: str) -> str:
    """The repository name a git URL points at."""
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail.removesuffix(".git")


def _installed_top_level_packages(dist: importlib.metadata.Distribution) -> list[Path]:
    """Absolute paths of the distribution's installed top-level package
    directories (those with an ``__init__.py``). Reads ``top_level.txt``
    when present, else infers top-level names from the recorded files.
    """
    names: list[str] = []
    top = dist.read_text("top_level.txt")
    if top:
        names = [line.strip() for line in top.splitlines() if line.strip()]
    else:
        derived: set[str] = set()
        for f in dist.files or []:
            parts = f.parts
            if len(parts) > 1 and not parts[0].endswith((".dist-info", ".data")):
                derived.add(parts[0])
        names = sorted(derived)
    dirs: list[Path] = []
    for name in names:
        loc = Path(str(dist.locate_file(name))).resolve()
        if loc.is_dir() and (loc / "__init__.py").is_file() and loc not in dirs:
            dirs.append(loc)
    return dirs


def _snapshot_installed_packages(pkg_dirs: list[Path], dest: Path) -> Path:
    """Copy installed top-level package dirs into ``dest`` — used for
    non-editable Tufa packages, which have no repo checkout."""
    dest.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for pkg_dir in pkg_dirs:
        shutil.copytree(pkg_dir, dest / pkg_dir.name, ignore=ignore, dirs_exist_ok=True)
    return dest


def detect_bundled_repo_name(start: Path | None = None) -> str:
    """The name :func:`snapshot_editable_sources` will bundle this checkout
    under: its enclosing repo-root directory name.

    The one right default for ``TufaSlurmTarget.main_project``, and derived
    rather than written down because the two must agree — a worktree, or a
    renamed checkout, changes the bundled name, and a literal that no longer
    matches sends the worker to ``uv sync`` a path that does not exist.
    """
    return _find_repo_root((start or Path.cwd()).resolve()).name


def _find_repo_root(start: Path) -> Path:
    """Walk up to the enclosing project, preferring a uv workspace root so a
    monorepo member brings its siblings."""
    for p in [start, *start.parents]:
        if (p / "pyproject.toml").is_file():
            return _uv_workspace_root_of(p) or p
    raise RuntimeError(f"no pyproject.toml found walking up from {start}")


def _uv_workspace_members(root: Path) -> list[Path] | None:
    """Member dirs of the uv workspace rooted at ``root``, or ``None`` when
    ``root/pyproject.toml`` declares no ``[tool.uv.workspace]``. Members
    are the ``members`` globs minus ``exclude``, keeping only dirs that
    carry a ``pyproject.toml`` (uv's own rule)."""
    import fnmatch

    pyproject = root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    with open(pyproject, "rb") as f:
        cfg = tomllib.load(f)
    workspace = cast(dict[str, Any], cfg.get("tool", {}).get("uv", {}).get("workspace") or {})
    if not workspace:
        return None
    exclude = [str(pattern) for pattern in cast(list[Any], workspace.get("exclude") or [])]
    members: list[Path] = []
    for pattern in cast(list[Any], workspace.get("members") or []):
        for candidate in sorted(root.glob(str(pattern))):
            if not candidate.is_dir() or not (candidate / "pyproject.toml").is_file():
                continue
            relative = str(candidate.relative_to(root))
            if any(fnmatch.fnmatchcase(relative, ex) for ex in exclude):
                continue
            if candidate not in members:
                members.append(candidate)
    return members


def _uv_workspace_root_of(project_root: Path) -> Path | None:
    """The uv-workspace root claiming ``project_root`` as a member, or
    ``None``. An editable workspace member must be bundled as its whole
    workspace: the member pyproject alone cannot resolve its
    ``{ workspace = true }`` sources on the worker."""
    for ancestor in project_root.parents:
        members = _uv_workspace_members(ancestor)
        if members is None:
            continue
        # The nearest workspace decides: either it claims the project, or
        # the project is standalone (uv workspaces don't nest).
        is_member = project_root.resolve() in {member.resolve() for member in members}
        return ancestor if is_member else None
    return None


def _members_to_snapshot(repo_root: Path) -> list[Path]:
    """A workspace's members to bundle: all of them, unless this venv installed only some.

    ``ullm`` is a member of the ARC3 project's workspace, whose ``platypus``, ``duck`` and
    ``sweeper`` share their import names with this project's own and would shadow them.
    """
    members = _uv_workspace_members(repo_root) or []
    editable = _editable_source_paths()
    installed = [member for member in members if member.resolve() in editable]
    return installed if installed else members


def _snapshot_one_repo(repo_root: Path, dest: Path) -> Path:
    """Copy one repo's importable sources and build files into ``dest``,
    preserving repo-relative paths so the bundle installs on the worker."""
    dest.mkdir(parents=True, exist_ok=True)
    # uv.lock is not importable source, but the worker's ``uv sync``
    # needs it. Include when present.
    for fname in ("pyproject.toml", "README.md", "uv.lock", "Makefile"):
        src = repo_root / fname
        if src.is_file():
            shutil.copy2(src, dest / fname)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".venv", ".git")
    for dirname in ("configs", "scripts"):
        src_dir = repo_root / dirname
        if src_dir.is_dir():
            shutil.copytree(src_dir, dest / dirname, ignore=ignore, dirs_exist_ok=True)
    for pkg_dir in _find_package_dirs(repo_root):
        # Preserve the package's path relative to the repo root so the
        # bundled pyproject's ``packages.find.where`` still resolves on
        # the worker (``src/taaf`` → ``dest/src/taaf``, flat ``re_arc``
        # → ``dest/re_arc``).
        rel = pkg_dir.relative_to(repo_root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(pkg_dir, target, ignore=ignore, dirs_exist_ok=True)
    # A uv-workspace root brings every member along at its workspace-
    # relative path, so the bundled root can `uv sync --all-packages` and
    # member-to-member `{ workspace = true }` sources resolve on the worker.
    for member in _members_to_snapshot(repo_root):
        _snapshot_one_repo(member, dest / member.relative_to(repo_root))
    return dest


def _find_package_dirs(repo_root: Path) -> list[Path]:
    """Locate importable package source tree(s). Reads
    ``[tool.setuptools.packages.find]`` (``where`` + ``include``) when
    present; falls back to scanning ``src/`` and the repo root for any
    directory with an ``__init__.py``. Returns top-level package roots
    only — subpackages come along via ``copytree``.
    """
    import fnmatch

    pyp = repo_root / "pyproject.toml"
    where: list[str] = ["."]
    includes: list[str] = ["*"]
    if pyp.is_file():
        with open(pyp, "rb") as f:
            cfg = tomllib.load(f)
        find_cfg = cast(
            dict[str, Any],
            cfg.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find", {}),
        )
        if isinstance(find_cfg.get("where"), list):
            where = [str(w) for w in find_cfg["where"]]
        if isinstance(find_cfg.get("include"), list):
            includes = [str(p) for p in find_cfg["include"]]

    found: list[Path] = []
    for w in where:
        base = (repo_root / w).resolve()
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir()):
            if not child.is_dir() or not (child / "__init__.py").is_file():
                continue
            # setuptools' ``include`` is glob-style on dotted package
            # names (e.g. ``re_arc*``). At the top level the dotted name
            # is just the dir name, so fnmatch on the first segment works.
            if any(fnmatch.fnmatchcase(child.name, pat.split(".")[0]) for pat in includes):  # noqa: SIM102 - the outer test is the cheap one
                if child not in found:
                    found.append(child)
    return found


def _git_info(repo: Path) -> tuple[str, str, str, bool]:
    """Return ``(short_commit, branch, subject, clean?)`` for ``repo``.
    Best-effort: returns ``("?", "?", "?", False)`` if git isn't available
    or the dir isn't a repo. ``branch`` is ``HEAD`` on detached checkouts.
    """
    try:
        commit = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        subject = subprocess.run(
            ["git", "-C", str(repo), "log", "-1", "--format=%s"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return commit, branch, subject, dirty == ""
    except Exception:
        return "?", "?", "?", False
