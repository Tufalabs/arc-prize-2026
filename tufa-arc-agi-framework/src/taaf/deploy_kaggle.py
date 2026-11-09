"""Kaggle deployment target (R2.24 / R2.51-R2.61).

The launcher writes two Kaggle bundles under ``benchmark.job_dir``:

1. a source dataset containing the pickled benchmark and source snapshots
   for Tufa repos;
2. a notebook kernel rendered from ``src/taaf/kaggle/taaf_kaggle_run.ipynb``
   with small deployment-specific placeholders filled in.
"""

from __future__ import annotations

import csv
import getpass
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
import time
import unicodedata
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import taaf.benchmark
import taaf.deploy
import taaf.support

COMPETITION_SLUG = "arc-prize-2026-arc-agi-3"
# Kaggle mounts the competition at /kaggle/input/competitions/<slug> on some kernels and at the
# older flat /kaggle/input/<slug> on others, from the same metadata (as it does datasets), so
# both notebooks install from the first of these that holds the wheels.
COMPETITION_WHEELHOUSES = (
    f"/kaggle/input/competitions/{COMPETITION_SLUG}/arc_agi_3_wheels",
    f"/kaggle/input/{COMPETITION_SLUG}/arc_agi_3_wheels",
)
DEFAULT_ACCELERATOR = "NvidiaRtxPro6000"
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
# The source dataset holds the bundle as this one zip, which the notebook unpacks: a dataset of
# many files is slow to attach, and Kaggle unpacks an uploaded `.zip` into its files, so the
# name must not end in `.zip`.
DATASET_BUNDLE_ARCHIVE = "taaf-kaggle-bundle.taafzip"
DEFAULT_SOURCE_DATASET_SLUG = "taaf-release-source"
DEFAULT_KERNEL_SLUG = "taaf-arc-agi3"
KAGGLE_WORKING_DIR = "/kaggle/working"
_SOFT_DEADLINE_BUFFER_S = 600.0
#: The ``KaggleTarget`` fields that set a startup deadline (R2.67), each of
#: which needs ``startup_stealth``.
_STARTUP_DEADLINE_FIELDS = (
    "startup_deadline_s",
    "submission_startup_deadline_s",
    "startup_deadline_after_boot_s",
    "submission_startup_deadline_after_boot_s",
)
NOTEBOOK_NAME = "taaf_kaggle_run.ipynb"
SHARE_NOTEBOOK_NAME = "taaf_kaggle_run_share.ipynb"
_NOTEBOOK_TEMPLATE = Path(__file__).resolve().parent / "kaggle" / NOTEBOOK_NAME
_SHARE_NOTEBOOK_TEMPLATE = Path(__file__).resolve().parent / "kaggle" / SHARE_NOTEBOOK_NAME
# Repos left out of the public share bundle (its env-files snapshot must not ship publicly).
_SHARE_EXCLUDE_REPOS = ("re-arc-3",)


@dataclass
class KaggleHandle(taaf.deploy.DeploymentHandle):
    """Handle to a Kaggle notebook version.

    Fields:

    - ``kernel_id``: Kaggle kernel id, ``owner/slug``.
    - ``dataset_ref``: source dataset id, ``owner/slug``.
    - ``uploaded``: false for dry-run packaging.
    - ``poll_interval_s``: cadence for ``wait()`` status polling.
    """

    kernel_id: str = ""
    dataset_ref: str = ""
    uploaded: bool = False
    poll_interval_s: float = 30.0
    output_dir: Path | None = None
    _cached_benchmark: taaf.benchmark.Benchmark | None = field(default=None, init=False, repr=False)

    def wait(self) -> taaf.benchmark.Benchmark:
        if not self.uploaded:
            raise RuntimeError("Kaggle dry-run handle has no remote notebook to wait for.")
        while not self.is_done:
            time.sleep(self.poll_interval_s)
        return self._load_or_pull_benchmark()

    def stop(self) -> None:
        # Kaggle has no TAAF graceful-stop path (R2.33 exception).
        pass

    @property
    def is_done(self) -> bool:
        if not self.uploaded:
            return True
        result = subprocess.run(
            ["kaggle", "kernels", "status", self.kernel_id],
            capture_output=True,
            text=True,
            check=False,
        )
        text = f"{result.stdout}\n{result.stderr}".lower()
        return any(token in text for token in ("complete", "error", "cancelled", "canceled", "failed"))

    @classmethod
    def _attach(cls, job_dir: Path, meta: dict[str, Any]) -> KaggleHandle:
        cfg = cast(dict[str, Any], meta.get("target_config") or {})
        kernel_id = str(meta.get("job_id") or cfg.get("kernel_id") or "")
        dataset_ref = str(cfg.get("source_dataset_ref") or cfg.get("dataset_ref") or "")
        uploaded = bool(cfg.get("uploaded", True))
        output_raw = cfg.get("output_dir")
        output_dir = Path(output_raw) if isinstance(output_raw, str) and output_raw else None
        return cls(
            job_dir=job_dir, kernel_id=kernel_id, dataset_ref=dataset_ref, uploaded=uploaded, output_dir=output_dir
        )

    def _load_or_pull_benchmark(self) -> taaf.benchmark.Benchmark:
        if self._cached_benchmark is not None:
            return self._cached_benchmark
        candidates = [self.job_dir / "benchmark.json"]
        output_dir = self.output_dir or (self.job_dir / "kaggle-output")
        candidates.append(output_dir / "benchmark.json")
        for path in candidates:
            if path.is_file():
                self._cached_benchmark = taaf.benchmark.Benchmark.from_json(path)
                return self._cached_benchmark

        output_dir.mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["kaggle", "kernels", "output", self.kernel_id, "-p", str(output_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and (output_dir / "benchmark.json").is_file():
            self._cached_benchmark = taaf.benchmark.Benchmark.from_json(output_dir / "benchmark.json")
            return self._cached_benchmark
        raise RuntimeError(
            f"Kaggle kernel {self.kernel_id} is done, but no benchmark.json was found. "
            "Submission-mode notebooks intentionally skip this file; otherwise check Kaggle logs."
        )


@dataclass
class KaggleTarget(taaf.deploy.DeploymentTarget):
    """Create/update a Kaggle notebook and source dataset.

    Fields:

    - ``username``: Kaggle owner. Defaults to env / ``~/.kaggle/kaggle.json``.
    - ``kernel_slug`` / ``kernel_title``: notebook destination. Defaults
      to a private per-user TAAF notebook.
    - ``dataset_ref``: source dataset destination. Defaults to
      ``<username>/taaf-release-source``.
    - ``additional_dataset_sources``: extra Kaggle datasets attached to
      the notebook. Solvers can also expose ``kaggle_dataset_sources``.
    - ``additional_model_sources``: Kaggle models attached to the notebook.
      A model is not a dataset: it mounts under
      ``/kaggle/input/models/<owner>/<slug>/<framework>/<variation>/<version>``
      and is the only way to reach weights someone published as one.
      Solvers can also expose ``kaggle_model_sources``.
    - ``additional_kernel_sources``: extra Kaggle utility-script / kernel
      sources attached to the notebook (e.g. a wheels utility script that
      a setup command then ``pip install``s from). Solvers can also expose
      ``kaggle_kernel_sources``.
    - ``setup_commands``: shell commands executed in the notebook before
      the benchmark pickle is loaded. Solvers can expose
      ``kaggle_setup_commands`` for solver-specific wheel/model setup.
      Commands receive ``PYTHON``, ``TAAF_KAGGLE_BUNDLE_DIR``,
      ``TAAF_KAGGLE_WORKING_DIR``, ``TAAF_KAGGLE_INPUT_PATHS`` (a JSON
      ``{ref: mount_path}`` map for attached datasets and utility scripts),
      and ``TAAF_KAGGLE_SETUP_ENV``. If a command writes a JSON object to
      ``TAAF_KAGGLE_SETUP_ENV``, those keys are merged into the runner
      process environment before the benchmark pickle is loaded.
    - ``teardown_commands``: best-effort shell commands executed after
      the benchmark finishes. Solvers can expose
      ``kaggle_teardown_commands`` for solver-specific cleanup.
    - ``extra_source_repos``: local repos forced into the source dataset.
    - ``exclude_source_repos``: repo directory names kept OUT of the source
      dataset (case-insensitive), on top of the share variant's own
      exclusions. Solvers can expose ``kaggle_exclude_source_repos`` for
      the repos their worker install supersedes. For such a repo the
      bundled copy is not merely dead weight: the runner prepends every
      bundled repo to ``sys.path``, so it would shadow the installed one.
    - ``bundle_files``: extra files copied verbatim into the source
      dataset's ``payload/`` directory, for setup commands to read at
      ``$TAAF_KAGGLE_BUNDLE_DIR/payload/<name>``. Solvers can expose
      ``kaggle_bundle_files``. Unlike a source repo a payload file is
      never interpreted — nothing puts it on ``sys.path`` — so it is the
      way to ship an offline artifact whose contents the kernel installs
      for itself.
    - ``run_as_submission``: emulate submission behavior. True
      competition reruns force this on inside the notebook.
    - ``make_share_version``: build the public-facing share variant
      (R2.61) — renders the minimal ``taaf_kaggle_run_share.ipynb``
      (Kaggle-only, no ``RUN_AS_SUBMISSION`` emulation; branches on the
      real ``KAGGLE_IS_COMPETITION_RERUN`` — live Arcade in a rerun, the
      competition's own bundled offline env-files otherwise), leaves
      ``re-arc-3`` out of the source dataset, and ``-share``-suffixes the
      default kernel/dataset slugs. Explicit ``kernel_slug`` /
      ``dataset_ref`` are left untouched.
    - ``public``: push the notebook public (default private, R2.52). This is
      refused at deploy time on every variant (R2.62), which also blocks
      uploading to a notebook/dataset that is already public on Kaggle — TAAF
      pushes privately and publishing is a deliberate manual later step.
    - ``cpu_only``: disables GPU for cheap public/offline smoke tests.
      The default honors R2.52: RTX6000 GPU with internet disabled.
    - ``docker_image``: the Kaggle image the notebook runs on, by digest
      (``gcr.io/kaggle-private-byod/python@sha256:...``), instead of whatever
      Kaggle's current image is when the version runs. A solver whose setup
      was built for one image (a wheelhouse, a recorded install) pins it by
      exposing ``kaggle_docker_image``; both may be set only if they agree.
      None leaves the image to Kaggle.
    - ``startup_stealth``: whether both notebooks start the solver before
      listing or opening any game (R2.66), on every run. False restores the
      order from before it: a rerun's games are listed before ``run``, every
      game opens before the solver starts, and in a submission the keepalive
      RESETs (R2.63) through the start-up. The startup deadlines need it, so
      the deploy refuses any of them without it.
    - ``startup_deadline_s``: under ``startup_stealth`` (R2.66), how long
      after notebook start the solver must be up and ready
      to play outside a submission; past it the notebook's process exits with
      no game opened (R2.67). None sets no deadline.
    - ``submission_startup_deadline_s``: the same in a submission, a
      competition rerun or a run with ``run_as_submission`` (R2.67). None
      sets no deadline.
    - ``startup_deadline_after_boot_s``: the same, counted from when the
      machine booted (``/proc/uptime``) rather than from notebook start, so
      it also ends a session that sat on its machine for long before the
      notebook ran; with both set, the earlier applies (R2.67). Outside a
      submission; None sets no such deadline.
    - ``submission_startup_deadline_after_boot_s``: the deadline from machine
      boot in a submission, a competition rerun or a run with
      ``run_as_submission``, in place of ``startup_deadline_after_boot_s``
      (R2.67). None sets no such deadline.
    - ``dry_run``: write bundles and metadata but do not call Kaggle.
    """

    username: str | None = None
    kernel_slug: str | None = None
    kernel_title: str | None = None
    dataset_ref: str | None = None
    additional_dataset_sources: list[str] = field(default_factory=lambda: list[str]())
    additional_model_sources: list[str] = field(default_factory=lambda: list[str]())
    additional_kernel_sources: list[str] = field(default_factory=lambda: list[str]())
    setup_commands: list[str] = field(default_factory=lambda: list[str]())
    teardown_commands: list[str] = field(default_factory=lambda: list[str]())
    extra_source_repos: list[Path] = field(default_factory=lambda: list[Path]())
    exclude_source_repos: list[str] = field(default_factory=lambda: list[str]())
    bundle_files: list[Path] = field(default_factory=lambda: list[Path]())
    run_as_submission: bool = False
    make_share_version: bool = False
    public: bool = False
    enable_internet: bool = False
    cpu_only: bool = False
    accelerator: str | None = DEFAULT_ACCELERATOR
    docker_image: str | None = None
    max_runtime_s: float = 12 * 3600.0
    startup_stealth: bool = True
    startup_deadline_s: float | None = None
    submission_startup_deadline_s: float | None = None
    startup_deadline_after_boot_s: float | None = None
    submission_startup_deadline_after_boot_s: float | None = None
    kernel_push_timeout_s: int | None = None
    dataset_version_message: str = "Update TAAF Kaggle source bundle."
    dry_run: bool = False

    # Populated by ``deploy`` / runner so solvers can inspect the real
    # Kaggle mode via ``solver.runtime_environment``.
    kernel_id: str = field(default="", init=False)
    source_dataset_ref: str = field(default="", init=False)
    uploaded: bool = field(default=False, init=False)
    actual_run_as_submission: bool = field(default=False, init=False)
    is_competition_rerun: bool = field(default=False, init=False)

    async def deploy(self, benchmark: taaf.benchmark.Benchmark) -> KaggleHandle:
        if benchmark.job_dir is None:
            raise ValueError("KaggleTarget requires benchmark.job_dir to be set for local staging (R2.34).")
        if not self.startup_stealth:
            # Refused here rather than by run() on Kaggle, after the install.
            deadlines = [name for name in _STARTUP_DEADLINE_FIELDS if getattr(self, name) is not None]
            if deadlines:
                raise ValueError(
                    f"KaggleTarget {', '.join(deadlines)} needs startup_stealth (R2.67): without it the games "
                    "open before the solver starts up, so a startup deadline could no longer keep them closed."
                )
        job_dir = benchmark.job_dir.resolve()
        taaf.deploy.check_job_dir_unused(job_dir)
        job_dir.mkdir(parents=True, exist_ok=True)

        _load_dotenv_upwards()
        username, key = _resolve_kaggle_credentials(self.username)
        share_suffix = "-share" if self.make_share_version else ""
        kernel_slug = slugify(self.kernel_slug or f"{DEFAULT_KERNEL_SLUG}{share_suffix}")
        default_title = "TAAF ARC AGI3 Share" if self.make_share_version else "TAAF ARC AGI3"
        kernel_title = self.kernel_title or (
            default_title if self.kernel_slug is None else kernel_slug.replace("-", " ")
        )
        if slugify(kernel_title) != kernel_slug:
            raise ValueError(
                "KaggleTarget kernel_title must slugify to kernel_slug. "
                f"Got kernel_slug={kernel_slug!r}, kernel_title={kernel_title!r}, "
                f"slugified title={slugify(kernel_title)!r}."
            )
        dataset_ref = normalize_dataset_ref(
            self.dataset_ref or f"{slugify(username)}/{DEFAULT_SOURCE_DATASET_SLUG}{share_suffix}"
        )
        self.kernel_id = f"{slugify(username)}/{kernel_slug}"
        self.source_dataset_ref = dataset_ref

        solver = benchmark.solver
        dataset_sources = _dedupe(
            normalize_dataset_ref(value)
            for value in [
                *self.additional_dataset_sources,
                *_solver_iterable_attr(solver, "kaggle_dataset_sources"),
            ]
            if str(value).strip()
        )
        model_sources = _dedupe(
            str(value).strip()
            for value in [
                *self.additional_model_sources,
                *_solver_iterable_attr(solver, "kaggle_model_sources"),
            ]
            if str(value).strip()
        )
        kernel_sources = _dedupe(
            normalize_kernel_ref(value)
            for value in [
                *self.additional_kernel_sources,
                *_solver_iterable_attr(solver, "kaggle_kernel_sources"),
            ]
            if str(value).strip()
        )
        setup_commands = _dedupe(
            [
                *self.setup_commands,
                *_solver_iterable_attr(solver, "kaggle_setup_commands"),
            ]
        )
        teardown_commands = _dedupe(
            [
                *self.teardown_commands,
                *_solver_iterable_attr(solver, "kaggle_teardown_commands"),
            ]
        )
        docker_image = _resolve_docker_image(self.docker_image, getattr(solver, "kaggle_docker_image", None))
        pip_packages = _dedupe(_solver_iterable_attr(solver, "kaggle_pip_install_packages"))
        if pip_packages:
            quoted = " ".join(_shell_quote(pkg) for pkg in pip_packages)
            setup_commands.append(f'"$PYTHON" -m pip install --no-deps {quoted}')

        staging_root = job_dir / "kaggle"
        source_bundle = staging_root / "source-dataset"
        source_upload = staging_root / "source-dataset-upload"
        kernel_bundle = staging_root / "kernel"
        taaf.deploy.write_git_status(job_dir)
        preamble = taaf.deploy.format_preamble(benchmark)
        _write_source_dataset_bundle(
            benchmark=benchmark,
            target=self,
            bundle_dir=source_bundle,
            preamble=preamble,
            setup_commands=setup_commands,
            teardown_commands=teardown_commands,
            make_share_version=self.make_share_version,
        )
        _write_source_dataset_upload(source_bundle, source_upload)
        _write_kernel_bundle(
            bundle_dir=kernel_bundle,
            kernel_id=self.kernel_id,
            kernel_title=kernel_title,
            dataset_sources=_dedupe([dataset_ref, *dataset_sources]),
            model_sources=model_sources,
            kernel_sources=kernel_sources,
            private=not self.public,
            enable_gpu=not self.cpu_only,
            enable_internet=self.enable_internet,
            accelerator=None if self.cpu_only else self.accelerator,
            docker_image=docker_image,
            run_as_submission=self.run_as_submission,
            make_share_version=self.make_share_version,
        )

        self.uploaded = False
        if not self.dry_run:
            if key is None:
                raise RuntimeError("Missing Kaggle API key. Set KAGGLE_KEY or ~/.kaggle/kaggle.json.")
            _ensure_kaggle_cli_available()
            _check_kaggle_auth(username, key)
            _assert_targets_not_public(self, username, key)
            _ensure_dataset(source_upload, dataset_ref, self.dataset_version_message, username, key)
            _push_kernel(
                kernel_bundle,
                accelerator=None if self.cpu_only else self.accelerator,
                timeout=self.kernel_push_timeout_s,
                username=username,
                key=key,
            )
            self.uploaded = True

        taaf.deploy.write_deploy_meta(
            job_dir=job_dir,
            target=self,
            handle_class=KaggleHandle,
            benchmark=benchmark,
            job_id=self.kernel_id,
        )
        return KaggleHandle(
            job_dir=job_dir,
            kernel_id=self.kernel_id,
            dataset_ref=dataset_ref,
            uploaded=self.uploaded,
        )


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    if not slug:
        raise ValueError(f"Could not derive a Kaggle slug from {value!r}.")
    return slug


def split_dataset_ref(dataset_ref: str) -> tuple[str, str]:
    parts = dataset_ref.strip().split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Expected Kaggle dataset ref in owner/slug format, got {dataset_ref!r}")
    owner, slug = parts
    if slugify(owner) != owner or slugify(slug) != slug:
        raise ValueError(f"Expected lowercase Kaggle owner/slug dataset ref, got {dataset_ref!r}")
    return owner, slug


def normalize_dataset_ref(dataset_ref: str) -> str:
    """Return a Kaggle dataset ref in ``owner/slug`` form."""
    text = str(dataset_ref).strip()
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc:
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or host not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError(f"Expected Kaggle dataset URL or owner/slug dataset ref, got {dataset_ref!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0] != "datasets":
            raise ValueError(f"Expected Kaggle dataset URL or owner/slug dataset ref, got {dataset_ref!r}")
        text = f"{parts[1]}/{parts[2]}"
    split_dataset_ref(text)
    return text


def normalize_kernel_ref(kernel_ref: str) -> str:
    """Return a Kaggle kernel (notebook / utility script) ref in ``owner/slug`` form."""
    text = str(kernel_ref).strip()
    parsed = urlparse(text)
    if parsed.scheme or parsed.netloc:
        host = (parsed.hostname or "").lower()
        if parsed.scheme not in {"http", "https"} or host not in {"kaggle.com", "www.kaggle.com"}:
            raise ValueError(f"Expected Kaggle kernel URL or owner/slug ref, got {kernel_ref!r}")
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 3 or parts[0] != "code":
            raise ValueError(f"Expected Kaggle kernel URL or owner/slug ref, got {kernel_ref!r}")
        text = f"{parts[1]}/{parts[2]}"
    split_dataset_ref(text)
    return text


def _dedupe(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in out:
            out.append(text)
    return out


def _resolve_docker_image(target_value: str | None, solver_value: object) -> str | None:
    """The image the notebook is pinned to: the target's or the solver's,
    which must agree when both are given; None when neither is."""
    values = _dedupe(str(value) for value in (target_value, solver_value) if value is not None)
    if len(values) > 1:
        raise ValueError(
            f"KaggleTarget.docker_image={target_value!r} disagrees with the solver's "
            f"kaggle_docker_image={solver_value!r}; a notebook runs on one image."
        )
    return values[0] if values else None


def _solver_iterable_attr(solver: object, attr_name: str) -> list[str]:
    if solver is None or not hasattr(solver, attr_name):
        return []
    value: object = getattr(solver, attr_name)
    if callable(value):
        value = value()
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        items = cast(Iterable[object], value)
        return [str(item) for item in items]
    return []


def _excluded_source_repos(solver: object, target: "KaggleTarget", *, make_share_version: bool) -> list[str]:
    """Repo directory names to keep out of the source dataset (R2.35).

    Resolved in one place so the uploaded bundle and the local-debug one
    exclude the same set; the share variant's own exclusion is part of it
    rather than a second thing a caller has to remember.
    """
    return _dedupe(
        [
            *(_SHARE_EXCLUDE_REPOS if make_share_version else ()),
            *target.exclude_source_repos,
            *_solver_iterable_attr(solver, "kaggle_exclude_source_repos"),
        ]
    )


def _bundle_payload_files(solver: object, target: "KaggleTarget") -> list[Path]:
    """Files to copy into the bundle's ``payload/`` directory.

    Resolved in one place so the uploaded bundle and the local-debug one
    carry the same payload.
    """
    paths = [*target.bundle_files, *(Path(value) for value in _solver_iterable_attr(solver, "kaggle_bundle_files"))]
    resolved: list[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.is_file():
            raise RuntimeError(f"kaggle bundle payload file missing: {candidate}")
        if candidate not in resolved:
            resolved.append(candidate)
    names = [candidate.name for candidate in resolved]
    if len(set(names)) != len(names):
        raise RuntimeError(f"kaggle bundle payload files collide by name: {sorted(names)}")
    return resolved


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _load_dotenv_upwards(start: Path | None = None) -> None:
    start_dir = (start or Path.cwd()).resolve()
    for directory in [start_dir, *start_dir.parents]:
        dotenv_path = directory / ".env"
        if dotenv_path.is_file():
            _load_dotenv_file(dotenv_path)
            return


def _load_dotenv_file(path: Path) -> None:
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        value = _normalize_credential(value)
        if name and value is not None and name not in os.environ:
            os.environ[name] = value


def _normalize_credential(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value or None


def _kaggle_config_file() -> Path:
    config_dir = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_dir:
        return Path(config_dir).expanduser() / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def _resolve_kaggle_credentials(configured_username: str | None) -> tuple[str, str | None]:
    username = configured_username or os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
    config_file = _kaggle_config_file()
    if config_file.is_file():
        try:
            data = cast(dict[str, Any], json.loads(config_file.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid Kaggle credentials file at {config_file}: {exc}") from exc
        username = username or cast(str | None, data.get("username"))
        key = key or cast(str | None, data.get("key"))
    username = _normalize_credential(username)
    key = _normalize_credential(key)
    if not username:
        raise RuntimeError("Missing Kaggle username. Set KAGGLE_USERNAME, ~/.kaggle/kaggle.json, or target.username.")
    return username, key


def _kaggle_env(username: str, key: str) -> dict[str, str]:
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = username
    env["KAGGLE_KEY"] = key
    # kaggle CLI >= 2.x authenticates with KAGGLE_API_TOKEN. Keep KAGGLE_KEY
    # for older clients and set both so the deployment works across versions.
    env["KAGGLE_API_TOKEN"] = key
    # The kaggle CLI stores resumable-upload state at a fixed
    # {tempdir}/.kaggle/uploads. On a shared host the first user to upload
    # owns that directory and everyone else gets EACCES, so give the CLI a
    # per-user tempdir instead.
    tmpdir = Path(tempfile.gettempdir()) / f"kaggle-cli-{getpass.getuser()}"
    tmpdir.mkdir(parents=True, exist_ok=True)
    env["TMPDIR"] = str(tmpdir)
    return env


def _ensure_kaggle_cli_available() -> None:
    if shutil.which("kaggle") is None:
        raise RuntimeError(
            "Kaggle CLI is not installed or not on PATH. Install it with `python -m pip install kaggle`."
        )


def _check_kaggle_auth(username: str, key: str) -> None:
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "1"],
        capture_output=True,
        env=_kaggle_env(username, key),
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return
    output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
    raise RuntimeError(f"Kaggle auth check failed:\n{output}")


def _kaggle_api(username: str, key: str) -> Any:
    """Authenticated Kaggle Python API client.

    The CLI exposes no dataset-visibility field, so the public-target guard
    (R2.62) reads ``is_private`` off the Python API instead. ``import kaggle``
    authenticates eagerly from the environment, so the credentials are set
    first; imported lazily to keep module import light and off the dry-run
    path. The env vars are restored afterwards (auth is cached on the api
    object), matching the module's throwaway-env pattern (``_kaggle_env``).
    """
    saved = {name: os.environ.get(name) for name in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN")}
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    os.environ["KAGGLE_API_TOKEN"] = key
    try:
        import kaggle

        kaggle.api.authenticate()
        return kaggle.api
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _fetch_is_private(api: Any, *, kind: str, owner: str, slug: str) -> bool:
    """Read ``is_private`` for the exact Kaggle ref via a direct get-by-ref.

    Uses the low-level ``kagglesdk`` client because neither the CLI nor the
    search endpoints return an exact resource's visibility. Raises on a
    missing/forbidden ref — the caller classifies that as 'not created yet'
    (R2.62).
    """
    with api.build_kaggle_client() as client:
        if kind == "notebook":
            from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

            request: Any = ApiGetKernelRequest()
            request.user_name = owner
            request.kernel_slug = slug
            return bool(client.kernels.kernels_api_client.get_kernel(request).metadata.is_private)
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetMetadataRequest

        request = ApiGetDatasetMetadataRequest()
        request.owner_slug = owner
        request.dataset_slug = slug
        return bool(client.datasets.dataset_api_client.get_dataset_metadata(request).info.is_private)


def _kaggle_ref_absent(exc: BaseException, *, owner: str, username: str) -> bool:
    """Whether a get-by-ref failure means the ref simply doesn't exist yet.

    Kaggle masks a missing resource as HTTP 403 (or 404); for a ref we own that
    is unambiguously 'not created yet', so the deploy may proceed and create it
    private by default (R2.52). Mirrors the 403/404 handling in
    ``_dataset_exists``; any other failure is an unresolved visibility (the
    caller fails closed).
    """
    text = str(exc).lower()
    if "404" in text or "not found" in text or "does not exist" in text:
        return True
    return ("403" in text or "forbidden" in text) and owner == slugify(username)


def _visibility_unresolved_hint(kind: str, ref: str, reason: str) -> str:
    label = "notebook" if kind == "notebook" else "dataset"
    return (
        f"Refusing to deploy: could not determine whether the target {label} '{ref}' is public — "
        f"{reason} (R2.62). A leak guard must not push to a target of unknown visibility. Check your "
        f"Kaggle credentials / connectivity and the ref, then retry; to publish deliberately, use the "
        f"share variant (make_share_version=True)."
    )


def _resource_is_public(api: Any, *, kind: str, ref: str, username: str) -> bool:
    """Whether the Kaggle ``ref`` (``owner/slug``) exists and is public.

    ``kind`` is ``"notebook"`` or ``"dataset"``. Reads ``is_private`` via a
    direct get-by-ref (not a search), so an existing public target can't be
    missed. A ref that doesn't exist yet reads as not-public (``False``) — the
    deploy will create it private by default (R2.52), so a first deploy
    proceeds. Any other failure to read visibility raises: a leak guard must
    fail closed, never silently treat an unresolved target as safe (R2.62).
    """
    owner, slug = split_dataset_ref(ref)
    try:
        is_private = _fetch_is_private(api, kind=kind, owner=owner, slug=slug)
    except Exception as exc:
        if _kaggle_ref_absent(exc, owner=owner, username=username):
            return False
        raise RuntimeError(_visibility_unresolved_hint(kind, ref, f"the Kaggle query failed ({exc})")) from exc
    return is_private is False


def _assert_targets_not_public(target: KaggleTarget, username: str, key: str) -> None:
    """Refuse to upload to a public Kaggle target (R2.62).

    TAAF always pushes privately; making an artefact public is a deliberate
    manual step taken after the run. So a target that is already public — or a
    request to push public (``public=True``) — is refused, for every variant
    including the share variant (R2.61): a public target at deploy time is an
    accident to catch, not the intended state. For the default (non-share)
    bundle this also prevents leaking TAAF's private offline env files and Tufa
    source snapshots (R2.35). Visibility is checked fail-closed via
    ``_resource_is_public`` (a not-yet-created target is created private, so
    first deploys proceed).
    """
    if target.public:
        raise RuntimeError(
            "Refusing to deploy with target.public=True: TAAF always pushes privately, and making a "
            "notebook public is a deliberate manual step taken after the run (R2.62). Deploy privately, "
            "then publish on Kaggle once you've reviewed it."
        )
    api = _kaggle_api(username, key)
    offenders: list[str] = []
    if _resource_is_public(api, kind="notebook", ref=target.kernel_id, username=username):
        offenders.append(f"notebook https://www.kaggle.com/code/{target.kernel_id}")
    if _resource_is_public(api, kind="dataset", ref=target.source_dataset_ref, username=username):
        offenders.append(f"dataset https://www.kaggle.com/datasets/{target.source_dataset_ref}")
    if offenders:
        raise RuntimeError(
            "Refusing to deploy to a public Kaggle target — TAAF pushes privately and publishing is a "
            "deliberate manual step after the run, so a public target is treated as an accident (R2.62). "
            "For the default bundle this also prevents leaking TAAF's private offline env files (R2.35). "
            "Public target(s):\n  "
            + "\n  ".join(offenders)
            + "\nMake them private on Kaggle first; publish manually after the run if that was intended."
        )


def _write_source_dataset_bundle(
    *,
    benchmark: taaf.benchmark.Benchmark,
    target: KaggleTarget,
    bundle_dir: Path,
    preamble: str,
    setup_commands: list[str],
    teardown_commands: list[str],
    make_share_version: bool = False,
) -> None:
    notebook_name = SHARE_NOTEBOOK_NAME if make_share_version else NOTEBOOK_NAME
    shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    taaf.deploy.snapshot_editable_sources(
        bundle_dir / "src",
        extra_repos=target.extra_source_repos,
        exclude_repos=_excluded_source_repos(benchmark.solver, target, make_share_version=make_share_version),
    )
    payload_files = _bundle_payload_files(benchmark.solver, target)
    if payload_files:
        payload_dir = bundle_dir / "payload"
        payload_dir.mkdir(parents=True, exist_ok=True)
        for payload_file in payload_files:
            shutil.copy2(payload_file, payload_dir / payload_file.name)
    taaf.support.atomic_pickle_dump(benchmark, bundle_dir / "benchmark_initial.pkl")
    taaf.support.atomic_pickle_dump(target, bundle_dir / "deploy_target.pkl")
    (bundle_dir / "preamble.txt").write_text(preamble, encoding="utf-8")
    # Bundled so the runner can drop it into /kaggle/working before run():
    # the worker can't regenerate git status (no .git / dist-info there), and
    # diagnostics.html embeds job_dir/git_status.txt.
    (bundle_dir / "git_status.txt").write_text(taaf.deploy.format_git_status(), encoding="utf-8")
    (bundle_dir / "setup_commands.json").write_text(json.dumps(setup_commands, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / "teardown_commands.json").write_text(json.dumps(teardown_commands, indent=2) + "\n", encoding="utf-8")
    (bundle_dir / DATASET_BUNDLE_MARKER).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": datetime.now().isoformat(),
                "benchmark_label": benchmark.label,
                "notebook": notebook_name,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (bundle_dir / "README.dataset.md").write_text(
        "# TAAF Kaggle Source Bundle\n\n"
        "Generated by TAAF. Contains source snapshots and a pickled Benchmark. "
        "The visible Kaggle notebook owns the run logic. This dataset intentionally "
        "does not contain Kaggle credentials.\n",
        encoding="utf-8",
    )
    (bundle_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": "TAAF Kaggle Source Bundle",
                "id": target.source_dataset_ref,
                "licenses": [{"name": "CC0-1.0"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_source_dataset_upload(bundle_dir: Path, upload_dir: Path) -> None:
    """The source dataset as it is uploaded: the bundle as one archive,
    ``DATASET_BUNDLE_ARCHIVE``, beside the dataset's metadata."""
    metadata = bundle_dir / "dataset-metadata.json"
    shutil.rmtree(upload_dir, ignore_errors=True)
    upload_dir.mkdir(parents=True)
    with zipfile.ZipFile(upload_dir / DATASET_BUNDLE_ARCHIVE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if path.is_file() and path != metadata:
                archive.write(path, path.relative_to(bundle_dir))
    shutil.copy2(metadata, upload_dir / metadata.name)


def _write_kernel_bundle(
    *,
    bundle_dir: Path,
    kernel_id: str,
    kernel_title: str,
    dataset_sources: list[str],
    model_sources: list[str],
    kernel_sources: list[str],
    private: bool,
    enable_gpu: bool,
    enable_internet: bool,
    accelerator: str | None,
    run_as_submission: bool,
    make_share_version: bool = False,
    docker_image: str | None = None,
) -> None:
    shutil.rmtree(bundle_dir, ignore_errors=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    notebook_name = SHARE_NOTEBOOK_NAME if make_share_version else NOTEBOOK_NAME
    template = _SHARE_NOTEBOOK_TEMPLATE if make_share_version else _NOTEBOOK_TEMPLATE
    (bundle_dir / notebook_name).write_text(
        _render_kaggle_notebook(
            run_as_submission=run_as_submission,
            dataset_sources=dataset_sources,
            kernel_sources=kernel_sources,
            enable_gpu=enable_gpu,
            template=template,
        ),
        encoding="utf-8",
    )
    metadata: dict[str, Any] = {
        "id": kernel_id,
        "title": kernel_title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": private,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": enable_internet,
        "competition_sources": [COMPETITION_SLUG],
        "dataset_sources": dataset_sources,
        "kernel_sources": kernel_sources,
        "model_sources": model_sources,
    }
    if accelerator:
        metadata["machine_shape"] = accelerator
    if docker_image:
        metadata["docker_image"] = docker_image
    (bundle_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def _render_kaggle_notebook(
    *,
    run_as_submission: bool,
    dataset_sources: Iterable[str] = (),
    kernel_sources: Iterable[str] = (),
    enable_gpu: bool = False,
    template: Path = _NOTEBOOK_TEMPLATE,
) -> str:
    notebook = cast(dict[str, Any], json.loads(template.read_text(encoding="utf-8")))
    replacements = {
        "__TAAF_RUN_AS_SUBMISSION__": "True" if run_as_submission else "False",
        "__TAAF_ENABLE_GPU__": "True" if enable_gpu else "False",
        "__TAAF_COMPETITION_WHEELHOUSES__": json.dumps(list(COMPETITION_WHEELHOUSES)),
        "__TAAF_DATASET_SOURCES__": json.dumps(list(dataset_sources)),
        "__TAAF_DATASET_BUNDLE_MARKER__": json.dumps(DATASET_BUNDLE_MARKER),
        "__TAAF_DATASET_BUNDLE_ARCHIVE__": json.dumps(DATASET_BUNDLE_ARCHIVE),
        "__TAAF_KERNEL_SOURCES__": json.dumps(list(kernel_sources)),
        "__TAAF_KAGGLE_WORKING_DIR__": json.dumps(KAGGLE_WORKING_DIR),
        "__TAAF_SOFT_DEADLINE_BUFFER_S__": repr(_SOFT_DEADLINE_BUFFER_S),
    }
    cells = notebook.get("cells", [])
    if not isinstance(cells, list):
        raise ValueError(f"{_NOTEBOOK_TEMPLATE} must contain a JSON list at cells.")
    for cell_obj in cast(list[object], cells):
        if not isinstance(cell_obj, dict):
            raise ValueError(f"{template} contains a non-object notebook cell.")
        cell = cast(dict[str, Any], cell_obj)
        # The noqa pragmas silence ruff on the undefined placeholder names in
        # the repo source; once placeholders are filled the names are defined,
        # so strip the now-pointless pragma from code cells before upload.
        strip_noqa = cell.get("cell_type") == "code"
        source = cell.get("source")
        if isinstance(source, list):
            source_lines = cast(list[object], source)
            cell["source"] = [_render_notebook_line(str(line), replacements, strip_noqa) for line in source_lines]
        elif isinstance(source, str):
            cell["source"] = _render_notebook_line(source, replacements, strip_noqa)
    return json.dumps(notebook, indent=1, ensure_ascii=True) + "\n"


_NOQA_RE = re.compile(r"[ \t]*#[ \t]*noqa\b.*$")


def _render_notebook_line(text: str, replacements: dict[str, str], strip_noqa: bool) -> str:
    text = _replace_notebook_placeholders(text, replacements)
    return _strip_noqa(text) if strip_noqa else text


def _strip_noqa(text: str) -> str:
    return "\n".join(_NOQA_RE.sub("", line) for line in text.split("\n"))


def _replace_notebook_placeholders(text: str, replacements: dict[str, str]) -> str:
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _ensure_dataset(bundle_dir: Path, dataset_ref: str, message: str, username: str, key: str) -> None:
    exists = _dataset_exists(dataset_ref, username=username, key=key)
    # Capture the live version marker before the push so we can confirm the
    # new version actually went live (not just that status went "ready").
    before = _dataset_last_updated(dataset_ref, username=username, key=key) if exists else None
    if exists:
        command = ["kaggle", "datasets", "version", "-p", str(bundle_dir), "-m", message, "-r", "zip"]
    else:
        command = ["kaggle", "datasets", "create", "-p", str(bundle_dir), "-r", "zip"]
    result = subprocess.run(command, env=_kaggle_env(username, key), text=True, check=False)
    if result.returncode != 0:
        # stdout/stderr stream through unchanged (upload progress), so the
        # CLI's own error text is above this traceback, not in it.
        raise RuntimeError(
            f"`{' '.join(command)}` failed with exit code {result.returncode}; "
            "see the kaggle CLI output above for the underlying error."
        )
    _wait_for_dataset_ready(dataset_ref, username=username, key=key)
    _wait_for_dataset_version_change(dataset_ref, before=before, username=username, key=key)


def _dataset_exists(dataset_ref: str, *, username: str, key: str) -> bool:
    result = subprocess.run(
        ["kaggle", "datasets", "status", dataset_ref],
        capture_output=True,
        env=_kaggle_env(username, key),
        text=True,
        check=False,
    )
    if result.returncode == 0:
        return True
    output = f"{result.stdout}\n{result.stderr}".lower()
    if "404" in output or "not found" in output or "does not exist" in output:
        return False
    if "403" in output or "forbidden" in output:
        owner, _slug = split_dataset_ref(dataset_ref)
        if owner == slugify(username):
            return False
    raise RuntimeError(f"`kaggle datasets status {dataset_ref}` failed:\n{output}")


def _wait_for_dataset_ready(dataset_ref: str, *, username: str, key: str, timeout_s: float = 300.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_output = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["kaggle", "datasets", "status", dataset_ref],
            capture_output=True,
            env=_kaggle_env(username, key),
            text=True,
            check=False,
        )
        output = "\n".join(part for part in [result.stdout.strip(), result.stderr.strip()] if part)
        last_output = output
        if result.returncode == 0 and "ready" in output.lower():
            return
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for Kaggle dataset {dataset_ref} to become ready:\n{last_output}")


def _dataset_last_updated(dataset_ref: str, *, username: str, key: str) -> str | None:
    """The dataset's ``lastUpdated`` timestamp from ``kaggle datasets list``,
    or ``None`` when the ref isn't listed. Used as a version proxy — the CLI
    exposes no version number, and ``lastUpdated`` advances on every push.
    """
    owner, slug = split_dataset_ref(dataset_ref)
    scope = ["--mine"] if owner == slugify(username) else ["--user", owner]
    result = subprocess.run(
        ["kaggle", "datasets", "list", *scope, "-s", slug, "--csv"],
        capture_output=True,
        env=_kaggle_env(username, key),
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    # The CLI prints warnings on stdout before the CSV; skip to the header.
    lines = result.stdout.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("ref,")), None)
    if header_idx is None:
        return None
    for row in csv.DictReader(lines[header_idx:]):
        if row.get("ref") == dataset_ref:
            return row.get("lastUpdated")
    return None


def _wait_for_dataset_version_change(
    dataset_ref: str, *, before: str | None, username: str, key: str, timeout_s: float = 300.0
) -> None:
    """After a version push + ``ready``, wait until the dataset listing's
    ``lastUpdated`` advances past ``before`` — the closest CLI-observable
    signal that the new version is actually live. ``kaggle kernels push``
    attaches the *current* version, and that pointer can briefly lag the
    ``ready`` status, which is the race that drops a just-pushed dataset.

    Best-effort: a dataset can report ready before its listing entry
    refreshes, and a private dataset owned by someone else isn't visible to
    ``datasets list`` at all. Rather than block the deploy, warn and proceed
    on timeout — the ``ready`` gate above already passed.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = _dataset_last_updated(dataset_ref, username=username, key=key)
        if current is not None and current != before:
            return
        time.sleep(5)
    print(
        f"taaf.kaggle: dataset {dataset_ref} reported ready but its listing lastUpdated "
        f"did not advance past {before!r} within {timeout_s:.0f}s; proceeding (a kernel "
        "push now may briefly attach the previous version).",
        flush=True,
    )


def _push_kernel(
    bundle_dir: Path,
    *,
    accelerator: str | None,
    timeout: int | None,
    username: str,
    key: str,
) -> None:
    command = ["kaggle", "kernels", "push", "-p", str(bundle_dir)]
    if accelerator:
        command.extend(["--accelerator", accelerator])
    if timeout is not None:
        command.extend(["-t", str(timeout)])
    result = subprocess.run(command, env=_kaggle_env(username, key), text=True, check=False)
    if result.returncode != 0:
        # stdout/stderr stream through unchanged (upload progress), so the
        # CLI's own error text is above this traceback, not in it.
        raise RuntimeError(
            f"`{' '.join(command)}` failed with exit code {result.returncode}; "
            "see the kaggle CLI output above for the underlying error."
        )


def package_for_local_debug(
    benchmark: taaf.benchmark.Benchmark,
    target: KaggleTarget,
    output_dir: Path,
) -> Path:
    """Build only the source dataset bundle for local notebook smoke tests.

    Honours ``target.make_share_version`` so a share target's local bundle
    reflects the share variant (``re-arc-3`` excluded, share notebook in the
    marker). Note that :func:`run_source_bundle_locally` still only executes
    the default variant — see its docstring.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if target.dataset_ref:
        target.source_dataset_ref = normalize_dataset_ref(target.dataset_ref)
    elif not target.source_dataset_ref:
        target.source_dataset_ref = "local/taaf-release-source"
    _write_source_dataset_bundle(
        benchmark=benchmark,
        target=target,
        bundle_dir=output_dir,
        preamble=taaf.deploy.format_preamble(benchmark),
        setup_commands=list(target.setup_commands),
        teardown_commands=list(target.teardown_commands),
        make_share_version=target.make_share_version,
    )
    return output_dir


def run_source_bundle_locally(
    bundle_dir: Path, working_dir: Path, *, timeout_s: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Execute the rendered Kaggle notebook against a source bundle locally.

    Always renders and executes the default ``taaf_kaggle_run.ipynb`` variant.
    The share notebook (R2.61) is Kaggle-only — its offline branch reads the
    competition's ``environment_files`` dataset, which is absent locally — so
    local execution does not cover it even when ``bundle_dir`` was packaged
    from a ``make_share_version`` target.
    """
    env = os.environ.copy()
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(bundle_dir)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(working_dir)
    target = _load_packaged_kaggle_target(bundle_dir)
    notebook = json.loads(
        _render_kaggle_notebook(
            run_as_submission=False,
            dataset_sources=_target_declared_dataset_sources(target) if target is not None else (),
            kernel_sources=_target_declared_kernel_sources(target) if target is not None else (),
            enable_gpu=False,
        )
    )
    code = "\n\n".join("".join(cell.get("source", [])) for cell in notebook["cells"] if cell.get("cell_type") == "code")
    env["TAAF_LOCAL_NOTEBOOK_CODE"] = code
    runner = (
        "import ast, asyncio, inspect, os\n"
        "code = os.environ.pop('TAAF_LOCAL_NOTEBOOK_CODE')\n"
        "compiled = compile(code, 'taaf_kaggle_run.ipynb', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)\n"
        "result = eval(compiled, {'__name__': '__main__'})\n"
        "if inspect.isawaitable(result):\n"
        "    asyncio.run(result)\n"
    )
    return subprocess.run(
        [sys_executable(), "-c", runner],
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def sys_executable() -> str:
    import sys

    return sys.executable


def _load_packaged_kaggle_target(bundle_dir: Path) -> KaggleTarget | None:
    path = bundle_dir / "deploy_target.pkl"
    if not path.is_file():
        return None
    with open(path, "rb") as file:
        restored = pickle.load(file)
    return restored if isinstance(restored, KaggleTarget) else None


def _target_declared_dataset_sources(target: KaggleTarget) -> list[str]:
    return _dedupe(
        normalize_dataset_ref(value)
        for value in [target.source_dataset_ref, *target.additional_dataset_sources]
        if str(value).strip()
    )


def _target_declared_kernel_sources(target: KaggleTarget) -> list[str]:
    return _dedupe(normalize_kernel_ref(value) for value in target.additional_kernel_sources if str(value).strip())
