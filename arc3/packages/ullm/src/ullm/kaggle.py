"""What a Kaggle kernel needs to run this workspace's engine.

A kernel has no package index, so the engine and its closure install
``--no-index`` from a wheelhouse attached to the kernel, and the fork this
workspace pins reaches the kernel as a delta laid over that stock install
(``ullm.vllm_overlay``). Everything an agent's deployment has to say about
the engine — which wheelhouse, which requirement set, whether the fork rides
along — is one ``KaggleEngine``; the shipped one is ``VLLM_FORK``, linked
from the agent's run configuration. Changing the engine line means changing
it here and the pin in the workspace's root ``pyproject.toml``; the test
suite checks that the two agree.
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import shlex
from collections.abc import Mapping

#: The stock wheelhouse on the engine line this workspace pins. It holds the
#: whole offline closure of the requirement set below, built for Kaggle's
#: image; the fork's overlay is built against the same precompiled base the
#: wheel is, which the kernel checks file by file before applying it. Beside
#: its ``wheels/`` it holds the record of installing them on the kernel's
#: image, which a kernel that matches replays instead of running pip
#: (``ullm.kaggle_site``). Built by ``tools/kaggle/arc3-site-0290``, a fork of
#: ``jeroencottaar/arc3-utils-0290`` that downloads the same way.
KAGGLE_VLLM_WHEELS = "jeroencottaar/arc3-site-0290"
#: The Kaggle image a kernel runs on, pinned (TAAF R2.65) rather than left to
#: whatever Kaggle's current one is: the wheelhouse is resolved on it and its
#: record starts from its installed state. The image the nine-hour RTX runs
#: played on, and the one the wheelhouse's notebook is pinned to
#: (``tools/kaggle/arc3-site-0290/kernel-metadata.json``; a test holds them
#: equal).
KAGGLE_DOCKER_IMAGE = (
    "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
)
KAGGLE_PILLOW_REQUIREMENT = "pillow==11.3.0"
#: What the kernel names when installing from the wheelhouse, which holds
#: the whole closure: pip resolves the rest from it, so only the engine and
#: what the closure cannot decide for us is named. The engine is pinned to
#: the line; pillow is held at the Kaggle image's version; the CUDA 13
#: nvcc/crt/nvvm pins hold the JIT toolchain to torch's cudart so
#: flashinfer's sm_120 kernels link on the kernel's card. transformers is
#: named because the workspace lock pins it: left to resolve, the kernel
#: takes whatever the wheelhouse holds and the two deployments render and
#: tokenize on different releases. A test holds this to the installed one.
KAGGLE_TRANSFORMERS_REQUIREMENT = "transformers==5.16.1"
#: FlashInfer's kernels, prebuilt. A kernel has no network at any point, so
#: the cubins the engine downloads from NVIDIA everywhere else cannot be
#: fetched there, and FlashInfer falls back to building them with the nvcc
#: pinned below -- which spawns one ``cicc`` per parallel job, several GiB
#: apiece, beside a model that already holds most of the box.
#:
#: These two are the same kernels, shipped. ``flashinfer-cubin`` is the
#: downloaded-cubin set and ``flashinfer-jit-cache`` the compiled modules.
#: Both are keyed to the FlashInfer version by content hash, so the version
#: is pinned HERE rather than left to resolve as a dependency of vllm: a
#: drift does not fail, it misses the cache and compiles anyway.
KAGGLE_FLASHINFER_VERSION = "0.6.18"
KAGGLE_VLLM_REQUIREMENTS = (
    "vllm==0.29.0",
    KAGGLE_TRANSFORMERS_REQUIREMENT,
    KAGGLE_PILLOW_REQUIREMENT,
    f"flashinfer-python=={KAGGLE_FLASHINFER_VERSION}",
    f"flashinfer-cubin=={KAGGLE_FLASHINFER_VERSION}",
    f"flashinfer-jit-cache=={KAGGLE_FLASHINFER_VERSION}+cu130",
    # The CuteDSL half of FlashInfer, which `has_flashinfer_b12x_moe()`
    # checks for. Nothing asks for b12x today, but a kernel installs `--no-index`, so a
    # package absent from the wheelhouse is a wheelhouse rebuild away
    # rather than a config change away. It rides along for that reason.
    "nvidia-cutlass-dsl==4.6.2",
    "nvidia-cuda-nvcc==13.0.88",
    "nvidia-cuda-crt==13.0.88",
    "nvidia-nvvm==13.0.88",
)
#: The repository the workspace pins the fork from, by directory name: kept
#: out of the source bundle whenever the fork ships as an overlay, since the
#: runner prepends every bundled repo to sys.path and a bundled fork would
#: shadow the stock install its overlay expects underneath.
VLLM_FORK_REPO = "arc-prize-2026-vllm"


def _mount_candidates(
    ref: str,
    *,
    input_root: pathlib.Path,
    usr_lib_root: pathlib.Path,
    notebooks_root: pathlib.Path,
) -> list[pathlib.Path]:
    """Every layout Kaggle has used for an attached source, dataset or
    utility alike, with the slug's hyphens both kept and normalised.

    Kaggle has attached one and the same source as a utility output and as
    a plain dataset at different times, so the kind an agent declares it as
    says nothing about where the mount lands; every layout is probed for
    every ref.
    """
    owner, slug = ref.split("/", maxsplit=1)
    candidates: list[pathlib.Path] = []
    for name in dict.fromkeys((slug, slug.replace("-", "_"))):
        candidates.extend(
            (
                input_root / name,
                input_root / "datasets" / owner / name,
                usr_lib_root / name,
                usr_lib_root / "notebooks" / owner / name,
                notebooks_root / owner / name,
            )
        )
    return candidates


def resolve_kaggle_wheel_dir(
    ref: str = KAGGLE_VLLM_WHEELS,
    *,
    input_paths: Mapping[str, str] | None = None,
    input_root: pathlib.Path = pathlib.Path("/kaggle/input"),
    usr_lib_root: pathlib.Path = pathlib.Path("/kaggle/usr/lib"),
    notebooks_root: pathlib.Path = pathlib.Path("/kaggle/input/notebooks"),
) -> pathlib.Path:
    """Return a populated wheel directory from Kaggle's possible mounts.

    TAAF's input map is tried first, then every mount layout Kaggle has
    used; the first directory with a wheel in it wins, so an empty mount
    directory shadowing a populated one is passed over.
    """
    if input_paths is None:
        raw_paths = os.environ.get("TAAF_KAGGLE_INPUT_PATHS", "{}")
        loaded_paths = json.loads(raw_paths)
        if not isinstance(loaded_paths, dict):
            raise RuntimeError("TAAF_KAGGLE_INPUT_PATHS must contain a JSON object")
        input_paths = {str(key): str(value) for key, value in loaded_paths.items()}

    if "/" not in ref:
        raise ValueError(f"Kaggle utility ref must be owner/slug, got {ref!r}")

    candidates: list[pathlib.Path] = []
    mapped_root = input_paths.get(ref)
    if mapped_root is not None:
        candidates.append(pathlib.Path(mapped_root) / "wheels")
    candidates.extend(
        mount / "wheels"
        for mount in _mount_candidates(
            ref, input_root=input_root, usr_lib_root=usr_lib_root, notebooks_root=notebooks_root
        )
    )

    unique_candidates = list(dict.fromkeys(candidates))
    for candidate in unique_candidates:
        if candidate.is_dir() and next(candidate.iterdir(), None) is not None:
            return candidate

    checked = " and ".join(str(candidate) for candidate in unique_candidates)
    raise RuntimeError(f"populated wheel directory missing; checked {checked}")


def kaggle_vllm_overlay_command() -> str:
    """Build the command that puts the vLLM fork over the installed stock.

    Ordered after the wheel install: it replaces files the wheels lay
    down, and refuses if what it finds is not the base wheel it was built
    against (``ullm.vllm_overlay``).
    """
    return "\"$PYTHON\" -c 'import ullm.vllm_overlay; ullm.vllm_overlay.apply_from_bundle()'"


def _check_ref(ref: str, what: str) -> None:
    """Refuse a ref that is not ``owner/slug`` or would break the shell command."""
    if "/" not in ref or any(ch in ref for ch in "'\" \\"):
        raise ValueError(f"Kaggle {what} ref must be owner/slug, got {ref!r}")


def kaggle_site_install_command(*requirements: str, wheels: str = KAGGLE_VLLM_WHEELS) -> str:
    """Build the command that installs ``requirements`` from the wheelhouse
    ``wheels``: the record beside its wheels replayed when it matches the
    kernel, one offline pip install from them otherwise (``ullm.kaggle_site``)."""
    if not requirements:
        raise ValueError("at least one wheel requirement is required")
    _check_ref(wheels, "wheelhouse")
    arguments = " ".join(shlex.quote(argument) for argument in (wheels, *requirements))
    return f'"$PYTHON" -m ullm.kaggle_site {arguments}'


def kaggle_prefetch_command(*checkpoints: str) -> str:
    """Build the command that starts reading ``checkpoints`` (directories, in
    the order the engine loads them) into page cache a bounded window ahead
    of the engine's loader, detached, so the install runs beside it
    (``ullm.kaggle_prefetch``). Its log and report land beside the run's
    outputs."""
    if not checkpoints:
        raise ValueError("at least one checkpoint directory is required")
    arguments = " ".join(shlex.quote(checkpoint) for checkpoint in checkpoints)
    log = '"${TAAF_KAGGLE_WORKING_DIR:-/kaggle/working}/shard_prefetch.log"'
    return f'setsid nohup "$PYTHON" -m ullm.kaggle_prefetch {arguments} > {log} 2>&1 < /dev/null &'


#: Where a kernel mounts a dataset, as a cell names it; a mount that landed
#: elsewhere is found through TAAF's input map (``kaggle_prefetch.resolve_checkpoint_dir``).
KAGGLE_DATASET_MOUNT = "/kaggle/input/datasets"
#: Where a kernel mounts a Kaggle model. TAAF's input map lists datasets and
#: scripts alone, so a cell names a model's mount exactly (``model_mount``).
KAGGLE_MODEL_MOUNT = "/kaggle/input/models"


def model_mount(ref: str) -> str:
    """Where a kernel mounts the model ``ref``, which names one version as
    ``<owner>/<slug>/<framework>/<variation>/<version>``: the same path, the
    framework in lower case."""
    parts = ref.split("/")
    if len(parts) != 5 or not all(parts) or not parts[4].isdigit():
        raise ValueError(f"Kaggle model ref {ref!r}: expected <owner>/<slug>/<framework>/<variation>/<version>")
    owner, slug, framework, variation, version = parts
    return f"{KAGGLE_MODEL_MOUNT}/{owner}/{slug}/{framework.lower()}/{variation}/{version}"


def is_model_version(ref: str) -> bool:
    """Whether ``ref`` names one version of a Kaggle model rather than a
    dataset: five parts against a dataset's ``owner/slug``."""
    return ref.count("/") == 4


#: Where a checkpoint split over datasets is put together: outside the kernel's
#: outputs, which hold only what a run writes.
KAGGLE_ASSEMBLED_ROOT = "/tmp/ullm-checkpoints"


@dataclasses.dataclass(frozen=True)
class KaggleCheckpoint:
    """A checkpoint as the Kaggle datasets that hold it.

    ``dataset`` holds the checkpoint's own files -- its config, tokenizer and
    index -- and whichever of its shards it carries. A shard its index names
    and it does not carry comes from the first of ``shards_from`` that has it:
    expert prunes of one base share every shard the prune leaves alone (for
    Flash-Next, the four that hold the PLE table), so a prune's dataset need
    not spend its owner's quota on them. ``dataset`` and every ``shards_from``
    entry is a dataset or one version of a Kaggle model (``model_mount``).
    The kernel puts the two together
    before anything reads them (``ullm.kaggle_assemble``). Empty for a
    checkpoint in one dataset or model.
    """

    dataset: str
    shards_from: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse a ref the mount or the setup command could not take."""
        for ref in (self.dataset, *self.shards_from):
            if is_model_version(ref):
                model_mount(ref)
            else:
                _check_ref(ref, "dataset")

    @property
    def sources(self) -> tuple[str, ...]:
        """The datasets the kernel mounts for it."""
        return tuple(ref for ref in (self.dataset, *self.shards_from) if not is_model_version(ref))

    @property
    def model_sources(self) -> tuple[str, ...]:
        """The model versions the kernel mounts for it."""
        return tuple(ref for ref in (self.dataset, *self.shards_from) if is_model_version(ref))

    @property
    def path(self) -> str:
        """Where the engine reads it: its own mount, or the directory its
        assembly makes."""
        if self.shards_from:
            return f"{KAGGLE_ASSEMBLED_ROOT}/{self.dataset}"
        if is_model_version(self.dataset):
            return model_mount(self.dataset)
        return f"{KAGGLE_DATASET_MOUNT}/{self.dataset}"

    def assemble_command(self) -> str | None:
        """The setup command that puts it together; None for one dataset."""
        if not self.shards_from:
            return None
        arguments = " ".join(shlex.quote(argument) for argument in (self.path, self.dataset, *self.shards_from))
        return f'"$PYTHON" -m ullm.kaggle_assemble {arguments}'


def kaggle_cutlass_dsl_path_command() -> str:
    """Put FlashInfer's CuteDSL dependency where an import can find it.

    ``nvidia-cutlass-dsl`` ships ``cutlass`` nested inside its own package
    and relies on a ``.pth`` to put that directory on the path. On Kaggle's
    image the ``.pth`` does not take effect: the wheel installs, the package
    is present, and ``import cutlass`` still fails. Two things then fail
    together and neither names the cause -- ``flashinfer.cute_dsl.utils``
    cannot import, and ``flashinfer.fused_moe`` hides ``b12x_fused_moe``
    behind the same import, so a 12.x MoE backend reports only that the
    "kernel does not support current device".

    Exported rather than computed in the deployment, because the directory
    is the kernel's site-packages and a launcher's would be a different
    machine's path.

    ``TAAF_KAGGLE_SETUP_ENV`` holds ONE JSON object that every setup command
    contributes to and the runner re-reads after each, so this merges into
    what is there. Appending would leave two objects and a decode error at
    the next command, not at this one.
    """
    script = (
        "import json, os, pathlib, nvidia_cutlass_dsl; "
        'path = pathlib.Path(os.environ["TAAF_KAGGLE_SETUP_ENV"]); '
        "env = json.loads(path.read_text()) if path.is_file() and path.read_text().strip() else {}; "
        'dsl = os.path.join(nvidia_cutlass_dsl.__path__[0], "dsl_packages"); '
        'existing = env.get("PYTHONPATH") or os.environ.get("PYTHONPATH", ""); '
        "entries = [p for p in existing.split(os.pathsep) if p and p != dsl]; "
        'env["PYTHONPATH"] = os.pathsep.join([dsl] + entries); '
        "path.write_text(json.dumps(env))"
    )
    return f'"$PYTHON" -c {shlex.quote(script)}'


@dataclasses.dataclass(frozen=True)
class KaggleEngine:
    """The engine a Kaggle kernel runs, as the deployment needs it said.

    An agent links one of these from its run configuration and says nothing
    else about the engine; the solver turns it into the kernel sources,
    setup commands, bundle exclusions and payload files TAAF's Kaggle target
    reads. ``fork_overlay`` is whether the workspace's fork is laid over the
    stock install, which only an agent that asks the engine for something
    stock lacks needs. ``prefetch`` is the checkpoint directories the engine
    loads, in its order, which a reader starts on before the install (the
    cell names them: ``ullm.backends.deployment``). ``assemble`` is the
    checkpoints split over datasets, which the kernel puts together before
    anything reads them. ``docker_image`` is the Kaggle image the kernel is
    pinned to, the one its wheelhouse was built for; None leaves it to Kaggle.
    """

    wheels: str
    requirements: tuple[str, ...]
    fork_overlay: bool
    prefetch: tuple[str, ...] = ()
    assemble: tuple[KaggleCheckpoint, ...] = ()
    docker_image: str | None = None

    def kernel_sources(self) -> tuple[str, ...]:
        """The utility outputs the kernel attaches: the wheelhouse."""
        return (self.wheels,)

    def setup_commands(self) -> tuple[str, ...]:
        """Any split checkpoint put together, before anything reads it; the
        prefetch, so the weights stream while the engine installs; the
        install; then the overlay when the fork ships: the overlay replaces
        files the wheels lay down, so the order is the contract."""
        commands = [command for checkpoint in self.assemble if (command := checkpoint.assemble_command())]
        if self.prefetch:
            commands.append(kaggle_prefetch_command(*self.prefetch))
        commands.append(kaggle_site_install_command(*self.requirements, wheels=self.wheels))
        # After the install, since it reads the installed package's location.
        commands.append(kaggle_cutlass_dsl_path_command())
        if self.fork_overlay:
            commands.append(kaggle_vllm_overlay_command())
        return tuple(commands)

    def exclude_source_repos(self) -> tuple[str, ...]:
        """Repos to keep out of the source bundle on the engine's account."""
        return (VLLM_FORK_REPO,) if self.fork_overlay else ()

    def bundle_files(self, workspace: pathlib.Path) -> list[str]:
        """The payload files the kernel's setup commands read: the fork's
        overlay, built on the launcher against ``workspace``'s pin."""
        if not self.fork_overlay:
            return []
        import ullm.vllm_overlay

        return [str(ullm.vllm_overlay.build_for_deploy(workspace=workspace))]


#: The shipped engine: the pinned fork over the stock wheelhouse.
VLLM_FORK = KaggleEngine(
    wheels=KAGGLE_VLLM_WHEELS,
    requirements=KAGGLE_VLLM_REQUIREMENTS,
    fork_overlay=True,
    docker_image=KAGGLE_DOCKER_IMAGE,
)
#: The same install without the fork, for an agent that asks the engine for
#: nothing a stock build lacks.
VLLM_STOCK = dataclasses.replace(VLLM_FORK, fork_overlay=False)
