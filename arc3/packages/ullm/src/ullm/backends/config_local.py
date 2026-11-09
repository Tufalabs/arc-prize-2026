"""Engine-arg pieces every local vLLM deployment needs, whatever it serves.

``ModelConfig`` is the bundle a factory is built from — engine args, the
backend class, its keywords — and the rest are the parts of an engine's
arguments that are nobody's checkpoint in particular: the host KV tier, the
job directory a connector writes to, the environment a JIT build wants, and
the CUDA/venv paths a cluster node may not have set.

Which checkpoint to serve, and where, is ``ullm.backends.deployment``: this
module has no opinion about either.
"""

import copy
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ullm.backend
import ullm.backends.vllmx
import ullm.support as uls
import vllm


@dataclass
class ModelConfig(uls.BaseClass):
    """Everything you need to stand up a vLLM-backed ullm backend for one model."""

    engine_args: vllm.AsyncEngineArgs  # pass to vllm.AsyncLLMEngine.from_engine_args
    sampling_params: ullm.backend.GenericSamplingParameters  # defaults per-family
    # VLLM or VLLMHarmony: both take the engine this config builds.
    backend_cls: "type[ullm.backends.vllmx.VLLM] | type[ullm.backends.vllmx.VLLMHarmony]"
    backend_kwargs: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())

    def build(
        self, *, jupyter_safe: bool = False, **backend_overrides: Any
    ) -> tuple[vllm.AsyncLLMEngine, ullm.backend.Backend]:
        """Construct the AsyncLLMEngine and wrap it in the family's backend.

        With `jupyter_safe=True`, installs the EngineCore SIGINT guard before
        engine construction so Jupyter's interrupt button doesn't kill the
        engine subprocess. `backend_overrides` are passed through to the
        backend constructor alongside `backend_kwargs` (e.g. `diagnostics=True`)."""
        if jupyter_safe:
            ullm.backends.vllmx.patch_engine_core_sigint_for_jupyter()
        _quiet_engine_logging()
        engine = vllm.AsyncLLMEngine.from_engine_args(self.engine_args)
        backend = self.backend_cls(engine=engine, **self.backend_kwargs, **backend_overrides)
        return engine, backend


DEFAULT_CPU_KV_OFFLOAD_GIB = 64.0
DEFAULT_CPU_KV_OFFLOAD_VICTIM = True


def enable_cpu_kv_offload(
    engine_args: vllm.AsyncEngineArgs,
    *,
    cpu_gib: float | None = None,
    victim: bool | None = None,
    connector_extra: dict[str, Any] | None = None,
) -> None:
    """Back the GPU KV pool with a host tier, so a lost prefix costs a fetch
    rather than a re-prefill.

    ``cpu_gib`` and ``victim`` accept ``None``, meaning the module defaults
    above — the one place the shipped values live, so a caller forwarding a
    solver-level knob does not restate them.

    The GPU pool alone bounds how many contexts are RESIDENT — a host tier
    cannot hold a decoding sequence — so what this buys is capacity for
    prefixes between their turns, and only to the extent the tier holds
    something the pool does not.

    ``victim`` picks which connector it is, and the victim connector is the
    default: it copies a block only as it nears the front of the GPU free
    queue, holds a finishing request's blocks for their copy when the pool
    is under pressure, ranks a recurrent model's checkpoints so a context's
    latest boundaries outlive everything else and the boundaries it has
    moved past are never copied, and keeps a copy only if its source block
    still carries the hash it was read under when the copy lands. At a pool
    smaller than the live set it holds the cached fraction of a pool twice
    its size (the measurement is in ``kv_offload.md``). ``victim=False``
    restores stock vLLM, which stores every block as it is computed.

    ``connector_extra`` merges into the connector's own settings: the spill
    band and per-step cap, the finish-hold watermark, the demote switch, and
    ``engine_log``, which puts the instrumentation in the engine log as well
    as in the job directory's trace.

    ``cpu_gib`` is claimed EAGERLY at startup, on top of weights and engine.
    It lands in ``/dev/shm``, which a cgroup counts as page cache while being
    unreclaimable without swap — so a host sized off "available" memory, or a
    monitor that treats page cache as reclaimable, will both call a job
    healthy right up until the OOM killer takes the engine. The default is
    larger than a Kaggle host; size it there.
    """
    import vllm.config.kv_transfer

    if cpu_gib is None:
        cpu_gib = DEFAULT_CPU_KV_OFFLOAD_GIB
    if victim is None:
        victim = DEFAULT_CPU_KV_OFFLOAD_VICTIM
    # kv_both: one engine writes the blocks out and reads them back. The
    # connector extends this engine's cache rather than shipping KV to a
    # different instance, so it is producer and consumer at once.
    extra: dict[str, Any] = {"cpu_bytes_to_use": int(cpu_gib * 1024**3)}
    if victim:
        import ullm.backends.victim_offload as _victim

        # The connector's own host policy: the eviction classes its
        # checkpoint classification feeds, the demote it can choose to use,
        # and the reuse-depth monitor. Importing the module also checks the
        # vLLM line the connector is written against.
        extra["eviction_policy"] = _victim.VICTIM_POLICY
    extra.update(connector_extra or {})
    connector = "VictimOffloadingConnector" if victim else "OffloadingConnector"
    engine_args.kv_transfer_config = vllm.config.kv_transfer.KVTransferConfig(
        kv_connector=connector,
        kv_role="kv_both",
        # An out-of-tree connector is named by class and resolved from this
        # module path; the in-tree one needs the path unset, since a set path
        # takes priority over the registry.
        kv_connector_module_path=("ullm.backends.victim_offload" if victim else None),
        kv_connector_extra_config=extra,
    )


def attach_job_dir(engine_args: Any, job_dir: "os.PathLike[str] | str") -> None:
    """Tell the KV connector where the run's job directory is.

    The connector's scheduler side runs in the engine subprocess and sees
    nothing of the launcher but the engine config, so the directory rides
    in the connector's extra config (``job_dir``). A no-op without a
    connector.
    """
    config = engine_args.kv_transfer_config
    if config is None:
        return
    config.kv_connector_extra_config["job_dir"] = str(job_dir)


def _ensure_path_has_venv_bin() -> None:
    """Put the current interpreter's bin dir on PATH so flashinfer JIT can find
    `ninja` / `nvcc` helpers when spawning child processes (test_runner does this)."""
    venv_bin = os.path.dirname(sys.executable)
    if venv_bin not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = venv_bin + os.pathsep + os.environ.get("PATH", "")


#: vLLM loggers the engine's processes keep below WARNING. During play the
#: engine's only line was the fork's, once per closed prefix-cache context, to
#: the stdout it inherits: in a notebook a pipe that one ipykernel thread
#: drains, and the engine blocked on a write to it once that thread had stopped
#: (2026-10-29: a Kaggle run froze at 3 h 53 m with no error).
QUIET_ENGINE_LOGGERS = ("vllm.v1.core.context_threads",)


def _quiet_engine_logging() -> None:
    """Hand the engine's processes vLLM's own logging config with
    ``QUIET_ENGINE_LOGGERS`` at WARNING, through ``VLLM_LOGGING_CONFIG_PATH``,
    which vLLM reads in every process it starts. Must run BEFORE the engine
    subprocess spawns; this process configured its own logging at import. A
    config path the caller set, or vLLM's logging turned off, is theirs."""
    import json

    import vllm.envs
    import vllm.logger

    if os.environ.get("VLLM_LOGGING_CONFIG_PATH") or not vllm.envs.VLLM_CONFIGURE_LOGGING:
        return
    config = copy.deepcopy(vllm.logger.DEFAULT_LOGGING_CONFIG)
    for name in QUIET_ENGINE_LOGGERS:
        config["loggers"][name] = {"level": "WARNING"}
    base = os.environ.get("TMPDIR") or "/tmp"
    os.makedirs(base, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="ullm_engine_logging_", suffix=".json", dir=base)
    with os.fdopen(fd, "w") as file:
        json.dump(config, file)
    os.environ["VLLM_LOGGING_CONFIG_PATH"] = path


def _prepend_path(var: str, entry: str) -> None:
    """Put ``entry`` first on a PATH-like variable, once."""
    parts = [p for p in os.environ.get(var, "").split(os.pathsep) if p and p != entry]
    os.environ[var] = os.pathsep.join([entry, *parts])


def _ensure_cuda_home() -> None:
    """Make a CUDA toolkit discoverable for flashinfer / deep_gemm JIT (nvcc).

    Cluster nodes are inconsistently provisioned: some ship a complete
    ``/usr/local/cuda``, but most have only an empty ``/usr/local/cuda-13.0``
    (no nvcc/headers) and no ``/usr/local/cuda`` symlink, so JIT compilation
    fails there. Resolution order: keep any toolkit already reachable
    (``CUDA_HOME``/``CUDA_PATH``, ``nvcc`` on ``PATH``, or a usable
    ``/usr/local/cuda``); otherwise point at the pip-installed
    ``nvidia-cuda-nvcc`` wheel (``nvidia/cu13``), which matches torch's cu13
    build. Stitches the bits nvcc's linker expects (lib64, unversioned
    ``.so`` names, a libcuda stub from the driver). Must run before the vLLM
    engine subprocess spawns. Idempotent."""
    # On Kaggle, never keep the image's toolkit — not even via the image's
    # own CUDA_HOME env: it points at an old /usr/local/cuda (< 12.9) that
    # cannot normalize SM 12.x, so flashinfer's arch probe logs "Failed to
    # get device capability" and then dies with a misleading "requires
    # sm75 or higher". Always rebuild from the pip cu13 wheels below —
    # exactly what sandbox's Kaggle branch does (unconditional
    # CUDA_HOME overwrite).
    if not os.path.isdir("/kaggle/working"):
        # Validate inherited toolkit paths before trusting them: the
        # launcher's CUDA_HOME (e.g. its login-node venv toolkit) travels
        # into SLURM jobs via --export=ALL, where that filesystem may not
        # exist — trusting it blindly leaves flashinfer's ninja pointing at
        # a nonexistent nvcc (exit 127). Keep only a toolkit that actually
        # has nvcc here; drop stale values and fall through to the local
        # pip toolkit.
        for var in ("CUDA_HOME", "CUDA_PATH"):
            configured = os.environ.get(var)
            if configured and not os.path.isfile(os.path.join(configured, "bin", "nvcc")):
                del os.environ[var]
        # A toolkit kept here must also put its nvcc on PATH: vLLM decides
        # whether FlashInfer is usable at all by `shutil.which("nvcc")`, which
        # never consults CUDA_HOME. Without it a node with a complete system
        # toolkit reports FlashInfer absent, and every FlashInfer MoE backend
        # is then refused as "kernel does not support current device" -- on
        # exactly the nodes provisioned best.
        configured = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if configured:
            _prepend_path("PATH", os.path.join(configured, "bin"))
            return
        if shutil.which("nvcc"):
            return
        if os.path.isfile("/usr/local/cuda/bin/nvcc"):
            os.environ["CUDA_HOME"] = "/usr/local/cuda"
            _prepend_path("PATH", "/usr/local/cuda/bin")
            return
    try:
        import nvidia  # noqa: PLC0415 — optional pip CUDA toolkit
    except ImportError:
        return
    cu13 = next(
        (Path(b) / "cu13" for b in nvidia.__path__ if (Path(b) / "cu13" / "bin" / "nvcc").is_file()),
        None,
    )
    if cu13 is None:
        return
    cu13_lib = cu13 / "lib"
    if not (cu13 / "lib64").exists():
        try:
            (cu13 / "lib64").symlink_to("lib")
        except OSError:
            pass
    # Wheels ship only versioned libs; nvcc's linker wants unversioned names.
    for so in cu13_lib.glob("lib*.so.*"):
        unversioned = cu13_lib / re.sub(r"\.so\..*$", ".so", so.name)
        if not unversioned.exists():
            try:
                unversioned.symlink_to(so.name)
            except OSError:
                pass
    # libcuda comes from the driver, not the wheel; stub it for the linker.
    stubs = cu13_lib / "stubs"
    stubs.mkdir(exist_ok=True)
    libcuda = next(
        (
            p
            for p in (
                "/usr/lib/x86_64-linux-gnu/libcuda.so",
                "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
                "/usr/local/nvidia/lib64/libcuda.so",
                "/usr/local/nvidia/lib64/libcuda.so.1",
            )
            if os.path.exists(p)
        ),
        None,
    )
    if libcuda and not (stubs / "libcuda.so").exists():
        try:
            (stubs / "libcuda.so").symlink_to(libcuda)
        except OSError:
            pass
    os.environ["CUDA_HOME"] = str(cu13)
    os.environ["PATH"] = f"{cu13 / 'bin'}{os.pathsep}" + os.environ.get("PATH", "")
    for v in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
        os.environ[v] = os.pathsep.join([str(cu13_lib), str(stubs), os.environ.get(v, "")]).strip(os.pathsep)
    # Parallelize the cold flashinfer JIT compile (defaults to 1 thread);
    # set here (not only in set_env) so factory users get it too.
    os.environ.setdefault("MAX_JOBS", str(min(os.cpu_count() or 8, 64)))
    os.environ.setdefault("FLASHINFER_NVCC_THREADS", str(min(os.cpu_count() or 8, 8)))


def set_env() -> None:
    """Set the os.environ entries an engine build wants, whatever it serves.
    Call BEFORE constructing the AsyncLLMEngine — env-var reads happen during
    engine subprocess spawn. Idempotent."""
    _ensure_path_has_venv_bin()
    _ensure_cuda_home()
    # Parallelize flashinfer JIT kernel compiles. Each nvcc+ptxas costs ~2-4 GB
    # RAM; 8 jobs × 4 GB = 32 GB headroom. setdefault so user overrides win.
    os.environ.setdefault("MAX_JOBS", "8")
    # Inductor's compile parallelism is not ours to set: vLLM assigns it one
    # thread at import, over anything set before (``settings.md``, U-0V).
