"""vLLM-backed backend factories (require the ``vllm`` extra).

Two ways to run a local model, both configured by plain
``vllm.AsyncEngineArgs`` (no CLI-flag re-encoding;
``EngineFactory.from_config`` fills a factory from a
``config_local.ModelConfig`` family preset):

- ``EngineFactory`` — an in-process ``AsyncLLMEngine`` wrapped in the
  family's backend, either ``mode="direct"`` (backend holds the engine;
  not picklable — single-process/debug) or ``mode="proxy"`` (the whole
  backend is served over a Unix socket via ``ullm.backends.proxy``; the
  returned ``ProxyBackend`` is picklable, keeps its own accounting, and
  needs no vllm imports client-side).
- ``InprocessServerFactory`` — vLLM's OpenAI-compatible HTTP server run in
  this process via its programmatic entry points (``setup_server`` /
  ``run_server_worker``), returning a picklable ``OpenAIServer`` backend.
  (The engine core is its own subprocess since vLLM V1, so "in-process"
  is only the HTTP layer.) For hand-driven CLI serving (``vllm serve``
  flags), use ``ullm.utils.VLLMServer`` directly — there is deliberately
  no factory for it, since a CLI subprocess can't take ``AsyncEngineArgs``.

Absorbed from sandbox's ``taaf_wrapper``: default socket naming,
the ``HF_HUB_OFFLINE`` flip after the engine warmed the tokenizer cache
(``hf_hub_offline_after_start``, for offline/Kaggle runs where workers
re-import on first generate), and the teardown ordering (stop serving →
shut down engine → close backend client handles; each step reported, not
silently swallowed, so a teardown bug surfaces without masking the
exception that ran the ``finally``).
"""

import asyncio
import contextlib
import copy
import dataclasses
import os
import pathlib
import shutil
import socket as socket_module
import tempfile
import traceback
import typing
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Optional

import ullm.backend
import ullm.backends.config_local
import ullm.engine_monitor
import ullm.factory
import vllm

if TYPE_CHECKING:
    import ullm.backends.vllmx


def default_socket_path(model: str) -> str:
    """/tmp socket path unique per model + job (SLURM job id, else pid).
    HF-style model ids contain ``/`` which would point into a non-existent
    subdir of /tmp — sanitized to ``_``."""
    safe = model.replace("/", "_")
    return f"/tmp/vllmx_{safe}_{os.environ.get('SLURM_JOB_ID') or os.getpid()}.sock"


def force_hf_hub_offline() -> None:
    """Flip HuggingFace Hub to offline for this process and its children.
    Call after the engine has warmed the tokenizer/processor cache; worker
    processes re-import on first generate and must not touch the network."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    import huggingface_hub.constants

    huggingface_hub.constants.HF_HUB_OFFLINE = True


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for a free TCP port (bind-then-close; small race window is
    acceptable for test/dev use — pass an explicit port in production)."""
    with socket_module.socket() as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _resolve_local_model_path(model: str) -> str:
    """A local-path model that doesn't exist at engine start may be a
    deploy-time guess of a dataset mount (Kaggle mount layouts vary by
    account/era). Resolve through the deploy layer's mount map
    (``TAAF_KAGGLE_INPUT_PATHS``, ref -> mounted path) by basename; keep
    the original on any miss so the engine raises its own clear error."""
    if not model.startswith("/") or os.path.exists(model):
        return model
    mapping_json = os.environ.get("TAAF_KAGGLE_INPUT_PATHS")
    if not mapping_json:
        return model
    import json

    mounts = json.loads(mapping_json)
    # A guess of /kaggle/input/datasets/<owner>/<slug> names its ref, which
    # the map keys exactly, whatever layout the mount landed in (a slug's
    # hyphens can come back as underscores).
    for ref, path in mounts.items():
        if model.rstrip("/").endswith("/" + ref) and os.path.exists(str(path)):
            print(f"[EngineFactory] model path {model} not mounted; using {path}", flush=True)
            return str(path)
    base = os.path.basename(model.rstrip("/"))
    for path in mounts.values():
        if os.path.basename(str(path).rstrip("/")) == base and os.path.exists(str(path)):
            print(f"[EngineFactory] model path {model} not mounted; using {path}", flush=True)
            return str(path)
    return model


async def _verify_hybrid_serving_bug(engine: "vllm.AsyncLLMEngine") -> None:
    """Assert the engine still serves warm hybrid cache hits inconsistently.

    The duck's scores depend on a characterized vLLM 0.25.1 behavior: with a
    KV connector present, the scheduler serves a warm prefix-cache hit at
    the full-attention page boundary even where no Mamba state snapshot
    exists, pairing valid attention KV with an unmaterialized recurrent
    state. The perturbation curbs rumination and is worth several points
    per game, so its silent disappearance must fail loudly.

    Design constraints (the writeup is arc3-experiments
    engine_probe/FINDINGS.md):

    - the probed pages must be laid down by a longer prior request;
    - the harm exists only where a prefix crosses a gap between Mamba
      state snapshots, and snapshot placement is entrypoint- and
      chunking-specific — so the check probes MANY lengths and concludes
      absence only if every usable length reads clean (single-length
      probes read "clean" inside masked stripes; measured: 13/20 lengths
      harmful under the async entrypoint, in bands);
    - no cold draw reference exists here (seeded draws are prompt-bound;
      cache resets are unreliable through this wrapper), so the clean
      reference is a NONCE-prefixed prompt_logprobs request — every page
      hash shifts, forcing a full recompute, proven structurally by the
      returned list's completeness;
    - positions that are too peaked mask the bug and too-flat ones drown
      the verdict, so each probed length is gated on its clean top-20 mass
      and skipped (not failed) outside the band.

    Verdict: ACTIVE if any usable length's warm draws are improbable under
    its clean distribution; refusal only when several usable lengths all
    read consistent.
    """
    import inspect
    import math
    import pathlib

    tokenizer = engine.get_tokenizer()
    if inspect.isawaitable(tokenizer):
        tokenizer = await tokenizer
    # The probe corpus is REAL duck-agent text (three distinct game
    # conversations from validated replay runs, committed as package data).
    # Synthetic corpora fail here twice over: generated text is predictable
    # by construction, and any repeated structure gets in-context-learned
    # into determinism exactly at the probe positions.
    corpus = (pathlib.Path(__file__).parent.parent / "data" / "serving_probe_corpus.txt").read_text()
    full_ids = tokenizer.encode(corpus)[:34_000]
    # Candidate cuts: dense inside the measured harmful bands, sparse
    # controls above, each snapped forward to a word boundary so the probed
    # position is a word choice rather than a word completion. Density is
    # what makes the verdict reliable — stripe layout is chunking-specific
    # and 0.25.1 selects kernels per boot, which moves positions across the
    # gates below, so the sweep must offer enough candidates that some
    # usable striped length survives; stop-at-first-displaced keeps the
    # bug-present path cheap regardless.
    targets = list(range(12_200, 17_900, 150)) + [18_800, 24_500, 30_000]
    probe_lengths = []
    for target in targets:
        cut = target
        while cut < len(full_ids) - 1 and not tokenizer.decode([full_ids[cut]]).startswith((" ", "\n")):
            cut += 1
        probe_lengths.append(cut)
    if len(full_ids) < probe_lengths[-1] + 1600:
        raise RuntimeError(f"serving-bug probe corpus too short ({len(full_ids)} tokens, last cut {probe_lengths[-1]})")
    n_draws = 16

    async def _last(ids: list[int], params: "vllm.SamplingParams", tag: str) -> "vllm.RequestOutput":
        last = None
        async for out in engine.generate({"prompt_token_ids": ids}, params, request_id=f"serving-bug-probe-{tag}"):
            last = out
        assert last is not None
        return last

    print(
        f"[EngineFactory] serving-bug check: full prefill ({len(full_ids)} tokens), then per probed "
        f"length {n_draws} seeded warm draws vs a nonce-recomputed clean distribution; "
        f"lengths {probe_lengths}",
        flush=True,
    )
    await _last(full_ids, vllm.SamplingParams(temperature=0.0, max_tokens=1), "prefill")

    usable = 0
    displaced = []
    for length in probe_lengths:
        probe_ids = full_ids[:length]
        warm_outs = await asyncio.gather(
            *(
                _last(
                    probe_ids,
                    vllm.SamplingParams(temperature=1.0, top_k=194, max_tokens=1, seed=seed),
                    f"warm-{length}-{seed}",
                )
                for seed in range(n_draws)
            )
        )
        warm = [int(o.outputs[0].token_ids[0]) for o in warm_outs]
        # Unique nonce per length: a shared nonce makes each length's clean
        # reference a cacheable prefix of the next one's, and the second
        # nonce request then prefix-hits the first's pages.
        nonce_ids = tokenizer.encode(f"X{length}Y") + probe_ids + [warm[0]]
        lp_out = await _last(
            nonce_ids,
            vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=20),
            f"cleanref-{length}",
        )
        assert lp_out.prompt_logprobs is not None
        if len(lp_out.prompt_logprobs) != len(nonce_ids):
            raise RuntimeError(
                "serving-bug check could not obtain a clean reference: the nonce-prefixed "
                f"prompt_logprobs request scored {len(lp_out.prompt_logprobs)} of {len(nonce_ids)} "
                "positions — the engine served part of it from cache, so the reference is not clean."
            )
        entry = lp_out.prompt_logprobs[-1]
        assert entry, "no logprobs at the probe position"
        clean_top = {int(t): math.exp(lo.logprob) for t, lo in entry.items()}
        # vLLM always includes the position's ACTUAL token in the dict, even
        # when it ranks below the top-k. That token is warm[0] here, so
        # leaving it in would guarantee one hit at every length (and inflate
        # the expectation) — a structural nudge toward "consistent". Drop it
        # when its presence is what pushed the dict past k.
        if len(clean_top) > 20:
            clean_top.pop(warm[0], None)
        clean_mass = sum(clean_top.values())
        top1 = max(clean_top.values())
        if clean_mass < 0.4 or top1 > 0.8:
            # Peaked positions are unusable twice over: near-delta
            # distributions are exactly where the perturbation is invisible,
            # and in this corpus they are grid content, where the nonce's
            # one-token shift genuinely changes the prediction (misaligned
            # rows) and invalidates the reference.
            print(
                f"[EngineFactory] serving-bug check: len {length}: position unusable "
                f"(top-20 mass {clean_mass:.2f}, top-1 {top1:.2f}) — skipped",
                flush=True,
            )
            continue
        usable += 1
        hits = sum(1 for t in warm if t in clean_top)
        expected = clean_mass * n_draws
        verdict = hits < 0.5 * expected
        if verdict:
            displaced.append(length)
        print(
            f"[EngineFactory] serving-bug check: len {length}: {hits}/{n_draws} warm draws in the "
            f"clean top-20 (consistent expects ~{expected:.0f}) -> " + ("DISPLACED" if verdict else "consistent"),
            flush=True,
        )
        if displaced:
            break  # one displaced length proves the behavior; skip the rest

    if displaced:
        print(
            f"[EngineFactory] serving-bug check: serving perturbation ACTIVE (displaced at "
            f"length {displaced[0]}), proceeding",
            flush=True,
        )
        return
    if usable < 3:
        raise RuntimeError(
            f"serving-bug check inconclusive: only {usable} probed lengths had usable positions; "
            "the probe text needs sharper positions before the verdict means anything."
        )
    raise RuntimeError(
        "The vLLM 0.25.1 hybrid warm-hit serving behavior this deployment depends on is NOT "
        f"active: all {usable} usable probed lengths served warm hits consistently with the "
        "recomputed clean distribution. The duck's scores rely on the connector-present scheduler "
        "serving warm cache hits beyond the materialized Mamba state (several points per game). "
        "Likely causes: a vLLM upgrade (fixed in 0.27.0), the CPU KV offload connector not "
        "configured, or an upstream scheduler change. See duck/design.md 'The engine serving "
        "dependence' and arc3-experiments engine_probe/FINDINGS.md before proceeding."
    )


def _set_jit_parallelism() -> None:
    """Cold-start JIT knobs (setdefault — user overrides win): flashinfer's
    version check off (matches sandbox's worker setup). Inductor's
    compile parallelism is not ours to set: vLLM pins it to one thread at
    import (``settings.md``, U-0V)."""
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def _isolate_flashinfer_workspace() -> str:
    """Point ``FLASHINFER_WORKSPACE_BASE`` at a fresh temp dir so the JIT
    cache lives and dies with this engine. The shared ``~/.cache/flashinfer``
    bakes the building venv's absolute source paths into ``build.ninja``;
    per-job venvs are ephemeral, so the next job that needs the same kernel
    hits ninja's "missing and no known rule to make it". Must run BEFORE the
    engine subprocess spawns. Returns the dir (caller removes it on close).

    Cost: every engine start pays a full flashinfer JIT compile (~150 s), which is what makes a GPU test column ~5 min rather than ~1.
    A per-node cache keyed by a stable venv path would recover that."""
    base = os.environ.get("TMPDIR") or "/tmp"
    os.makedirs(base, exist_ok=True)
    workspace = tempfile.mkdtemp(prefix="flashinfer_run_", dir=base)
    os.environ["FLASHINFER_WORKSPACE_BASE"] = workspace
    return workspace


# Engine arguments a deployment may leave for its host to settle at start,
# because their right value follows from configuration the engine cannot see:
# the window from the agent's context window, the sequence cap from the
# harness concurrency. Recorded as the factory was BUILT so a host settling
# them can tell "nobody chose this" from "the caller chose this, leave it".
# Nothing here settles them; that is the host's business (the duck does it in
# DuckAgent.prepare_for_backend).
# ``mm_processor_kwargs`` is on the list for a host whose task bounds its own
# images: the processor's largest image is what the start-up profile reserves
# activation for, and a host that renders every image itself can state a
# ceiling from its own rules. Recorded whole, so a caller's own processor
# arguments are told from a derived one the same way.
DERIVABLE_ENGINE_ARGS = ("max_model_len", "max_num_seqs", "mm_processor_kwargs")


def derivation_baseline(engine_args: Any) -> dict[str, Any]:
    """The derivable arguments as built, which a host updates as it settles.

    Copied, not referenced: one of them is a dict, and a baseline holding the
    engine's own object would follow a caller's in-place edit of it, so the
    edit would read as the value the baseline was built with and be settled
    over instead of refused.
    """
    return {name: copy.deepcopy(getattr(engine_args, name, None)) for name in DERIVABLE_ENGINE_ARGS}


@dataclass
class EngineProfile:
    """Torch-profiler windows over a running engine, opened by its state.

    Each window records ``iterations`` engine steps -- GPU kernels and the host
    work around them -- and writes one Chrome trace, plus vLLM's per-op table of
    the window, into ``directory`` (``packages/ullm/tools/engine_profile_read.py``
    reads one), and packs that directory into one ``<directory>.tar`` beside it
    each time a window's trace is complete and again at shutdown: one file to
    carry off a box where that is done by hand. A window opens when the KV pool is at least
    ``kv_cache_above`` full with requests running: the phase worth seeing is the
    saturated one, where the pool holds all it can and the host tier carries
    the rest, and when a run reaches it depends on the card, the concurrency
    and the boot, so it is watched for rather than timed. Later windows wait
    ``spacing_s`` after the one before, so they sample the phase apart.

    The profiler rides the engine's own mechanism (``profiler_config``), which
    only records while a window is open; outside one it costs a check per
    step. Inside, it slows the steps it records, and each window stalls the
    engine briefly while its trace is written. Stacks and shapes make both far
    worse and the traces far larger, so they stay off unless a question needs
    them.
    """

    # A fraction of the pool, as the engine status line reports it.
    kv_cache_above: float = 0.85
    # No window opens before this long after the engine is ready, whatever the
    # pool says. A small pool can cross its threshold on the first wave of
    # prompts, while kernels are still compiling on first use -- a trace of
    # warm-up, not of play.
    not_before_s: float = 300.0
    windows: int = 2
    spacing_s: float = 600.0
    # Steps per window: enough decode steps to average over, few enough that a
    # trace opens in a browser.
    iterations: int = 100
    # How often the engine's metrics are read while waiting for the phase.
    poll_s: float = 5.0
    with_stack: bool = False
    record_shapes: bool = False
    # None -> ``engine_profile`` under the run's job directory.
    directory: Optional[str] = None


@dataclass
class LocalEngineFields(ullm.factory.BackendFactory):
    """What the two factories that build an engine of their own both settle,
    for an engine neither of them has constructed yet.

    Applied at START rather than at construction, which is what lets them read
    engine arguments a caller is still free to change -- and, for the tier,
    puts the connector's import in the process that will use it, since that
    import checks the vLLM line the connector is written against.

    The tier's whole configuration is four fields: whether it is on, how
    large, which connector, and any connector settings. The connector object
    they resolve into is an implementation detail of ``_start``.
    """

    cpu_kv_offload: bool = False
    cpu_kv_offload_gib: Optional[float] = None  # None -> config_local's shipped size
    cpu_kv_offload_victim: Optional[bool] = None  # None -> config_local's shipped connector
    cpu_kv_offload_extra: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    dense_cuda_capture: bool = True
    profile: Optional[EngineProfile] = None
    _profile_task: Optional["asyncio.Task[None]"] = field(default=None, init=False, repr=False)
    _profile_directory: Optional[str] = field(default=None, init=False, repr=False)
    # The watcher opens no window while held (hold_profile_windows).
    _profile_held: bool = field(default=False, init=False, repr=False)

    def use_minimal_diagnostics(self) -> None:
        """The host tier's park log off: its connector writes one by default."""
        self.cpu_kv_offload_extra = {**self.cpu_kv_offload_extra, "park_log": False}

    def _apply_cpu_kv_offload(self, engine_args: Any) -> None:
        """Write the tier into the engine args, refusing to displace another
        connector: two of them cannot both hold the KV, and quietly replacing
        one is how a run ends up with a tier nobody asked for."""
        if not self.cpu_kv_offload:
            return
        if engine_args.kv_transfer_config is not None:
            raise ValueError(
                "cpu_kv_offload is on but engine_args.kv_transfer_config is already set. "
                "Both describe who holds the KV. Use the factory's cpu_kv_offload_* fields, "
                "or set cpu_kv_offload=False and configure the connector yourself."
            )
        ullm.backends.config_local.enable_cpu_kv_offload(
            engine_args,
            cpu_gib=self.cpu_kv_offload_gib,
            victim=self.cpu_kv_offload_victim,
            connector_extra=self.cpu_kv_offload_extra or None,
        )

    def _apply_dense_cuda_capture(self, engine_args: Any) -> None:
        """Give every decode width its own CUDA graph, and leave the rest alone.

        A captured graph serves one exact batch width, so a decode batch that
        falls between rungs of the engine's ladder (1, 2, 4, then every
        eighth) is padded up and the model computes filler it discards --
        averaging 3.2 sequences, paid in the per-sequence term of every step.
        Decode is never dispatched above ``max_num_seqs`` requests, so that
        range is the only one worth making dense; the ladder stays beside it,
        because mixed batches land on it and prefill overflows past it into no
        graph at all.

        With a drafter, a decode step carries ``1 + num_speculative_tokens``
        tokens per request and is dispatched to a graph only at an exact
        multiple of that width, so the dense rungs are those multiples, up to
        ``max_num_seqs`` requests' worth -- a rung between them would never be
        used, and the widest drafted batch would otherwise sit above every
        graph.

        Read at start, and from the engine arguments rather than from anything
        that derives them, so it sees the width the engine will actually admit
        however late it was decided. An unset width is left alone rather than
        guessed: vLLM's own default for it depends on the serving topology,
        so a guess would be wrong under one of them.
        """
        if not self.dense_cuda_capture:
            return
        width = getattr(engine_args, "max_num_seqs", None)
        if not width:
            return
        import vllm.config

        compilation = engine_args.compilation_config
        if not isinstance(compilation, vllm.config.CompilationConfig):
            compilation = vllm.config.CompilationConfig(**(compilation or {}))
            engine_args.compilation_config = compilation
        if compilation.cudagraph_capture_sizes:
            return  # a caller who named the rungs chose them
        query_len = 1 + _num_speculative_tokens(engine_args)
        decode_widths = [query_len * n for n in range(1, width + 1)]
        # vLLM requires a named ceiling to equal the widest rung, so a caller's
        # ceiling bounds the rungs; otherwise the ladder reaches at least the
        # widest decode batch.
        named_ceiling = compilation.max_cudagraph_capture_size
        ceiling = named_ceiling or max(512, decode_widths[-1])
        ladder = [1, 2, 4, *range(8, 256, 8), *range(256, ceiling + 1, 16)]
        sizes = sorted({size for size in (*decode_widths, *ladder) if size <= ceiling})
        if named_ceiling and sizes[-1] != named_ceiling:
            sizes.append(named_ceiling)
        compilation.cudagraph_capture_sizes = sizes

    def _apply_profile(self, engine_args: Any) -> None:
        """Arm the engine's torch profiler, which then waits for a window.

        Refuses a profiler the caller configured themselves: the windows are
        opened against this configuration, and two owners of one profiler
        would each close the other's."""
        if self.profile is None:
            return
        existing = getattr(engine_args, "profiler_config", None)
        if existing is not None and getattr(existing, "profiler", None) is not None:
            raise ValueError(
                "profile is set but engine_args.profiler_config already names a profiler. Use one or the other."
            )
        p = self.profile
        if p.windows <= 0 or p.iterations <= 0 or p.poll_s <= 0 or p.spacing_s < 0 or p.not_before_s < 0:
            raise ValueError(
                f"profile needs positive windows, iterations and poll_s, and non-negative spacing_s and not_before_s, got {p}"
            )
        if not 0.0 <= p.kv_cache_above <= 1.0:
            raise ValueError(f"profile.kv_cache_above is a fraction of the pool, got {p.kv_cache_above}")
        directory = self.profile.directory
        if directory is None:
            if self.job_dir is None:
                raise ValueError("profile writes under the job directory; set job_dir or profile.directory")
            directory = str(self.job_dir / "engine_profile")
        os.makedirs(directory, exist_ok=True)
        self._profile_directory = directory
        from vllm.config import ProfilerConfig

        engine_args.profiler_config = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=directory,
            # The engine closes each window itself after this many steps, so a
            # window that opens during a quiet spell still ends on steps.
            max_iterations=self.profile.iterations,
            # The front end's own trace does not count steps and would record
            # from the first window to shutdown.
            ignore_frontend=True,
            torch_profiler_with_stack=self.profile.with_stack,
            torch_profiler_record_shapes=self.profile.record_shapes,
        )

    def hold_profile_windows(self) -> None:
        """Keep the profiler windows shut until ``delay_profile_windows``
        lifts the hold: for a caller's untimed phase (the traffic replay's
        decode gate), which can outlast ``not_before_s`` and must not have a
        window open inside it. The watcher reads the hold at every poll."""
        self._profile_held = True

    def delay_profile_windows(self, seconds: float) -> None:
        """Lift a hold, and open the profiler windows ``seconds`` later than
        ``not_before_s`` said: for a caller whose timed phase starts after an
        untimed one (the traffic replay's decode gate), since the windows
        count from the engine's readiness. The watcher reads ``not_before_s``
        at every poll."""
        self._profile_held = False
        if self.profile is not None:
            self.profile.not_before_s += seconds

    def _open_profile_windows(
        self,
        open_window: Callable[[], Awaitable[Any]],
        read_metrics: Callable[[], Awaitable[Any]],
    ) -> None:
        """Watch the engine's metrics and open each window when the phase arrives."""
        if self.profile is None:
            return
        import ullm.engine_monitor as monitor

        profile = self.profile

        async def run() -> None:
            loop = asyncio.get_running_loop()
            ready = loop.time()
            last_opened: Optional[float] = None
            opened = 0
            while opened < profile.windows:
                await asyncio.sleep(profile.poll_s)
                if self._profile_held or loop.time() - ready < profile.not_before_s:
                    continue
                if last_opened is not None and loop.time() - last_opened < profile.spacing_s:
                    continue
                try:
                    metrics = await read_metrics()
                except monitor.MetricsUnavailable as exc:
                    print(f"[profile] no engine metrics to watch, so no windows: {exc}", flush=True)
                    return
                except Exception:  # a metrics hiccup is not a reason to stop watching
                    continue
                usage = metrics.scalars.get(monitor.KV_CACHE_USAGE, 0.0)
                running = metrics.scalars.get(monitor.REQUESTS_RUNNING, 0.0)
                if usage < profile.kv_cache_above or running <= 0:
                    continue
                last_opened = loop.time()
                try:
                    await open_window()
                except Exception as exc:  # a profile must never take the run down
                    print(f"[profile] window {opened + 1} failed to open: {exc!r}", flush=True)
                    continue
                opened += 1
                print(
                    f"[profile] window {opened}/{profile.windows} opened: "
                    f"kv_cache={100.0 * usage:.1f}% running={running:.0f}",
                    flush=True,
                )
                await self._bundle_when_written(loop)

        self._profile_task = asyncio.create_task(run())

    def _profile_traces(self) -> dict[str, int]:
        """The trace files in the profile directory, by name, with their sizes."""
        directory = self._profile_directory
        if directory is None or not os.path.isdir(directory):
            return {}
        return {
            name: os.path.getsize(os.path.join(directory, name))
            for name in os.listdir(directory)
            if name.endswith((".json", ".json.gz"))
        }

    async def _bundle_when_written(self, loop: asyncio.AbstractEventLoop) -> None:
        """Pack the traces once the window just opened has written its own.

        The engine writes a trace some time after the window's last step, from
        its own process, so "written" is a trace this watcher has not seen
        whose size holds still across two polls. Bounded, so a window that
        never records does not stop the next one from being watched for."""
        assert self.profile is not None
        if self._profile_directory is None:
            return  # armed by nothing: no directory to watch
        before = self._profile_traces()
        deadline = loop.time() + 600.0
        last: dict[str, int] = {}
        while loop.time() < deadline:
            await asyncio.sleep(max(self.profile.poll_s, 1.0))
            new = {name: size for name, size in self._profile_traces().items() if name not in before and size > 0}
            if new and new == last:
                try:
                    await asyncio.to_thread(self._bundle_profile)
                except Exception as exc:  # packing is a convenience; the next window is not
                    print(f"[profile] could not pack the traces: {exc!r}", flush=True)
                return
            last = new

    def _bundle_profile(self) -> None:
        """Write ``<directory>.tar`` holding the directory, replacing any older
        one whole, so a reader never sees a half-written archive."""
        directory = self._profile_directory
        if directory is None or not self._profile_traces():
            return
        import tarfile

        root = directory.rstrip("/")
        partial = root + ".tar.partial"
        with tarfile.open(partial, "w") as archive:
            archive.add(root, arcname=os.path.basename(root))
        os.replace(partial, root + ".tar")
        print(f"[profile] traces packed into {root}.tar", flush=True)

    async def _close_profile_windows(self) -> None:
        """Stop the window watcher and pack whatever traces have landed."""
        if self._profile_task is not None:
            self._profile_task.cancel()
            with contextlib.suppress(BaseException):
                await self._profile_task
            self._profile_task = None
        # Once more at shutdown, for a window whose trace landed after the
        # watcher stopped waiting for it.
        try:
            await asyncio.to_thread(self._bundle_profile)
        except Exception as exc:  # packing is a convenience, never a failure
            print(f"[profile] could not pack the traces: {exc!r}", flush=True)

    def _settle_at_start(self, engine_args: Any) -> None:
        """Everything these fields write into the engine args, on every start
        path. The tier comes before the run's job directory is attached to it:
        ``attach_job_dir`` writes into the connector's config, so a connector
        that does not exist yet takes nothing and the run's trace is silently
        off. One method rather than a call per field per path, so the order
        cannot drift between them."""
        self._apply_cpu_kv_offload(engine_args)
        self._apply_dense_cuda_capture(engine_args)
        self._apply_profile(engine_args)
        if self.job_dir is not None:
            ullm.backends.config_local.attach_job_dir(engine_args, self.job_dir)


@dataclass
class EngineFactory(LocalEngineFields):
    """Builds an ``AsyncLLMEngine`` from ``engine_args`` and wraps it in
    ``backend_cls`` (``VLLM`` or ``VLLMHarmony``). Use ``from_config`` to
    fill engine args + backend class/kwargs from a family preset."""

    engine_args: vllm.AsyncEngineArgs = field(kw_only=True)
    # VLLM or VLLMHarmony: both take the engine the factory builds, which is
    # what distinguishes them from a Backend in general.
    backend_cls: "type[ullm.backends.vllmx.VLLM] | type[ullm.backends.vllmx.VLLMHarmony]" = field(kw_only=True)
    backend_kwargs: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    mode: Literal["proxy", "direct"] = "proxy"
    socket_path: Optional[str] = None  # proxy mode; None -> default_socket_path(model)
    jupyter_safe: bool = False  # install the EngineCore SIGINT guard (Jupyter interrupt survival)
    # Opt-in (offline/Kaggle presets set it): flips HF_HUB_OFFLINE for the
    # WHOLE process, so with it on, no later factory in the same process can
    # fetch anything from the hub.
    hf_hub_offline_after_start: bool = False
    # Env vars applied on the worker just before the engine subprocess
    # spawns (launcher-side os.environ never travels to the worker). Used
    # for engine toggles vLLM reads only at init, e.g. VLLM_USE_V2_MODEL_RUNNER.
    env_overrides: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    # Opt-in startup check that the vLLM 0.25.1 hybrid warm-hit serving
    # behavior the duck's scores depend on is actually active — see
    # ``_verify_hybrid_serving_bug``. OFF by default and asserted by no
    # test in either direction: the verdict is a statistical read of
    # boot-varying engine behavior (0.25.1 selects kernels per boot, which
    # moves probe positions across the verdict's gates; the harmful stripes
    # move with chunking), so it carries a false-verdict rate that neither
    # a scored run nor CI can absorb. Enable it manually when validating a
    # serving stack, and read its verdict as a measurement.
    verify_serving_bug: bool = False
    # The expert routing histogram (``ullm.expert_log``): counts and gate-weight
    # mass per (layer, expert), written to this directory by the engine core.
    # The input of the pruning tool; nothing about routing changes. With
    # ``expert_norms`` the gate weight times each expert's output norm is
    # recorded beside it, at the price of a second MoE kernel call per layer.
    expert_log_dir: Optional[str] = None
    expert_norms: bool = False
    # Host and device memory sampled from inside the engine core, every N
    # seconds and around each startup phase (``VLLM_MEMORY_LOG_S``, which the
    # pinned fork reads). A cgroup OOM kill leaves no traceback, so the only
    # account of what filled the cgroup is one printed before it did.
    memory_log_s: Optional[float] = None
    # Everything the engine's processes print, also in ``engine_output.log``
    # under the job directory (ullm.engine_output): where the console is the
    # only other record of a worker's lines -- a Kaggle kernel run by hand.
    engine_output_log: bool = False
    _engine: Optional[vllm.AsyncLLMEngine] = field(default=None, init=False, repr=False)
    _engine_host: Optional[Any] = field(default=None, init=False, repr=False)
    _proxy: Optional[Any] = field(default=None, init=False, repr=False)  # this process's ProxyBackend client
    _wrapped_backend: Optional[ullm.backend.Backend] = field(
        default=None, init=False, repr=False
    )  # host-side real backend (proxy mode)
    _backend: Optional[ullm.backend.Backend] = field(default=None, init=False, repr=False)
    _flashinfer_workspace: Optional[str] = field(default=None, init=False, repr=False)
    _derivation_baseline: dict[str, Any] = field(default_factory=lambda: dict[str, Any](), init=False, repr=False)

    def __post_init__(self) -> None:
        self._derivation_baseline = derivation_baseline(self.engine_args)
        super().__post_init__()

    @classmethod
    def from_config(cls, config: ullm.backends.config_local.ModelConfig, **field_overrides: Any) -> "EngineFactory":
        """Build a factory from a ``config_local`` family preset: engine
        args, backend class, and backend kwargs all come from the config
        (its ``sampling_params`` remain the caller's to apply at call
        time). ``field_overrides`` set any other factory field."""
        return cls(
            engine_args=config.engine_args,
            backend_cls=config.backend_cls,
            backend_kwargs=dict(config.backend_kwargs),
            **field_overrides,
        )

    async def _start(self) -> ullm.backend.Backend:
        """Build the in-process engine and return it behind a proxy client."""
        import ullm.backends.vllmx as _vllmx

        if self.jupyter_safe:
            _vllmx.patch_engine_core_sigint_for_jupyter()
        self._apply_expert_log()
        self._apply_env_overrides()
        # Cluster nodes are inconsistently CUDA-provisioned; make the
        # pip-installed toolkit discoverable for flashinfer/deep_gemm JIT
        # before the engine subprocess spawns (no-op when a toolkit is
        # already reachable).
        ullm.backends.config_local._ensure_path_has_venv_bin()
        ullm.backends.config_local._ensure_cuda_home()
        _set_jit_parallelism()
        ullm.backends.config_local._quiet_engine_logging()
        self._flashinfer_workspace = _isolate_flashinfer_workspace()
        self._settle_at_start(self.engine_args)
        self.engine_args.model = _resolve_local_model_path(self.engine_args.model)
        # A drafter's weights are a dataset mount too (its own on Kaggle).
        speculative = self.engine_args.speculative_config
        if isinstance(speculative, dict) and isinstance(speculative.get("model"), str):
            speculative["model"] = _resolve_local_model_path(speculative["model"])
        if self.memory_log_s is not None:
            # Also in THIS process: the engine core's sampler starts when the
            # engine core does, and everything before that -- this process's
            # own footprint, and whatever the box was already holding -- is
            # unlit. They share a cgroup, so either one sees the total; only
            # this one sees it early.
            from vllm.utils import memory_log

            memory_log.maybe_start()
            memory_log.report("factory: about to start the engine")
        with self._engine_output():
            engine = self._engine = vllm.AsyncLLMEngine.from_engine_args(self.engine_args)
        # So a dead engine says how it died: vLLM logs only that it did.
        ullm.engine_monitor.watch_engine_exits(engine)
        if self.memory_log_s is not None:
            # The engine core stops its own when start-up ends; this one is
            # in a process that goes on to play for hours, so it stops here.
            from vllm.utils import memory_log

            memory_log.stop("factory: engine started")
        if self.mode == "direct":
            self._backend = self.backend_cls(engine=self._engine, **self.backend_kwargs)
        else:
            import ullm.backends.proxy as _proxy

            # The real backend runs HERE (host side), serving its whole
            # pipeline over the socket; workers and this process use
            # ProxyBackend clients that need no vllm imports.
            socket_path = self.socket_path or default_socket_path(self.engine_args.model)
            self._wrapped_backend = self.backend_cls(engine=self._engine, **self.backend_kwargs)
            self._engine_host = await _proxy.start_backend_server(self._wrapped_backend, socket_path)
            self._proxy = _proxy.ProxyBackend(socket_path=socket_path)
            self._backend = self._proxy
        if self.hf_hub_offline_after_start:
            force_hf_hub_offline()
        if self.verify_serving_bug:
            await _verify_hybrid_serving_bug(self._engine)
        self._open_profile_windows(engine.start_profile, ullm.engine_monitor.inprocess_source)
        assert self._backend is not None  # every branch above assigns it
        # The engine's own profile, reachable from its backend: the factory a
        # solver starts is a copy of the agent's, so the agent's spec is not
        # the one the watcher reads.
        self._backend.hold_profile_windows = self.hold_profile_windows
        self._backend.delay_profile_windows = self.delay_profile_windows
        return self._backend

    def _engine_output(self) -> contextlib.AbstractContextManager[None]:
        """While the engine starts, the descriptors its processes inherit, if
        ``engine_output_log`` asks for their output in the job directory."""
        if not self.engine_output_log:
            return contextlib.nullcontext()
        if self.job_dir is None:
            raise ValueError("engine_output_log writes under the job directory; set job_dir")
        import ullm.engine_output

        return ullm.engine_output.capture(pathlib.Path(self.job_dir) / "engine_output.log")

    def _apply_expert_log(self) -> None:
        """Ship the histogram request to the engine core through the environment.

        The plugin that reads it binds vLLM's capturer itself, so
        ``enable_return_routed_experts`` stays off: that flag builds a
        per-request routing buffer vLLM refuses to run beside a KV connector.
        """
        import ullm.expert_log

        if self.expert_norms and not self.expert_log_dir:
            raise ValueError("expert_norms needs the histogram: set expert_log_dir, which is where norms are written")
        if self.expert_log_dir and self.env_overrides.get("XKB_XROUTER", os.environ.get("XKB_XROUTER")) == "1":
            raise ValueError(
                "expert_log_dir derives gate mass from each MoE layer's router logits, and XKB_XROUTER hands "
                "the experts packed routing (weights and ids) in their place: set XKB_XROUTER=0 for a "
                "calibration run"
            )
        if self.expert_log_dir:
            self.env_overrides[ullm.expert_log.ENV_LOG_DIR] = self.expert_log_dir
        if self.expert_norms:
            self.env_overrides[ullm.expert_log.ENV_NORMS] = "1"

        if self.memory_log_s is not None:
            if self.memory_log_s <= 0:
                raise ValueError(f"memory_log_s is a sampling interval in seconds, got {self.memory_log_s}")
            # The engine's own variable: the sampler lives in the fork, which
            # is the only thing that reliably runs inside the engine core.
            # A plugin cannot -- vLLM discovers those through installed entry
            # points, and the source checkout a Kaggle kernel runs has no
            # dist-info to be discovered by.
            self.env_overrides["VLLM_MEMORY_LOG_S"] = str(self.memory_log_s)

    def _apply_env_overrides(self) -> None:
        """Put the engine's environment in place on the worker, before the
        engine subprocess spawns (the launcher's environment never travels).

        The histogram's two variables are the factory's own: the plugin reads
        them in the spawned engine core at each start, so they are set when
        the fields ask and taken out of the process when they do not, and a
        calibration factory's setting never reaches the next engine this
        process starts. The other overrides are toggles vLLM reads at import,
        which no later factory can change.
        """
        import ullm.expert_log

        os.environ.update(self.env_overrides)
        for key in (ullm.expert_log.ENV_LOG_DIR, ullm.expert_log.ENV_NORMS):
            if key not in self.env_overrides:
                os.environ.pop(key, None)

    def _monitored_backend(self, backend: ullm.backend.Backend) -> ullm.backend.Backend:
        """In proxy mode the returned backend is a CLIENT; the engine lives in
        the host-side backend this process also holds, so that is what the
        status line reads."""
        return self._wrapped_backend or backend

    async def _close(self) -> None:
        # Order: disconnect this process's proxy client first (the host's
        # wait_closed() waits on live connections — Python >= 3.12.1 — so a
        # same-process client would deadlock the host close), then stop
        # serving, then shut down the engine; close the backend's client
        # handles last. Steps are independent — report a failing one and
        # keep going.
        """Shut down in dependency order: this process's proxy client first, then
        the host, then the engine. The host's ``wait_closed()`` waits on live
        connections, so closing it while a same-process client is still attached
        deadlocks.
        """
        await self._close_profile_windows()
        if self._proxy is not None:
            try:
                await self._proxy.close()
            except Exception:
                print("[EngineFactory] proxy client close() failed:", flush=True)
                traceback.print_exc()
            self._proxy = None
        if self._engine_host is not None:
            try:
                await self._engine_host.close()
            except Exception:
                print("[EngineFactory] engine_host.close() failed:", flush=True)
                traceback.print_exc()
            self._engine_host = None
        if self._engine is not None:
            try:
                self._engine.shutdown()
            except Exception:
                print("[EngineFactory] engine.shutdown() failed:", flush=True)
                traceback.print_exc()
            self._engine = None
        if self._wrapped_backend is not None:
            try:
                await self._wrapped_backend.close()
            except Exception:
                print("[EngineFactory] wrapped backend close() failed:", flush=True)
                traceback.print_exc()
            self._wrapped_backend = None
        if self._backend is not None and self._backend is not self._proxy:
            try:
                await self._backend.close()
            except Exception:
                print("[EngineFactory] backend.close() failed:", flush=True)
                traceback.print_exc()
            self._backend = None
        if self._flashinfer_workspace is not None:
            shutil.rmtree(self._flashinfer_workspace, ignore_errors=True)
            self._flashinfer_workspace = None


def _num_speculative_tokens(engine_args: Any) -> int:
    """Draft tokens per request, from a speculative config as either a dict or
    vLLM's own object; zero with no drafter."""
    spec = getattr(engine_args, "speculative_config", None)
    if spec is None:
        return 0
    count = (
        spec.get("num_speculative_tokens") if isinstance(spec, dict) else getattr(spec, "num_speculative_tokens", None)
    )
    return int(count or 0)


async def _post_start_profile(base: str) -> None:
    """Open a profile window on an in-process server, through its own route.

    Async for the reason ``_served_models`` is: the server answers from this
    process's event loop."""
    import httpx

    async with httpx.AsyncClient(timeout=60.0) as client:
        (await client.post(f"{base}/start_profile")).raise_for_status()


def _serve_args_from_engine_args(
    engine_args: vllm.AsyncEngineArgs, host: str, port: int, overrides: dict[str, Any]
) -> Any:
    """Build the argparse Namespace vLLM's API server entry points expect,
    seeded with the serve CLI's defaults and overlaid with our
    ``AsyncEngineArgs`` field by field (the server rebuilds its engine args
    via ``AsyncEngineArgs.from_cli_args(args)``, which reads the same field
    names off the namespace — a faithful round-trip, no flag encoding)."""
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    assert args is not None  # parse_args([]) on a fresh parser always yields a namespace
    for f in dataclasses.fields(type(engine_args)):
        setattr(args, f.name, getattr(engine_args, f.name))
    args.host = host
    args.port = port
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@dataclass
class InprocessServerFactory(LocalEngineFields):
    """Runs vLLM's OpenAI-compatible server in this process (HTTP layer
    only; the engine core subprocess is vLLM's own) and returns an
    ``OpenAIServer`` backend pointing at it."""

    # OpenAIServer continues a history by tokens over /v1/completions, which
    # takes no image input (``OpenAIServer._unwitnessable``).
    continues_images: ClassVar[bool] = False

    engine_args: vllm.AsyncEngineArgs = field(kw_only=True)
    host: str = "127.0.0.1"
    port: Optional[int] = None  # None -> pick a free port at start
    serve_overrides: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # Extra attrs set on the server args namespace (e.g. served_model_name,
    # enable_auto_tool_choice, tool_call_parser).
    backend_kwargs: dict[str, Any] = field(default_factory=lambda: dict[str, Any]())
    # A last-resort bound, not a tuned wait: a cold torch.compile of a large
    # model on a cold cache runs into the tens of minutes, and a bound that
    # fires on one turns a slow start into a failed run. Sized like the
    # transport's own inactivity watchdog (ullm.backends.proxy), which the
    # same reasoning produced.
    startup_timeout: float = 3600.0  # seconds to wait for /health
    # Env vars applied before the server (and the engine subprocess it spawns)
    # start; launcher-side os.environ never travels to the worker. Same role as
    # EngineFactory.env_overrides, e.g. VLLM_USE_V2_MODEL_RUNNER, which vLLM
    # reads only at engine init.
    env_overrides: dict[str, str] = field(default_factory=lambda: dict[str, str]())
    # Opt-in (offline/Kaggle presets set it): flips HF_HUB_OFFLINE for the
    # WHOLE process, so with it on, no later factory in the same process can
    # fetch anything from the hub.
    hf_hub_offline_after_start: bool = False
    # Serve ONE LoRA adapter on top of the base and point the returned client
    # at it. Both or neither: the adapter is mounted under lora_adapter_name
    # (vLLM only addresses adapters by lora_modules name) and the client's
    # model is set to that name. Deliberately NOT via served_model_name:
    # naming the BASE after the adapter makes vLLM's is_base_model resolution
    # serve the untrained base weights under the adapter's name — an eval
    # that runs, scores, and silently measures nothing.
    lora_adapter_path: Optional[str] = None
    lora_adapter_name: Optional[str] = None  # required when lora_adapter_path is set
    max_lora_rank: int = 16  # vLLM validates against its supported-rank Literal at engine build
    _serve_task: Optional["asyncio.Task[None]"] = field(default=None, init=False, repr=False)
    _backend: Optional[ullm.backend.Backend] = field(default=None, init=False, repr=False)
    _flashinfer_workspace: Optional[str] = field(default=None, init=False, repr=False)
    _derivation_baseline: dict[str, Any] = field(default_factory=lambda: dict[str, Any](), init=False, repr=False)

    def __post_init__(self) -> None:
        self._derivation_baseline = derivation_baseline(self.engine_args)
        super().__post_init__()

    async def _start(self) -> ullm.backend.Backend:
        """Start vLLM's OpenAI-compatible server in-process and return a client
        pointed at it, once it answers.
        """
        import httpx
        from vllm.entrypoints.launchers.api_server import entry as api_server

        # Fail on an inconsistent adapter spec before anything is spawned.
        self._validate_lora()
        # Engine toggles vLLM reads only at init, applied on the worker before
        # the server spawns its engine core (launcher env never travels).
        os.environ.update(self.env_overrides)
        # Same JIT-toolkit provisioning as EngineFactory — the server spawns
        # the same engine core underneath.
        ullm.backends.config_local._ensure_path_has_venv_bin()
        ullm.backends.config_local._ensure_cuda_home()
        _set_jit_parallelism()
        ullm.backends.config_local._quiet_engine_logging()
        self._flashinfer_workspace = _isolate_flashinfer_workspace()
        port = self.port if self.port is not None else find_free_port(self.host)
        # prompt_tokens_details is off by default in vLLM's server; without
        # it, usage never reports cached_tokens and prefix-cache accounting
        # is blind. Callers can still override via serve_overrides.
        serve_overrides: dict[str, Any] = {"enable_prompt_tokens_details": True, **self.serve_overrides}
        # Adapter mounting is start-time only: lora_modules is a server flag,
        # and the in-process engine path has no way to name an adapter at all.
        self._apply_lora(serve_overrides)
        # Before the namespace is built: it copies the engine args field by
        # field, so a connector applied after it never reaches the server.
        self._settle_at_start(self.engine_args)
        args = _serve_args_from_engine_args(self.engine_args, self.host, port, serve_overrides)
        # run_server = setup_server (bind socket, validate) + serve worker.
        # Run it as a task; cancellation propagates into uvicorn shutdown and
        # the engine-client context exit. setup_server installs a SIGTERM
        # handler, so start() must run on the main thread.
        self._serve_task = asyncio.create_task(api_server.run_server(args))

        base = f"http://{self.host}:{port}"
        async with httpx.AsyncClient() as client:
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            while True:
                if self._serve_task.done():
                    self._serve_task.result()  # raises the server's startup error
                    raise RuntimeError("vLLM API server exited before becoming healthy")
                with contextlib.suppress(httpx.HTTPError):
                    if (await client.get(f"{base}/health")).status_code == 200:
                        break
                if asyncio.get_running_loop().time() > deadline:
                    await self._abort_start()
                    raise TimeoutError(f"vLLM API server not healthy after {self.startup_timeout}s")
                await asyncio.sleep(1.0)

        try:
            if self.hf_hub_offline_after_start:
                force_hf_hub_offline()
            if self.lora_adapter_name is not None:
                await self._verify_adapter_served(base)
            self._open_profile_windows(
                lambda: _post_start_profile(base), ullm.engine_monitor.http_source(f"{base}/metrics")
            )
            return self._finish_start(base)
        except BaseException:
            # start() marks the factory started only when _start RETURNS, so
            # a failure here would otherwise leave close() a no-op and the
            # server running with no owner.
            await self._abort_start()
            raise

    async def _verify_adapter_served(self, base: str) -> None:
        """The one legitimate route to the adapter: verified served, then
        requested by its own name. Refusing here is what keeps a broken mount
        from silently scoring the base weights."""
        served = await self._served_models(base)
        if self.lora_adapter_name not in served:
            raise RuntimeError(
                f"adapter {self.lora_adapter_name!r} is not served (/v1/models has {sorted(served)}). "
                "Refusing to run: requests would fall back to the base model and an eval "
                "would silently score untrained weights."
            )

    def _finish_start(self, base: str) -> ullm.backend.Backend:
        """Construct the client against the now-verified server."""
        import ullm.backends.oai as _oai

        model_name = self.lora_adapter_name or self.serve_overrides.get("served_model_name") or self.engine_args.model
        # vLLM's server rejects the legacy "max_tokens" wire name, and renders
        # past reasoning from the assistant-message "reasoning" field; callers
        # can still override both via backend_kwargs. Nothing is needed here
        # for exact counting: the backend asks this server's own /tokenize.
        backend_kwargs: dict[str, Any] = {
            "token_limit_param": "max_completion_tokens",
            "reasoning_history_field": "reasoning",
            **self.backend_kwargs,
        }
        self._backend = _oai.OpenAIServer(base_url=f"{base}/v1", model=model_name, **backend_kwargs)
        return self._backend

    async def _abort_start(self) -> None:
        """Tear down everything ``_start`` acquired, on its failure path.

        Cancellation is awaited, not fired: a cancel left running keeps the
        port bound and uvicorn mid-shutdown while a caller retries. The
        FlashInfer workspace goes with it, as in ``_close``."""
        await self._close_profile_windows()
        if self._serve_task is not None:
            self._serve_task.cancel()
            with contextlib.suppress(BaseException):
                await self._serve_task
            self._serve_task = None
        if self._flashinfer_workspace is not None:
            shutil.rmtree(self._flashinfer_workspace, ignore_errors=True)
            self._flashinfer_workspace = None

    def _validate_lora(self) -> None:
        """Refuse an adapter spec missing either half, before any spend."""
        if (self.lora_adapter_path is None) != (self.lora_adapter_name is None):
            raise ValueError(
                "lora_adapter_path and lora_adapter_name go together: the adapter is "
                "mounted at the path and is only addressable by the name."
            )

    def _apply_lora(self, serve_overrides: dict[str, Any]) -> None:
        """Mount the adapter on the engine args and the server flags, from the
        one (name, path) spelling the client will later be pointed at."""
        if self.lora_adapter_path is None or self.lora_adapter_name is None:
            return
        from vllm.config.lora import MaxLoRARanks
        from vllm.entrypoints.openai.models.protocol import LoRAModulePath

        # One adapter by design: per-checkpoint sweeps run one job per
        # checkpoint, and a second adapter would need its own addressing story.
        self.engine_args.enable_lora = True
        self.engine_args.max_lora_rank = typing.cast(MaxLoRARanks, self.max_lora_rank)
        self.engine_args.max_loras = 1
        serve_overrides["lora_modules"] = [LoRAModulePath(name=self.lora_adapter_name, path=self.lora_adapter_path)]

    @staticmethod
    async def _served_models(base: str, timeout: float = 60.0) -> set[str]:
        """The model ids the server actually serves.

        Must be async: the server runs in THIS process's event loop, so a
        blocking HTTP call here deadlocks it against its own request — the
        server cannot answer while the loop is blocked waiting on the answer.
        """
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(f"{base}/v1/models")
            resp.raise_for_status()
            payload = resp.json()
        return {entry["id"] for entry in payload.get("data", [])}

    async def _close(self) -> None:
        """Stop the server."""
        await self._close_profile_windows()
        if self._serve_task is not None:
            self._serve_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._serve_task
            self._serve_task = None
        if self._backend is not None:
            try:
                await self._backend.close()
            except Exception:
                print("[InprocessServerFactory] backend.close() failed:", flush=True)
                traceback.print_exc()
            self._backend = None
        if self._flashinfer_workspace is not None:
            shutil.rmtree(self._flashinfer_workspace, ignore_errors=True)
            self._flashinfer_workspace = None
