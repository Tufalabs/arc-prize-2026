"""The sweep agent: one run, several configurations of a wrapped agent.

A sweep point is a **pass**. The agent wraps another agent and, inside each
game, configures its copy of that agent from ``arm_seeds[pass_index]`` — so a
benchmark of 25 games at 10 passes plays 10 configurations over the same 25
games, under one allocation and one served model. Nothing in the hosting
layer changes: this is an ordinary agent, deepcopied per (game, pass) like
any other, and the solver already gives every game its own backend client.

Two batches, because the two granularities are real: the solver reads some
configuration off the **template** before any game starts, and the rest off
the **per-game copy**.

- ``run`` knobs are drawn from ``run_seed`` and applied at construction, so
  they reach engine settings and hosting knobs. One job is one draw of them,
  and their effects are recovered across pooled jobs.
- ``arm`` knobs are drawn from ``arm_seeds[pass_index]`` and applied inside
  the game process, so they reach only what is read from the per-game copy.

Putting an arm knob on a template-only field is the failure this module
works hardest to prevent: it produces a knob column with no effect, which
does not look like a bug — it looks like a knob that did not matter.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Self

import taaf.game
from platypus import Agent, GameOutcome, GiveUp, LLMAgent, MoveContext
from sweeper.knobs import (
    RECORD,
    Design,
    Knob,
    apply_values,
    check_picklable,
    draw_values,
    set_by_path,
    unresolved_segment,
)

#: Prefix on the stdout lines that carry a job's design. A Kaggle submission
#: returns a leaderboard score and nothing else — no benchmark.json, no
#: sidecars — so the kernel log is the only record of what was configured,
#: and it has to be machine-readable.
SWEEP_LINE = "[sweep]"


def _arm_target_refusal(path: str) -> str | None:
    """Why this dotted path cannot be a per-arm knob, or ``None`` if it can.

    The rule is not a taste: it is where the value is READ. Per-arm knobs are
    applied inside the game process, to that game's copy, so a field the
    solver read off the template before the fan-out is already spent.

    That includes ``max_runtime_s_per_game``, which reads per GAME but not per
    MOVE: ``run_game`` freezes it into its budget before ``do_move``, and the
    arm batch is applied inside that call. The mutation lands one read too
    late, so the column would vary while the wall budget did not.
    """
    if path.startswith("run_config."):
        return "RunConfig is read before the arm batch is applied, so setting it per arm changes nothing"
    if path.startswith("llm_config."):
        return "the host backend is built from the template's llm_config before any game starts"
    if path.startswith(("inner.run_config.", "inner.llm_config.backend_factory")):
        return "the wrapped agent's own run_config/backend_factory is inert — the solver reads the wrapper's"
    if not path.startswith("inner."):
        return "a per-arm knob must target the wrapped agent (inner.*)"
    return None


def _check_arm_targets(knobs: Mapping[str, Knob]) -> None:
    """Refuse per-arm knobs aimed at values that are read before the fan-out."""
    for name in sorted(knobs):
        apply = knobs[name].apply
        path = getattr(apply, "path", None)
        if not isinstance(path, str):
            continue  # a custom apply is opaque; its author owns this rule
        _refuse_arm_path(path, f"arm knob {name!r}")


def _check_paths_resolve(agent: Any, paths: Iterable[tuple[str, str]]) -> None:
    """Refuse a ``(what, path)`` whose path names nothing on the agent it
    will be applied to.

    The other checks ask whether a path is allowed to vary per arm. This one
    asks whether it exists, which the first cannot see: ``setattr`` creates an
    attribute rather than failing, so a knob aimed at a renamed or misspelled
    field configures a stray attribute nothing reads. It still draws, still
    records, still fills a column — and reports a null, which is
    indistinguishable from having measured one.
    """
    for what, path in sorted(paths):
        stopped = unresolved_segment(agent, path)
        if stopped is not None:
            raise ValueError(
                f"{what} targets {path!r}, but {stopped!r} does not exist on the agent. "
                "Setting it would create an attribute nothing reads, so the knob would draw, "
                "record, and change nothing — a measured null that is not one."
            )


def _refuse_arm_path(path: str, what: str) -> None:
    """Raise if ``path`` cannot take effect per arm."""
    refusal = _arm_target_refusal(path)
    if refusal is not None:
        raise ValueError(
            f"{what} targets {path!r}, which cannot vary per arm: {refusal}. "
            "Move it to the run batch (one draw per job, recovered across pooled jobs)."
        )


def _check_derivation(knobs: Mapping[str, Knob], derive: object | None, *, batch: str) -> None:
    """Refuse a batch whose recording knobs have nothing to consume them.

    A ``RECORD`` knob writes no field, so without a derivation it varies in
    every table and changes nothing — the same silent nothing the target check
    exists to prevent, arrived at from the other direction.
    """
    recording = sorted(name for name, knob in knobs.items() if isinstance(knob.apply, type(RECORD)))
    if recording and derive is None:
        raise ValueError(
            f"{batch} knobs {recording} record only, but the batch declares no derivation — "
            "they would vary in every table and configure nothing. Give the batch a derive, "
            "or point the knobs at fields."
        )


def _kv_safe(value: Any) -> Any:
    """A knob value as something diagnostics kv accepts (JSON scalars only)."""
    return value if isinstance(value, bool | int | float | str) or value is None else repr(value)


@dataclass(kw_only=True)
class SweepCore:
    """The sweep mechanism, free of any LLM knowledge.

    Mixed into :class:`SweepAgent` and :class:`LLMSweepAgent` rather than
    living on one of them: the machinery is identical, and only the base
    class differs — an ``LLMAgent`` wrapper is required for an ``LLMAgent``
    inner, because ``LLMAgentSolver`` type-checks the template and builds the
    backend from ITS ``llm_config``.
    """

    #: The agent being swept. Configuration lands on this object, per game.
    inner: Agent
    #: What varies, in two batches.
    design: Design = field(default_factory=Design)
    #: One draw of the run batch per job.
    run_seed: int = 0
    #: One arm per pass. A single seed broadcasts to every pass, which is the
    #: Kaggle shape: one design point per submission, passes as replicates.
    arm_seeds: tuple[int, ...] = (0,)
    #: Print the design at construction. The kernel log is a submission's only
    #: record, so this is on by default and off for tests that assert output.
    announce: bool = True

    #: Resolved run-level values, drawn at construction and carried on the
    #: template — so they are in the pickled solver the job dir keeps.
    run_values: dict[str, Any] = field(default_factory=lambda: dict[str, Any](), init=False)
    #: This copy's arm, filled inside the game process on the first move.
    arm_values: dict[str, Any] = field(default_factory=lambda: dict[str, Any](), init=False)
    arm_seed: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate the design, draw the run batch, apply it, announce it.

        At construction rather than at setup, so the whole design is a value
        on the template: inspectable before an allocation is spent, carried
        into the job dir by the solver pickle, and drawn exactly once — a
        second draw site could disagree with the first.
        """
        if not self.arm_seeds:
            raise ValueError("arm_seeds must name at least one arm")
        check_picklable(self.design.run, batch="run")
        check_picklable(self.design.arm, batch="arm")
        _check_arm_targets(self.design.arm)
        _check_derivation(self.design.run, self.design.derive_run, batch="run")
        _check_derivation(self.design.arm, self.design.derive_arm, batch="arm")
        # Existence checks run in APPLICATION order, each against the graph
        # its write will actually see — a knob aimed at nothing is a silent
        # null discovered a whole job later, but a path can also be valid
        # only once an earlier write has landed. Run knob targets are checked
        # against the construction graph they are applied to; the run
        # derivation is validated inside ``_configure``, on the exact map it
        # applies, after the direct run assignments (a run knob may have
        # rebuilt the container a derived path lands in).
        run_targets: list[tuple[str, str]] = []
        arm_targets: list[tuple[str, str]] = []
        for targets, batch, knobs in ((run_targets, "run", self.design.run), (arm_targets, "arm", self.design.arm)):
            for name, knob in knobs.items():
                path = getattr(knob.apply, "path", None)
                if isinstance(path, str):
                    targets.append((f"{batch} knob {name!r}", path))
        _check_paths_resolve(self, run_targets)
        self.run_values = draw_values(self.design.run, self.run_seed)
        self._configure("run", self.run_values)
        # Arm knob targets against the post-run graph: the first arm writes
        # land on a fresh copy of exactly this graph.
        _check_paths_resolve(self, arm_targets)
        # Each arm's derivation, validated by SIMULATING that arm on a copy —
        # knobs applied first, the derived paths resolved against the knobbed
        # graph — the order ``_configure`` will apply them in. Resolved
        # against the template itself, a derivation targeting what an arm
        # knob just rebuilt would be falsely refused. ``_configure`` checks
        # the map it actually applies again at game time, which is what
        # covers a ``Derive`` that does not return the same map twice.
        for seed in self.arm_seeds:
            values = self.design.arm_values_for(seed)
            probe = copy.deepcopy(self)
            apply_values(self.design.arm, values, probe)
            derived = dict(self.design.derived("arm", values))
            for path in derived:
                _refuse_arm_path(path, f"arm derivation (seed {seed})")
            _check_paths_resolve(probe, [(f"arm derivation (seed {seed})", path) for path in derived])
            # The whole arm, its derived writes landed on the knobbed probe,
            # put to the wrapped agent's own checks. A combination the agent
            # refuses would otherwise be refused inside every game of that
            # pass, after the allocation is spent.
            for path in sorted(derived):
                set_by_path(probe, path, derived[path])
            try:
                probe._check_arm()
            except ValueError as error:
                raise ValueError(f"arm seed {seed} {values} is refused by the wrapped agent: {error}") from error
        if self.announce:
            print(self.describe(), flush=True)

    def describe(self) -> str:
        """The design as machine-parseable lines — what a Kaggle log carries.

        One line per batch draw, JSON with sorted keys, so a later analysis
        can recover the design from a kernel log and check it against its own
        redraw of the same seeds.
        """
        lines = [
            f"{SWEEP_LINE} run seed={self.run_seed} values={json.dumps(self.run_values, sort_keys=True, default=repr)}"
        ]
        for index, seed in enumerate(self.arm_seeds):
            values = self.design.arm_values_for(seed)
            lines.append(
                f"{SWEEP_LINE} arm {index} seed={seed} values={json.dumps(values, sort_keys=True, default=repr)}"
            )
        return "\n".join(lines)

    def _configure(self, batch: str, values: Mapping[str, Any]) -> dict[str, Any]:
        """Apply one batch's knobs, then its derivation. Returns what was derived.

        The derivation runs after the whole batch is drawn and applied, once,
        so a coordinate transform sees every value it needs and no knob's
        position in the ordering can change the result.
        """
        knobs = self.design.arm if batch == "arm" else self.design.run
        apply_values(knobs, values, self)
        derived = dict(self.design.derived(batch, values))
        # The exact map about to be applied, validated after this batch's
        # direct assignments: a knob may have rebuilt the container a derived
        # path lands in, and a ``Derive`` has no purity contract, so a map
        # validated anywhere else is not necessarily this one. For the arm
        # batch that includes per-arm legality — an impure derive could hand
        # application a template-only path the construction probe never saw,
        # which would land as exactly the measured null this module exists
        # to prevent.
        if batch == "arm":
            for path in derived:
                _refuse_arm_path(path, "arm derivation (at application)")
        _check_paths_resolve(self, [(f"{batch} derivation", path) for path in derived])
        for path in sorted(derived):
            set_by_path(self, path, derived[path])
        return derived

    def _check_arm(self) -> None:
        """Put this copy's arm to the wrapped agent's own checks; an LLM-free
        agent declares none."""

    def _arm_seed_for(self, pass_index: int) -> int:
        """This pass's arm seed; one seed covers every pass."""
        if len(self.arm_seeds) == 1:
            return self.arm_seeds[0]
        if pass_index >= len(self.arm_seeds):
            raise ValueError(
                f"pass_index {pass_index} has no arm: {len(self.arm_seeds)} arm seeds for a run with more "
                "passes. arm_seeds must have one entry per pass, or exactly one to cover them all."
            )
        return self.arm_seeds[pass_index]

    def _arm(self, ctx: MoveContext) -> None:
        """Configure this game's copy from its pass's seed.

        Inside the game process, on the move: the template carries the
        seeds, not N configured agents, so a run with many arms does not
        pickle every arm into every child.
        """
        self.arm_seed = self._arm_seed_for(ctx.pass_index)
        self.arm_values = draw_values(self.design.arm, self.arm_seed)
        derived = self._configure("arm", self.arm_values)
        with ctx.diag.span("sweep/arm", f"arm {ctx.pass_index}") as span:
            span.kv(
                pass_index=ctx.pass_index,
                arm_seed=self.arm_seed,
                run_seed=self.run_seed,
                **{f"arm_{name}": _kv_safe(value) for name, value in sorted(self.arm_values.items())},
                **{f"run_{name}": _kv_safe(value) for name, value in sorted(self.run_values.items())},
                # What the coordinates became: the analysis regresses on the
                # knobs, but a reader of one game wants the realised fields.
                **{f"derived_{path}": _kv_safe(value) for path, value in sorted(derived.items())},
            )

    async def _do_move(self, ctx: MoveContext, view: taaf.game.GameView) -> GiveUp | None:
        """Arm on the first call, then hand the move to the wrapped agent.

        The wrapped agent's PUBLIC ``do_move`` runs, so it keeps the framework
        half it was written against (give-up conversion, return validation);
        this agent's own wrapper then validates the same result again, which
        costs nothing and keeps the contract identical either way.
        """
        if self.arm_seed is None:
            self._arm(ctx)
        return await self.inner.do_move(ctx, view)

    async def _on_finish(self, ctx: MoveContext, outcome: GameOutcome) -> None:
        """Hand the terminal call to the wrapped agent."""
        await self.inner.on_finish(ctx, outcome)

    def status_reporters(self) -> list[Callable[[], str]]:
        """The wrapped agent's status reporters, if it publishes any.

        Duck-typed by ``AgentSolver`` off the hosted agent, which is this
        wrapper, so an inner agent's reporters would otherwise disappear.
        """
        hook: Callable[[], list[Callable[[], str]]] | None = getattr(self.inner, "status_reporters", None)
        return list(hook()) if callable(hook) else []


@dataclass(kw_only=True)
class SweepAgent(SweepCore, Agent):
    """Sweeps an LLM-free agent. Hosted by ``platypus.AgentSolver``."""


@dataclass(kw_only=True)
class LLMSweepAgent(SweepCore, LLMAgent):
    """Sweeps an ``LLMAgent``. Hosted by ``platypus.LLMAgentSolver``.

    The wrapper carries the ``llm_config`` the solver reads to build the one
    host-side backend; the inner agent's own ``llm_config`` is what its calls
    resolve sampling profiles from, so profile knobs belong on
    ``inner.llm_config.sampling_profiles`` and the backend factory belongs
    here.
    """

    def without_engine(self) -> Self:
        """This agent and the agent it wraps, each without the engine spec.

        The wrapped agent holds an ``llm_config`` of its own — usually the very
        object the wrapper carries — so dropping the wrapper's spec alone would
        leave the factory on the inner agent, and every game child would
        unpickle it.
        """
        child = super().without_engine()
        if isinstance(child.inner, LLMAgent):
            child.inner = child.inner.without_engine()
        return child

    def _check_arm(self) -> None:
        """The wrapped agent settles this arm's configuration against the
        engine the sweep states, which it may check but not write: one engine
        serves every arm."""
        if isinstance(self.inner, LLMAgent):
            self.inner.prepare_for_backend(settle_engine=False)

    def _validate_host_backend(self, backend: Any) -> None:
        """Validate every arm against the shared engine after its probes connect."""
        super()._validate_host_backend(backend)
        for seed in self.arm_seeds:
            probe = copy.deepcopy(self)
            probe._configure("arm", self.design.arm_values_for(seed))
            if isinstance(probe.inner, LLMAgent):
                probe.inner._validate_host_backend(backend)


__all__ = ["SWEEP_LINE", "LLMSweepAgent", "SweepAgent", "SweepCore"]
