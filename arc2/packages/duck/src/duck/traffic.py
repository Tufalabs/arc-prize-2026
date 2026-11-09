"""The traffic benchmark's duck: the engine a duck column serves, measured on
recorded traffic instead of games.

``TrafficDuck`` is built by the same ``build(engine, environment, ...)`` as the
duck, so its engine, host tier, step budget, concurrency and sampling are the
column's. Where the solver would start the games, it replays a workload
(``ullm.traffic``) against the started engine and leaves the games unplayed;
the reading lands in the job dir as ``traffic_result.json`` and
``traffic_calls.jsonl``.

Nothing is prefilled ahead of the window: the engine reaches the recording's
state by serving the lanes, so the window opens cold and its first
``settle_s`` is the warm-up the reading leaves out.

The workload is one file, carried to a Kaggle kernel as a payload file of the
deploy's bundle. Only this agent's bundle carries it: ``build`` adds it to the
run config's payload files, which a duck leaves as the column set them.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import ullm.backend

from duck.agent import DuckAgent


@dataclass
class TrafficDuck(DuckAgent):
    """A duck column's engine, replaying a recorded workload in a timed window."""

    workload: str = ""  # a workload file, as ``ullm.traffic.Workload.save`` wrote it
    window_s: float = 1500.0  # the timed window
    # The window's warm-up, which the reading leaves out. Ten minutes because
    # a cold pool takes that long to reach a steady state: the four L4s filled
    # their running set four minutes in and first waited on capacity twelve
    # minutes in (2026-09-26), and ARC3's RTX flattened at ~8 minutes.
    settle_s: float = 600.0
    # A lane's pause between one call returning and its next: none, as play
    # waits none since 2026-09-28. Readings on the 09-19 and 09-20 workloads
    # ran 5 s; a replay of those passes 5 to compare with them.
    request_gap_s: float = 0.0
    # Lanes keep playing this long past the window, counted in nothing: room
    # for profiler windows that would otherwise disturb the reading.
    tail_s: float = 0.0
    lanes: int | None = None  # lanes replayed; None replays run_config.max_concurrent of them
    record_outputs: bool = False  # each call's reply ids in its row (``ullm.traffic.replay``)
    # Every lane's first prompt prefilled on its own thread before the window,
    # untimed, its first call continuing it: the window opens on parked
    # conversations rather than a cold pool (``ullm.traffic.replay``'s
    # decode_gate).
    decode_gate: bool = False

    @classmethod
    def build(cls, **kwargs: Any) -> Self:
        """The duck ``DuckAgent.build`` makes from the same arguments, with the
        workload among the Kaggle bundle's payload files."""
        agent = super().build(**kwargs)
        if agent.workload:
            agent.run_config.kaggle_bundle_files = (*agent.run_config.kaggle_bundle_files, agent.workload)
        return agent

    def prepare_for_backend(self, *, settle_engine: bool = True) -> None:
        """Prepare as the duck does, then read the workload: one that is
        missing or unreadable fails the run here, before the engine loads,
        rather than once it has."""
        super().prepare_for_backend(settle_engine=settle_engine)
        # Aliased: a bare ``import ullm.traffic`` here would rebind ``ullm``
        # in this function and hide the module-level ``ullm.backend``.
        from ullm import traffic as _traffic

        if not self.workload:
            raise ValueError("TrafficDuck.workload names no workload file")
        _traffic.Workload.load(self._workload_file())

    def _workload_file(self) -> Path:
        """The workload where it was named, or in a Kaggle kernel, the copy in
        the bundle's payload.

        The kernel's runner publishes where it mounted each attached dataset,
        the source bundle first (``taaf.deploy_kaggle``).
        """
        named = Path(self.workload)
        if named.is_file():
            return named
        inputs = os.environ.get("TAAF_KAGGLE_INPUT_PATHS")
        sources = os.environ.get("TAAF_KAGGLE_DATASET_SOURCES")
        if inputs and sources:
            shipped = Path(json.loads(inputs)[json.loads(sources)[0]]) / "payload" / named.name
            if shipped.is_file():
                return shipped
        raise FileNotFoundError(f"workload {self.workload} is neither here nor in a Kaggle bundle's payload")

    async def run_on_host(self, backend: ullm.backend.Backend, job_dir: Path | None) -> bool:
        """Replay the workload against the host's engine, in place of the games."""
        # Aliased: a bare ``import ullm.traffic`` here would rebind ``ullm``
        # in this function and hide the module-level ``ullm.backend``.
        from ullm import traffic as _traffic

        if not hasattr(backend, "serve_ids"):
            raise TypeError(
                f"the traffic replay serves id prompts, which {type(backend).__name__} cannot; "
                "it needs the in-process engine (serving='direct')"
            )
        path = self._workload_file()
        workload = _traffic.Workload.load(path)
        profile = self.llm_config.sampling_profiles[self.profile]
        if not isinstance(profile, ullm.backend.GenericSamplingParameters):
            raise TypeError(f"sampling profile {self.profile!r} is {type(profile).__name__}")
        lanes = self.lanes if self.lanes is not None else self.run_config.max_concurrent
        print(
            f"[traffic] replaying {lanes} of {len(workload.lanes)} lanes from {path} "
            f"for {self.window_s:.0f} s, {self.request_gap_s:.1f} s between a lane's calls",
            flush=True,
        )
        await _traffic.replay(
            backend,  # type: ignore[arg-type]  # checked for serve_ids above
            workload,
            lanes=lanes,
            window_s=self.window_s,
            gap_s=self.request_gap_s,
            sampling_params=copy.deepcopy(profile),
            out_dir=job_dir,
            settle_s=self.settle_s,
            tail_s=self.tail_s,
            placeholder_id=backend.image_token_id() if hasattr(backend, "image_token_id") else None,  # type: ignore[attr-defined]
            record_outputs=self.record_outputs,
            decode_gate=self.decode_gate,
            # Reported as the window closes, before a tail whose profiler
            # windows could stall or kill the engine.
            on_window=_report_reading,
        )
        return True


def _report_reading(summary: dict[str, Any]) -> None:
    """The replay's reading, in the lines the archive tool reads."""
    if summary["settled_generated_tok_s"] is not None:
        print(
            f"[traffic] settled {summary['settled_s']:.0f} s from {summary['settled_at_s']:.0f} s in: "
            f"gen {summary['settled_generated_tok_s']:.1f} tok/s, uncached prompt "
            f"{summary['settled_uncached_prompt_tok_s']:.1f} tok/s",
            flush=True,
        )
    print(
        f"[traffic] window {summary['elapsed_s']:.0f} s: generated {summary['generated_tokens']} tokens "
        f"({summary['generated_tok_s']:.1f} tok/s), uncached prompt {summary['uncached_prompt_tokens']} "
        f"({summary['uncached_prompt_tok_s']:.1f} tok/s); {summary['calls_returned']} calls returned",
        flush=True,
    )
