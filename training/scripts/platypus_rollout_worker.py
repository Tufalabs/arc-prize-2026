"""One training rollout window, hosted by the platypus harness.

Runs INSIDE the platypus venv (invoked as
``<platypus_venv>/bin/python scripts/platypus_rollout_worker.py --spec spec.json``)
and loads only the selected harness adapter: the contract
between trainer and worker is the spec JSON in and the benchmark/trace
artifacts out.

Spec schema (all required unless noted)::

    {
      "window_id": str,               # label only
      "job_dir": str,                 # window output dir (benchmark.json, traces/)
      "games": [str, ...],            # env_name per rollout slot (repeats allowed)
      "rng_seed": int,                # window seed; Platypus derives slot seeds
      "slot_seeds": [int, ...] | null,  # optional: per-rollout model RNG seeds,
                                      # parallel to "games". Each slot has a
                                      # unique external game id, including
                                      # repeated environments. Absent/null:
                                      # rng_seed + game index derivation.
      "max_concurrent": int,
      "max_runtime_s_per_game": float | null,
      "base_url": str,                # native vLLM gateway root URL
      "served_model": str,            # served alias (runtime LoRA export) or base id
      "tokenizer_id": str,            # HF id or local checkpoint dir for output-render classification
      "engine": str | absent,         # duck engine preset (default "qwen3.8-27b"; Flash sends "qwen3.8-flash-next")
      "reasoning_effort": str | null, # duck knob; null keeps the preset default
      "context_window_tokens": int,       # duck decision window
      "summary_forbidden_token_ids": [int, ...], # exact over-trigger -100 logit biases
      "max_summary_tokens": int,          # over-trigger completion cap (0 disables)
      "max_served_sequence_tokens": int,  # prompt + completion ceiling
      "malformed_retries": 0,             # one draw per model request
      "accept_malformed_output": bool,    # keep an empty reply as the model's turn
      "malformed_call_penalty": float,    # ARC-2: terminal charge per malformed call
      "max_episode_penalty": float | null, # ARC-2: most an episode's penalties take
      "terminate_on_empty_reply": bool,   # ARC-2: an empty reply ends the game
      "compact_to_tokens": int,           # trainer-owned reset ceiling
      "compaction_trigger_tokens": int,   # exact pre-trim prompt threshold
      "safety_margin_tokens": int,        # explicit resolved Duck margin
      "results_url": str | null,      # trainer gateway results endpoint
      "augmentation": {...} | null,   # training windows only: the window's
                                      # augmentation parameters
                                      # (harness_support/augmentation.py)
      "sampling": {                   # applied to EVERY duck sampling profile
        "temperature": float, "top_p": float, "top_k": int | null
      }
    }

``sampling`` is mandatory for training: the trainer scores rollout tokens
at temperature 1.0 with the admitted token bias, so the harness must sample
from the same distribution or PPO's importance ratios are silently biased. The
ducks' own analyzer profiles sample at 0.8 (ARC-2) and 0.6 (ARC-3); neither is
used here.

When ``results_url`` is set, the worker ends the window by POSTing one
``RunResult`` per game run to the trainer's rollout gateway: the game
outcome plus, per model step, the request-stamp nonce (ullm stamps every
request's ``request_id`` field with ``run_id#nonce``), the authorized
``action_index``, and an output-token checksum. The trainer joins these
against its gateway captures — the harness's artifact files never cross
the trainer boundary (they remain the viewer/debug surface).

The worker exits non-zero on any hosting failure. Malformed model replies
remain ordinary recorded turns. An agent error ends only its own game
(``on_agent_error="crash_game"``, the deployment's policy): Benchmark teardown
marks it ``crashed`` and scores its standing answer before ``benchmark.json``
is written, so its result rides in the window's one result batch with its
``solver_note`` (the traceback).
"""

import argparse
import asyncio
import json
import sys
import time
import traceback
import urllib.request
import zlib
from pathlib import Path

from training.rollout_slots import _slot_game_ids


def _token_ids_crc32(token_ids) -> int:
    """Canonical output checksum; mirrored by training/rollout_gateway.py."""
    return zlib.crc32(",".join(str(int(t)) for t in token_ids).encode("ascii"))


def _step_reports(trace_dir: Path) -> list:
    """Distill one run's steps.jsonl into the results-contract step list."""
    steps_path = trace_dir / "steps.jsonl"
    reports = []
    if not steps_path.exists():
        # A run that crashed (or gave up) before its first model call
        # leaves no trace folder; it is still a legitimate zero-step,
        # zero-score outcome the trainer must see, not a worker failure.
        return reports
    with steps_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # platypus's collector files the request args under the step's
            # provider_validation block (R5); tolerate a top-level copy too.
            request_args = (
                (record.get("provider_validation") or {}).get("request_args")
                or record.get("request_args")
                or {}
            )
            stamp = request_args.get("request_id")
            if not isinstance(stamp, str) or "#" not in stamp:
                raise RuntimeError(
                    f"{steps_path}: trace step lacks the run#nonce request "
                    "stamp — the RL results contract requires stamped calls"
                )
            output_ids = record.get("output_token_ids") or []
            reports.append(
                {
                    "nonce": stamp.rpartition("#")[2],
                    "decision_elapsed_seconds": (
                        (
                            (request_args.get("request_context") or {}).get(
                                "arc3_progress"
                            )
                            or {}
                        ).get("data")
                        or {}
                    ).get("decision_elapsed_seconds"),
                    "action_index": record.get("action_index"),
                    "output_len": len(output_ids),
                    "output_crc32": _token_ids_crc32(output_ids),
                    # ullm's token continuation: the prompt extended the
                    # previous step's tokens (True) or was rendered afresh
                    # (False: first turn, compaction).
                    # Keep missing/invalid evidence visible to the trainer;
                    # defaulting a missing flag to False hides broken traces.
                    "continued": (record.get("extra") or {}).get("continued"),
                    "history_epoch": (record.get("extra") or {}).get("history_epoch"),
                    "exact_history": (record.get("extra") or {}).get("exact_history"),
                    # The backend kept the reply as malformed (ullm
                    # accept_malformed_output: no final content, or text it
                    # could not parse). ARC-3 reports it as the step's
                    # format_break, a diagnostic that charges no reward
                    # (format_break_penalty must be 0). ARC-2 penalizes from
                    # its processed-response journal instead (malformed_call_count
                    # and empty_reply, harness_support/arc2) and derives its
                    # own format_break from them.
                    "format_break": bool((record.get("extra") or {}).get("malformed")),
                }
            )
    return reports


def _post_results(spec: dict) -> None:
    """Report every run's outcome + step map to the trainer gateway."""
    from harness_support.augmentation import applied_record
    from training.harness import get_harness

    job_dir = Path(spec["job_dir"])
    benchmark = json.loads((job_dir / "benchmark.json").read_text(encoding="utf-8"))
    names = list(spec["games"])
    env_names = dict(zip(_slot_game_ids(names), names, strict=True))
    runs_payload = []
    for run in benchmark.get("game_runs") or []:
        run_id = str(run["run_id"])
        history = run.get("history") or []
        external_game_id = str(run["game_id"])
        if external_game_id not in env_names:
            raise ValueError(
                f"results contain unknown rollout slot {external_game_id!r}"
            )
        runs_payload.append(
            {
                "run_id": run_id,
                "game_id": env_names[external_game_id],
                "state": str(run.get("state", "")),
                # Preserve JSON types so the trainer can reject malformed
                # evidence instead of receiving silently truncated counts.
                "levels_completed": run.get("levels_completed", 0),
                "number_of_levels": run.get("number_of_levels", 0),
                "actions_per_level": run.get("actions_per_level") or [],
                "base_actions_per_level": run.get("base_actions_per_level") or [],
                "final_wallclock_seconds": run.get("final_wallclock_seconds"),
                "action_wallclock_seconds": [
                    entry.get("wallclock_seconds", 0.0) for entry in history
                ],
                "steps": _step_reports(job_dir / "traces" / run_id),
                # What the slot's task or game was augmented with, if anything.
                "augmentation": applied_record(spec, external_game_id),
                # The harness's own note on how the run ended: a crashed run's
                # traceback ("agent error:" or "game subprocess died:"), a
                # give-up's reason; null when it has none.
                "solver_note": run.get("solver_note"),
            }
        )
    if len(runs_payload) != len(names) or len(
        {run["run_id"] for run in runs_payload}
    ) != len(names):
        raise RuntimeError(
            "benchmark did not return exactly one result per rollout slot"
        )
    get_harness(spec.get("environment", "arc3")).enrich_results(
        spec, runs_payload, benchmark.get("game_runs") or []
    )
    _send_results(spec["results_url"], runs_payload)


def _send_results(results_url: str, runs_payload: list[dict]) -> None:
    """Post one complete result batch and require full gateway admission."""
    request = urllib.request.Request(
        results_url,
        data=json.dumps({"runs": runs_payload}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        accepted = json.loads(response.read()).get("accepted")
    if accepted != len(runs_payload):
        raise RuntimeError(
            f"trainer gateway accepted {accepted} of {len(runs_payload)} run results"
        )


def _build_benchmark(spec: dict):
    """Construct the selected harness without importing the other one."""
    from training.harness import get_harness

    return get_harness(spec.get("environment", "arc3")).build_benchmark(spec)


async def _run(spec: dict) -> None:
    """Run the benchmark to completion."""
    benchmark = _build_benchmark(spec)
    await benchmark.run(soft_end_time=None)
    if getattr(benchmark.solver, "teardown_errors", None):
        raise RuntimeError(
            f"harness teardown failed: {benchmark.solver.teardown_errors}"
        )


def _run_window(spec: dict) -> dict:
    """Run one window end to end; report rather than raise so a serving
    worker keeps its (expensive) imports for the next window."""
    try:
        asyncio.run(_run(spec))
        if spec.get("results_url"):
            _post_results(spec)
        return {"status": "ok"}
    except BaseException as exc:  # noqa: BLE001 - reported to the trainer
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def serve(serve_dir: Path, *, poll_seconds: float = 0.5) -> int:
    """Persistent mode: importing the harness costs minutes, so one worker
    serves every window of a run. The trainer drops ``<window_id>.spec.json``
    into ``serve_dir``; the worker consumes specs in name order, runs each
    window, and answers with ``<window_id>.done.json`` (status ok/error).
    A ``STOP`` file ends the loop once the queue is drained."""
    serve_dir.mkdir(parents=True, exist_ok=True)
    print(f"[serve] ready in {serve_dir}", flush=True)
    while True:
        pending = sorted(serve_dir.glob("*.spec.json"))
        if not pending:
            if (serve_dir / "STOP").exists():
                print("[serve] stop requested; exiting", flush=True)
                return 0
            time.sleep(poll_seconds)
            continue
        spec_path = pending[0]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        window_id = str(spec["window_id"])
        # Consume BEFORE running so a re-poll can never run a window twice.
        spec_path.rename(spec_path.with_suffix(".consumed"))
        print(f"[window {window_id}] start", flush=True)
        status = _run_window(spec)
        if status["status"] != "ok":
            print(status.get("traceback", status["error"]), flush=True)
        done_path = serve_dir / f"{window_id}.done.json"
        staging = done_path.with_suffix(".tmp")
        staging.write_text(json.dumps(status), encoding="utf-8")
        staging.replace(done_path)
        print(f"[window {window_id}] {status['status']}", flush=True)


def main() -> int:
    """CLI entry: ``--spec`` runs one window and exits; ``--serve DIR`` keeps
    the process alive across windows (see ``serve``)."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec")
    group.add_argument("--serve")
    args = parser.parse_args()
    if args.serve:
        return serve(Path(args.serve))
    spec = json.loads(Path(args.spec).read_text(encoding="utf-8"))
    status = _run_window(spec)
    if status["status"] != "ok":
        print(status.get("traceback", status["error"]), file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
