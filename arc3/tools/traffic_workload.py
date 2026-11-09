"""Cut a traffic workload out of a recording's traces, and say what it holds.

    uv run python tools/traffic_workload.py RECORDING_JOB_DIR OUT_FILE \\
        [--horizon-tokens 60000] [--snapshot-min M] [--skip-in-flight]

The recording is a ``traffic_launch.py record`` job (ids-only traces with slot
links). The snapshot defaults to the middle of the recording's settled span
(``ullm.traffic.build_workload``); ``--snapshot-min`` places it that many
minutes after the recording's first call instead.
"""

from __future__ import annotations

import argparse
import pathlib
import statistics

import ullm.traffic

import platypus.traffic


def main() -> None:
    """Build, save and describe one workload."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=pathlib.Path)
    parser.add_argument("out", type=pathlib.Path)
    parser.add_argument("--horizon-tokens", type=int, default=60_000)
    parser.add_argument("--snapshot-min", type=float, default=None)
    # Start each lane at the call after the one in flight at the snapshot,
    # whose reply is length-biased long (ullm.traffic.build_workload).
    parser.add_argument("--skip-in-flight", action="store_true")
    args = parser.parse_args()

    runs, media = platypus.traffic.recorded_runs(args.recording / "traces")
    start = min(call.finished_at for run in runs for call in run.calls)
    snapshot = None if args.snapshot_min is None else start + 60 * args.snapshot_min
    workload = ullm.traffic.build_workload(
        runs,
        media,
        horizon_tokens=args.horizon_tokens,
        snapshot=snapshot,
        meta={"recording": str(args.recording)},
        skip_in_flight=args.skip_in_flight,
    )
    workload.save(args.out)

    meta = workload.meta
    held = [len(lane.history) for lane in workload.lanes if lane.history is not None]
    calls = [call for lane in workload.lanes for call in lane.calls]
    print(f"{len(runs)} runs, {sum(len(r.calls) for r in runs)} calls recorded; {meta['recorded_lanes']} lanes")
    print(
        f"snapshot at {(meta['snapshot'] - start) / 60:.1f} min "
        f"(settled from {(meta['settled_at'] - start) / 60:.1f}, horizon held until "
        f"{(meta['horizon_held_until'] - start) / 60:.1f})"
    )
    print(
        f"history: {len(held)} of {len(workload.lanes)} lanes hold one, "
        f"{sum(held):,} tokens (median {statistics.median(held) if held else 0:,.0f})"
    )
    print(
        f"{len(calls)} calls after the snapshot, {sum(c.output_len for c in calls):,} reply tokens, "
        f"{sum(1 for c in calls if not c.shared)} opening a sequence; {len(workload.media)} images"
    )
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
