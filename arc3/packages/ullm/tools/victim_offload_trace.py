"""Summarise a victim offload trace (`kv_offload_trace.jsonl`) by time window.

The connector's scheduler side writes the trace into the run's job directory
(see kv_offload.md, Instrumentation). This reads it back as the questions a
collapse raises: per window, how much of each request's prompt was
recomputed and why, how the recurrent hit stood against the attention hit,
how long turns took and whether their checkpoints were still on the GPU at
finish, what left the free queue and at what age, and what the pool and its
eviction front were made of.

usage: victim_offload_trace.py <trace.jsonl> [window_minutes=5]
"""

import collections
import json
import sys

CHUNK = 1568


def pct(values: list, q: float):
    """The q-quantile of `values`, or "-" when there are none."""
    if not values:
        return "-"
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))]


def main() -> None:
    """Print the windowed summary of the trace named on the command line."""
    path = sys.argv[1]
    window = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 300.0
    events, bad = [], 0
    for line in open(path, errors="replace"):
        try:
            events.append(json.loads(line))
        except Exception:
            bad += 1
    if not events:
        print("no events")
        return
    t0 = events[0]["t"]
    kinds = collections.Counter(e["ev"] for e in events)
    print(f"{len(events)} events ({bad} unparsable), {(events[-1]['t'] - t0) / 60:.1f} min, kinds: {dict(kinds)}")
    for e in events:
        if e["ev"] == "start":
            print("start:", {k: v for k, v in e.items() if k not in ("t", "ev")})
            break
    by_win = collections.defaultdict(list)
    for e in events:
        by_win[int((e["t"] - t0) // window)].append(e)

    print(
        "\n== lookups: n, full recompute, recomputed% of prompt tokens, divergent, kinds, ext>0, linked, prev-turn p50 for hits/misses"
    )
    for w in sorted(by_win):
        ls = [e for e in by_win[w] if e["ev"] == "lookup" and "computed" in e]
        if not ls:
            continue
        toks = sum(e.get("tokens") or 0 for e in ls)
        rec = sum(e.get("recomputed") or 0 for e in ls)
        full = sum(1 for e in ls if e["computed"] == 0)
        div = sum(1 for e in ls if e.get("divergent"))
        kind = collections.Counter(e.get("kind", "served" if (e.get("ext") or 0) > 0 else "?") for e in ls)
        ext = sum(1 for e in ls if (e.get("ext") or 0) > 0)
        linked = [e for e in ls if e.get("prev_turn_s") is not None]
        hits = [e["prev_turn_s"] for e in linked if (e.get("recomputed") or 0) <= 2 * CHUNK]
        misses = [e["prev_turn_s"] for e in linked if (e.get("recomputed") or 0) > 2 * CHUNK]
        fills = sum(1 for e in by_win[w] if e["ev"] == "fill")
        filled = sum(1 for e in ls if e.get("fill"))
        print(
            f"  {w * window / 60:5.0f}m n={len(ls):4d} full={full:3d} rec={100 * rec / max(toks, 1):5.1f}% div={div:3d} "
            f"fill={fills}/{filled} {dict(kind)} ext={ext} linked={len(linked)} prev_turn p50 hit={pct(hits, 0.5)} "
            f"miss={pct(misses, 0.5)} (n={len(misses)})"
        )

    print(
        "\n== att_hit - rec_g0 at lookup (0 = recurrent state at the attention boundary; none = nothing within the probe)"
    )
    for w in sorted(by_win):
        ls = [e for e in by_win[w] if e["ev"] == "lookup" and "rec_g0" in e]
        if ls:
            gap: collections.Counter[str] = collections.Counter(
                ("none" if e["rec_g0"] < 0 else str(e["att_hit"] - e["rec_g0"])) for e in ls
            )
            print(f"  {w * window / 60:5.0f}m {dict(sorted(gap.items()))}")

    print("\n== finishes: n, turn_s p50/p90/max, out tokens p50, no checkpoint on GPU (g0), held, stored")
    for w in sorted(by_win):
        fs = [e for e in by_win[w] if e["ev"] == "finish"]
        if not fs:
            continue
        turns = [e["turn_s"] for e in fs if e.get("turn_s") is not None]
        outs = [e["out"] for e in fs if e.get("out") is not None]
        gone = sum(1 for e in fs if e.get("chunks", 0) > 0 and not e.get("ck_g0"))
        print(
            f"  {w * window / 60:5.0f}m n={len(fs):4d} turn p50={pct(turns, 0.5)} p90={pct(turns, 0.9)} max={pct(turns, 1.0)} "
            f"out p50={pct(outs, 0.5)} gone_g0={gone} held={sum(1 for e in fs if e.get('held'))} stored={sum(e.get('stored', 0) for e in fs)}"
        )

    print("\n== evictions (unreused departures): by class, kept-block evictions, age p50 by class")
    for w in sorted(by_win):
        ev = [e for e in by_win[w] if e["ev"] == "evict"]
        if not ev:
            continue
        cls = collections.Counter(e["cls"] for e in ev)
        kept = sum(1 for e in ev if "req" in e)
        ages = collections.defaultdict(list)
        for e in ev:
            if "age" in e:
                ages[e["cls"]].append(e["age"])
        print(
            f"  {w * window / 60:5.0f}m n={len(ev):5d} {dict(cls)} kept_evicted={kept} "
            f"age p50 {{{', '.join(f'{c}: {pct(a, 0.5)}' for c, a in sorted(ages.items()))}}}"
        )

    print(
        "\n== pool samples: free min/p50, used max, known p50, kept p50, missing max, gone, preempts, moved totals; last q and band"
    )
    for w in sorted(by_win):
        ss = [e for e in by_win[w] if e["ev"] == "step"]
        if not ss:
            continue
        moved = collections.Counter()
        for e in ss:
            moved.update(e.get("moved", {}))
        last = ss[-1]
        print(
            f"  {w * window / 60:5.0f}m free min={min(e['free'] for e in ss)} p50={pct([e['free'] for e in ss], 0.5)} "
            f"used max={max(e['used'] for e in ss)} known p50={pct([e['known'] for e in ss], 0.5)} "
            f"kept p50={pct([e['kept'] for e in ss], 0.5)} missing max={max(e['missing'] for e in ss)} gone={last['gone']} "
            f"pre={sum(e['pre'] for e in ss)} moved={dict(moved)}\n        q={last['q']} band={last['band']}"
        )

    if any(e["ev"] == "tally" for e in events):
        print(
            "\n== tally, last per window: spill counters | lookups hit/miss_evicted/fill_accepted | asked depth p50/p99/run_max"
        )
        for w in sorted(by_win):
            ts = [e for e in by_win[w] if e["ev"] == "tally"]
            if not ts:
                continue
            t = ts[-1]
            s, lk, d = t["spill"], t.get("lookups") or {}, t.get("depth") or {}
            keys = ("spilled", "dropped", "held", "kept", "missing", "gone", "kept_stored", "filled")
            print(
                f"  {w * window / 60:5.0f}m "
                + " ".join(f"{k}={s.get(k)}" for k in keys)
                + f" | hit={lk.get('hit', 0)} miss_evicted={lk.get('miss_evicted', 0)} fill_accepted={lk.get('fill_accepted', 0)}"
                + f" | depth p50={d.get('p50')} p99={d.get('p99')} run_max={d.get('run_max')}"
            )

    stores = [e for e in events if e["ev"] == "store"]
    drops = [e for e in events if e["ev"] == "drop"]
    print(
        f"\n== stores: {len(stores)} jobs, {sum(e['blocks'] for e in stores)} blocks "
        f"({dict(collections.Counter(e['origin'] for e in stores))}); drops: {sum(e['dropped'] for e in drops)} blocks in {len(drops)} jobs"
    )
    other = collections.Counter(e["ev"] for e in events if e["ev"] in ("resume", "preempt"))
    if other:
        print("resume/preempt:", dict(other))


if __name__ == "__main__":
    main()
