"""Read one engine-profile window: where a step's time goes.

``EngineProfile`` (``ullm.backends.vllm_factory``) writes a torch-profiler
trace per window under the job directory, in ``engine_profile/``. This
reports, over one window: the steps vLLM tagged and their prefill/decode mix,
GPU busy against GPU idle across the span (idle inside the span is the GPU
waiting on the host), GPU time by kernel family, host-device copies, and --
the comparable unit across windows -- kernel time per step for decode-only
and for mixed steps separately.

A kernel is attributed to a step through its launch: every GPU kernel carries
the correlation id of the CPU-side call that queued it (one call can queue a
whole CUDA graph), and that call lies inside exactly one of vLLM's step marks.
A step owns everything launched from its mark to the next one's, so sampling
and a drafter's passes -- which run after vLLM's mark closes -- count toward the
step they serve; only kernels launched before the first mark are unattributable. Families are regexes over kernel names; check them
against ``--names`` before trusting a split on a new model or backend.

    uv run python packages/ullm/tools/engine_profile_read.py <trace.json.gz> [--top N] [--names]
"""

import argparse
import collections
import gzip
import json
import re

FAMILIES = [
    ("copy DtoH", r"memcpy dtoh"),
    ("copy HtoD", r"memcpy htod"),
    ("copy other", r"memcpy|memset"),
    # FP4 expert GEMMs (TRT-LLM names them by dtype: E2m1 is FP4), Marlin,
    # and the MoE runner's routing/permute/finalize kernels.
    ("moe", r"e2m1|moe|marlin|trtllm|expert|routing|fused_experts|grouped|permute"),
    ("gdn / linear attn", r"gdn|delta|fla_|chunk_gated|recurrent|causal_conv|conv1d|mamba|ssm|short_conv"),
    ("attention (qsa)", r"qsa|flash|attn|indexer|mla|sparse|paged|kv_cache|reshape_and_cache"),
    ("hyper-connections", r"_hc_"),
    (
        "sampling",
        r"sampl|topk|top_k|argmax|multinomial|gumbel|logit|softmax|sort|penalt|temperature|draft_tokens|rejected",
    ),
    ("ple", r"_ple_|ngram"),
    (
        "input prep",
        r"slot_mapping|block_tables|prepare_|positions|post_update|zero_kv|apply_write|bincount|scatter_num|fused_post_conv",
    ),
    # bf16 dense GEMMs: cuBLASLt (nvjet, splitKreduce) and cutlass bf16.
    ("dense bf16 gemm", r"nvjet|cublas|gemm|gemv|sgemm|ampere_|sm80|matmul|splitkreduce"),
    ("nccl", r"nccl"),
    ("norm", r"norm"),
    ("triton / elementwise", r"triton_|elementwise|vectorized|reduce|index|cat|copy|fill|arange|where|cumsum"),
]
STEP = re.compile(r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)")


def family(name: str) -> str:
    """The kernel family a name falls in: the first match in FAMILIES, else "other"."""
    low = name.lower()
    for fam, pat in FAMILIES:
        if re.search(pat, low):
            return fam
    return "other"


def union_ms(intervals: list[tuple[float, float]]) -> float:
    """Total length of the union of [start, end) intervals, in milliseconds."""
    total = 0.0
    merged: list[float] = []  # [start, end] of the run being merged
    for s, e in sorted(intervals):
        if not merged or s > merged[1]:
            if merged:
                total += merged[1] - merged[0]
            merged = [s, e]
        else:
            merged[1] = max(merged[1], e)
    if merged:
        total += merged[1] - merged[0]
    return total / 1000.0


def main() -> None:
    """Read one trace and print the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--names", action="store_true", help="print every distinct kernel name with its family")
    a = ap.parse_args()
    opener = gzip.open if a.trace.endswith(".gz") else open
    with opener(a.trace, "rt") as f:
        events = json.load(f)["traceEvents"]
    cats = collections.Counter(e.get("cat", "?") for e in events)
    print("event categories:", dict(cats.most_common(12)))

    steps = []
    for e in events:
        if e.get("cat") in ("user_annotation", "gpu_user_annotation") and e.get("ph") == "X":
            m = STEP.search(e.get("name", ""))
            if m and e.get("cat") == "user_annotation":
                steps.append((e["ts"], e["ts"] + e["dur"], *map(int, m.groups())))
    gpu = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and e.get("ph") == "X"]
    if not gpu:
        print("no GPU events in the trace")
        return
    g0 = min(e["ts"] for e in gpu)
    g1 = max(e["ts"] + e["dur"] for e in gpu)
    span_ms = (g1 - g0) / 1000.0
    busy_ms = union_ms([(e["ts"], e["ts"] + e["dur"]) for e in gpu])

    print(f"\nsteps tagged: {len(steps)}")
    if steps:
        durs = sorted((s[1] - s[0]) / 1000.0 for s in steps)
        decode_only = [s for s in steps if s[2] == 0]
        print(f"  host-side step time ms: median {durs[len(durs) // 2]:.1f}, p90 {durs[int(len(durs) * 0.9)]:.1f}")
        print(f"  decode-only steps: {len(decode_only)}; mixed/prefill: {len(steps) - len(decode_only)}")
        gen = sorted(s[4] for s in steps)
        print(f"  decode requests per step: median {gen[len(gen) // 2]}, min {gen[0]}, max {gen[-1]}")
        if len(steps) > 1:
            wall = (steps[-1][1] - steps[0][0]) / 1000.0
            print(f"  wall per step over the window: {wall / len(steps):.1f} ms")
    print(
        f"\nGPU span {span_ms:.0f} ms, busy {busy_ms:.0f} ms ({100 * busy_ms / span_ms:.0f}%), "
        f"idle {span_ms - busy_ms:.0f} ms ({100 * (1 - busy_ms / span_ms):.0f}%)  <- GPU waiting on the host"
    )

    by_fam = collections.Counter()
    by_name = collections.Counter()
    count = collections.Counter()
    for e in gpu:
        name = e.get("name", "?")
        by_fam[family(name)] += e["dur"]
        by_name[name] += e["dur"]
        count[name] += 1
    total = sum(by_fam.values())
    print(f"\nGPU time by family (sum of kernel time {total / 1000:.0f} ms; overlaps across streams count twice):")
    for fam, d in by_fam.most_common():
        print(f"  {fam:22} {d / 1000:9.1f} ms  {100 * d / total:5.1f}%")
    # Per step type. Each GPU kernel carries the correlation id of the CPU-side
    # launch that queued it (one launch can queue a whole CUDA graph), and that
    # launch lies inside exactly one CPU step mark, which says whether the
    # step was decode-only. Decode-only steps are the comparable unit across
    # windows; mixed steps carry prefill.
    import bisect

    launch_ts = {}
    for e in events:
        if e.get("cat") in ("cuda_runtime", "cuda_driver"):
            corr = (e.get("args") or {}).get("correlation")
            if corr is not None:
                launch_ts[corr] = e["ts"]
    marks = sorted((st[0], st[1], st[2] == 0, st[4]) for st in steps)
    if marks:
        starts = [m[0] for m in marks]
        per_step = collections.defaultdict(collections.Counter)
        unassigned = 0.0
        for e in gpu:
            corr = (e.get("args") or {}).get("correlation")
            t = launch_ts.get(corr)
            i = bisect.bisect_right(starts, t) - 1 if t is not None else -1
            # A step owns everything launched from its start to the next
            # step's: sampling and a drafter's passes run after vLLM's mark
            # closes, and belong to the step they serve.
            if i >= 0 and (i + 1 == len(marks) or t < marks[i + 1][0]):
                per_step[i][family(e.get("name", "?"))] += e["dur"]
            else:
                unassigned += e["dur"]
        print(f"\nkernel time not attributable to a step: {unassigned / 1000:.0f} ms")
        for kind, label in ((True, "decode-only steps"), (False, "mixed / prefill steps")):
            idx = [i for i, m in enumerate(marks) if m[2] == kind]
            if not idx:
                continue
            fams = collections.Counter()
            for i in idx:
                fams.update(per_step[i])
            tot = sum(fams.values()) / 1000.0
            r = sorted(marks[i][3] for i in idx)
            n = len(idx)
            print(f"\n{label}: {n}, decode requests median {r[len(r) // 2]}, kernel time {tot / n:.1f} ms/step")
            for fam, d in fams.most_common():
                print(f"  {fam:22} {d / 1000 / n:7.2f} ms/step  {100 * d / 1000 / tot:5.1f}%")
    print(f"\ntop {a.top} kernels:")
    for name, d in by_name.most_common(a.top):
        print(f"  {d / 1000:8.1f} ms  {count[name]:6}x  [{family(name)}]  {name[:110]}")
    if a.names:
        print("\nall kernel names:")
        for name in sorted(by_name):
            print(f"  [{family(name)}] {name[:160]}")


if __name__ == "__main__":
    main()
