"""Split a duck run's uncached prompt tokens into what a KV tier can and cannot save.

The engine's ``prompt_tokens - prompt_tokens_cached`` mixes four things, and
only the last is what a host KV tier addresses:

- first-time content: the tool result or observation, computed once ever;
- block tail: a block caches only once full, so the partial tail of the
  previous prompt is recomputed every call — up to one block, here 1568
  tokens, and no cache size changes it;
- compaction: a prompt that shrank is a rewritten history, new content;
- eviction: previously computed, lost from both tiers, computed again.

Per game, calls are ordered by time. With nothing evicted, the cacheable
prefix at a call is the full-block prefix of the previous prompt plus the
previous completion (decode fills blocks too), so anything uncached beyond
``prompt - that`` was evicted. Reported as totals and as a per-minute series,
because the average hides the troughs and the troughs are what a run under
pressure is judged on.

    uv run python packages/ullm/tools/victim_offload_decompose.py <job_dir> [<job_dir> ...]
"""

from __future__ import annotations

import collections
import json
import os
import sys

# Tokens per KV block for the model the A/B runs. The attention page is padded
# up to the Mamba page, which is what makes it this large.
BLOCK_TOKENS = 1568

# A minute in which more than this share of the prompt was eviction recompute
# counts as a trough.
TROUGH_FRACTION = 0.15


def llm_calls(game_dir: str) -> list[tuple[float, int, int, int]]:
    """One (time, uncached, cached, completion) per real LLM call, time-ordered."""
    out: list[tuple[float, int, int, int]] = []
    with open(os.path.join(game_dir, "spans.jsonl")) as f:
        for line in f:
            event = json.loads(line)
            if event.get("kind") != "llm/response":
                continue
            values = event["values"]
            if values.get("synthetic"):
                continue
            out.append(
                (
                    event["t"],
                    values["uncached_input_tokens"],
                    values["cached_input_tokens"],
                    values["completion_tokens"],
                )
            )
    out.sort()
    return out


def decompose(job_dir: str) -> list[tuple[float, int, int, int, int, int, int]]:
    """Per call: (time, prompt, uncached, first_time, block_tail, compaction, eviction)."""
    rows: list[tuple[float, int, int, int, int, int, int]] = []
    games_dir = os.path.join(job_dir, "games")
    for game in os.listdir(games_dir):
        prev_prompt: int | None = None
        prev_completion = 0
        for t, uncached, cached, completion in llm_calls(os.path.join(games_dir, game)):
            prompt = uncached + cached
            if prev_prompt is None:
                first, tail, compaction, eviction = uncached, 0, 0, 0
            elif prompt < prev_prompt:
                first, tail, compaction, eviction = 0, 0, uncached, 0
            else:
                cacheable = ((prev_prompt + prev_completion) // BLOCK_TOKENS) * BLOCK_TOKENS
                expected = max(0, prompt - cacheable)
                new_content = max(0, prompt - (prev_prompt + prev_completion))
                tail = max(0, expected - new_content)
                first = max(0, min(uncached, expected)) - tail
                compaction = 0
                eviction = max(0, uncached - expected)
            rows.append((t, prompt, uncached, first, tail, compaction, eviction))
            prev_prompt, prev_completion = prompt, completion
    rows.sort()
    return rows


def report(job_dir: str) -> None:
    """Print totals and the per-minute eviction series for one run."""
    rows = decompose(job_dir)
    if not rows:
        print(f"{job_dir}: no LLM calls found")
        return
    t0 = rows[0][0]
    prompt = sum(r[1] for r in rows)
    parts = {
        name: sum(r[i] for r in rows)
        for i, name in ((3, "first-time"), (4, "block-tail"), (5, "compaction"), (6, "eviction"))
    }
    uncached = sum(r[2] for r in rows)
    print(
        f"{os.path.basename(job_dir)}: {len(rows)} calls, {prompt / 1e6:.2f}M prompt tokens, {100 * uncached / prompt:.1f}% uncached"
    )
    for name, tokens in parts.items():
        print(f"  {name:11} {100 * tokens / prompt:5.1f}%  ({tokens / 1e6:.2f}M)")

    minutes: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0])
    for t, p, _, _, _, _, ev in rows:
        b = int((t - t0) // 60)
        minutes[b][0] += p
        minutes[b][1] += ev
    fractions = [minutes[b][1] / minutes[b][0] if minutes[b][0] else 0.0 for b in range(max(minutes) + 1)]
    ranked = sorted(fractions, reverse=True)
    n = len(ranked)
    print(
        f"  per-minute eviction: troughs(>{100 * TROUGH_FRACTION:.0f}%)={sum(1 for x in ranked if x > TROUGH_FRACTION)}/{n}"
        f"  worst={100 * ranked[0]:.0f}%  p90={100 * ranked[n // 10]:.0f}%  median={100 * ranked[n // 2]:.1f}%"
    )
    print("  " + " ".join(f"{100 * x:3.0f}" for x in fractions))


def main() -> None:
    """Report each job directory given on the command line."""
    for job_dir in sys.argv[1:]:
        report(job_dir)


if __name__ == "__main__":
    main()
