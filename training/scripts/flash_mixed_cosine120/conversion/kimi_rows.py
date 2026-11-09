"""SFT rows from tools/kimi_steps.py output (stdlib only, no model).

python3 tools/kimi_rows.py --steps STEPS_DIR --holdout GAME[,GAME...] --max-completions 32 --max-length 192000 \
    --probe-per-game 2 --out DIR
python3 tools/kimi_rows.py --steps NAME=STEPS_DIR NAME=STEPS_DIR ... --exclude GAME[,GAME...] ...

A run's steps form chains: a step continues the chain when its prompt starts with the previous step's prompt +
completion token for token and its images extend the previous step's. A chain is cut at step boundaries into
rows of at most --max-completions trained completions; each row is carried by its last step and packs the earlier
ones (``packed_completions``: [prompt tokens, completion tokens], each checked inside the carrier's prompt).
Not trained: a step stopped by ``max_tokens`` (its completion is cut off) and a step whose own prompt + completion
exceeds --max-length; either ends its chain. Rows of the --holdout games go to DIR/validation.jsonl, the others to
DIR/rows.jsonl; DIR/validation-probe.jsonl holds --probe-per-game held-out rows per game (hash order), the fixed
set the trainer scores every few updates. Writes DIR/ROWS_READY.json.

Several sources: each --steps given as NAME=STEPS_DIR (one tools/kimi_steps.py output per recorded job); their runs
are pooled, every row carries its ``source_set`` NAME and the receipt reports each source. --exclude drops the
named games from both files (e.g. near copies of held-out games).

--mask-steps FILE (JSON, run_id -> step indices, from tools/stuck_turns.py mask): those completions get no loss. A
masked step inside a row stays in the carrier's prompt as context, it is just not listed as trained; a masked step at
the end of a row is cut, the row's last trained step carries it; a row left with no trained step is not written.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from grpo_rows import VOCAB, sha, tokens_sha

RENDER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json")


def chains(run, cap):
    """The run's chains of trainable steps, and the counts of what was left out."""
    lines = (run / "steps.jsonl").read_text().splitlines()
    counts, groups, group, previous = Counter(), [], [], None
    for line in lines:
        step = json.loads(line)
        inp, out = step["input_token_ids"], step["output_token_ids"]
        assert all(type(t) is int and 0 <= t < VOCAB for t in inp + out), "invalid token ids"
        reason = "max_tokens" if step["stop"] == "max_tokens" else "over_cap" if len(inp) + len(out) > cap else None
        if reason:
            counts[f"calls_{reason}"] += 1
            counts[f"tokens_{reason}"] += len(out)
            if group:
                groups.append(group)
            group, previous = [], None
            continue
        extends = (
            previous is not None
            and len(inp) >= previous["total"]
            and tokens_sha(inp[: previous["total"]]) == previous["sha"]
            and step["images"][: len(previous["images"])] == previous["images"]
        )
        if not extends and group:
            groups.append(group)
            group = []
        previous = dict(total=len(inp) + len(out), sha=tokens_sha(inp + out), images=step["images"])
        group.append(step)
    if group:
        groups.append(group)
    return groups, counts


def rows_of(run, game, groups, limit, mm_kwargs, media, masked=frozenset(), counts=None):
    """Rows of one run: each chain cut into pieces of at most ``limit`` completions, ``masked`` steps not trained."""
    rows = []
    for number, chain in enumerate(groups):
        for piece, start in enumerate(range(0, len(chain), limit)):
            members = chain[start : start + limit]
            if masked:
                kept = [m for m in members if m["step_index"] not in masked]
                if counts is not None:
                    counts["calls_masked"] += len(members) - len(kept)
                    counts["tokens_masked"] += sum(len(m["output_token_ids"]) for m in members if m["step_index"] in masked)
                    counts["rows_all_masked"] += not kept
                members = kept
                if not members:
                    continue
            carrier = members[-1]
            prompt = carrier["input_token_ids"]
            for m in members[:-1]:
                p = len(m["input_token_ids"])
                assert prompt[p : p + len(m["output_token_ids"])] == m["output_token_ids"], "packed completion mismatch"
            images = []
            for relative in carrier["images"]:
                path = (run / relative).resolve()
                media.setdefault(str(path), sha(path))
                images.append({"path": str(path), "sha256": media[str(path)]})
            row = {
                "input_token_ids": prompt,
                "output_token_ids": carrier["output_token_ids"],
                "images": images,
                "mm_processor_kwargs": mm_kwargs,
                "run_id": run.name,
                "game_id": game,
                "chain": number,
                "piece": piece,
                "steps": [m["step_index"] for m in members],
                "source": str(run / "steps.jsonl"),
            }
            if len(members) > 1:
                row["packed_completions"] = [[len(m["input_token_ids"]), len(m["output_token_ids"])] for m in members[:-1]]
            rows.append(row)
    return rows


def stats(rows):
    if not rows:
        return dict(rows=0)
    seq = sorted(len(r["input_token_ids"]) + len(r["output_token_ids"]) for r in rows)
    spans = [len(r.get("packed_completions", [])) + 1 for r in rows]
    completions = [len(r["output_token_ids"]) + sum(n for _, n in r.get("packed_completions", [])) for r in rows]
    pct = lambda q: seq[min(len(seq) - 1, int(q * len(seq)))]
    return dict(rows=len(rows), completions=sum(spans), completion_tokens=sum(completions),
                sequence_tokens=sum(seq), spans_max=max(spans), spans_mean=sum(spans) / len(rows),
                completion_tokens_per_row_mean=sum(completions) / len(rows),
                sequence=dict(p10=pct(.1), p50=pct(.5), p90=pct(.9), max=seq[-1]),
                games=len({r["game_id"] for r in rows}))


def render_files(readies):
    """sha256 of the files the rows' token ids depend on, for the trainer's rows_render_files; None when a source
    records no checkpoint. The sources must agree."""
    checkpoints = sorted({r.get("checkpoint") or "" for r in readies.values()})
    if "" in checkpoints:
        return None
    found = [{name: sha(Path(c) / name) for name in RENDER_FILES} for c in checkpoints]
    assert all(f == found[0] for f in found), f"sources rendered with different tokenizers: {checkpoints}"
    return found[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", nargs="+", required=True, help="STEPS_DIR, or NAME=STEPS_DIR per source")
    p.add_argument("--holdout", required=True)
    p.add_argument("--exclude", default="", help="games left out of both files")
    p.add_argument("--max-completions", type=int, required=True)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--probe-per-game", type=int, required=True)
    p.add_argument("--mask-steps", type=Path, help="JSON run_id -> step indices trained with no loss")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)  # accepted for the CPU job template; the build is serial
    a = p.parse_args()
    assert a.max_completions >= 1 and a.max_length > 0 and a.probe_per_game >= 1
    assert not a.out.exists(), f"{a.out} exists"
    named = all("=" in s for s in a.steps)
    assert named or len(a.steps) == 1, "several sources need NAME=STEPS_DIR each"
    sources = dict(s.split("=", 1) for s in a.steps) if named else {None: a.steps[0]}
    assert len(sources) == len(a.steps), "source names repeat"
    readies = {name: json.loads((Path(d) / "STEPS_READY.json").read_text()) for name, d in sources.items()}
    holdout = set(a.holdout.split(","))
    exclude = set(filter(None, a.exclude.split(",")))
    assert not holdout & exclude, "a game is both held out and excluded"
    games, where = {}, {}
    for name, ready in readies.items():
        for r in ready["runs"]:
            assert r["run_id"] not in games, f"run {r['run_id']} in two sources"
            games[r["run_id"]], where[r["run_id"]] = r["game_id"], name
    assert holdout <= set(games.values()), f"held-out games not in the steps: {sorted(holdout - set(games.values()))}"
    assert exclude <= set(games.values()), f"excluded games not in the steps: {sorted(exclude - set(games.values()))}"
    kwargs = {json.dumps(r["mm_processor_kwargs"], sort_keys=True) for r in readies.values()}
    assert len(kwargs) == 1, "sources rendered with different mm_processor_kwargs"
    mm_kwargs = next(iter(readies.values()))["mm_processor_kwargs"]
    mask = {run_id: frozenset(steps) for run_id, steps in json.loads(a.mask_steps.read_text()).items()} if a.mask_steps else {}
    assert set(mask) <= set(games), f"masked runs not in the steps: {sorted(set(mask) - set(games))}"
    assert not {games[r] for r in mask} & (holdout | exclude), "a masked run is in a held-out or excluded game"
    a.out.mkdir(parents=True)
    train, held, counts, media = [], [], Counter(), {}
    for run_id in sorted(games):
        if games[run_id] in exclude:
            counts["runs_excluded"] += 1
            continue
        run = Path(sources[where[run_id]]) / run_id
        groups, c = chains(run, a.max_length)
        counts.update(c)
        counts["chains"] += len(groups)
        rows = rows_of(run, games[run_id], groups, a.max_completions, mm_kwargs, media, mask.get(run_id, frozenset()), counts)
        if named:
            for r in rows:
                r["source_set"] = where[run_id]
        (held if games[run_id] in holdout else train).extend(rows)
    by_game = defaultdict(list)
    for i, r in enumerate(held):
        by_game[r["game_id"]].append(i)
    probe = []
    for game, indices in sorted(by_game.items()):
        ranked = sorted(indices, key=lambda i: hashlib.sha256(f"probe:{game}:{held[i]['run_id']}:{held[i]['chain']}:{held[i]['piece']}".encode()).hexdigest())
        probe.extend(held[i] for i in ranked[: a.probe_per_game])
    files = {}
    for name, rows in (("rows.jsonl", train), ("validation.jsonl", held), ("validation-probe.jsonl", probe)):
        with (a.out / name).open("w") as f:
            for r in rows:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")
        files[name] = dict(stats(rows), data_sha256=sha(a.out / name))
        if named:
            files[name]["by_source"] = {s: stats([r for r in rows if r["source_set"] == s]) for s in sources}
    if named:
        where_from = dict(sources={name: dict(steps=d, steps_ready_sha256=sha(Path(d) / "STEPS_READY.json"))
                                   for name, d in sources.items()}, exclude=sorted(exclude))
    else:
        where_from = dict(steps=sources[None], steps_ready_sha256=sha(Path(sources[None]) / "STEPS_READY.json"))
    receipt = dict(
        **where_from, holdout=sorted(holdout),
        train_games=sorted(set(games.values()) - holdout - exclude), max_completions=a.max_completions,
        max_length=a.max_length, probe_per_game=a.probe_per_game, mm_processor_kwargs=mm_kwargs,
        counts=dict(counts), media_files=len(media), files=files, render_files=render_files(readies),
        data_sha256=files["rows.jsonl"]["data_sha256"], rows=files["rows.jsonl"]["rows"],
        spans_max=max(f.get("spans_max", 0) for f in files.values()),
        completion_tokens_per_row_mean=files["rows.jsonl"]["completion_tokens_per_row_mean"])
    if a.mask_steps:
        receipt["mask_steps"] = dict(path=str(a.mask_steps), sha256=sha(a.mask_steps), runs=len(mask),
                                     steps=sum(map(len, mask.values())))
    (a.out / "ROWS_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
