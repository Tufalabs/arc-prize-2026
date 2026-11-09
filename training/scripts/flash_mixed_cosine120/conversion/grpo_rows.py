"""Policy-gradient training rows from grpo_branch.py rollouts (stdlib only, no model).

python3 tools/grpo_rows.py --out DIR --alpha 0.1 [--alpha-actions 0.2] --max-length 192000 SHARD_DIR [SHARD_DIR ...]

A point's group is its finished rollouts (state won or gave_up, the branch reached, no replay failure, a
_SUCCESS trace); crashed, cancelled and replay-failed rollouts are left out. Per rollout W = level won and
L = the completion tokens of its live calls (the calls after the replayed prefix). The reward is
Arora & Zanette's length-penalised correctness (arXiv 2502.04463):

    R = W * (1 - alpha * sigmoid((L - mean) / std)),   mean, std (ddof 0) over the point's winning rollouts

(the sigmoid term is 0.5 when the point has fewer than two wins or std 0). Opt-in --alpha-actions (default 0:
the reward above) adds the same term for the level's actions K (the level's level_actions, the actions before
the branch included, a constant within a point): R = W * (1 - alpha sigma(z_L) - alpha_actions sigma(z_K)).
The advantage is
leave-one-out: A = R - mean of the group's other rewards, without std normalisation. Points with fewer than
two finished rollouts and rows with A = 0 are dropped.

Rows: a rollout's live calls form chains, a call continuing the chain when its prompt starts with the
previous call's prompt + completion token for token, its images extend the previous call's and the image
processor settings are equal. Each chain is one row carried by its last call; the earlier calls of the chain
are packed completions (``packed_completions``: [prompt tokens, completion tokens], each checked inside the
carrier's prompt). The replayed calls are context only.

Opt-in --validation-points MIXED:ALL_WON (default none) holds that many points out of training (mixed and
all-won outcomes, hash order) and writes DIR/validation.jsonl: per held-out point 4 rollouts whose whole rollout
is one row of at most --validation-max-length tokens, the 2 with the highest and the 2 with the lowest advantage;
their ``advantage`` is re-centred to sum to zero within the point (``advantage_raw`` keeps the original), so a
change common to the point cancels in an advantage-weighted score. The held-out points' rows never enter rows.jsonl. A call that fails the recording checks (exact
history, input token and payload reconstruction) ends its chain and is not trained; a call whose own
sequence exceeds --max-length is dropped with the rest of its chain. Every row carries the rollout's
``advantage``. Writes DIR/rows.jsonl, DIR/rollouts.jsonl (W, L, R, A per rollout) and DIR/ROWS_READY.json.
"""

import argparse
import hashlib
import json
import math
import os
import struct
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

VOCAB = 248320
FINISHED = ("won", "gave_up")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tokens_sha(tokens):
    return hashlib.sha256(struct.pack(f"<{len(tokens)}I", *tokens)).hexdigest()


def image_paths(messages):
    """Recorded relative image paths of a request, in prompt order."""
    paths = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if part.get("type") in ("video", "video_url", "input_video"):
                raise ValueError("video requests are unsupported")
            if part.get("type") not in ("image", "image_url", "input_image"):
                continue
            source = part.get("image_url", part)
            path = source.get("path", source.get("url")) if isinstance(source, dict) else source
            if not isinstance(path, str) or Path(path).is_absolute() or ":" in path:
                raise ValueError("expected a recorded relative image path")
            paths.append(path)
    return paths


def recorded(step):
    """The recording checks a trained call must pass (the ids and images are what the model saw)."""
    extra, v = step.get("extra") or {}, step.get("provider_validation") or {}
    return bool(
        extra.get("exact_history")
        and not extra.get("malformed")
        and v.get("input_token_reconstruction_status") == "ok"
        and v.get("original_payload_sha256")
        and v["original_payload_sha256"] == v.get("reconstructed_payload_sha256")
        and step["input_token_ids"]
        and step["output_token_ids"]
    )


def penalty(value, wins):
    """sigmoid((value - mean) / std) over the point's winning values; 0.5 without a spread."""
    if len(wins) < 2:
        return 0.5
    mean = sum(wins) / len(wins)
    std = math.sqrt(sum((x - mean) ** 2 for x in wins) / len(wins))
    return 0.5 if std == 0 else 1.0 / (1.0 + math.exp(-(value - mean) / std))


def reward(won, length, actions, wins, win_actions):
    """Length- (and opt-in action-) penalised correctness, with its two penalty terms."""
    if not won:
        return 0.0, None, None
    f, g = penalty(length, wins), penalty(actions, win_actions)
    return 1.0 - ALPHA * f - ALPHA_ACTIONS * g, f, g


def live_steps(item):
    """(lengths, live call metadata) of one finished rollout, read from its trace.

    A rollout's trace can lack its last counted live call (seen on ~1 in 400 won rollouts): its recorded
    live calls are then the steps after the replayed prefix, and the missing ones are counted.
    """
    trace, live = Path(item["trace"]), item["live_calls"]
    lines = (trace / "steps.jsonl").read_text().splitlines()
    recorded_live = len(lines) - item["steps_replayed"]
    assert 0 < recorded_live <= live, f"{trace}: {len(lines)} steps for {item['steps_replayed']} + {live}"
    steps = [json.loads(line) for line in lines[-recorded_live:]]
    indices = [s["step_index"] for s in steps]
    assert indices == sorted(indices) and len(set(indices)) == recorded_live, f"{trace}: step order"
    return dict(item, recorded_live=recorded_live, missing_live=live - recorded_live,
                completion_tokens=sum(len(s["output_token_ids"]) for s in steps))


def chains(item):
    """The rollout's rows (one per chain of live calls) and its counts."""
    trace, live = Path(item["trace"]), item["recorded_live"]
    engine, cap = item["engine_kwargs"], item["max_length"]
    lines = (trace / "steps.jsonl").read_text().splitlines()[-live:]
    counts = Counter()
    groups, group, previous = [], [], None
    for line in lines:
        step = json.loads(line)
        inp, out = step["input_token_ids"], step["output_token_ids"]
        assert all(type(t) is int and 0 <= t < VOCAB for t in inp + out), "invalid token ids"
        if not recorded(step):
            counts["calls_unrecorded"] += 1
            counts["tokens_unrecorded"] += len(out)
            if group:
                groups.append(group)
            group, previous = [], None
            continue
        if len(inp) + len(out) > cap:
            counts["calls_over_cap"] += 1
            counts["tokens_over_cap"] += len(out)
            if group:
                groups.append(group)
            group, previous = [], None
            continue
        kwargs = dict(engine, **(step["provider_validation"].get("request_args", {}).get("mm_processor_kwargs") or {}))
        paths = image_paths(step["messages"])
        extends = (
            previous is not None
            and len(inp) >= previous["total"]
            and tokens_sha(inp[: previous["total"]]) == previous["sha"]
            and paths[: len(previous["images"])] == previous["images"]
            and kwargs == previous["kwargs"]
        )
        if not extends and group:
            groups.append(group)
            group = []
        previous = dict(total=len(inp) + len(out), sha=tokens_sha(inp + out), images=paths, kwargs=kwargs)
        group.append(dict(step=step, prompt=len(inp), completion=len(out), kwargs=kwargs, paths=paths))
    if group:
        groups.append(group)
    rows, media = [], {}
    for number, members in enumerate(groups):
        carrier = members[-1]
        step = carrier["step"]
        prompt = step["input_token_ids"]
        for m in members[:-1]:
            assert prompt[m["prompt"] : m["prompt"] + m["completion"]] == m["step"]["output_token_ids"], (
                "packed completion mismatch"
            )
        images = []
        for relative in carrier["paths"]:
            path = (trace / relative).resolve()
            assert path.is_relative_to(trace.resolve()), "media path escapes the trace"
            media.setdefault(str(path), sha(path))
            images.append({"path": str(path), "sha256": media[str(path)]})
        row = {
            "input_token_ids": prompt,
            "output_token_ids": step["output_token_ids"],
            "images": images,
            "mm_processor_kwargs": carrier["kwargs"],
            "advantage": item["advantage"],
            "group": item["point"],
            "rollout": item["name"],
            "run_id": item["run_id"],
            "chain": number,
            "steps": [m["step"]["step_index"] for m in members],
            "level_won": item["level_won"],
            "reward": item["reward"],
            "game_id": item["game_id"],
            "level": item["level"],
            "source": str(trace / "steps.jsonl"),
            "request_payload_sha256": step["provider_validation"]["original_payload_sha256"],
        }
        if len(members) > 1:
            row["packed_completions"] = [[m["prompt"], m["completion"]] for m in members[:-1]]
        rows.append(json.dumps(row, separators=(",", ":")))
        counts["calls_trained"] += len(members)
        counts["tokens_trained"] += sum(m["completion"] for m in members)
    return rows, counts, [
        dict(sequence=m[-1]["prompt"] + m[-1]["completion"], spans=len(m), completion=sum(x["completion"] for x in m))
        for m in groups
    ]


def validation_split(held, kept, want, max_length):
    """The held-out points (in pool order, by outcome) and their 4 validation rows each, as JSON lines."""
    by_point = defaultdict(list)
    for m in kept:
        by_point[m["point"]].append(m)
    chosen, lines = set(), []
    ranked = sorted(held, key=lambda pt: hashlib.sha256(f"validation:{pt}".encode()).hexdigest())
    for label, test in (("mixed", lambda w: w == {True, False}), ("all_won", lambda w: w == {True})):
        taken = 0
        for point in ranked:
            if taken == want[label] or not test({m["level_won"] for m in by_point[point]}):
                continue
            # A rollout qualifies when it is a single row of at most max_length tokens.
            single = [(item, rows[0], s[0]) for item, rows, c, s in held[point]
                      if len(rows) == 1 and s[0]["sequence"] <= max_length]
            single.sort(key=lambda e: e[0]["advantage"])
            if len(single) < 4 or (label == "mixed" and (single[1][0]["advantage"] >= 0 or single[-2][0]["advantage"] <= 0)):
                continue
            picked = single[:2] + single[-2:]
            mean = sum(e[0]["advantage"] for e in picked) / 4
            for item, line, _ in picked:
                row = json.loads(line)
                row["advantage_raw"], row["advantage"] = row["advantage"], row["advantage"] - mean
                lines.append(json.dumps(row, separators=(",", ":")))
            chosen.add(point)
            taken += 1
        assert taken == want[label], f"only {taken} of {want[label]} {label} validation points qualify"
    return chosen, lines


def main():
    global ALPHA, ALPHA_ACTIONS
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--alpha-actions", type=float, default=0.0)
    p.add_argument("--validation-points", default=None)
    p.add_argument("--validation-max-length", type=int, default=64000)
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--points", type=Path, required=True)
    p.add_argument("shards", type=Path, nargs="+")
    a = p.parse_args()
    ALPHA, ALPHA_ACTIONS = a.alpha, a.alpha_actions
    assert ALPHA >= 0 and ALPHA_ACTIONS >= 0 and ALPHA + ALPHA_ACTIONS < 1 and a.max_length > 0
    assert not a.out.exists(), f"{a.out} exists"
    points = {json.loads(line)["point"]: json.loads(line) for line in a.points.read_text().splitlines()}
    items, excluded, engine = [], Counter(), None
    for shard in a.shards:
        config = json.loads((shard / "run_config.json").read_text())
        kwargs = config["engine_args"].get("mm_processor_kwargs") or {}
        assert engine in (None, kwargs), "shards differ in image processor settings"
        engine = kwargs
        model = config["engine_args"]["model"]
        for row in json.loads((shard / "branch-report.json").read_text())["rows"]:
            trace = shard / "traces" / row["run_id"]
            reason = (
                "not_finished" if row["state"] not in FINISHED
                else "not_reached" if not row["reached_branch"]
                else "point_failed" if row["point_failed"]
                else "no_success" if not (trace / "_SUCCESS").is_file()
                else "no_live_calls" if not row["live_calls"]
                else None
            )
            if reason:
                excluded[reason] += 1
                continue
            assert json.loads((trace / "trace.json").read_text())["model_id"] == model
            point = points[row["point"]]
            items.append(dict(
                trace=str(trace), name=row["name"], run_id=row["run_id"], point=row["point"],
                live_calls=row["live_calls"], steps_replayed=row["steps_replayed"],
                level_won=bool(row["level_won"]), level_actions=row["level_actions"],
                state=row["state"], game_id=point["game_id"],
                level=point["level"], category=point["category"], engine_kwargs=engine,
                max_length=a.max_length))
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    with Pool(workers) as pool:
        items = pool.map(live_steps, items, chunksize=4)
        groups = defaultdict(list)
        for item in items:
            groups[item["point"]].append(item)
        kept, outcome = [], Counter()
        for point, members in sorted(groups.items()):
            if len(members) < 2:
                outcome["under_two_finished"] += 1
                continue
            wins = [m["completion_tokens"] for m in members if m["level_won"]]
            win_actions = [m["level_actions"] for m in members if m["level_won"]]
            for m in members:
                m["reward"], m["length_term"], m["action_term"] = reward(
                    m["level_won"], m["completion_tokens"], m["level_actions"], wins, win_actions)
            total = sum(m["reward"] for m in members)
            for m in members:
                m["advantage"] = m["reward"] - (total - m["reward"]) / (len(members) - 1)
            won = len(wins)
            outcome["mixed" if 0 < won < len(members) else "all_won" if won else "all_lost"] += 1
            kept.extend(members)
        trained = [m for m in kept if m["advantage"] != 0]
        zero_advantage = len(kept) - len(trained)
        # Opt-in validation: a candidate pool of points (hash order) whose rows are held back until chosen.
        want, pool_points = {"mixed": 0, "all_won": 0}, set()
        if a.validation_points:
            want["mixed"], want["all_won"] = map(int, a.validation_points.split(":"))
            kind = {}
            for m in kept:
                kind.setdefault(m["point"], set()).add(m["level_won"])
            ranked = sorted(kind, key=lambda pt: hashlib.sha256(f"validation:{pt}".encode()).hexdigest())
            for label, test in (("mixed", lambda w: w == {True, False}), ("all_won", lambda w: w == {True})):
                pool_points |= set([pt for pt in ranked if test(kind[pt])][: 4 * want[label]])
        held = defaultdict(list)
        a.out.mkdir(parents=True)
        counts, shapes = Counter(), []
        # Streamed in rollout order: the rows never sit in memory together (the candidate pool's are held).
        with (a.out / "rows.jsonl").open("w") as f:
            for item, (rows, c, s) in zip(trained, pool.imap(chains, trained, chunksize=2)):
                if item["point"] in pool_points:
                    held[item["point"]].append((item, rows, c, s))
                    continue
                counts.update(c)
                shapes.extend(s)
                for line in rows:
                    f.write(line + "\n")
            chosen, validation = validation_split(held, kept, want, a.validation_max_length)
            for point, entries in held.items():
                if point in chosen:
                    continue
                for item, rows, c, s in entries:
                    counts.update(c)
                    shapes.extend(s)
                    for line in rows:
                        f.write(line + "\n")
        if a.validation_points:
            with (a.out / "validation.jsonl").open("w") as f:
                for line in validation:
                    f.write(line + "\n")
        trained = [m for m in trained if m["point"] not in chosen]
    with (a.out / "rollouts.jsonl").open("w") as f:
        for m in kept:
            f.write(json.dumps({k: m[k] for k in (
                "name", "run_id", "point", "category", "game_id", "level", "state", "level_won",
                "completion_tokens", "level_actions", "live_calls", "missing_live", "reward", "length_term",
                "action_term", "advantage")}) + "\n")
    sequence = sorted(s["sequence"] for s in shapes)
    pct = lambda xs, q: xs[min(len(xs) - 1, int(q * len(xs)))]
    advantages = sorted(m["advantage"] for m in trained)
    receipt = dict(
        alpha=ALPHA, alpha_actions=ALPHA_ACTIONS, max_length=a.max_length, engine_mm_processor_kwargs=engine, shards=[str(s) for s in a.shards],
        points=str(a.points), points_sha256=sha(a.points), rollouts_excluded=dict(excluded),
        rollouts_finished=len(items), points_by_outcome=dict(outcome),
        rollouts_missing_live_calls=sum(m["missing_live"] > 0 for m in items),
        live_calls_missing=sum(m["missing_live"] for m in items), rollouts_in_groups=len(kept),
        rollouts_zero_advantage=zero_advantage, rollouts_trained=len(trained),
        rows=len(shapes), counts=dict(counts),
        validation=None if not a.validation_points else dict(
            points=sorted(chosen), rows=len(validation), max_length=a.validation_max_length,
            requested=want, data_sha256=sha(a.out / "validation.jsonl"),
            rollouts_held_out=sum(m["point"] in chosen for m in kept)),
        spans_max=max(s["spans"] for s in shapes), spans_mean=sum(s["spans"] for s in shapes) / len(shapes),
        completion_tokens_per_row_mean=counts["tokens_trained"] / len(shapes),
        sequence_tokens=dict(mean=round(sum(sequence) / len(sequence)), p10=pct(sequence, .1),
                             p50=pct(sequence, .5), p90=pct(sequence, .9), max=sequence[-1]),
        advantage=dict(min=advantages[0], p50=pct(advantages, .5), max=advantages[-1],
                       positive=sum(x > 0 for x in advantages), negative=sum(x < 0 for x in advantages)),
        data_sha256=sha(a.out / "rows.jsonl"), rollouts_sha256=sha(a.out / "rollouts.jsonl"))
    (a.out / "ROWS_READY.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


ALPHA = ALPHA_ACTIONS = None


if __name__ == "__main__":
    main()
