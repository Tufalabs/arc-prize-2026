"""Fetch the fastest public human replay per LEVEL per official game and emit replay-agent data.

Data source: the ARC-AGI-3 human study public demo set
(https://arcprize.org/blog/arc-agi-3-human-dataset), which links the top 10
replays per environment. Session metadata and full step recordings are served
by arcprize.org's public API.

For each official game this script:
1. downloads all linked sessions (winning and not — non-winners still
   contribute per-level segments),
2. derives trace variants per recording — spliced (drop abandoned level
   attempts), death-spliced (drop only post-GAME_OVER attempts), frame-spliced
   (cut any RESET back to the frame it lands on, handling checkpoint-style
   resets), loop-cut (additionally cut walked-in-circles wandering) and raw —
   each replayed and verified against the local port at seed 0,
3. stitches the final trace per level from the fastest verified death-free
   human segment available for that level (segments start from the level's
   deterministic initial state, so they compose across players), retrying
   next-fastest segments when composite verification localizes a divergence,
4. checks the result wins using at most metadata baseline_actions[i] actions
   on every level i (the arc_agi scorecard==100 bar that CI enforces),
5. writes re_arc/dsl/human_replays/<base>.json with per-level action
   segments plus provenance.

Falls back to the fastest single fitting replay when stitching fails.
Officials' metadata.json (baseline_actions included) is never written.

Usage:
    python -m scripts.fetch_official_human_replays            # games missing traces
    python -m scripts.fetch_official_human_replays --all      # also report ft09/ls20/lp85
    python -m scripts.fetch_official_human_replays --game ar25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

BLOG_URL = "https://arcprize.org/blog/arc-agi-3-human-dataset"
SESSION_URL = "https://arcprize.org/api/sessions/{guid}"
RECORDING_URL = "https://arcprize.org/api/recordings/{game_id}/{guid}"
USER_AGENT = "Mozilla/5.0 (re-arc-3 official-replay fetcher)"
DEFAULT_CACHE_DIR = Path("/tmp/arc3_human_replays")
OUTPUT_DIR = REPO_ROOT / "re_arc" / "dsl" / "human_replays"
GAMES_WITH_EXISTING_AGENTS = {"ft09", "ls20", "lp85"}


def _http_get(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _cached_get(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    payload = _http_get(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(payload)
    return payload


def fetch_blog_guids(cache_dir: Path) -> list[str]:
    html = _cached_get(BLOG_URL, cache_dir / "blog.html").decode("utf-8", errors="replace")
    guids = sorted(set(re.findall(r"/replay/([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", html)))
    if not guids:
        raise RuntimeError("No replay GUIDs found on the blog page.")
    return guids


def fetch_session(guid: str, cache_dir: Path) -> dict:
    raw = _cached_get(SESSION_URL.format(guid=guid), cache_dir / "sessions" / f"{guid}.json")
    return json.loads(raw)


def fetch_recording(hosted_game_id: str, guid: str, cache_dir: Path) -> list[dict]:
    raw = _cached_get(
        RECORDING_URL.format(game_id=hosted_game_id, guid=guid),
        cache_dir / "recordings" / f"{hosted_game_id}.{guid}.jsonl",
    )
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def session_candidates(guids: list[str], cache_dir: Path) -> dict[str, list[dict]]:
    """Group all public sessions by game base.

    Winning sessions first (by total actions ascending); non-winning sessions
    after (by levels completed descending) — they cannot provide a whole-game
    trace but still contribute per-level segments for stitching.
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        sessions = list(pool.map(lambda g: (g, fetch_session(g, cache_dir)), guids))

    by_base: dict[str, list[dict]] = {}
    for guid, session in sessions:
        for environment in session.get("environments") or []:
            for run in environment.get("runs") or []:
                hosted_game_id = str(run.get("id") or environment.get("id") or "")
                base = hosted_game_id.split("-", 1)[0].lower()
                by_base.setdefault(base, []).append(
                    {
                        "guid": guid,
                        "hosted_game_id": hosted_game_id,
                        "won": str(run.get("state")) == "WIN" and bool(run.get("completed")),
                        "actions": int(run.get("actions") or 0),
                        "resets": int(run.get("resets") or 0),
                        "levels_completed": int(run.get("levels_completed") or 0),
                    }
                )
    for candidates in by_base.values():
        candidates.sort(key=lambda c: (not c["won"], c["actions"] if c["won"] else -c["levels_completed"]))
    return by_base


def _last_layer_signature(frame: object) -> str:
    import hashlib

    layers = frame if isinstance(frame, list) else []
    last = layers[-1] if layers else frame
    return hashlib.sha1(json.dumps(last, separators=(",", ":")).encode("utf-8")).hexdigest()


def _action_id(raw: object) -> int:
    """Recordings encode actions as ints ('id': 4) or names ('id': 'ACTION4'/'RESET')."""
    if isinstance(raw, str):
        name = raw.strip().upper()
        if name == "RESET":
            return 0
        if name.startswith("ACTION"):
            return int(name.removeprefix("ACTION"))
        raise ValueError(f"Unrecognized action id {raw!r}")
    return int(raw)


def extract_actions(entries: list[dict]) -> tuple[str, list[dict]]:
    """Initial frame signature + per-step records (action, state/levels/frame AFTER it).

    Skips the initial RESET entry (covered by env.reset()) and the trailing
    summary entry without an action_input id. Signatures hash the recording's
    own (hosted) frames, so frame comparisons are self-consistent regardless
    of any hosted/local rendering differences.
    """
    initial_signature = _last_layer_signature(entries[0]["data"].get("frame"))
    steps = []
    for entry in entries[1:]:
        data = entry["data"]
        action_input = data.get("action_input") or {}
        action_id = action_input.get("id")
        if action_id is None:
            continue
        payload = {k: int(v) for k, v in (action_input.get("data") or {}).items() if k in ("x", "y")}
        steps.append(
            {
                "action": (_action_id(action_id), payload),
                "state": str(data.get("state")),
                "levels": int(data.get("score") or 0),
                "signature": _last_layer_signature(data.get("frame")),
            }
        )
    return initial_signature, steps


def splice_failed_attempts(steps: list[dict], *, deaths_only: bool = False) -> list[tuple[int, dict[str, int]]]:
    """Drop abandoned level attempts, keeping the pass that completed each level.

    A RESET (action id 0) abandons the current attempt: drop everything kept
    since the last anchor (level transition or, with deaths_only, a kept
    voluntary reset). With deaths_only=True only resets that follow GAME_OVER
    trigger the drop; voluntary resets are kept verbatim as ordinary actions
    (some games' RESET has checkpoint semantics rather than restoring the
    level start). Truncate after WIN.
    """
    kept: list[tuple[int, dict[str, int]]] = []
    anchor = 0
    prev_levels = 0
    prev_state = ""
    for step in steps:
        action_id, payload = step["action"]
        if action_id == 0:
            if not deaths_only or prev_state == "GAME_OVER":
                del kept[anchor:]
            else:
                kept.append((action_id, payload))
                anchor = len(kept)
            prev_state = step["state"]
            continue
        kept.append((action_id, payload))
        if step["levels"] > prev_levels:
            prev_levels = step["levels"]
            anchor = len(kept)
        prev_state = step["state"]
        if step["state"] == "WIN":
            break
    return kept


def replay_locally(
    local_game_id: str,
    actions: list[tuple[int, dict[str, int]]],
    *,
    stop_at_game_over: bool = True,
    collect_signatures: bool = False,
) -> dict:
    """Replay actions on the local port; return final state and per-level counts.

    With stop_at_game_over=False (diagnostic mode for raw human episodes that
    contain deaths) the replay continues through GAME_OVER states, mirroring
    how the human played on through a RESET. With collect_signatures=True the
    result carries a per-step timeline of frame signatures (plus the initial
    frame) for frame-matching splices.
    """
    from re_arc import EnvSampler
    from re_arc.dsl import resolve_action, unpack_step_result
    from re_arc.dsl.core import frame_signature, state_name

    sampler = EnvSampler(augment=False, seed=0)
    env = sampler.make(game_id=local_game_id, seed=0)
    try:
        observation = env.reset()
        total_levels = int(observation.win_levels)
        prev_level = int(observation.levels_completed)
        actions_in_level = 0
        per_level: list[int] = []
        level_segments: list[list[tuple[int, dict[str, int]]]] = [[] for _ in range(total_levels)]
        used = 0
        saw_game_over = False
        initial_signature = frame_signature(observation) if collect_signatures else None
        timeline: list[dict] = []

        level_had_death = [False] * total_levels

        for action_id, payload in actions:
            state = state_name(observation)
            if state == "WIN" or (stop_at_game_over and state == "GAME_OVER"):
                break
            action = resolve_action(env, action_id)
            observation, _, done, _ = unpack_step_result(env.step(action, data=dict(payload)))
            used += 1
            actions_in_level += 1
            if prev_level < total_levels:
                level_segments[prev_level].append((action_id, dict(payload)))
            new_level = int(observation.levels_completed)
            if new_level > prev_level:
                for solved_idx in range(new_level - prev_level):
                    if len(per_level) >= total_levels:
                        break
                    per_level.append(actions_in_level if solved_idx == 0 else 1)
                actions_in_level = 0
            if state_name(observation) == "GAME_OVER":
                saw_game_over = True
                if prev_level < total_levels:
                    level_had_death[prev_level] = True
            prev_level = max(prev_level, new_level)
            if collect_signatures:
                timeline.append(
                    {"signature": frame_signature(observation), "state": state_name(observation), "level": prev_level}
                )
            if done and stop_at_game_over:
                break

        return {
            "final_state": state_name(observation),
            "total_levels": total_levels,
            "per_level": per_level,
            "level_segments": level_segments,
            "actions_used": used,
            "saw_game_over": saw_game_over,
            "level_had_death": level_had_death,
            "initial_signature": initial_signature,
            "timeline": timeline,
        }
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


def official_games() -> dict[str, dict]:
    """base -> {game_id, baseline_actions} for official-tagged local games."""
    from re_arc.dsl.precomputed_actions import metadata_index

    officials: dict[str, dict] = {}
    for metadata_path in sorted(set(metadata_index().values())):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "official" not in (metadata.get("tags") or []):
            continue
        game_id = str(metadata["game_id"])
        officials[game_id.split("-", 1)[0].lower()] = {
            "game_id": game_id,
            "baseline_actions": [int(v) for v in metadata.get("baseline_actions") or []],
        }
    return officials


def _over_budget(per_level: list[int], baseline: list[int]) -> list[int]:
    return [index for index, count in enumerate(per_level) if index < len(baseline) and count > baseline[index]]


def splice_resets_by_frame(
    initial_signature: str, steps: list[dict], *, dedupe_all: bool = False
) -> list[tuple[int, dict[str, int]]]:
    """Cut segments that return to an already-seen frame, by frame matching.

    Uses the recording's own frames: for an action landing on frame F in level
    L that some earlier kept step in level L already produced (or the
    episode-initial frame for level 0), drop the whole loop — everything after
    that step, the current action included. By default only RESET actions are
    treated this way (handles attempt-restarts and checkpoint-style resets);
    with dedupe_all=True every action is, additionally cutting walked-in-
    circles wandering. A RESET whose landing frame matches nothing (e.g. a
    layout reroll) is kept as an ordinary action. Truncates after WIN. Frame
    equality cannot see hidden state, so the result must be verified by a
    local replay.
    """
    kept: list[int] = []
    for index, step in enumerate(steps):
        action_id, _ = step["action"]
        if action_id == 0 or dedupe_all:
            target_signature = step["signature"]
            target_level = step["levels"]
            match = None
            if step["state"] == "NOT_FINISHED":
                for position, kept_index in enumerate(kept):
                    prior = steps[kept_index]
                    if (
                        prior["levels"] == target_level
                        and prior["state"] == "NOT_FINISHED"
                        and prior["signature"] == target_signature
                    ):
                        match = position
                        break
            if match is not None:
                del kept[match + 1 :]
                continue
            if step["state"] == "NOT_FINISHED" and target_level == 0 and target_signature == initial_signature:
                kept.clear()
                continue
        kept.append(index)
        if step["state"] == "WIN":
            break
    return [steps[i]["action"] for i in kept]


def _write_replay_file(base: str, local_game_id: str, source: dict, level_segments: list[list]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "game_id": local_game_id,
        "source": {
            "dataset": "ARC-AGI-3 human study public demo (arcprize.org/blog/arc-agi-3-human-dataset)",
            **source,
        },
        "levels": [[[action_id, payload_] for action_id, payload_ in segment] for segment in level_segments],
    }
    out_path = OUTPUT_DIR / f"{base}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _evaluate_candidate(
    local_game_id: str, baseline: list[int], candidate: dict, initial_signature: str, steps: list[dict]
) -> list[dict]:
    """Try trace variants for one recording, cheapest-to-replay first.

    Returns verdicts for every variant that was attempted; a verdict with
    ok=True is usable as a trace directly, one with final_state == WIN (even
    over budget) still contributes level segments to stitching.
    """
    raw = [step["action"] for step in steps]
    seen: list[list] = []
    variants: list[tuple[str, list]] = []
    for mode, actions in (
        ("spliced", splice_failed_attempts(steps)),
        ("death-spliced", splice_failed_attempts(steps, deaths_only=True)),
        ("frame-spliced", splice_resets_by_frame(initial_signature, steps)),
        ("loop-cut", splice_resets_by_frame(initial_signature, steps, dedupe_all=True)),
        ("raw", raw),
    ):
        if actions not in seen:
            seen.append(actions)
            variants.append((mode, actions))

    verdicts = []
    for mode, actions in variants:
        verdict = {
            "candidate": candidate,
            "mode": mode,
            "raw_actions": len(steps),
            "spliced_actions": len(actions),
            "final_state": None,
            "per_level": [],
            "level_segments": [],
            "level_had_death": [],
            "over_budget_levels": [],
            "ok": False,
        }
        try:
            # The raw episode is replayed through deaths exactly as recorded:
            # it is only usable as a trace when death-free, but its death-free
            # levels still provide verified segments for stitching.
            result = replay_locally(local_game_id, actions, stop_at_game_over=(mode != "raw"))
        except ValueError as exc:
            verdict["final_state"] = f"unreplayable ({exc})"
            verdicts.append(verdict)
            continue
        over_budget = _over_budget(result["per_level"], baseline)
        verdict.update(
            {
                "spliced_actions": result["actions_used"],
                "final_state": result["final_state"],
                "per_level": result["per_level"],
                "level_segments": result["level_segments"],
                "level_had_death": result["level_had_death"],
                "over_budget_levels": over_budget,
                "ok": (
                    result["final_state"] == "WIN"
                    and not result["saw_game_over"]
                    and len(result["per_level"]) == result["total_levels"]
                    and not over_budget
                ),
            }
        )
        verdicts.append(verdict)
    return verdicts


def process_game(base: str, official: dict, candidates: list[dict], cache_dir: Path, write: bool) -> dict:
    local_game_id = official["game_id"]
    baseline = official["baseline_actions"]
    attempts: list[dict] = []
    for candidate in candidates:
        try:
            recording = fetch_recording(candidate["hosted_game_id"], candidate["guid"], cache_dir)
        except Exception as exc:
            print(f"        [warn] {base}: failed to fetch {candidate['guid'][:8]}: {exc}")
            continue
        initial_signature, steps = extract_actions(recording)
        attempts.extend(_evaluate_candidate(local_game_id, baseline, candidate, initial_signature, steps))

    # Primary: per-level fastest human segments, stitched. Fallback: the
    # fastest single replay that fits (covers stitch verification failures).
    chosen, stitch_reason = _try_stitch(local_game_id, baseline, attempts)
    if chosen is None:
        single = [v for v in attempts if v["ok"]]
        if single:
            chosen = min(single, key=lambda v: v["spliced_actions"])

    if chosen is None:
        print(f"        [info] {base}: stitch failed: {stitch_reason}")
        # Disambiguate the failure for reporting: does the raw episode
        # (played through deaths, exactly as recorded) win on our port at all?
        for verdict in attempts:
            if verdict["final_state"] == "WIN" or str(verdict["final_state"]).startswith("unreplayable"):
                continue
            candidate = verdict["candidate"]
            recording = fetch_recording(candidate["hosted_game_id"], candidate["guid"], cache_dir)
            _, steps = extract_actions(recording)
            try:
                diag = replay_locally(local_game_id, [s["action"] for s in steps], stop_at_game_over=False)
            except ValueError as exc:
                verdict["diagnosis"] = f"full episode unreplayable: {exc}"
            else:
                verdict["diagnosis"] = (
                    f"port matches through deaths (full episode {diag['final_state']})"
                    if diag["final_state"] == "WIN"
                    else f"port divergence: full episode ends {diag['final_state']} "
                    f"after {diag['actions_used']} actions, levels={diag['per_level']}"
                )
        return {"base": base, "ok": False, "chosen": None, "attempts": attempts, "baseline": baseline}

    if write:
        if chosen["mode"] == "stitched":
            source = {
                "mode": "stitched",
                "note": "per-level composite of the fastest public human replay segments",
                "hosted_game_id": attempts[0]["candidate"]["hosted_game_id"],
                "per_level_sources": chosen["sources"],
                "spliced_actions": chosen["spliced_actions"],
            }
        else:
            candidate = chosen["candidate"]
            source = {
                "mode": chosen["mode"],
                "session_guid": candidate["guid"],
                "replay_url": f"https://arcprize.org/replay/{candidate['guid']}",
                "hosted_game_id": candidate["hosted_game_id"],
                "raw_actions": chosen["raw_actions"],
                "spliced_actions": chosen["spliced_actions"],
            }
        _write_replay_file(base, local_game_id, source, chosen["level_segments"])
    return {"base": base, "ok": True, "chosen": chosen, "attempts": attempts, "baseline": baseline}


def _try_stitch(local_game_id: str, baseline: list[int], attempts: list[dict]) -> tuple[dict | None, str]:
    """Compose each level from the fastest human segment available for it.

    A level segment is usable from ANY locally-verified replay (winning or
    not) as long as that replay completed the level without dying in it: every
    segment starts from the level's deterministic initial state, so segments
    from different replays compose. The composite is verified by a full local
    replay; when verification localizes a diverging level (frame equality
    cannot see hidden state), that level's next-fastest segment is tried.
    """
    if not attempts:
        return None, "no replayable candidates"
    total_levels = len(baseline)
    options: list[list[tuple]] = [[] for _ in range(total_levels)]
    for attempt in attempts:
        per_level = attempt["per_level"]
        for level in range(min(len(per_level), total_levels)):
            if len(attempt["level_segments"][level]) != per_level[level]:
                continue  # completed via a multi-level jump; segment/count mismatch
            if attempt["level_had_death"][level]:
                continue  # segment contains a GAME_OVER; unusable in a trace
            options[level].append(
                (per_level[level], attempt["level_segments"][level], attempt["candidate"], attempt["mode"])
            )

    for level in range(total_levels):
        unique, seen_segments = [], []
        for option in sorted(options[level], key=lambda o: o[0]):
            if option[1] in seen_segments:
                continue
            seen_segments.append(option[1])
            unique.append(option)
        fitting = [o for o in unique if o[0] <= baseline[level]]
        if not fitting:
            have = f"fastest available is {unique[0][0]}" if unique else "no death-free segment"
            return None, f"level {level}: no human segment within baseline {baseline[level]} ({have})"
        options[level] = fitting

    choice = [0] * total_levels
    for _ in range(40):
        expected = [options[level][choice[level]][0] for level in range(total_levels)]
        composite = [action for level in range(total_levels) for action in options[level][choice[level]][1]]
        try:
            result = replay_locally(local_game_id, composite)
        except ValueError:
            result = None
        if (
            result is not None
            and result["final_state"] == "WIN"
            and not result["saw_game_over"]
            and result["per_level"] == expected
        ):
            sources = [
                {
                    "level": level,
                    "actions": options[level][choice[level]][0],
                    "session_guid": options[level][choice[level]][2]["guid"],
                    "mode": options[level][choice[level]][3],
                }
                for level in range(total_levels)
            ]
            return {
                "candidate": {"guid": "stitched", "hosted_game_id": attempts[0]["candidate"]["hosted_game_id"]},
                "mode": "stitched",
                "raw_actions": sum({a["candidate"]["guid"]: a["raw_actions"] for a in attempts}.values()),
                "spliced_actions": len(composite),
                "final_state": result["final_state"],
                "per_level": result["per_level"],
                "level_segments": result["level_segments"],
                "over_budget_levels": [],
                "sources": sources,
                "ok": True,
            }, ""
        failing_level = 0
        if result is not None:
            failing_level = next(
                (
                    level
                    for level in range(total_levels)
                    if level >= len(result["per_level"]) or result["per_level"][level] != expected[level]
                ),
                0,
            )
        if choice[failing_level] + 1 >= len(options[failing_level]):
            return None, (
                f"level {failing_level}: all {len(options[failing_level])} fitting segments fail composite verification"
            )
        choice[failing_level] += 1
    return None, "stitch retry budget exhausted"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--game", action="append", default=[], help="Game base prefix (e.g. ar25). Repeatable.")
    parser.add_argument(
        "--all", action="store_true", help="Process all officials, reporting (without writing) games with agents."
    )
    args = parser.parse_args()

    officials = official_games()
    guids = fetch_blog_guids(args.cache_dir)
    print(f"{len(guids)} replay GUIDs from blog; {len(officials)} official games locally.")
    candidates_by_base = session_candidates(guids, args.cache_dir)

    requested = [g.lower() for g in args.game] or sorted(officials)
    failures = []
    for base in requested:
        if base not in officials:
            print(f"[skip] {base}: not an official game locally")
            continue
        has_agent = base in GAMES_WITH_EXISTING_AGENTS
        if has_agent and not (args.all or args.game):
            continue
        candidates = candidates_by_base.get(base) or []
        if not candidates:
            print(f"[fail] {base}: no WIN sessions among public replays")
            failures.append(base)
            continue
        write = not has_agent
        result = process_game(base, officials[base], candidates, args.cache_dir, write=write)
        chosen = result["chosen"]
        if result["ok"]:
            c = chosen
            margin = " ".join(
                f"{observed}/{expected}" for observed, expected in zip(c["per_level"], result["baseline"], strict=False)
            )
            label = "report-only" if not write else "written"
            print(
                f"[ok  ] {base}: {c['mode']} {c['candidate']['guid'][:8]} raw={c['raw_actions']} "
                f"trace={c['spliced_actions']} per-level obs/baseline: {margin} ({label})"
            )
        else:
            failures.append(base)
            print(f"[fail] {base}: no candidate passed; attempts:")
            for attempt in result["attempts"]:
                diagnosis = f" [{attempt['diagnosis']}]" if attempt.get("diagnosis") else ""
                print(
                    f"        {attempt['candidate']['guid'][:8]} {attempt['mode']} raw={attempt['raw_actions']} "
                    f"trace={attempt['spliced_actions']} state={attempt['final_state']} "
                    f"per_level={attempt['per_level']} over={attempt['over_budget_levels']}{diagnosis}"
                )
    if failures:
        print(f"FAILED games: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
