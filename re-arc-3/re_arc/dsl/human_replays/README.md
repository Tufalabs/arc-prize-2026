# Human replay data

One JSON per game: a winning playthrough split into per-level action segments,
played back verbatim by `HumanReplayAgent`
(re_arc/dsl/agents/_human_replay.py). `make precompute_dsl_actions` turns them
into the canonical traces under re_arc/dsl/precomputed_actions/.

Two sources:

- **Official games** — segments assembled from the fastest public human
  replays of ARC-AGI-3 (the human-study public demo set,
  https://arcprize.org/blog/arc-agi-3-human-dataset). Each file's `source`
  block records the session GUID(s), the hosted game version, and the
  extraction mode (`spliced` / `death-spliced` / `frame-spliced` / `loop-cut`
  / `raw` for a single session, `stitched` with `per_level_sources` when
  levels come from different sessions). Regenerate or extend with:

  ```bash
  python scripts/fetch_official_human_replays.py [--game <base>] [--all]
  ```

- **In-house games** — record one yourself in the player: `make play`, then
  **Record** → solve every level → **"Use recording for DSL agent"**, which
  writes `<base>.json` here and creates the game's agent module.

Every file should be verified by a local replay at seed 0: it must reach WIN
without any GAME_OVER and use at most `baseline_actions[i]` actions on every
level `i` (the arc_agi scorecard then reports exactly 100.0, which CI asserts).

The official games' files are not shipped here: `scripts/fetch_official_human_replays.py` rebuilds them.
