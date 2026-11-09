# Environment Authoring Guide

Use this workflow when you add a new environment.

## 1. Add the Environment Files

Create a folder for the game slug and version:

```text
re_arc/environment_files/<game_slug>/<variant>/
```

Example:

```text
re_arc/environment_files/mygame/0001/
```

Add these files in that folder:

1. `metadata.json` (required)
2. `<game_slug>.py` (required)
3. `manifest.json` (optional, used by many generated/spec-based games)

`metadata.json` must include a `game_id` (for example `mygame-0001`).
For custom games, mirror existing metadata files and include fields like `title` and `class_name`.

In `<game_slug>.py`, define your `ARCBaseGame` environment and keep the score contract below.

## 2. Add the DSL Agent

Create a matching agent module:

```text
re_arc/dsl/agents/<game_slug>.py
```

The module should:

1. Define a class that subclasses `DslAgent`
2. Implement `next_action(self, env, observation)`
3. Export `AGENT_CLASS = YourAgentClass`

The loader resolves by base game id, so `mygame-0001` maps to `re_arc.dsl.agents.mygame`.

## 3. Keep the Score Contract

Do not mutate `self._score` or `self._win_score` in game logic.

Use a game-specific variable for internal points, for example:

```python
self._route_score = 0
```

When a level is solved, call:

```python
self.next_level()
```

Why: transition reward is computed from level progress:

```text
(new_levels_completed - prev_levels_completed) / win_levels
```

If `_score` is used for anything else, reward accounting breaks.

## 4. Smoke-Test Your Game Locally

Before running the full test suite, point `config.env` at your new game:

```text
GAME_ID=mygame-0001
```

Then run:

```bash
make play
```

Use this to manually play the game and confirm controls/mechanics feel correct.

Next run:

```bash
make dsl
```

This runs the DSL policy and writes a GIF rollout (by default under `runs/`).

## 5. Validate Before Merge

Run:

```bash
make precompute_dsl_actions
```

This runs each environment with its DSL agent and saves precomputed action traces used to speed up test runs.
It also refreshes `metadata.json["baseline_actions"]` from the deterministic replay so metadata stays aligned with CI.
For faster generation on multi-core machines, use parallel workers, for example:

```bash
make precompute_dsl_actions PRECOMPUTE_DSL_ARGS="--jobs 8"
```

Then run:

```bash
make test
```

This verifies (among other checks) that DSL agents:

1. Reach `WIN`
2. Finish within 1000 steps
3. Keep each environment step under 10 ms
4. End with episode reward/score of `1.0`

## 6. Publish Replay Assets

Before finishing, add the generated replay assets for the new game under:

```text
docs/replays/
```

This should include both:

1. The representative GIF
2. The matching `.replay.json` trace

The replay trace is required for `re-arc --policy replay --replay-dir docs/replays` to discover and display the game.
