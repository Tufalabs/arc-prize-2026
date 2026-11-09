# Agent Notes For `re-arc-3`

When editing files under `re_arc/environment_files/`:

1. Do not mutate `self._score` or `self._win_score` directly.
2. Use a game-specific variable for internal points (for example `self._route_score`).
3. Use `self.next_level()` to represent solved-level progression.

Reason: `TransitionRewardEnv` derives reward from `levels_completed / win_levels`, which are backed by `ARCBaseGame` score fields.

Before finishing environment changes, run:

- `pytest -q tests/test_environment_score_contract.py`

Before finalizing any code changes, run the full CI suite:

- `make prepare`

## Commit Message Style

Use Conventional Commit style for all commits:

- `feat:` new user-facing functionality
- `fix:` bug fixes / behavior corrections
- `chore:` maintenance work (tooling, refactors without behavior change, docs-only updates)
