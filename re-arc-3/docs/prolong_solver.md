# Local PRO-LONG solver

This repository contains a local, non-Docker implementation of the PRO-LONG
control loop for RE-ARC. The evaluator uses its own minimal trusted runtime and
does not depend on a visual-player or prompt-evolution system.

The runner:

1. appends the initial board and every later action, outcome, animation frame,
   settled board, level transition, and analyzer briefing to
   `workspace/logs.txt`;
2. gives a locally running Codex coding agent persistent access to that log and
   ordinary file, shell, and Python tools, with network access disabled;
3. reads a bounded `workspace/actions.json` plan from the agent;
4. executes the plan one primitive action at a time without another model call;
5. invalidates remaining actions when the level or environment state changes.

Run one game from the repository root:

```bash
make pro-long PRO_LONG_ARGS="ft09-0d8bbf25 --timeout-seconds 1200 --max-episode-actions 500"
```

Results are written under `runs/pro-long-solver/<run-id>/`. The main artifacts
are `workspace/logs.txt`, `trace.jsonl`, `codex.events.jsonl`,
`token_usage.json`, `result.json`, and `manifest.json`.

For a quick local validation:

```bash
uv run pytest -q tests/test_prolong_player.py
```
