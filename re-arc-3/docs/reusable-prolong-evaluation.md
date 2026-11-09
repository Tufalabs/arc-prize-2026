# Reusable local PRO-LONG evaluation

This directory packages a game batch so another person can evaluate every game on
their laptop and push a readable report and traces to the branch they are on.

## Tester setup

The tester needs macOS or Linux, Git, Node.js/npm, their own Codex account, and
GitHub push access to this repository (or a fork). The script installs repo-local
copies of `uv` and the Codex CLI when either is missing. Never send, copy, or
commit `auth.json`; every tester must authenticate their own account. The script
starts device authentication if needed.

```bash
git clone https://github.com/Tufalabs/arc-prize-2026.git
cd arc-prize-2026/re-arc-3
./evaluate
```

## Run the batch

The launcher bootstraps the locked Python environment, checks Codex
authentication, and starts the first configured game as a canary. The remaining
games start only after the canary writes its first valid action. The checked-in
panel gives each game 30 minutes and 500 total actions with no more than eight
games active at once. When the batch finishes, the same command sanitizes the
results, creates the Markdown report, commits only that bundle, rebases, and
pushes it to the checked-out branch. There is no second submission command.

Reduce concurrency on a smaller laptop with:

```bash
./evaluate --concurrency 3
```

Ctrl-C stops active solvers and still packages all partial traces. Raw working
state remains ignored under `runs/prolong-panel/`. The reusable, sanitized
bundle is written under:

```text
results/prolong/<protocol>/<tester>-<timestamp>/
```

Its `README.md` is the batch report. Each game links to `logs.txt`, which shows
the initial board, Codex reasoning and plan, actions, outcomes, and resulting
boards. The package also retains `trace.jsonl`, `codex.events.jsonl`,
`codex.stderr.log`, `result.json`, `manifest.json`, and token usage when those
files exist.

The submit phase rejects common credential markers and large files. The owner
can review the generated Markdown directly in the existing PR or retrieve it
with `git pull`. If the tester lacks upstream write access, add their fork and
run `./evaluate --remote fork`; the result will use the same branch name in the
fork and can be opened as a PR.

Use `--no-submit` only when intentionally collecting results without committing
or pushing them. If a final push fails after the result was committed, the
script prints the exact `git push` recovery command; rerunning the paid
evaluation is unnecessary.

## Reuse for the next game batch

Copy `configs/prolong/short-games-action7-v1.json` to a new descriptive name.
Change these fields:

- `protocol`: a unique, filesystem-safe batch name; this selects the result
  directory.
- `description`: a short human description for the generated report.
- `games`: the ordered list of `<game_slug>-<variant>` IDs. The first is the
  canary.

Usually no Python or shell code needs editing. Run the new panel with:

```bash
./evaluate --panel configs/prolong/my-next-batch.json
```

Before sending a branch to a tester, verify its configuration without launching
any paid solver process:

```bash
./evaluate --dry-run
```
