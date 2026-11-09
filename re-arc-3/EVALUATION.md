# Reusable PRO-LONG evaluation

This directory includes a laptop-first evaluator for the 30 short ACTION7 games.
After cloning, the tester runs one command from here:

```bash
./evaluate
```

It runs the visual solver, creates a Markdown report plus sanitized traces,
commits only that packaged result, and pushes it to the current branch.
Complete setup, recovery, output, and reuse details are in
[`docs/reusable-prolong-evaluation.md`](docs/reusable-prolong-evaluation.md).

To reuse the evaluator for another batch, copy and edit only the `protocol`,
`description`, and `games` fields in a panel JSON such as
[`configs/prolong/short-games-action7-v1.json`](configs/prolong/short-games-action7-v1.json),
then make it the default panel or run `./evaluate --panel path/to/panel.json`.
