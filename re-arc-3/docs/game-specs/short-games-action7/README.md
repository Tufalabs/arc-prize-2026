# Short games with native undo

This branch retains 27 games generated from the GPT-5.6 Pro specification batch.
Each promoted package contains one seven-level environment, its DSL agent, and
a seed-0 precomputed winning action trace.

The retained package is evaluation-qualified: every included visual version
completed at least 3/7 levels in a full 30-minute `gpt-5.6-sol` PRO-LONG run.
When multiple fully evaluated versions qualified, the version with the higher
official ARC score was selected. Containment Walls, Freeze Rings, and Spark
Web were removed because no completed evaluation reached 3/7 levels.

## Provenance

- Base branch: `codex/pipeline-prolong-clean`
- Base commit: `42bed61161d2f8e9646817c2f2096d517ad7e2c9`
- Specification run: `20260811-141359-short-game-gpt56-pro-specs`
- Implementation run:
  `20260812-095912-short-game-gpt56-pro-action7-implementations`
- Implementation model: `gpt-5.6-sol`, medium reasoning
- Required control: `GameAction.ACTION7` / action id 7 is native undo

The source implementation run completed 30/30 games with zero failures. After
evaluation curation, the retained 27 deterministic DSL replays solve 189/189
levels at normalized score 1.0.
