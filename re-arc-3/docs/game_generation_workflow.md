# Game generation and PRO-LONG evaluation

This branch keeps one narrow workflow: prepare game ideas, run them through the
standard generation pipeline, and evaluate generated games with PRO-LONG. It
does not include automatic prompt or difficulty evolution.

## 1. Prepare game ideas

The optional pre-pipeline prompts are:

1. `pipeline/prompts/axis_generation.md` defines a broad possibility space.
2. `pipeline/prompts/sample_from_axis.md` samples 30 coherent game seeds.
3. `pipeline/prompts/idea_prompt.md` expands each short seed into one faithful
   game idea.

The pipeline consumes a JSON array containing `title`, `description`, and
`num_levels` for each idea. The recent compact-axis input is retained at
`pipeline/game_ideas/compact_axis_seed_ideas_20260805.json`.

## 2. Generate specification prompts

The compact-axis configuration uses human specification generation. This
command writes one prompt per idea and leaves the run waiting for completed
Markdown specs:

```bash
make pipeline-specs \
  PIPELINE_RUN_ID=compact-axis-specs \
  PIPELINE_ARGS="--config pipeline/configs/compact_axis_seed_gpt_pro_human.yaml"
```

Complete the generated prompts and place the resulting Markdown files in the
run's `specs/` directory.

## 3. Implement completed specs

```bash
make pipeline-implement \
  PIPELINE_RUN_ID=compact-axis-games \
  PIPELINE_ARGS="--spec-dir pipeline/runs/compact-axis-specs/specs --implementation-prompt pipeline/prompts/codex_check_visual_prompt.md --codex-reasoning-effort medium"
```

The implementation workers receive `pipeline/prompts/codex_check_visual_prompt.md`.
That prompt requires canonical actions, game-native undo, a DSL solver, worker
validation, and representative visual review.

For a non-human configuration, `make pipeline-run` performs specification and
implementation stages together.

## 4. Evaluate a generated game

After integrating a generated environment and its metadata, run:

```bash
make pro-long \
  PRO_LONG_ARGS="<game-id> --timeout-seconds 1200 --max-episode-actions 500"
```

PRO-LONG writes its interaction log, execution trace, Codex events, token
usage, result, and manifest under `runs/pro-long-solver/<run-id>/`. Pass
`--resume-from <prior-run-directory>` to reconstruct a previous run by replaying
its actions and continue the same Codex thread.
