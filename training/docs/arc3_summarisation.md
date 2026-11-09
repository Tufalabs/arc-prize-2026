# ARC-3 native summarisation

The worker selects DuckAgent's native `summarize` compaction when
`inference.agent_max_summary_tokens` is positive. Zero selects pruning.
Existing pruning recipes remain explicit even though the harness now defaults
to summarisation.

## Retained draws and dependencies

The harness pin is arc3-platypus PR #113's merge,
`e53d822d262c12a3b2b7fb31b52b47f8ce905d67`: it includes
the retained-failed-draw fix and clamps summary requests to the connected
engine's actual window, and pins TAAF `55eb62c9`, which the
trainer lockfile follows. The remote worker sets `derive_max_pixels=False`:
the server owns image processor limits, and there are no local engine
arguments to derive them onto. Summary training enables `preserve_failed_draws`
with `malformed_retries=0`: an invalid or malformed summary keeps its draw and
ends the episode without retrying or rewriting history. The worker refuses
older harness versions that lack this contract. CPU integration tests cover
empty summaries, tool markup, tool calls and parser refusals through the
trainer observer and gateway ledger.

## Duck settings the worker inherits

The per-level completion-token budget (`max_tokens_per_level`: the Duck gives
up a game once one level has consumed it) is the engine preset's in training,
never set by the worker: at `e53d822d` the class default is `(90_000,)`, the
Flash cell on `slurm-b200` sets `(175_000,)` and the Kaggle environment none.
The worker reads the value off the agent it built and every run reports it as
`agent_settings.max_tokens_per_level` (null: no budget), which the trainer
keeps in each trajectory's metadata. `scripts/eval_base_vs_lora.py` keeps
disabling it (an owner ruling of 2026-09-01, 6aad3b0f, that the per-game
budget be the eval's only token limit), so training and the standalone eval
play under different give-up rules; each eval episode line records the value
it played under.

The prompt injection is the preset's too: `configure_training_agent` no longer
sets `inject_text_on_first_prompt`, `inject_text_on_later_prompts` or
`prompt_injection_schedule`. At `e53d822d` the 27B preset keeps the class
defaults (`"We need solve"` on a game's first prompt, `"We"` on every later
one, schedule `game`) and Flash serves none. Training and the eval driver
served no injection on either engine before 2026-10-10, so ARC-3 27B rollouts
now differ from earlier runs'. Every run reports the three settings in
`agent_settings`; ARC-3 checkpoint identities do not pin them.

## Configuration

For a 32,768-token ordinary window and 4,096-token reply reserve, use these
fields in an ARC-3 recipe:

```json
{
  "inference": {
    "agent_context_window": 32768,
    "agent_max_completion_tokens": 4096,
    "agent_safety_margin_tokens": 512,
    "agent_compact_to_tokens": 28160,
    "agent_max_summary_tokens": 4096,
    "agent_max_served_sequence_tokens": 49152,
    "agent_summary_forbidden_token_ids": [248058]
  },
  "training": {
    "max_sequence_tokens": 49152
  }
}
```

The summary cap is twice `window // 16`, and the reset target is the ordinary
prompt budget (`window - reply reserve - safety margin`). The server ceiling
must accommodate at least the window, reply reserve and 8,192 extra tokens.
The worker uses nonthinking summaries and the Qwen tool-call token ban.
Evaluation window overrides resize the summary cap, reset target and server
ceiling together. The served ceiling keeps the larger of the configured room
above window plus summary cap and above window plus reply reserve (at least
8,192), so a smaller evaluation window never falls below ARC-3's minimum. Summary token biases
continue through the existing gateway and learner scoring path.

This integration uses the new compaction implementation. It does not use the
removed SummaryDuckAgent or the earlier summarisation feature branch.
