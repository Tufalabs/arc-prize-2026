# sweeper

*Register: living — the map. Behaviour is documented by the module
docstrings; this says what is here and what the pieces are for.*

One run, several agent configurations. A `SweepAgent` wraps another agent
and configures its per-game copy from the **pass** it is playing, so a
benchmark of 25 games at 10 passes becomes 10 configurations over the same
25 games — one allocation, one served model, one job dir.

```python
design = Design(
    run={"kv_cache": Knob(draw=Choice(("auto", "fp8_e4m3")), apply=Field("llm_config...."), missing="auto")},
    arm={"effort": Knob(draw=Choice(("medium", "high")), apply=Field("inner.reasoning_effort"), missing="high")},
)
inner = DuckAgent.build()
# The wrapper needs its OWN llm_config: LLMAgentSolver reads the hosted agent's
# to build the one host-side backend, and the hosted agent is the wrapper. Pass
# the inner agent's own object so both resolve sampling profiles from it and the
# preset stays authoritative — `inner.llm_config` alone is inert.
agent = LLMSweepAgent(
    inner=inner, design=design, run_seed=7, arm_seeds=tuple(range(40, 50)), llm_config=inner.llm_config
)
```

| module | what it is |
|---|---|
| `knobs.py` | what varies: how a value is drawn, where it is applied, and what it was before it existed |
| `agent.py` | the wrapper: run batch at construction, arm batch per pass |
| `seeds.py` | how a sweep allocates seed space — arm blocks per job, non-overlapping play seeds |
| `jobs.py` | what a job dir says: finished / in flight / not submitted, and the design stamp that gates reuse |
| `collect.py` | finished job dirs back to rows — scored at a common budget, aggregated at the unit each knob was drawn at |
| `example_analysis.py` + `notebooks/example_analysis.ipynb` | a worked sweep of the random agent, end to end |

The package stops at the mechanical half. Regressions, plots and effect
sizes are the experiment's, and live in the notebook.

**The engine arguments are the sweep's to state.** The duck settles its own
at `prepare_for_backend`, and the solver calls that on the WRAPPER, so none of
them reach a wrapped duck: the window and the sequence cap stay whatever the
sweep put on the engine. That is the right answer rather than a gap — one
engine is shared by every arm, while the window is an `inner.*` knob that
varies per arm, so no single agent's window is the engine's. Size the engine
for the widest arm. A sweep that wants the duck's derivations back has to run
one agent per engine, not one engine per sweep. Each Duck arm's generation
cap is `inner.reply_cap_tokens`; it also sets that arm's context reserve.
The solver checks every arm's window against the connected engine before
games start.

**The real sweeps are not here.** Which knobs, which ranges, which runs and
every statistic live in the `arc3-experiments` repo, which depends on
this package as an editable path. That split is deliberate: a design is an
experiment that changes every round, while this package is machinery that
should not. The worked example above exists so the mechanism has a test that
does not need the other repo.

## The two batches

The split is not stylistic — it is where the solver *reads* a value.

- **run** knobs are drawn from `run_seed` and applied at construction, before
  anything reads the template, so they reach engine settings and hosting
  knobs. One job is one draw of them, so their effects are only visible
  across pooled jobs.
- **arm** knobs are drawn from `arm_seeds[pass_index]` and applied inside the
  game process, so they reach only what is read after that: the wrapped agent's
  own fields and its sampling profiles. Nothing on `run_config` qualifies — the
  move loop reads it before the first `do_move`, which is where the arm batch
  runs.

An arm knob aimed at a template-only value is refused at construction. The
failure it prevents is silent: the knob varies, nothing happens, and the
analysis reports that it does not matter.

## Rules worth knowing before writing a design

- **Knobs must pickle.** They travel into game subprocesses with the agent, so
  a `lambda` or a closure cannot make the trip; the helpers are frozen
  dataclasses for exactly this reason, and `check_picklable` refuses the rest
  at construction rather than from inside a spawned child.
- **Every knob draws from its own stream**, seeded from `(seed, name)`. Adding
  a knob therefore leaves every other knob's value unchanged, which is what
  lets jobs recorded under different knob sets pool — `Knob.missing` supplies
  the baseline for the ones an older job never had.
- **A queue-driven budget uncap does not belong in a local sweep.** Anything
  that lifts a limit once `ctx.slots.drained` makes spend depend on where a
  game was scheduled rather than on its knobs; the agents' local presets turn
  it off, and `shuffle_games=True` keeps what is left from attaching to a
  particular arm.
- **Compare arms at a spend you choose, not where they stopped.** With a
  per-level token allowance, a configuration that reaches more levels earns
  more budget, so its terminal score is measured at a spend the others never
  had — and the sweep flatters whatever wins levels early. `collect_rows`
  therefore reads every lane off its score-vs-tokens curve at one budget
  (`DEFAULT_BUDGET_TOKENS`, 120k generated tokens per game), using TAAF's own
  primitives rather than a second implementation of them. The budget is the
  **ensemble's**: a single game may overrun as long as the fleet averages to
  it, which is what `ceiling_for_budget` arranges. Reading past a lane's end
  is exact when the lane ended on its own, and an extrapolation only when
  something outside stopped it — `extrapolated_fraction` says how much of an
  answer rests on the latter.
- **Count observations at the unit a knob was drawn at.** `by_arm` / `by_job`
  do this from the design, because it is the easiest way to manufacture
  significance: a run-level knob decided ten times, tested across a thousand
  games, reports an `n` in the thousands.
- **Two seeding rules, both silent when broken** (`seeds.py`): every job takes
  its own block of arm seeds, or N jobs replay K design points instead of
  drawing NxK; and play seeds are spaced by more than one job's entry count,
  or two jobs hand out the same randomness with nothing recording it.
- **A sweep run's `diagnostics.html` is not a result.** TAAF's per-pass
  statistics assume every pass shared a configuration; here they measure the
  thing being swept.

## Kaggle

A submission returns a leaderboard score and writes no `benchmark.json`, so
the design is printed to stdout at construction — one machine-parseable line
per batch — and the kernel log is the record. `parse_sweep_lines` reads it
back and `verify_arms` re-draws the same seeds to check that the knob set has
not moved since. One design point per submission: a single arm seed
broadcasts to every pass, which makes the passes replicates.
