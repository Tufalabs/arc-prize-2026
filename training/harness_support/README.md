# Harness support

`arc2/` and `arc3/` contain only the differences between the two harnesses.
Everything else in the trainer consumes the shared contract and must not
know which harness produced a result.

Each harness has two adapters, selected lazily by `training/harness.py`:

- `worker.py` (worker side, `get_harness`): task construction, harness API
  calls, progress snapshots and result translation into the shared gateway
  contract. It runs in the harness's own worker environment and imports that
  harness's `duck`, `platypus` and `taaf` packages; loading one adapter must
  never import the other harness.
- `trainer.py` (trainer side, `get_trainer_support`): pure Python with no
  harness imports. It declares the harness's media capabilities, deadline
  states and whether runs echo a behavior-policy snapshot, validates recipe
  settings the harness cannot honor, resolves datasets and the checkpoint
  identity facts, independently re-validates reported outcomes for the
  gateway, and translates a verified run into a `RolloutTrajectory`. Facts
  only the harness knows (the agent settings its preset resolved, which the
  worker reports with every run as `agent_settings`) reach the checkpoint
  identity through `rollout_identity`, pinned at the first window
  (`training/rollout_identity.py`).

ARC-3's score rules (the official game score, its per-level deltas and
their attribution to steps, stage features) are `arc3/scoring.py`, one copy
that imports nothing from either side: the worker's callback
(`arc3/results.py`) reports rewards with it and the trainer re-derives them
with it (`tests/test_arc3_scoring.py` compares it with TAAF's own score).

Shared, and deliberately not here: the rollout worker loop, backend
configuration, exact-history contract, gateway verification (token
continuation, checksums, media evidence, policy identity), draw journaling,
episode packing, the training algorithms, checkpointing, and the pinned
image-processing identity (`training/multimodal.py`, applied whenever an
adapter declares images). Experiment settings stay in trainer configuration;
this directory defines no training recipes.

Agent behavior and reusable lifecycle hooks belong in the Platypus repos.
Both ARC-3 and ARC-2 use ordinary Duck with public context controls and the
shared training observer in `scripts/serving_contract.py`. ARC-2's callable
response observer journals processed-draw metadata before tool execution.
Per-game backend extensions use `configure_game_backend`; no trainer adapter
overrides a private harness lifecycle method.

Adding a harness means a `worker.py`, a `trainer.py`, an entry in
`training/harness.py`'s `ENVIRONMENTS`, and recipes; the gateway and the
training pipeline are unchanged.
