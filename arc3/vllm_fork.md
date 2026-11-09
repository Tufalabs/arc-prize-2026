# The vLLM fork: what it is and how it ships

*Register: as-built.*

Platypus runs on a fork of vLLM, `https://github.com/Tufalabs/arc-prize-2026-vllm`
(private): upstream v0.29.0 plus what Tufa's agents need of the engine,
`SamplingParams.zero_recurrent_state` first -- deliberately blanking a
request's recurrent (Mamba/GDN) state at a chosen prompt position while its
attention KV stays intact (why an agent would want that is the agent's
story, `duck.DuckAgent.zero_recurrent`). What the fork carries, how its
repo and release line work and how its PRs are reviewed is the fork's own
`TUFA.md`: read it at the pinned revision, which it describes exactly. This
document is about how platypus consumes the fork: how the pin is built,
what serving it needs, the deployment rules the engine cannot enforce, and
how it ships to Kaggle.

## How platypus consumes it

Two blocks in the root `pyproject.toml`, and they must move together:

- `override-dependencies`: `vllm @ git+https://github.com/Tufalabs/arc-prize-2026-vllm@<sha>`
  — a full revision, never a branch name, so `uv.lock` and the pin agree
  by construction.
- `[tool.uv.extra-build-variables]`: `VLLM_USE_PRECOMPILED=1` and
  `VLLM_PRECOMPILED_WHEEL_COMMIT=<v0.29.0 commit>`.

The fork is pure Python on top of v0.29.0 — its compiled payload is
byte-identical to stock — so the "build" just unpacks upstream's
precompiled wheel for that exact base commit and overlays our `.py`
files. Seconds, not a CUDA compile, in every node-local venv.

The opt-in Flash-Next throughput stack also carries SM120 CUDA images inside
its Python overlay. Their sources and build provenance live in the fork
(`csrc/xkb`, described by `TUFA.md`); the precompiled base extensions remain
the wheel's. Its guards keep those images on capability 12.0 and retain the
existing kernels elsewhere.

**Both variables, always.** `VLLM_USE_PRECOMPILED=1` alone resolves the
base wheel by guessing a merge-base, and in an isolated build that guess
once fell back to **nightly**: 0.28-rc binaries under 0.27.1 Python,
imported fine, ran fine, was wrong. Declaring the pair in
`pyproject.toml` is the fix — it is reviewed and locked instead of
living in whoever's shell. When the fork moves to a new upstream tag,
the COMMIT variable moves with it in the same diff as the pin.

**The uv floor is a statement, not a guard.** `extra-build-variables`
arrived in uv 0.8.6, and `required-version = ">=0.8.6"` records that.
It does not stop an older uv: measured with 0.6.6, the unknown field
makes uv discard the whole `[tool.uv]` table — the floor included — warn,
and carry on resolving without the variables. Nothing inside the file
can refuse a uv too old to read it, so the real control is keeping
workers on a current uv; the floor only refuses a uv that parses the
table but predates the variables.

The installed version reads `0.29.1.devN+g<sha>`: `N` commits past the
`v0.29.0` tag, at fork revision `<sha>`. That string lands on every
trace as `provider_version`, so a run's engine is identifiable from its
artifacts alone.

Changing the fork is therefore: branch on the fork repo → PR into
`tufa/v0.29.0`, with the change's groups in its `TUFA.md` → merge → bump the
`@<sha>` here → `uv lock`. Nothing
else to regenerate, and a forgotten bump changes nothing silently — the
old engine keeps running until the pin moves.

The same pin update must review `engine_proxy.sampling_to_wire` against
`msgspec.structs.fields(vllm.SamplingParams)`, including fork-only fields.
Its explicit public-field set is the remote constructor contract: omitted
private or derived state must be reproduced exactly by reconstruction, and
any unknown non-neutral field fails closed. Run `test_engine_proxy.py` when
moving the pin; silently dropping a new sampling control is not supported.

### The pin stops at this workspace root

Every `[tool.uv]` setting above — the override, the build variables, the
`required-version` floor — is read from the **workspace root only**. A
member's own `[tool.uv]` is ignored. Measured, not assumed: a member
declaring `override-dependencies = ["idna==3.4"]` inside a parent
workspace resolves to idna 3.19 in the parent's lock, with no overrides
recorded at all.

So a repo that adopts these packages as members of ITS workspace gets
our agent with whatever vLLM it pins for itself. That is not
hypothetical — a training workspace pinning stock vLLM would take the
duck and none of the engine it needs.

Two things follow for an adopter:

- Copy the override, both build variables AND the `required-version`
  floor into the adopting workspace's own root. A member's
  `project.dependencies` is the only thing that travels, and a git
  dependency there would still build without the variables — an
  hours-long CUDA compile, or the nightly fallback.
- Expect the runtime checks to fire meanwhile rather than silence:
  `DuckAgent` refuses at `prepare_for_backend` with `zero_recurrent=True`
  when the installed `vllm.SamplingParams` has no `zero_recurrent_state`,
  and refuses a server whose `/version` is not this client's build. Those hold regardless of whose root won, which is why
  they exist.

## Topologies

Both serving paths carry the blank, and each has a condition.

**In-process engine** (`serving="direct"`, what every run so far used).
`ullm.backends.vllmx` splats `extra_params` into `vllm.SamplingParams`,
so the fields are set by name. A stock engine raises `TypeError` on the
unknown keys — loud, though only as a side effect of constructing a
typed object.

**OpenAI-compatible server** (`serving="server"`, or an external server
via a supplied `backend_factory`). The fork declares both fields on
`ChatCompletionRequest` and `CompletionRequest` and forwards them to
`SamplingParams`. Two conditions:

- The server must run with `VLLM_USE_V2_MODEL_RUNNER=1`. The blank is
  implemented only by the V2 runner, and a hybrid model is not in the
  default-V2 set, so a server without it **refuses** every blanked
  request. A caller who owns the server command line sets this environment variable.
- The server must be this client's own build. A server on any other
  revision accepts the fields as unknown extras — `OpenAIBaseModel`
  allows them — and serves the request unblanked with nothing in the
  response to say so. The agent therefore asks `/version` and refuses a
  mismatch, or a server it cannot ask at all.

## The deployment invariant: one mode, one target, per cache domain

vLLM's prefix cache hashes token ids (plus LoRA and multimodal identity),
never sampling parameters. A blanked request therefore writes its
perturbed recurrent state, and past the next block boundary its
downstream attention KV, under the same keys an unblanked request would
read. So on one engine, two requests that share a prefix through the
blank position and disagree about the blank — off versus on, or two
different targets — serve each other's state, silently, in either
order. Turning the CPU KV tier off does not change this: the GPU prefix
cache is what carries it.

The engine does not enforce this, by decision: the parameter is
per-request precisely so that agents with different system prompts can
ask for different positions on one engine, and a guard that refused a
second configuration was tried and reverted. The rule is therefore the
deployment's:

- **One reset mode and one absolute target per shared-prefix cache
  domain.** A control arm, or a different target, runs on its own engine
  or in its own cache namespace, never beside the treated arm on a
  shared server.
- Every run so far has been homogeneous — every request on, one target
  — which is a property of how it was launched, not of the engine.

## Training: an explicit choice before adoption

ARC3-Training consumes these packages as members of its own workspace,
so everything in the section above applies: its root must carry the
override, both build variables and the floor, or it keeps the stock vLLM
it pins today. Two further gates are specific to it, and no Training
change is asked for now — this is the checklist for when it adopts.

**The trainer has no blank.** `PPOTrainer._token_scores()` runs the
Hugging Face decoder once over the ordinary full sequence, with no
recurrent reset at any position. With `zero_recurrent=True` on the
rollout, its `old_token_logprobs` come from the blanked serving policy
while the trainer recomputes `new` logprobs from the unblanked one, and
the replay evidence puts first-token differences at up to ~5.6 nats —
same-policy startup parity should fail, and bypassing it would bias the
PPO ratios. Adoption must therefore choose, explicitly:

1. Leave `zero_recurrent` at its default (off) in
   `platypus_rollout_worker.py`, keeping the trainer's semantics and
   forgoing the blank; or
2. Implement the identical absolute-position recurrent reset in the
   differentiable trainer forward, carry the mode and target through the
   rollout–training contract, and pass same-policy parity before
   optimizing.

**A shared Training server is one cache domain.** The invariant above
applies with force here: rollouts with the blank on cannot share a
server with control or different-target traffic whose prompts share the
blanked prefix. Isolate by server or cache namespace before mixing.

**The server side of Training is hand-launched.** Its launcher must export `VLLM_USE_V2_MODEL_RUNNER=1` itself; without it the fork refuses
every blanked request. And resolving the fork in Training's single
shared environment moves Linux from torch 2.11 / Triton 3.6 to the
fork's torch 2.13 / Triton 3.7.1 — not a server package swap. That
needs the parent lock regenerated and, on the exact fork with V2 forced:
trainer forward/backward, FA4 import and kernels, runtime LoRA load and
swap, and serving/trainer parity. The existing generic 0.27.1 parity
result does not cover that configuration.

## Kaggle: the fork, overlaid on a stock install

The engine a kernel runs is said once, in `ullm.kaggle`: the shipped
descriptor `VLLM_FORK` names the wheelhouse
(`arc3-site-0290`), the requirement set the kernel
installs from it `--no-index` (or replays from the record of that
install beside the wheels, `ullm.kaggle_site`), and that the fork rides
along as an overlay. An agent links it from `RunConfig.kaggle_engine` and says
nothing else about the engine; the solver derives the kernel source,
the two setup commands, the bundle exclusion and the payload file from
it. The stock install owns the whole offline dependency closure, which
is the hard part on a kernel with no index; the second setup command
replaces the engine's own files with the fork's, so what runs on Kaggle
is the pinned revision, not a stock build. `ullm.vllm_overlay` is both
halves of that. The wheels are looked for under every mount layout
Kaggle has used for an attached source, dataset or utility alike, since
Kaggle has attached the same wheelhouse both ways at different times.
A test holds the wheelhouse's slug, the requirement set's engine pin, the
connector's line guard, the venv's installed engine and the workspace's
fork pin to one line, so a version bump that forgets one of them fails
before a deploy.

**The install is replayed, not run** (`ullm.kaggle_site`). pip took ~190 s
of every RTX boot. The wheelhouse notebook (`arc3-site-0290`,
built by `tools/kaggle/arc3-site-0290/`) runs `arc3-utils-0290`'s download
cells as they are, its versions pinned by constraint to that wheelhouse's,
then installs them on the pinned image and records the install: what it
removed, an archive of what it wrote, and the installed distributions'
digest before and after. A kernel checks that its own site-packages are the
record's start state and installs with pip, untouched, where they are not;
otherwise it replays the record and checks the end state, and a replay that
fails or ends elsewhere fails the kernel, since pip cannot repair a
half-unpacked distribution. Its report is `site_install.json` (70.4 s on the
RTX: 144,747 files unpacked, 17,926 removed). The record is the image's, so
every kernel runs on the image it was made on (`KaggleEngine.docker_image`,
through TAAF's `KaggleTarget.docker_image`): Kaggle otherwise runs a kernel
on whatever image it last ran on.

**The checkpoint is read ahead of the loader** (`ullm.kaggle_prefetch`).
The RTX Flash cell loads lazily, and the kernel's first setup command reads
its shards into the page cache a 24 GiB window ahead of the shard the
loader has open, in the loader's order, dropping those it has passed: the
target's weights loaded in 133 s instead of 208 on the RTX, the drafter's
in under a second. It finds the loader through `/proc/*/maps` and counts it
done when it stops mapping new shards, since the lazy load keeps every one
mapped; its report is `shard_prefetch.json`. The four-L4 cell uses the
engine's own prefetch instead: the reader moves with the loaders it has
seen, and four walk the shards there, the second stage's passing the first
stage's shards before the first stage starts, so the reader dropped what
those had yet to read (535 s of load against the engine prefetch's 252).

**The delta is taken against the base wheel, not against git.** The
launcher hashes its installed `vllm/` against the precompiled wheel for
`VLLM_PRECOMPILED_WHEEL_COMMIT` — read from uv's own record of the build
variables in the installed dist-info, so it is what this venv was built
with rather than what a pyproject says today — and ships every file that
differs, plus a manifest naming, for each of them, the digest the base
wheel holds and the digest the fork holds.

A name list from `git diff base..pin` would not do. Much of the
installed tree — the vendored trees the build extracts from the wheel
(below), `vllm_flash_attn/cute/`, the generated `_version.py` — has no
counterpart in the fork's git tree at all, so a bumped vendored dependency
is invisible to it: git would show a submodule pointer while the changed
bytes went unshipped. Hashing the
two artifacts that actually meet on Kaggle has no such blind spot, and
costs one ~300 MB wheel download per base commit, cached.

**Replaced in place, never on `sys.path`.** The payload rides in the
TAAF bundle's `payload/` (the engine's `bundle_files`, which TAAF
copies verbatim and never interprets) and is applied over
site-packages. Games run in spawned children and the engine spawns its
own workers, so a path-order trick would have to hold across every one
of them; there is instead exactly one `vllm` on disk, and
`vllm/_version.py` travels with it, so `provider_version` on a Kaggle
trace names the fork revision like any other run's.

**It refuses rather than half-land.** Before writing anything the apply
step checks the installed version against the base the overlay was built
on, and every file it is about to overwrite against that wheel's digest;
afterwards it checks what it wrote. The failure it prevents is silent —
a stock engine takes `zero_recurrent_state` as an unknown extra and
serves the request unblanked — so a mismatch fails the kernel's setup
instead. Re-running the command is not a mismatch: a payload already in
place reports itself and stops.

The wheelhouse must therefore be the fork's precompiled base. The
manifest records the base wheel it was built from by sha256, and every
kernel checks the pre-image digests file by file before applying the
overlay — so a wheelhouse built from a different build of the same
version fails loudly there rather than running as something else.

**What the precompiled build means for fork work.** With
`VLLM_USE_PRECOMPILED=1` the build takes the compiled extensions, the
Rust frontend, the vendored trees it extracts (`third_party/` `deep_gemm`,
`flashmla`, `fmha_sm100`, `tml_fa4` and `triton_kernels`) and the
`vllm_flash_attn` Python (all but its `__init__` and interface) *from
the base wheel*, overwriting the checkout. A fork commit touching
`.cu`/`.cpp`/CMake or a file in those trees is inert in this venv, on
Slurm as on Kaggle, and nothing warns; the vendored Python the build does
not extract (`third_party/flash_linear_attention`, which the FP8 recurrent
state edits) builds from the checkout like any fork file. The overlay's
delta is the check: a change that is not in it was not built. The same mechanism is why the
delta can grow from half a megabyte to hundreds without a change of
ours — a bumped vendored tree or a changed `.so` is picked up against
the base wheel — and the deploy prints the replace/add/remove counts and
the payload size so a move is visible.

**The payload is base64 over a gzipped tar, and the encoding is
load-bearing.** Kaggle's dataset ingestion recurses into anything it
recognises as an archive and puts every inner name in front of its
filename validation, which the engine's fused-MoE config files (names
with `=` and `,`) fail by the hundred. An archive by neither name nor
content passes untouched.

**Two guards, one of them topology-dependent.** The overlay refuses at
setup when the tree is not its base; separately, the direct-mode engine
raises `TypeError` on the unknown sampling key when a stock engine is
somehow serving. A server topology accepts unknown extras silently, so
moving Kaggle to `serving="server"` would leave only the first guard
plus the agent's `/version` check.
