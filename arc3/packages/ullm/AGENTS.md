# ullm (platypus fork)

*Register: living.*

The platypus LLM layer — a hard fork of the standalone ullm repo (which
the whale's sandbox keeps using; the two never share a venv). The
`aaf` dependency is dropped (`ullm.shared_semaphore` is vendored
in-tree). Architecture: `design.md` in this directory. Open fork work:
`roadmap.md` here.

Gate posture: the package — tests and the vllm-extra modules included —
is checked by the workspace pyright gate, with one carve-out: a
per-directory executionEnvironment relaxes the unknown-type family,
reportMissingTypeArgument and reportUnnecessaryIsInstance. That is
deferred work rather than policy (`roadmap.md`): closing it is ~100
findings across the package's dynamic surfaces. Formatting and lint are
the workspace's, like every other package. GPU serving deps (vllm + cu13
toolchain) are plain dependencies — every install gets them; importing
`ullm.backends.vllmx*` works anywhere, running an engine needs a GPU.

## Coding standards

These are ullm's house style, kept so the package stays consistent with
its own existing code. **Where they differ from the root `AGENTS.md`
guidelines, these win inside this package.** The known divergences:
comment density (~1 line per code block here, vs. root's
concise-and-why-focused), and the from-import ban (stricter than root,
which mandates no import style). Root rules not contradicted here still
apply — docstrings everywhere, and never narrating past bugs. Layout is
not among the divergences: the workspace formatter owns it.

- Use type hints on function headers.
- Use type hints on class properties, which are always dataclass fields.
- Avoid the "from A import B" pattern; rather, "import A" and use "A.B"
  in the code. Allowed exceptions: dataclasses, typing.
- Class properties are always fields.
- Include comments on class properties if not 100% obvious, typically
  in-line.
- Try to have ~1 line of explanation comment for every code block.

## Type checking and constraints

- All classes extend `BaseClass` (support.py), a dataclass with
  attribute freezing (no new attributes after `__init__`) and runtime
  type/constraint checking.
- `debugging_mode` is a global (0-3) controlling strictness. 0 = off,
  1 = type checks only, 2+ = also recursively checks nested BaseClass
  instances. Default is 2.
- `check_constraints()` is NOT called automatically at construction.
  Instances may be incomplete after `__init__` (e.g., needing
  configuration). Constraints must hold by the time the object is
  passed around or used.
- Subclasses override `_check_constraints()` for custom validation
  logic beyond type checking.
- We use fairly strict type checking — patterns like
  `extra_params: dict[str, Any] = field(default_factory=lambda:
  dict[str, Any]())`.

## Tests

The cost tiers (`make test` / `test-gpu-srun` / `test-api` /
`test-full`) are **workspace-wide**, defined in the root `AGENTS.md` —
gpu- and api-marked tests also live in the root suite (reference
regeneration, the trace end-to-ends), not only here.

What is ullm-specific is the backend x test matrix
(`packages/ullm/tests/test_ullm_backend_matrix.py` + column definitions
in `conftest.py`), this package's main consumer of those tiers: free
columns (dummy/stub) run in the cheap tier; the vLLM columns (direct /
proxy / in-process server, on Qwen3.5-9B: small, thinking and
vision-language) run as one SLURM job per column under the gpu tier;
the paid columns (OpenRouter, Anthropic) under the api tier. Rows skip
columns via capability flags; scripted columns consume per-test scripts
so each row exercises the same code path everywhere.
