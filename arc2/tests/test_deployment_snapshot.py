"""Every deployment's whole configuration, frozen and diffed.

**Not a defaults test.** No single number here is the assertion — the repo
forbids asserting that a declared default equals itself, and rightly. What
this pins is the COMPOSITION: three layers (ullm's coordinate blocks, the
duck's own, and the settings derived at ``prepare_for_backend``) resolved
across every column into an agent, its ``RunConfig`` and ``LLMConfig``, and
the backend factory with its engine arguments. No other test asks whether the
pipeline still
produces the same deployment; each asks whether one value it names is what it
expects.

That gap is the dangerous direction — model-visible or engine-visible but
unrecorded. The committed replay references catch any change to what a model
was sent and nothing at all about the ENGINE, so a window, a watermark or a KV
connector can move and the whole suite stays green.

The snapshot is taken at the moment the backend WOULD be constructed:
``LLMAgent.prepare_for_backend()``, which the solver calls immediately before
it copies and starts the factory spec — so it sees every setting as the engine
will, derivations included.

A diff here is meant to be READ, not merely to match: it is the list of what a
change did to every deployment, which is exactly what a vLLM bump, an ullm
repin or a preset restructure wants reviewing. Regenerate deliberately, and
put the diff in the PR::

    PLATYPUS_REGENERATE_REFERENCES=1 make test TEST_ARGS='-k deployment_snapshot'

Ported from arc3-platypus ``tests/test_deployment_snapshot.py``. What differs
here is the column list, and only because the deployments differ: the
four-L4 Kaggle box in place of the RTX one, and no summary duck, which
takes its training column with it. The server and external columns are
text-only ducks: an image is continued by tokens by an engine and not by a
chat server, and every column promises ``exact_history``.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import os
import pathlib
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

pytest.importorskip("vllm", reason="the deployment snapshot builds real engine args")
deepdiff = pytest.importorskip("deepdiff", reason="the snapshot is compared with deepdiff")

from duck.agent import DuckAgent
from duck.multi_take import MultiTakeDuck, TieredTakeDuck

from platypus.llm import LLMAgent

REFERENCE = pathlib.Path(__file__).resolve().parent / "references" / "deployment_snapshot.json"

# Values that differ run to run and say nothing about configuration. Replaced
# by name wherever they appear, so a real change to the surrounding object is
# still visible.
_VOLATILE_FIELDS = frozenset(
    {
        "engine_id",  # KVTransferConfig stamps a fresh uuid4 per construction
    }
)


# --------------------------------------------------------------------------
# the deployments
#
# Every column a run is launched from today: one row per engine and
# environment ARC2 runs in, plus the knobs that vary within one.
# --------------------------------------------------------------------------


def _no_tier(agent: DuckAgent) -> DuckAgent:
    """The same column with the host KV tier off."""
    factory: Any = agent.llm_config.backend_factory
    factory.cpu_kv_offload = False
    return agent


def _tp(agent: DuckAgent, size: int) -> DuckAgent:
    """The same column sharded across more GPUs."""
    factory: Any = agent.llm_config.backend_factory
    factory.engine_args.tensor_parallel_size = size
    return agent


DEPLOYMENTS: dict[str, Callable[[], LLMAgent]] = {
    "slurm-3.8-direct": lambda: DuckAgent.build(engine="qwen3.8-27b"),
    "slurm-3.6-direct": lambda: DuckAgent.build(engine="qwen3.6-27b"),
    "slurm-flash-direct": lambda: DuckAgent.build(engine="qwen3.8-flash-next"),
    "slurm-3.8-server": lambda: DuckAgent.build(engine="qwen3.8-27b", serving="server", images="off"),
    "slurm-3.8-no-offload": lambda: _no_tier(DuckAgent.build(engine="qwen3.8-27b")),
    "slurm-3.8-tp2": lambda: _tp(DuckAgent.build(engine="qwen3.8-27b"), 2),
    "kaggle-3.8": lambda: DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4"),
    "kaggle-3.6": lambda: DuckAgent.build(engine="qwen3.6-27b", environment="kaggle-4xl4"),
    "kaggle-flash": lambda: DuckAgent.build(engine="qwen3.8-flash-next", environment="kaggle-4xl4"),
    "kaggle-3.8-no-offload": lambda: _no_tier(DuckAgent.build(engine="qwen3.8-27b", environment="kaggle-4xl4")),
    # The wrapper over two ducks, on the default engine: its run_config and
    # llm_config are the first take's, so the row shows both takes and what
    # the engine got from them.
    "slurm-flash-two-takes": lambda: MultiTakeDuck.build(),
    # The demo's Kaggle agent: the submission's shape.
    "kaggle-flash-tiered-takes": lambda: TieredTakeDuck.build(environment="kaggle-4xl4"),
}


# --------------------------------------------------------------------------
# canonicalisation
# --------------------------------------------------------------------------


# Structure markers. "~" because deepdiff's ignore_private_variables skips any
# key starting with "__", which would make a marker named __list__ -- and
# everything it wraps -- invisible to the diff; and because "~" cannot occur in
# a dataclass field name, so these collide with nothing.
_CLASS = "~class"
_DIGEST = "~digest"
_LIST = "~list"
_TUPLE = "~tuple"
_SET = "~set"


def _qualname(value: Any) -> str:
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _digest(tree: Any) -> str:
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode()).hexdigest()[:16]


def _canonical(value: Any, *, path: tuple[int, ...] = ()) -> Any:
    """One deployment's configuration as JSON-safe, order-stable nesting.

    Recurses through dataclasses, mappings and sequences — which is every
    container configuration is written in here — and stops at anything else
    with a type marker. Stopping matters: a live engine handle has no
    configuration in it and unbounded reach.

    Runtime state is skipped rather than snapshotted: underscore-prefixed
    fields and ``init=False`` ones are where an agent keeps its conversation,
    its counters and its started handles.
    """
    # Enums first, and not as their value: vLLM writes several as str/int
    # subclasses, so a scalar test matches them and stores the bare "backed"
    # -- which reloads from JSON as a plain string and reads as a type change
    # against the live object every run.
    if isinstance(value, enum.Enum):
        return f"<enum {_qualname(value)}.{value.name}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes sha256={hashlib.sha256(bytes(value)).hexdigest()[:16]} len={len(value)}>"
    # The `elements`/`items` locals are annotation sites, not style: these
    # containers arrive from vLLM's engine args as bare collections whose
    # element type the checker cannot see, and a comprehension has nowhere to
    # say Any.
    if isinstance(value, (set, frozenset)):
        members: list[Any] = list(cast("set[Any]", value))
        return {_SET: sorted(repr(_canonical(v)) for v in members)}
    if isinstance(value, (list, tuple)):
        elements: list[Any] = list(cast("list[Any]", value))
        return {(_TUPLE if isinstance(value, tuple) else _LIST): [_canonical(v) for v in elements]}
    if isinstance(value, Mapping):
        items: list[tuple[Any, Any]] = list(cast("Mapping[Any, Any]", value).items())
        return {str(k): _canonical(v) for k, v in sorted(items, key=lambda kv: str(kv[0]))}
    if isinstance(value, type):
        return f"<class {_qualname(value)}>"
    if dataclasses.is_dataclass(value):
        if id(value) in path:  # a cycle: an agent's tool holding the agent, say
            return f"<cycle {_qualname(value)}>"
        fields = {
            f.name: _canonical(getattr(value, f.name, None), path=path + (id(value),))
            for f in dataclasses.fields(value)
            if f.init and not f.name.startswith("_")
        }
        # The prompt bodies have their own golden tests (prompt_golden/); what
        # belongs here is only whether a deployment changed which prompts it
        # deploys, so they ride as a digest instead of thousands of lines.
        if type(value).__name__.endswith("Prompts"):
            return {_CLASS: _qualname(value), _DIGEST: _digest(fields)}
        return {_CLASS: _qualname(value), **fields}
    if callable(value):
        return f"<callable {getattr(value, '__qualname__', _qualname(value))}>"
    return f"<{_qualname(value)}>"


def _scrub_volatile(tree: Any) -> Any:
    """Blank the per-construction values, in place of dropping their keys: a
    field that STOPS being written is then still a visible difference."""
    if isinstance(tree, dict):
        mapping: dict[str, Any] = cast("dict[str, Any]", tree)
        return {k: ("<volatile>" if k in _VOLATILE_FIELDS else _scrub_volatile(v)) for k, v in mapping.items()}
    if isinstance(tree, list):
        return [_scrub_volatile(v) for v in cast("list[Any]", tree)]
    return tree


def _row(agent: LLMAgent) -> Any:
    """One deployment's row, in exactly the form the reference file holds.

    The JSON round-trip is not cosmetic: the reference is JSON, so anything
    the canonical form still carries that JSON cannot (a str subclass, an int
    key) would read as a type change against the file on every run. Both sides
    go through the same door.
    """
    # The moment this test is about: the last point an engine argument can
    # still change, which is where derived settings will resolve.
    agent.prepare_for_backend()
    return json.loads(json.dumps(_scrub_volatile(_canonical(agent)), sort_keys=True))


def _snapshot() -> dict[str, Any]:
    """Every deployment, settled and canonicalised."""
    return {name: _row(build()) for name, build in DEPLOYMENTS.items()}


def _diff(reference: Any, current: Any) -> Any:
    """Every difference, with nothing excluded: a change is read, not silenced."""
    return deepdiff.DeepDiff(
        reference,
        current,
        # Configuration is written as literals; 1 and 1.0 for a watermark are
        # the same setting, and a bool that became an int is not.
        ignore_numeric_type_changes=True,
        # OFF, against the default: it skips every key starting with "__",
        # which is meant for Python privates but applies to dict keys too, and
        # a snapshot is data where nothing is private.
        ignore_private_variables=False,
        verbose_level=2,
    )


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_every_deployment_matches_the_reference() -> None:
    """A change to any deployment is behaviour-neutral until a diff says otherwise."""
    current = _snapshot()
    if os.environ.get("PLATYPUS_REGENERATE_REFERENCES"):
        REFERENCE.write_text(json.dumps(current, indent=1, sort_keys=True) + "\n")
        pytest.skip(f"regenerated {REFERENCE.name}")
    assert REFERENCE.is_file(), f"{REFERENCE} is missing; regenerate with PLATYPUS_REGENERATE_REFERENCES=1"
    reference = json.loads(REFERENCE.read_text())

    difference = _diff(reference, current)
    assert not difference, (
        "a deployment's configuration moved.\n"
        "Intended? regenerate, and put this diff in the PR — it is the list of what "
        "the change did to every column. Unintended? that is what this exists for.\n"
        f"{difference.pretty()}"
    )


def test_the_snapshot_is_reproducible() -> None:
    """Twice in one process must agree, or a difference means nothing.

    This is what pins ``_VOLATILE_FIELDS``: the per-construction uuid inside
    the KV connector's config would otherwise make every run a failure.
    """
    assert not _diff(_snapshot(), _snapshot())


def test_the_reference_covers_every_deployment() -> None:
    """A row added to DEPLOYMENTS without regenerating is a silent gap."""
    reference = json.loads(REFERENCE.read_text())
    assert sorted(reference) == sorted(DEPLOYMENTS)


def _factory_of(agent: DuckAgent) -> Any:
    factory = agent.llm_config.backend_factory
    assert factory is not None
    return factory


# One mutation per layer the snapshot has to reach. A layer it cannot reach
# goes on passing quietly, so each is exercised rather than assumed.
_MUTATIONS: dict[str, Callable[[DuckAgent], None]] = {
    # Not a derived argument: settling one refuses a chosen value, so mutating
    # it here would exercise the guard rather than the snapshot's reach.
    "engine arg": lambda a: setattr(_factory_of(a).engine_args, "max_num_batched_tokens", 4096),
    "factory field": lambda a: setattr(_factory_of(a), "verify_serving_bug", True),
    "run config": lambda a: setattr(a.run_config, "max_concurrent", 3),
    "agent field": lambda a: setattr(a, "compact_to_tokens", 1234),
    "tuple field": lambda a: setattr(a, "compaction_forbidden_token_ids", (1234,)),
    "dict field": lambda a: _factory_of(a).env_overrides.__setitem__("MOVED", "1"),
    "sampling": lambda a: setattr(a.llm_config.sampling_profiles["play"], "temperature", 0.123),
    "prompts": lambda a: setattr(a.prompts, "puzzle_overview_addendum", "moved"),
    # The declaration is what this snapshot reaches; the connector it resolves
    # into is built on the worker at start, from these fields.
    "kv tier": lambda a: setattr(_factory_of(a), "cpu_kv_offload_gib", 7.0),
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_a_moved_setting_is_caught(name: str) -> None:
    """The test only earns its place if it fails when a setting moves."""
    mutate = _MUTATIONS[name]
    reference = _snapshot()
    agent = cast(DuckAgent, DEPLOYMENTS["slurm-3.8-direct"]())
    mutate(agent)
    current = copy.deepcopy(reference)
    current["slurm-3.8-direct"] = _row(agent)

    difference = _diff(reference, current)
    assert difference, f"a changed {name} left the snapshot identical"


@pytest.mark.parametrize(
    "before, after",
    [
        ([1, 2, 3], [1, 2, 4]),
        ((1, 2, 3), (1, 2, 4)),
        ({"a": 1}, {"a": 2}),
        (frozenset({1, 2}), frozenset({1, 3})),
        ([[1, 2]], [[1, 3]]),
    ],
)
def test_a_change_inside_a_container_is_visible(before: Any, after: Any) -> None:
    """The structure markers must not hide what they wrap: a marker deepdiff
    skips takes the whole container's contents out of the comparison."""
    assert _diff(_canonical(before), _canonical(after))
