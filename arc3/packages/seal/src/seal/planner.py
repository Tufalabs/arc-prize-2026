"""Search over a world model for what to do next.

Given a model and the trace so far, find a sequence of actions worth
executing: a win if one is reachable, otherwise the most informative place
to walk to — the edge of what the model claims to know, an action never
tried, a state never visited. When nothing is worth walking to, say so, and
say which kind of nothing it was, because the answer determines what the
model should be asked next.

Sync and pure, which is exactly what lets the agent ship it into a
subprocess: search sometimes exhausts memory running model-written code
thousands of times, and that has to take down a cheap child rather than the
game.

Failure modes are **return values**, not exceptions. Exhaustion, repetition
and an invisible edge are outcomes of a search, not errors — and it is what
makes the process boundary clean, since a raise cannot carry diagnostics
back alongside it without glueing them onto the exception.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import heapq
import numbers
import os
import pickle
import pprint
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Literal, Sequence, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from platypus.contract import Action
from seal.model import WorldModel
from seal.move_lister import MoveLister
from seal.rng import derive_rng
from seal.trace import LevelTrace

#: What a returned path was aiming at, highest priority first. A win is
#: always preferred; the rest are only reached when no win was found.
Target = Literal["win", "boundary", "untried_action", "unvisited", "untried_pair"]

#: Why a search came back empty-handed. Each asks the model for a different
#: kind of fix, which is why they are distinct.
Tag = Literal["exhausted", "repetition", "invisible_edges", "timeout"]

#: Never a search target in its own right: RESET always lands at
#: ``initial_state()``, so a "novel" RESET teaches nothing.
_RESET = "RESET"

#: A path as a chain of ``(prefix, action)`` links, innermost last, with
#: ``None`` for the empty path at the root. A child's path is its parent's link
#: plus one action, which is O(1) to build and shares the entire prefix —
#: where copying a list costs O(depth) in time and memory for every child, and
#: the frontier holds one path per queued node.
_PathLink: TypeAlias = "tuple[_PathLink | None, Action]"


def _path_actions(link: _PathLink | None) -> list[Action]:
    """The actions a chain spells out, root first."""
    out: list[Action] = []
    while link is not None:
        link, action = link
        out.append(action)
    out.reverse()
    return out


def _page_size() -> int:
    """The system page size, for turning RSS pages into bytes."""
    try:
        return os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 4096  # only reached if /proc/self/statm exists at all


_PAGE_SIZE = _page_size()


@dataclass(frozen=True)
class PlanDiagnostics:
    """What the search did, for the span. Never load-bearing."""

    exit_reason: str = "?"
    search_result: str = "?"
    nodes_explored: int = 0
    seen_states: int = 0
    path_depth: int | None = None
    max_depth_reached: int = -1
    best_first: bool = True
    best_first_disable_reason: str | None = None
    boundary_type: str | None = None
    mode_used: int | None = None
    max_frame_repetition: int | None = None
    render_errors: tuple[str, ...] = ()
    #: The most click regions on one searched board, and how many boards had
    #: more than the click cap (``MoveLister.most_regions``,
    #: ``boards_over_cap``). Zero in a game without clicks.
    most_click_regions: int = 0
    boards_over_click_cap: int = 0
    summary: str = ""


@dataclass(frozen=True)
class Plan:
    """A sequence of actions to execute, and what it was aiming at."""

    actions: tuple[Action, ...]
    target: Target
    #: This is the shortest path to a target of this type and a better one
    #: definitely does not exist. ``False`` is the *don't know* answer — it
    #: covers both "a shorter one exists but was not searched for" and "the
    #: search stopped early and cannot rule one out" — so it is safe to act
    #: on when ``True`` and carries no claim when ``False``.
    optimal: bool
    diagnostics: PlanDiagnostics


@dataclass(frozen=True)
class NoPlan:
    """No target was found, and which kind of nothing it was."""

    tag: Tag
    reason: str
    #: Everything a report of it needs. The planner renders what only it can
    #: see -- the states, which do not leave the search -- and hands back
    #: everything else raw. In particular ``detail["action"]`` is an
    #: ``Action``, not a string: how an action is spelled to the model is the
    #: agent's vocabulary, and rendering it here would put model-facing text
    #: in the planner.
    detail: dict[str, Any]
    diagnostics: PlanDiagnostics


def action_signature(action: Action, frame: npt.NDArray[np.int8] | None = None) -> tuple[Any, ...]:
    """Cross-game "action type" key for the untried-action target.

    Simple actions key on the name alone; a click additionally keys on the
    *colour* of the cell it lands on, so clicking a blue cell and a red one
    count as different experiments. ``frame`` is the board the action acts
    on — the real pre-action frame agent-side, the model's render inside
    search. Absent or out of range, a click degrades to name-only, which is
    still a valid signature, just coarser.
    """
    if action.name == "ACTION6" and frame is not None and action.row is not None and action.col is not None:
        try:
            return (action.name, int(frame[action.row, action.col]))
        except (IndexError, TypeError, ValueError):
            pass
    return (action.name,)


def _action_key(action: Action) -> tuple[Any, ...]:
    """Identity of an action for the untried-pair target: two clicks at
    different cells are different experiments."""
    return (action.name, action.row, action.col)


def state_values_equal(a: Any, b: Any) -> bool:
    """Compare two state values without letting numpy's elementwise ``==``
    turn a boolean test into an array."""
    a_is_array, b_is_array = isinstance(a, np.ndarray), isinstance(b, np.ndarray)
    if a_is_array or b_is_array:
        if not (a_is_array and b_is_array):
            return False
        return bool(np.array_equal(cast("npt.NDArray[Any]", a), cast("npt.NDArray[Any]", b)))
    try:
        return bool(a == b)
    except Exception:
        return False


class _CappedRepr:
    """Stands in for an oversized value so ``pprint`` renders a truncated
    string instead of a 64x64 array. Substitutes for the *object*, not for
    its repr, so nothing gets quoted."""

    __slots__ = ("_r",)

    def __init__(self, fixed_repr: str) -> None:
        self._r = fixed_repr

    def __repr__(self) -> str:
        return self._r


def _cap_value(value: Any, cap: int = 1000, edge: int = 100) -> Any:
    """``value`` if its repr is short enough, else an elided stand-in — search
    diagnostics must not carry a model's whole state dict.
    """
    r = repr(value)
    if len(r) <= cap:
        return value
    omitted = len(r) - 2 * edge
    return _CappedRepr(f"{r[:edge]}…<truncated {omitted} chars>…{r[-edge:]}")


def _cap_dict_values(d: dict[str, Any]) -> dict[str, Any]:
    """Per-value cap, top level only: the dict is assumed to have a sane
    number of keys; the cap protects against one field being huge."""
    return {k: _cap_value(v) for k, v in d.items()}


#: Held open across calls: the poll runs once per node expansion, where
#: opening the file costs more than reading it. ``None`` means "not opened in
#: this process yet".
_statm_fd: int | None = None


def _reset_statm_fd() -> None:
    """Drop the descriptor in a forked child.

    ``/proc/self`` resolves to ``/proc/<pid>`` at open time, so an inherited
    descriptor reports the *parent's* memory. Spawned children re-import and
    start at ``None`` anyway; this covers a fork.
    """
    global _statm_fd
    fd, _statm_fd = _statm_fd, None
    if fd is not None:
        with contextlib.suppress(OSError):
            os.close(fd)


os.register_at_fork(after_in_child=_reset_statm_fd)


def _current_rss_bytes() -> int:
    """Resident set size via ``/proc/self/statm``; 0 when unavailable, which
    callers treat as "unknown, skip the check"."""
    global _statm_fd
    try:
        if _statm_fd is None:
            _statm_fd = os.open("/proc/self/statm", os.O_RDONLY)
        return int(os.pread(_statm_fd, 128, 0).split()[1]) * _PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


#: Returned unchanged by :func:`_deepcopy_state`, matching
#: ``copy._deepcopy_atomic``. Deliberately excludes ``frozenset``,
#: ``bytearray`` and ``set``: ``copy.deepcopy`` rebuilds those, and sharing
#: what it would have rebuilt changes the state's pickle.
_ATOMIC_TYPES = frozenset({type(None), bool, int, float, complex, str, bytes})


def _deepcopy_state(x: Any, memo: dict[int, Any]) -> Any:
    """``copy.deepcopy`` specialised for the shapes a model state holds.

    One copy happens per child, which makes this among the hottest calls in the
    search; for a dict of scalars ``copy.deepcopy``'s dispatch and keep-alive
    bookkeeping cost more than the copying does. Every type not handled here is
    delegated, so the result is ``copy.deepcopy``'s for any input.

    Two details are load-bearing rather than cosmetic, because a state's dedup
    key is its pickle and pickle encodes a repeated object as a back-reference:
    aliasing is preserved through ``memo``, and a tuple whose items all copy to
    themselves is returned unchanged, exactly as ``copy._deepcopy_tuple`` does.

    No ``_keep_alive``: that guards against an original being freed mid-copy and
    its id reused, which cannot happen here — the caller holds the root for the
    whole call, so every object reachable from it stays alive.
    """
    cls = cast("type[Any]", type(x))
    if cls in _ATOMIC_TYPES:
        return x
    known = memo.get(id(x))
    if known is not None:
        return known
    if cls is dict:
        out: dict[Any, Any] = {}
        memo[id(x)] = out
        for k, v in x.items():
            out[k if type(k) in _ATOMIC_TYPES else _deepcopy_state(k, memo)] = _deepcopy_state(v, memo)
        return out
    if cls is list:
        items: list[Any] = []
        memo[id(x)] = items
        for v in x:
            items.append(_deepcopy_state(v, memo))
        return items
    if cls is tuple:
        copied = [_deepcopy_state(v, memo) for v in x]
        if id(x) in memo:
            # Copying the items reached this tuple again, so a copy of it
            # already exists and every reference has to be that one. Building
            # a second here would break the cycle the state was holding —
            # `copy.deepcopy` re-checks at exactly this point for the same
            # reason, and the dedup key is a digest of its bytes.
            return memo[id(x)]
        result: Any = x
        for original, new in zip(x, copied):
            if original is not new:
                result = tuple(copied)
                break
        memo[id(x)] = result
        return result
    return copy.deepcopy(x, memo)


#: Digest width for a state key, in bytes. At 128 bits a false dedup over a
#: 10M-state search sits around 1e-25, orders below the error rate of the
#: hardware the search runs on; at 64 bits it would be near 1e-2 per run,
#: which is not a margin. Widen rather than narrow if this is ever revisited.
_KEY_DIGEST_SIZE = 16


def _state_key(state: dict[str, Any]) -> bytes:
    """The identity a state is deduped on: a digest of its pickle.

    The pickle is what distinguishes two states; the digest is only how that
    identity is *stored*. Every use is an equality test — ``seen`` membership,
    the visited-in-history set, the untried-pair set — and the bytes are never
    read back, so keeping 16 of them instead of the whole pickle halves what a
    large search holds. ``seen`` retains one key per state ever reached and is
    never pruned, which makes it the term that decides where the memory
    ceiling lands.

    Two error modes, of opposite sign and very different size. Pickling is not
    a perfect value-equality test — two logically identical states can pickle
    differently through dict insertion order or a list re-allocation — so
    dedup *misses* those. That is the safe direction, costing work rather than
    losing a state, and the digest inherits it unchanged. The digest adds the
    other direction, a collision deduping two distinct states and dropping one
    subtree, bounded by the width above.
    """
    return hashlib.blake2b(pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), digest_size=_KEY_DIGEST_SIZE).digest()


def _make_dedup(
    energy_label: str | None,
) -> tuple[
    Callable[[dict[str, Any]], bytes], Callable[[Any, dict[str, Any], int], tuple[bytes, bool]], Callable[[], Any]
]:
    """Build the dedup primitives, specialised once per search.

    Returns ``(key_fn, dedup_check, seen_factory)``. An arrival survives if
    nothing already recorded for its state beats it, and storing it evicts
    whatever it beats. What "beats" means is the whole of the difference
    between the two forms: without an energy field it is arriving no later,
    so each state keeps its cheapest cost; with one, the field is zeroed in
    the key so visits at different energy share a state, and each state
    keeps the frontier of arrivals that are non-dominated in *both* cost and
    energy.

    Cost has to be part of it because the frontier is ordered by the model's
    ``distance_to_win``, not by cost, so a costlier route can reach a state
    first. Dedup that kept only the first arrival would then delete the
    cheaper one before it was ever queued — and win-finding is
    anytime-optimal, so the phase that would have found the shorter path
    searches a queue the shorter path was never put on, empties, and reports
    the longer win as shortest.

    A surviving arrival is pushed again, which is what reopening a state
    costs here: no separate machinery, since "not beaten" already means
    "worth continuing from".

    Keys come from :func:`_state_key`, whose docstring carries what the
    identity is and what it can get wrong. Search accepts the occasional
    missed dedup because the per-node cost is small; where the answer has to
    be right, :func:`state_values_equal` does a real per-field comparison
    instead.
    """
    if energy_label is None:

        def key_fn(s: dict[str, Any]) -> bytes:
            return _state_key(s)

        def dedup_check(seen: Any, state: dict[str, Any], cost: int) -> tuple[bytes, bool]:
            k = _state_key(state)
            prev = seen.get(k)
            if prev is not None and prev <= cost:
                return k, True
            seen[k] = cost
            return k, False

        return key_fn, dedup_check, dict

    lbl = energy_label

    def _missing(state: dict[str, Any]) -> RuntimeError:
        return RuntimeError(
            f"state[{lbl!r}] is missing; the planner has energy_label={lbl!r} "
            f"but the learned state does not contain it (no fallback)."
        )

    def key_fn(s: dict[str, Any]) -> bytes:
        # Swap energy to a sentinel, pickle, restore — cheaper than a dict
        # comprehension per call.
        if lbl not in s:
            raise _missing(s)
        original = s[lbl]
        s[lbl] = 0
        try:
            return _state_key(s)
        finally:
            s[lbl] = original

    def dedup_check(seen: Any, state: dict[str, Any], cost: int) -> tuple[bytes, bool]:
        if lbl not in state:
            raise _missing(state)
        e = state[lbl]
        state[lbl] = 0
        try:
            k = _state_key(state)
        finally:
            state[lbl] = e
        frontier = seen.get(k)
        if frontier is None:
            seen[k] = [(cost, e)]
            return k, False
        # Dominated by anything already here: no cheaper AND no more energy.
        if any(kept_cost <= cost and kept_energy >= e for kept_cost, kept_energy in frontier):
            return k, True
        # Kept, so drop whatever it dominates -- the frontier holds only the
        # arrivals that are still worth continuing from.
        survivors = [
            (kept_cost, kept_energy)
            for kept_cost, kept_energy in frontier
            if not (cost <= kept_cost and e >= kept_energy)
        ]
        survivors.append((cost, e))
        seen[k] = survivors
        return k, False

    return key_fn, dedup_check, dict


@dataclass
class _Targets:
    """The best candidate found for each target type, one search's worth.

    A win outranks every fallback, and is recorded rather than returned:
    under best-first the first win is rarely the shortest, so the search
    continues until everything cheaper is exhausted. ``win_cost`` is the
    bound the three prunes compare against — ``inf`` until a win exists,
    which makes the strict and non-strict comparisons correct with no
    None-guard.

    The fallbacks are listed in the order :meth:`BFSPlanner._finish` prefers
    them, which is also mode order: boundary, untried action, unvisited
    state, untried (state, action) pair.

    Paths are held as :data:`_PathLink` chains and materialised only by
    :meth:`BFSPlanner._finish`. ``None`` therefore reads unambiguously as
    "nothing recorded": every target is reached by at least one action, so a
    recorded link is never the empty path.
    """

    win: _PathLink | None = None
    win_cost: float = float("inf")
    boundary: tuple[_PathLink, str] | None = None
    untried_action: _PathLink | None = None
    unvisited: _PathLink | None = None
    novel: _PathLink | None = None

    def record_win(self, path: _PathLink, cost: int) -> None:
        """Keep ``path`` if it beats the best win so far."""
        if cost < self.win_cost:
            self.win = path
            self.win_cost = cost

    def note_boundary(self, path: _PathLink, kind: str) -> None:
        """Record the first boundary transition and ignore the rest.

        First is cheapest: the frontier is cost-monotonic below the best
        win, so nothing recorded later can be shorter.
        """
        if self.boundary is None:
            self.boundary = (path, kind)


@dataclass
class _Progress:
    """How far the search got, and why it stopped.

    Everything here is written during the loop and read once it ends, which
    is what lets :meth:`BFSPlanner._finish` take one argument for all of it.
    """

    nodes_explored: int = 0
    seen_states: int = 0
    #: Deepest cost popped. Starts below zero so an empty search reports it.
    max_depth_reached: int = -1
    best_first: bool = True
    disable_reason: str | None = None
    #: One repr per state the model could not draw; the search continues.
    render_errors: list[str] = field(default_factory=list["str"])
    early_exit: bool = False
    hit_timeout: bool = False
    hit_expansions: bool = False
    hit_memory: bool = False
    rss_at_exit: int = 0

    @property
    def hit_bound(self) -> bool:
        """Whether a resource, rather than the state space, ended the search."""
        return self.hit_timeout or self.hit_expansions or self.hit_memory

    def disable_best_first(self, reason: str) -> None:
        """Fall back to cost order for the rest of this search.

        One-way: the model's heuristic is not consulted again once it has
        misbehaved, so a search cannot alternate between two orderings.
        """
        self.best_first = False
        self.disable_reason = reason


@dataclass
class BFSPlanner:
    """Best-first search with an anytime-optimal win hunt and a cascade of
    fallback targets.

    The frontier is a min-heap keyed ``(priority, cost, push_counter, ...)``.
    ``priority`` is ``distance_to_win`` when the model defines a well-behaved
    one and 0 otherwise, so without a heuristic the ordering collapses to
    ``(cost, push_counter)`` — cost-monotonic, FIFO within a cost, exactly a
    plain BFS. ``push_counter`` is unique, so the heap never compares a state
    dict.

    Win-finding is anytime-optimal: the first win is recorded, not returned.
    Search continues with three prunes that keep the frontier strictly below
    the best known win's cost, so the queue self-empties once every cheaper
    candidate is explored — at which point the win is provably shortest.
    """

    #: Target ceiling. 0 = win and boundary only; 1 adds untried action
    #: types; 2 adds unvisited states; 3 adds untried (state, action) pairs.
    mode_max: int = 1
    #: Wall-clock budget for one search, in seconds. Checked between node
    #: expansions, so a single model-written ``do_move`` that hangs is not
    #: caught here — that is ``search_process``'s hard timeout.
    max_duration: float = 600.0
    #: Wall-clock budget for the residual search *after* the first win.
    #:
    #: The win hunt is anytime-optimal, so it keeps going past the first win
    #: looking for something shorter. At depth that residual search can eat
    #: the rest of ``max_duration`` for little marginal benefit, so the
    #: deadline tightens to ``now + this`` the moment the first win lands.
    #: Stamped once: later wins inherit the cap and do not extend it.
    #: ``None`` searches until exhaustion or ``max_duration``.
    max_duration_after_first_win: float | None = 60.0
    #: Deterministic companion to ``max_duration``: a bound that does not
    #: depend on how fast the machine is, so a recorded run and its replay
    #: explore the same space. ``None`` = unbounded.
    max_expansions: int | None = None
    #: Deterministic companion to ``max_duration_after_first_win``, with the
    #: same stamped-once semantics counted in expansions.
    max_expansions_after_first_win: int | None = None
    #: Reject a path that visits the same rendered frame more often than
    #: this. ``None`` disables the check.
    allow_repetition: int | None = 3
    prune_invisible_edges: bool = False
    #: Polled inside the loop, so a state-space blow-up exits with full
    #: diagnostics instead of being OOM-killed. Linux-only; ``None``
    #: disables.
    memory_limit_bytes: int | None = 10 * 2**30
    #: Name of the per-state energy scalar, used for Pareto dedup. When set,
    #: every state reached during search must contain it — strict, no
    #: fallback. ``None`` disables energy-aware dedup.
    energy_label: str | None = "energy"
    #: Re-raise a model crash out of the search instead of recording it as
    #: an error boundary. Off by default and inert in any recorded run; it
    #: exists so a post-mortem debugger lands on the model's own frame
    #: rather than on a boundary target several moves later.
    raise_model_errors: bool = False
    move_lister: MoveLister = field(default_factory=MoveLister)
    #: Action-type signatures already executed for real. Cross-level and
    #: cross-game, owned by the agent, which adds one for every action it
    #: plays and never resets it; search only reads it.
    tried_action_signatures: set[tuple[Any, ...]] = field(default_factory=set[tuple[Any, ...]])

    #: Mode floor for this search: it exits as soon as it holds a target at
    #: or below the floor, so a model already known to have nothing at the
    #: lower modes does not pay to re-search them.
    #:
    #: **Read here, owned by the agent.** ``plan`` runs in a spawned child
    #: with a pickled copy of this planner, so a floor raised here would die
    #: with that child; the agent sets it before dispatch and carries it
    #: forward from ``PlanDiagnostics.mode_used``.
    min_mode: int = 0

    def plan(
        self,
        model: WorldModel,
        trace: LevelTrace,
        available_actions: Sequence[str],
        *,
        seed: int,
    ) -> Plan | NoPlan:
        """Search ``model`` from where ``trace`` currently stands.

        ``seed`` is this planning call's derived seed; per-node move
        orderings derive from it and a node counter, so the number of draws
        never escapes the search however large it gets (see ``seal.rng``).
        """
        with model.active():
            return self._search(model, trace, available_actions, seed=seed)

    def _search(  # noqa: C901 — one loop, deliberately not split
        self,
        model: WorldModel,
        trace: LevelTrace,
        available_actions: Sequence[str],
        *,
        seed: int,
    ) -> Plan | NoPlan:
        """The search itself: a best-first frontier when the model's heuristic
        behaves, anytime-optimal win finding, the fallback-target cascade, and
        the resource ceilings. One loop, deliberately not split.
        """
        start_state = trace.current_state()
        n_list_moves = 0  # per-node rng coordinate within this call
        self.move_lister.reset_counts()
        #: The game is over and lost, so the root admits RESET and nothing
        #: else. Read from the trace rather than passed in: the loss is a
        #: property of the observed history, and the agent deliberately does
        #: not special-case it.
        lost_at_root = bool(trace.steps) and trace.steps[-1].outcome == "lost"

        targets = _Targets()
        progress = _Progress()

        # Best-first via the model's own heuristic. Optional and
        # model-written, so nothing is pre-checked: try it, and let any
        # failure — absent, raises, non-numeric — flip the flag off for the
        # rest of the search. Once off, the function is never called again.
        def priority(s: dict[str, Any]) -> Any:
            if not progress.best_first:
                return 0
            try:
                d = model.distance_to_win(s)
            except Exception as e:
                progress.disable_best_first(f"distance_to_win raised: {type(e).__name__}: {e}")
                return 0
            if d is None:
                progress.disable_best_first("distance_to_win is not defined by this model")
                return 0
            if not isinstance(d, numbers.Real):
                progress.disable_best_first(f"distance_to_win returned non-numeric (got {type(d).__name__})")
                return 0
            # Returned as-is, not coerced: an exact integer heuristic beyond
            # float precision would otherwise collapse distinct priorities
            # into one.
            return d

        key_fn, dedup_check, seen_factory = _make_dedup(self.energy_label)

        # Energy-masked render: drop the per-state energy fill so an energy
        # bar ticking down alone does not make two otherwise-identical frames
        # look different. Skipped when there is no energy label, or when
        # initial_state misbehaves on the lookup.
        initial_energy_for_mask: Any = None
        if self.prune_invisible_edges and self.energy_label is not None:
            initial_energy_for_mask = self._initial_energy(model)

        def frame_key(s: dict[str, Any]) -> bytes:
            return self._frame_key(model, s, initial_energy_for_mask)

        mode_max = self.mode_max
        min_mode = self.min_mode

        tried_signatures = self.tried_action_signatures

        visited_in_history: set[bytes] = set()
        if mode_max >= 2:
            visited_in_history = {key_fn(s) for s in trace.states}

        tried_pairs: set[tuple[Any, ...]] = set()
        if mode_max >= 3:
            for i, step in enumerate(trace.steps):
                if i == 0 or step.action is None:
                    continue
                tried_pairs.add((key_fn(trace.states[i - 1]), *_action_key(step.action)))

        # cost == the path's length. RESET costs 1 like any other action,
        # matching the engine's per-level counter.
        queue: list[tuple[Any, int, int, dict[str, Any], "_PathLink | None", bytes]] = []
        seen = seen_factory()
        root_key, _ = dedup_check(seen, start_state, 0)
        push_counter = 0
        heapq.heappush(queue, (priority(start_state), 0, push_counter, start_state, None, root_key))
        push_counter += 1

        # RESET is enumerated only at the root: MoveLister strips it, and
        # from any deeper state it lands at initial_state(), which the root
        # already explored, so dedup would drop it anyway.
        reset_action = Action(_RESET)
        has_click = "ACTION6" in available_actions

        # Both stamped once, on the first win, and never reset.
        post_win_deadline: float | None = None
        post_win_ceiling: int | None = None
        t0 = time.monotonic()

        while queue:
            if self._bound_hit(progress, t0=t0, deadline=post_win_deadline, ceiling=post_win_ceiling):
                break
            if self._can_exit_early(progress, targets, min_mode):
                break

            _, path_cost, _, state, path, state_key = heapq.heappop(queue)
            progress.nodes_explored += 1

            # Lazily computed the first time a child of this parent reaches
            # the invisible-edge check, so a parent whose children are all
            # leaves never pays the render.
            parent_frame_key: bytes | None = None

            # Prune at pop: every child of this node costs path_cost + 1, so
            # if that already matches the best win, no expansion can improve
            # and the whole move enumeration is skipped.
            if path_cost + 1 >= targets.win_cost:
                continue

            progress.max_depth_reached = max(progress.max_depth_reached, path_cost)

            if lost_at_root and path is None:
                # Game over: the engine refuses anything but RESET. No
                # `list_moves` call — it would advance `n_list_moves`, and
                # every later node's rng coordinate derives from that.
                parent_render = None
                actions = [reset_action]
            else:
                try:
                    parent_render = model.render(state) if has_click else None
                except Exception as e:
                    # A reachable state the model cannot draw, in a game whose
                    # clicks are listed from the board. A target of its own
                    # kind — the move there is predicted, its board is not —
                    # and not expanded, since its clicks cannot be listed. A
                    # game without clicks never draws a state here at all.
                    if path is not None:
                        targets.note_boundary(path, "render_error")
                    progress.render_errors.append(repr(e))
                    continue

                actions = self.move_lister.list_moves(
                    available_actions, parent_render, derive_rng(seed, "node", n_list_moves)
                )
                n_list_moves += 1
                if path is None:
                    actions = [reset_action] + actions

            for action in actions:
                # Prune at the sibling level: all of this parent's children
                # share a cost, so once a win at or below it exists, every
                # remaining sibling is doomed — break, not continue.
                if path_cost + 1 >= targets.win_cost:
                    break

                if (
                    mode_max >= 1
                    and targets.untried_action is None
                    and action.name != _RESET
                    and action_signature(action, parent_render) not in tried_signatures
                ):
                    targets.untried_action = (path, action)

                # Checked before simulating: cost-monotonicity means the
                # first untried pair recorded is the cheapest path to one.
                if (
                    mode_max >= 3
                    and targets.novel is None
                    and action.name != _RESET
                    and (state_key, *_action_key(action)) not in tried_pairs
                ):
                    targets.novel = (path, action)

                child_state = _deepcopy_state(state, {})
                prediction = model.step(child_state, action)
                if self.raise_model_errors and prediction.kind == "error":
                    # The object when it survived, the traceback string when
                    # it did not — the string is the guaranteed field.
                    raise prediction.exception or RuntimeError(prediction.error)
                child_path: _PathLink = (path, action)
                child_cost = path_cost + 1

                if prediction.outcome == "won":
                    # Recorded, not returned: under best-first the first win
                    # is rarely the shortest, so the search keeps going until
                    # everything cheaper is exhausted.
                    targets.record_win(child_path, child_cost)
                    # Tightened on the FIRST win only; later wins inherit the
                    # caps and do not extend them.
                    if post_win_deadline is None and self.max_duration_after_first_win is not None:
                        post_win_deadline = time.monotonic() + self.max_duration_after_first_win
                    if post_win_ceiling is None and self.max_expansions_after_first_win is not None:
                        post_win_ceiling = progress.nodes_explored + self.max_expansions_after_first_win
                    continue

                if prediction.outcome == "lost":
                    # Never expanded — from a lost state the engine accepts
                    # only RESET, which the root already enumerates. The
                    # transition can still be a target, though.
                    if mode_max >= 2:
                        # Keyed unconditionally, before the `is None` test:
                        # the strict energy contract is enforced by key_fn
                        # raising, and short-circuiting past it would let a
                        # state missing its energy field through unnoticed.
                        child_key = key_fn(child_state)
                        if child_key not in visited_in_history and targets.unvisited is None:
                            targets.unvisited = child_path
                    continue

                if prediction.kind in ("error", "unknown"):
                    targets.note_boundary(child_path, prediction.kind)
                    continue

                child_key, dominated = dedup_check(seen, child_state, child_cost)
                if dominated:
                    continue

                if mode_max >= 2 and child_key not in visited_in_history and targets.unvisited is None:
                    targets.unvisited = child_path

                if self.prune_invisible_edges and action.name != _RESET:
                    try:
                        if parent_frame_key is None:
                            parent_frame_key = frame_key(state)
                        child_frame_key = frame_key(child_state)
                    except Exception as e:
                        targets.note_boundary(child_path, "render_error")
                        progress.render_errors.append(repr(e))
                        continue
                    if child_frame_key == parent_frame_key and self._changed_keys(state, child_state):
                        progress.seen_states = len(seen)
                        return self._invisible_edge(
                            state,
                            action,
                            child_state,
                            self._diagnostics(
                                progress,
                                exit_reason="invisible_edges",
                                search_result="invisible_edges",
                            ),
                        )

                # No prune needed here: the sibling break above already
                # rejected every candidate that cannot beat the best win.
                heapq.heappush(
                    queue,
                    (
                        priority(child_state),
                        child_cost,
                        push_counter,
                        child_state,
                        child_path,
                        child_key,
                    ),
                )
                push_counter += 1

        progress.seen_states = len(seen)
        return self._finish(model, start_state, targets, progress)

    def _bound_hit(self, progress: _Progress, *, t0: float, deadline: float | None, ceiling: int | None) -> bool:
        """Whether a resource bound has fired, recording which one on ``progress``.

        ``deadline`` and ``ceiling`` are the tighter post-win caps, stamped
        once when the first win is found. A clock and a memory limit depend
        on the machine while an expansion budget does not, which is why the
        three stay distinguishable here even though they share a give-up tag.
        """
        now = time.monotonic()
        if now - t0 > self.max_duration or (deadline is not None and now > deadline):
            progress.hit_timeout = True
            return True
        expansions = progress.nodes_explored
        if (self.max_expansions is not None and expansions >= self.max_expansions) or (
            ceiling is not None and expansions >= ceiling
        ):
            progress.hit_expansions = True
            return True
        if self.memory_limit_bytes:
            rss = _current_rss_bytes()
            if rss > self.memory_limit_bytes:
                progress.hit_memory = True
                progress.rss_at_exit = rss
                return True
        return False

    def _can_exit_early(self, progress: _Progress, targets: _Targets, min_mode: int) -> bool:
        """Whether the sticky floor lets this search stop with what it has.

        At floor 0 it never does — a deeper win could still appear, so the
        search runs to exhaustion. At a higher floor the previous call
        established that this model has no target below it, so the first
        thing in hand at or under the floor is the answer. The conditions
        are cumulative: at floor N every earlier-mode slot also counts.
        """
        found = (
            min_mode >= 1 and (targets.boundary is not None or targets.untried_action is not None),
            min_mode >= 2 and targets.unvisited is not None,
            min_mode >= 3 and targets.novel is not None,
        )
        if any(found):
            progress.early_exit = True
            return True
        return False

    # --- outcome assembly -------------------------------------------------

    def _finish(
        self,
        model: WorldModel,
        start_state: dict[str, Any],
        targets: _Targets,
        progress: _Progress,
    ) -> Plan | NoPlan:
        """Turn what the search accumulated into a plan, or a give-up."""

        def diagnostics(**kwargs: Any) -> PlanDiagnostics:
            return self._diagnostics(progress, **kwargs)

        # A win outranks every fallback. If the loop ran to natural
        # exhaustion the win is provably shortest: every cheaper candidate
        # was either explored or pruned because it could not beat it. Under
        # an early exit or a resource bound the win is valid but unproven.
        if targets.win is not None:
            path = _path_actions(targets.win)
            if progress.hit_timeout:
                exit_reason = "timeout_after_win"
            elif progress.hit_expansions:
                exit_reason = "expansions_after_win"
            elif progress.hit_memory:
                exit_reason = "memory_limit_after_win"
            elif progress.early_exit:
                exit_reason = "early_exit_after_win"
            else:
                exit_reason = "win_optimal"
            diag = diagnostics(
                exit_reason=exit_reason,
                search_result="win",
                path_depth=len(path),
                mode_used=0,
            )
            return self._returning(model, start_state, path, "win", exit_reason == "win_optimal", diag)

        if progress.early_exit:
            exit_reason = "early_exit"
        elif progress.hit_timeout:
            exit_reason = "timeout"
        elif progress.hit_expansions:
            exit_reason = "expansions"
        elif progress.hit_memory:
            exit_reason = "memory_limit"
        else:
            exit_reason = "exhausted"

        # The sticky floor only gated the early exit while searching. Now
        # that the loop is done, take the best target found regardless: if a
        # boundary turned up while running at a raised floor, the assumption
        # that this model had none was simply wrong.
        boundary = targets.boundary
        cascade: list[tuple[_PathLink | None, Target, str, int, str | None]] = [
            (boundary[0] if boundary else None, "boundary", "boundary", 0, boundary[1] if boundary else None),
            (targets.untried_action, "untried_action", "untried_action", 1, None),
            (targets.unvisited, "unvisited", "unvisited", 2, None),
            (targets.novel, "untried_pair", "novel_move", 3, None),
        ]
        for link, target, search_result, mode, boundary_type in cascade:
            if link is None:
                continue
            path = _path_actions(link)
            diag = diagnostics(
                exit_reason=exit_reason,
                search_result=search_result,
                path_depth=len(path),
                boundary_type=boundary_type,
                mode_used=mode,
            )
            # Not claimed optimal: the fallback targets return the first hit
            # rather than searching on for a shorter one.
            return self._returning(model, start_state, path, target, False, diag)

        # Nothing at any tracked mode. One tag for all three resource bounds;
        # which bound fired stays in ``exit_reason`` -- and that distinction
        # matters downstream: timing and memory depend on the machine, an
        # expansion count does not, so only the first two make a run
        # unreproducible.
        if progress.hit_bound:
            if progress.hit_timeout:
                reason = f"search timed out after {self.max_duration:.1f}s without finding a target"
            elif progress.hit_expansions:
                reason = f"search hit its expansion budget ({self.max_expansions}) without finding a target"
            else:
                limit = self.memory_limit_bytes or 0
                reason = (
                    f"search hit its memory limit ({progress.rss_at_exit / 1024**3:.2f} GiB > "
                    f"{limit / 1024**3:.2f} GiB) without finding a target"
                )
            return NoPlan(
                tag="timeout",
                reason=reason,
                detail={},
                diagnostics=diagnostics(exit_reason=exit_reason, search_result=exit_reason),
            )
        return NoPlan(
            tag="exhausted",
            reason="search exhausted all reachable states without finding a target",
            detail={},
            diagnostics=diagnostics(exit_reason=exit_reason, search_result="exhausted"),
        )

    def _returning(
        self,
        model: WorldModel,
        start_state: dict[str, Any],
        path: list[Action],
        target: Target,
        optimal: bool,
        diag: PlanDiagnostics,
    ) -> Plan | NoPlan:
        """Last gate before a path is handed back: the repetition check."""
        max_rep, rejection = self._check_repetition(model, start_state, path)
        if max_rep is not None:
            diag = replace(diag, max_frame_repetition=max_rep)
        if rejection is not None:
            return replace(rejection, diagnostics=diag)
        return Plan(actions=tuple(path), target=target, optimal=optimal, diagnostics=diag)

    # --- helpers ----------------------------------------------------------

    def _initial_energy(self, model: WorldModel) -> Any:
        """The energy the model starts with, or ``None`` if it has no energy field
        or cannot say.
        """
        try:
            initial = model.initial_state()
        except Exception:
            return None
        if self.energy_label is not None and self.energy_label in initial:
            return initial[self.energy_label]
        return None

    def _frame_key(self, model: WorldModel, s: dict[str, Any], initial_energy: Any) -> bytes:
        """A state's render as a dedup key, with energy neutralised — two visits to
        the same board differing only in energy must collide, or the Pareto
        dedup cannot compare them.
        """
        if initial_energy is None or self.energy_label is None:
            return model.render(s).tobytes()
        original = s.get(self.energy_label)
        s[self.energy_label] = initial_energy
        try:
            return model.render(s).tobytes()
        finally:
            s[self.energy_label] = original

    def _changed_keys(self, parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
        """Non-energy state fields that actually differ. The authoritative
        comparison — unlike the dedup key, which is fast-but-fuzzy.

        Sorted, because the result is listed to the model in the
        invisible-edge and repetition prompts: a set of strings iterates in
        ``PYTHONHASHSEED`` order, which nothing pins, so an unsorted walk
        makes that prompt unreproducible from one process to the next.
        """
        changed: dict[str, Any] = {}
        for k in sorted(set(parent.keys()) | set(child.keys())):
            if k == self.energy_label:
                continue
            if not state_values_equal(parent.get(k), child.get(k)):
                changed[k] = child.get(k)
        return changed

    def _invisible_edge(
        self,
        parent_state: dict[str, Any],
        action: Action,
        child_state: dict[str, Any],
        diag: PlanDiagnostics,
    ) -> NoPlan:
        """An edge that moved hidden state without changing the picture.

        The model is wrong in a way that makes further search pointless — no
        useful target is reachable when ``do_move`` mutates state the render
        never shows — so the search stops here rather than continuing.
        """
        changed = self._changed_keys(parent_state, child_state)
        return NoPlan(
            tag="invisible_edges",
            reason=(
                f"search hit a do_move edge (action={action!r}) that changed state keys "
                f"{sorted(changed.keys())} without changing the energy-masked render — "
                f"the world model is incorrect"
            ),
            detail={
                "example_state_repr": pprint.pformat(
                    _cap_dict_values(parent_state), indent=2, width=80, sort_dicts=False
                ),
                "action": action,
                "changed_keys": _cap_dict_values(changed),
            },
            diagnostics=diag,
        )

    def _check_repetition(
        self,
        model: WorldModel,
        start_state: dict[str, Any],
        path: list[Action],
    ) -> tuple[int | None, NoPlan | None]:
        """Re-walk ``path`` rendering at each step; reject it if it visits
        the same frame too often.

        Returns ``(max_frame_repetition, rejection)`` — the count is for
        diagnostics and is reported even when the path is accepted; the
        rejection carries placeholder diagnostics the caller replaces.

        This catches the "do this a thousand times to win" failure mode the
        model is prone to. Both conditions must hold to reject, and the
        second matters as much as the first:

        - the frame-repetition threshold is crossed;
        - **at least one repeating transition really changed non-energy
          state**. A path can cross the threshold purely through dedup
          misses (states logically equal that pickled differently); those
          are a search inefficiency, not a model bug.

        Renders normalise the energy field to its initial value, so an
        energy bar ticking down does not make identical frames look
        distinct.
        """
        if self.allow_repetition is None or not path:
            return None, None

        initial_energy = self._initial_energy(model) if self.energy_label is not None else None

        state = copy.deepcopy(start_state)
        counts: dict[bytes, int] = {}
        try:
            key = self._frame_key(model, state, initial_energy)
        except Exception:
            # Cannot draw the start state; the check is advisory and the
            # render failure surfaces downstream when the path is executed.
            return None, None
        counts[key] = 1
        max_rep = 1
        example: tuple[dict[str, Any], Action, dict[str, Any]] | None = None

        for action in path:
            previous = copy.deepcopy(state)
            prediction = model.step(state, action)
            # Past the end of a sensible path there are no meaningful frames.
            if prediction.kind in ("error", "unknown"):
                break
            if prediction.outcome in ("won", "lost"):
                break
            try:
                key = self._frame_key(model, state, initial_energy)
            except Exception:
                break
            counts[key] = counts.get(key, 0) + 1
            max_rep = max(max_rep, counts[key])
            if counts[key] >= 2 and example is None and self._changed_keys(previous, state):
                example = (previous, action, copy.deepcopy(state))
            if max_rep > self.allow_repetition and example is not None:
                break

        if not (max_rep > self.allow_repetition and example is not None):
            return max_rep, None

        previous, action, current = example
        # Between N+1 identical frames there are N transitions that advanced
        # the state without changing the render — that is the count the
        # prompt calls out.
        n_moves = max_rep - 1
        changed = self._changed_keys(previous, current)
        return max_rep, NoPlan(
            tag="repetition",
            reason=(
                f"the path visits the same rendered frame {max_rep} times "
                f"(> allow_repetition={self.allow_repetition}); the model's rules likely "
                f"require meaningless repetition to reach the target"
            ),
            detail={
                "n_moves": n_moves,
                "example_state_repr": pprint.pformat(_cap_dict_values(previous), indent=2, width=80, sort_dicts=False),
                "action": action,
                "changed_keys": _cap_dict_values(changed),
            },
            diagnostics=PlanDiagnostics(),
        )

    def _diagnostics(
        self,
        progress: _Progress,
        *,
        exit_reason: str = "?",
        search_result: str = "?",
        path_depth: int | None = None,
        boundary_type: str | None = None,
        mode_used: int | None = None,
    ) -> PlanDiagnostics:
        """A report of what the search did (from ``progress``) and what it
        decided (from the keywords, which only the caller knows)."""
        diag = PlanDiagnostics(
            exit_reason=exit_reason,
            search_result=search_result,
            nodes_explored=progress.nodes_explored,
            seen_states=progress.seen_states,
            path_depth=path_depth,
            max_depth_reached=progress.max_depth_reached,
            best_first=progress.best_first,
            best_first_disable_reason=progress.disable_reason,
            boundary_type=boundary_type,
            mode_used=mode_used,
            render_errors=tuple(progress.render_errors),
            most_click_regions=self.move_lister.most_regions,
            boards_over_click_cap=self.move_lister.boards_over_cap,
        )
        return replace(diag, summary=_summary(diag, progress.rss_at_exit, self.memory_limit_bytes))


def _summary(diag: PlanDiagnostics, rss_at_exit: int = 0, memory_limit_bytes: int | None = None) -> str:
    """The one-line search summary carried in diagnostics."""
    rss_gib = rss_at_exit / 1024**3
    limit_gib = (memory_limit_bytes or 0) / 1024**3
    exit_labels = {
        "win_optimal": "win found and proven shortest (queue exhausted)",
        "timeout_after_win": "win found, then the search timed out — shortest unverified",
        "expansions_after_win": "win found, then the expansion budget ran out — shortest unverified",
        "memory_limit_after_win": (
            f"win found, then memory limit hit ({rss_gib:.2f} GiB > {limit_gib:.2f} GiB) — shortest unverified"
        ),
        "early_exit_after_win": (
            f"win found, then sticky-mode early exit (mode {diag.mode_used} target in hand) — shortest unverified"
        ),
        "exhausted": "queue emptied (full search space explored)",
        "timeout": "wall-clock timeout",
        "expansions": "expansion budget exhausted",
        "memory_limit": f"memory limit hit ({rss_gib:.2f} GiB > {limit_gib:.2f} GiB)",
        "early_exit": f"sticky-mode early exit (mode {diag.mode_used} target in hand)",
        "invisible_edges": "aborted: a do_move edge changed hidden state without changing the render",
    }
    result_labels = {
        "win": "returning winning path",
        "boundary": f"returning path to boundary ({diag.boundary_type})",
        "untried_action": "returning path to untried action type",
        "unvisited": "returning path to unvisited state",
        "novel_move": "returning path to untried (state, action)",
        "exhausted": "GIVING UP — no viable target in full search space",
        "timeout": "GIVING UP — timed out before finding a target",
        "expansions": "GIVING UP — expansion budget ran out before finding a target",
        "memory_limit": "GIVING UP — hit memory limit before finding a target",
        "invisible_edges": "GIVING UP — model bug, search aborted",
    }
    lines = [
        f"search exit: {exit_labels.get(diag.exit_reason, diag.exit_reason)}",
        f"result:      {result_labels.get(diag.search_result, diag.search_result)}",
        f"stats:       {diag.nodes_explored} nodes, {diag.seen_states} states",
    ]
    if diag.max_depth_reached >= 0:
        lines[-1] += f", max depth {diag.max_depth_reached}"
    lines.append(f"path:        {'none' if diag.path_depth is None else f'length {diag.path_depth}'}")
    return "\n".join(lines)
