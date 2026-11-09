"""Agent-facing contract types.

These do not replace TAAF's data model. An agent answers with a
``taaf.game.Submission`` of ``taaf.grid.Grid``s and reads a
``taaf.game.GameView``, all of which are ours already and validated at
construction. What lives here is what those types do not say: the give-up
channel, the terminal summary, and the hosting configuration.

The boundary is by kind rather than by package. Agents may use TAAF **value**
types — ``Grid``, ``Pair``, ``GameView``, ``Submission`` — and never TAAF
**hosting** types: ``Game``, ``GameRun``, ``Benchmark``, ``Solver``. The
solver remains the only code that touches a live game, which is what keeps an
agent from reaching the scoring it must not see.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import ullm.backends.proxy
import ullm.kaggle


class ContractViolationError(Exception):
    """An agent broke the contract (submitted after its call ended, a
    mis-shaped answer, ...).

    Raised by the framework-owned wrappers; handled per
    ``RunConfig.on_agent_error`` like any other agent exception.
    """


class GiveUpError(Exception):
    """Raising this from any depth inside ``_do_move`` is equivalent to
    returning ``GiveUp`` — the public ``do_move`` converts it."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class GiveUp:
    """An agent's decision to stop playing this game early.

    Whatever it last submitted still stands and is still scored — giving up
    frees the slot, it does not withdraw the answer. ``reason`` is free text
    and lands on the move span's outcome.
    """

    reason: str


@dataclass(frozen=True)
class GameOutcome:
    """How a game ended, handed to ``on_finish`` on every terminal path.

    Named apart from ``taaf.game.GameResult``, which is the *scoring* of an
    answer and deliberately never reaches an agent: an answer's result is
    the ground truth, so an agent that could read it would have a
    correctness oracle.

    ``state`` is deliberately **not** TAAF's ``GameRun.state``, which
    distinguishes ``won`` from ``submitted`` — and "won" means at least one
    pair came back right, which is the same oracle wearing a different hat.
    These four say only how the agent's own call ended, which is knowable
    without scoring anything:

    - ``finished``: ``do_move`` returned normally.
    - ``gave_up``: it returned ``GiveUp`` (or raised ``GiveUpError``).
    - ``cancelled``: it was cut off — the wall budget or a run deadline.
    - ``crashed``: it raised.

    Whatever it had submitted still stands and is still scored in every one
    of those cases; the agent simply does not get to see how that went.
    """

    state: Literal["finished", "gave_up", "cancelled", "crashed"]
    n_submitted: int


#: Slot hint (``MoveContext.yield_slot``): which take of its game an agent is
#: playing, 0 for the first. The ranking charges ``take_handicap_tokens`` per
#: take after the first.
TAKE_INDEX_HINT = "take_index"

#: Slot hint: this offer sits between two takes, where no context is resident,
#: so a swap costs nothing — free in the sense ``compacted`` is.
TAKE_BOUNDARY_HINT = "take_boundary"

#: Slot hint: tokens the game asks the ranking to put behind its spend, on top
#: of any take handicap — the duck's ``audit_handicap_tokens``, once its audit
#: has compacted. Absent is zero.
HANDICAP_HINT = "handicap_tokens"


@dataclass
class RunConfig:
    """Hosting knobs; lives ON the agent, read by ``AgentSolver`` from the
    template once at setup (per-game copies carry it inertly). A plain
    mutable dataclass: tweak fields freely while composing an experiment;
    the solver reads the values at run start, so edits after that have no
    effect on a running solver.

    ``rng_seed``: ``None`` (default) draws each game's seed from OS
    entropy; a fixed value makes runs reproducible (per-game seed =
    ``rng_seed`` + the game's index in the run). Either way the per-game
    seed is a concrete integer, resolved once by ``AgentSolver._game_seed``
    and recorded, so any finished run can be replayed with the seeds it
    actually used.

    ``shuffle_games`` / ``shuffle_seed``: play the run's (game, pass)
    entries in a random order rather than the benchmark's. Worth having
    under a hard deadline: whichever tasks get cut should be a random
    sample rather than the tail of a fixed order. ``shuffle_seed=None``
    derives the seed from the current date (2026-07-01 -> 20260701), which
    is what lets the same notebook be submitted repeatedly and still name
    the order it played; the resolved value is printed at run start either
    way.

    ``play_priority``: per-``game_id`` priority, highest started first,
    applied stably after any shuffle; see the field.

    ``rng_seed_per_game``: explicit per-game seeds by ``game_id``, taking
    precedence over the derived value. This is how a run reproduces
    randomness it did not generate itself — a captured reference, or a run
    recorded by different code entirely. Keyed like
    ``llm_config.replay_dir`` and sharing its limitation: with
    ``n_passes > 1`` every pass copy of a game would take the same seed.
    """

    label: str = ""  # "" -> agent class name
    max_concurrent: int = 32
    # Floor on the gap between a game's consecutive requests, from the end
    # of one to the start of the next; the time the game spends in the
    # kernel counts toward it, so a game slower than this to act waits
    # nothing. Under an engine that queues admissions (ullm's kv_offload.md)
    # the wait is spent in the queue the game would sit in anyway, and it
    # moves the continuation past the moment its blocks are leaving the GPU
    # with their copies in flight: it finds them resident or in the tier,
    # never between. Off by default: under the tier's park mode, which every
    # engine that keeps conversations live runs, the tier itself holds a
    # lookup until its conversation's copy has landed (a step or few, not
    # seconds), so the floor only slows the games. 5 s was the value under
    # the tier's band.
    min_request_gap_s: float = 0.0
    # The share of ``max_concurrent`` that must hold a slot for that floor to
    # apply. The gap protects a pool at its admission line with the tier
    # carrying the queue; once most games have finished there is no queue to
    # protect and it only slows the games left. Decided from the run's slot
    # tally (``platypus.slots``) as a game's call starts, and again before
    # each LLM call by an agent with a per-round hook (the duck's); 0 paces
    # always, and an unknown tally keeps the floor.
    min_request_gap_running_fraction: float = 0.5
    # Floor on the gap between consecutive game STARTS across the run. Every
    # game that holds a slot at the start of a run spawns its child process
    # at once, and a couple of hundred of those spawning together broke the
    # host; spreading them costs nothing a game notices (its turns run
    # minutes) and reaches every column, since starting is the harness's.
    start_gap_s: float = 0.5
    # Per-request host watchdog: error if the backend produces nothing for
    # this long. None disables it for an externally budgeted deployment.
    backend_inactivity_timeout_s: float | None = ullm.backends.proxy.DEFAULT_INACTIVITY_TIMEOUT_S
    # The run's own watchdog over the engine (``platypus.watchdog``): end
    # the run, the way the soft deadline does, once the engine has generated
    # nothing for this long while it still reports requests, or has died.
    # What a stuck engine otherwise costs is the rest of the budget -- the
    # per-request bound above is an hour, and a submission then produces
    # nothing for its last hours. None disables it.
    engine_stall_timeout_s: float | None = 600.0
    # Who gets a slot when one comes free (``platypus.priority``).
    # "queue_order" is the run's own play order, which is FIFO — a plain
    # concurrency cap.
    #
    # "least_tokens" ranks by what a game has spent so far, so the machine
    # follows progress rather than arrival: a game that has spent little
    # is cheap and holds its slot, one grinding the same task slides
    # behind the queue.
    #
    # It only bites when games yield (``ctx.yield_slot``) and there are
    # more games than slots, which is a submission's shape and not a local
    # run's — hence the neutral default.
    slot_policy: Literal["queue_order", "least_tokens"] = "queue_order"
    # How often the scheduler says where the slots stand, on stdout and on
    # the run stream. Purely a heartbeat: ranking happens at every grant,
    # from the numbers as they stand then, so nothing waits on this and
    # changing it changes only how often drift no single event caused
    # reaches the log.
    slot_report_interval_s: float = 30 * 60
    # Longest ONE game holds a slot without a free swap point. An agent
    # offers its slot back wherever it could be swapped out and says
    # whether the swap is free there (its cache is already gone); a free
    # offer is always taken up, and this bounds the wait for a game that
    # never reaches one. Measured per game from its own last grant and
    # reset by every grant, so it fires only for the game that has gone
    # this long without one — never as a run-wide reshuffle, which would
    # throw away every prefix at once. The clock is infra's because
    # agents do not read clocks. Not a runtime limit: the game keeps
    # playing either way. A swap that is not free costs the game its
    # resident context, and with a host KV tier bringing it back is a load
    # rather than a recompute but still a wait. ``None`` retires the
    # fallback: only a free offer is ever taken, at a compaction, a take
    # boundary or a finish. Off by default (2026-10-16): a
    # submission ends at a wall clock, so a non-free swap costs a game its
    # whole resident context for nothing bought -- the deadline does not
    # move, and the slot it frees goes to a game that must then build a
    # context of its own -- and off a deadline the fallback is not needed.
    # The free swap point is compaction, which an ARC2 task at this budget
    # rarely reaches, so a run that wants the hold to bite at all sets it
    # short.
    slot_hold_s: float | None = None
    # What each take after a game's first costs it in the ranking
    # (``platypus.multi_take``), in tokens: a game in its second take ranks
    # as if it had spent this many more than it has. First answers are
    # bought before second ones, and a second take still overtakes a first
    # that has spent more than it plus this. Read by "least_tokens" alone.
    take_handicap_tokens: int = 150_000
    # Agents run in a spawned per-game child by default; "inprocess" is the
    # single-process debug mode. Identical agent code either way.
    process_mode: Literal["subprocess", "inprocess"] = "subprocess"
    # RLIMIT_AS per child. A real cap on the deployment targets we use: the
    # Kaggle box has 173 GB and a SLURM node is sized per job.
    subprocess_memory_limit_bytes: int = 32 * 2**30
    # Uncaught agent exception: "crash_game" contains it (this game ends
    # "crashed", others continue), "raise" propagates — the run dies and a
    # debugger lands on the original frame. An agent that wants to forfeit
    # on internal failure catches its own exception and returns GiveUp.
    # Either way whatever it had already submitted still scores.
    on_agent_error: Literal["crash_game", "raise"] = "crash_game"
    # Per-game wall-clock budget, enforced by infra (agents still never
    # read clocks): the in-flight call is CANCELLED at the deadline. Time
    # the game spends parked in ``ctx.yield_slot`` does not count — it holds
    # no slot then — so under a ranking policy the net still measures the
    # game's own tenure. The game finishes "cancelled", scored on whatever
    # it had submitted by then. None = unlimited.
    #
    # Unlimited by default, because a budget belongs to a deployment rather
    # than to the contract: it is one harness's operating point, and an
    # agent that inherits it silently gets its games cut off on someone
    # else's schedule.
    max_runtime_s_per_game: float | None = None
    shuffle_games: bool = False
    shuffle_seed: int | None = None
    # Which games start first: a priority per ``game_id``, highest first, a
    # game not named ranking as 0. Applied after the shuffle and stably, so
    # games of one priority keep the shuffled order among themselves. Under
    # ``slot_policy="queue_order"`` the play order is the start order, so a
    # run that knows its long tasks starts them first and does not end on a
    # tail of them started late. Empty by default: the order is the
    # benchmark's, or the shuffle's.
    play_priority: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    rng_seed: int | None = None
    rng_seed_per_game: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())
    # Seconds between per-game status lines on stdout (platypus.status);
    # None disables the report. Matches the vLLM engine monitor's cadence so
    # the two lines interleave one for one.
    status_interval_s: float | None = 60.0
    # Memory reporting on that same line's cadence. A page-table walk per
    # process, so a run with many game subprocesses may want it off.
    report_memory: bool = True
    # A host stall on demand, to test what one does to a run -- the engine
    # draining, the KV tier, the games' timelines: the host's event loop is
    # held ``loop_hold_s`` every ``loop_hold_every_s``, from that long into
    # play (``platypus.host_monitor.hold_loop``). Off at zero.
    loop_hold_s: float = 0.0
    loop_hold_every_s: float = 600.0
    # Where the run writes its span streams and diagnostic blocks while it
    # plays, environment variables expanded; they land in the job dir as
    # numbered zip parts (``platypus.diagnostics.DiagnosticsArchiver``) —
    # finished games every ``diagnostics_archive_interval_s``, the rest at the
    # end. A node-local disk by default (TAAF's Slurm preamble sets the
    # variable): a stream appends a line per event and a large block is a file
    # of its own, which a network filesystem makes slow to write, and slow to
    # delete by the tens of thousands. Unset or empty once expanded (a run off
    # Slurm, a test) writes into the job dir directly, as always, unarchived.
    # A game on scratch cannot be watched live from another node.
    diagnostics_scratch: str = "${TAAF_LOCAL_JOB_ROOT}"
    diagnostics_archive_interval_s: float = 30 * 60
    # Seconds between engine-status lines (ullm.engine_monitor), overriding
    # the cadence the backend factory was built with. None = leave the
    # factory's own setting alone, so the engine line has ONE owner by
    # default and this is the run saying it wants a different rate.
    engine_status_interval_s: float | None = None
    # Kaggle deploy hooks — not LLM-specific (any agent may need kernel /
    # utility-script sources, dataset mounts, or setup commands in the
    # kernel). taaf.deploy_kaggle duck-reads these attribute names off the
    # SOLVER; AgentSolver forwards them from here (config lives on the
    # agent).
    # The engine a Kaggle kernel runs, said once in ullm and linked from
    # here: the wheelhouse it installs from, the requirement set, and
    # whether the workspace's fork is overlaid on the stock install. The
    # solver derives the kernel sources, setup commands, bundle exclusions
    # and payload files from it, ahead of the agent's own below. None for an
    # agent that runs no engine on the kernel.
    kaggle_engine: ullm.kaggle.KaggleEngine | None = None
    # The agent's own kernel sources, dataset mounts and setup commands,
    # beyond what its engine needs: the model snapshot above all.
    kaggle_kernel_sources: tuple[str, ...] = ()
    kaggle_dataset_sources: tuple[str, ...] = ()
    # Kaggle MODELS the kernel mounts, which are not datasets: a model lands
    # under /kaggle/input/models/<owner>/<slug>/<framework>/<variation>/
    # <version>, and weights published as one are reachable no other way.
    kaggle_model_sources: tuple[str, ...] = ()
    kaggle_setup_commands: tuple[str, ...] = ()
    # Repos to keep out of the deploy's source bundle, by directory name. The
    # runner prepends every bundled repo to sys.path, so a repo the kernel
    # installs for itself has to stay out or the bundled copy shadows it.
    #
    # The vLLM fork is that case for every agent here, engine or no engine:
    # this workspace pins it from a Tufalabs URL, so a deploy would bundle
    # its installed tree, while a Kaggle kernel installs stock vLLM from
    # wheels. Bundled, the fork's Python would be served on binaries it was
    # not built against. The fork reaches Kaggle through the engine's
    # overlay, not by emptying this.
    kaggle_exclude_source_repos: tuple[str, ...] = (ullm.kaggle.VLLM_FORK_REPO,)
    # Files copied into the deploy bundle's payload/ directory, for a setup
    # command to read at $TAAF_KAGGLE_BUNDLE_DIR/payload/<name>. Unlike a
    # bundled source repo nothing puts these on sys.path, which is what makes
    # them the way to carry an artifact the kernel installs over itself. The
    # engine's own payload, its overlay, is added by the LLM solver.
    kaggle_bundle_files: tuple[str, ...] = ()


__all__ = [
    "HANDICAP_HINT",
    "TAKE_BOUNDARY_HINT",
    "TAKE_INDEX_HINT",
    "ContractViolationError",
    "GameOutcome",
    "GiveUp",
    "GiveUpError",
    "RunConfig",
]
