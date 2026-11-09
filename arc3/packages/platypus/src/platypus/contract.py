"""Agent-facing contract types (requirements.md §1–§3).

These do not replace the TAAF/arcengine data model: ``Observation``
embeds the ``taaf.game.GameState`` (raw engine data included) and
``taaf.game.Frame``, and ``Action`` maps 1:1 onto
``arcengine.ActionInput``. They exist to carry what agents need and the
substrate doesn't define — per-observation outcome/level and
level-transition splitting, the (row, col) convention with its checked
engine bijection, the give-up channel, config-on-agent — while keeping
agents off taaf/arcengine entirely (I1: agents never touch the engine;
the solver is the only code that does).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping

import arcengine
import taaf.game
import ullm.backends.proxy
import ullm.kaggle

SIMPLE_ACTIONS = ("RESET", "ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION7")  # 7 = undo
COORDINATE_ACTIONS = ("ACTION6",)
ENGINE_ACTIONS = SIMPLE_ACTIONS + COORDINATE_ACTIONS

Outcome = Literal["playing", "level_won", "game_won", "lost"]


class ContractViolationError(Exception):
    """An agent broke the contract (bad return type, unavailable action, ...).

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
    """An agent's decision to stop playing this game. Returned from
    ``_do_move`` (or raised as ``GiveUpError`` from any depth); the game
    finishes ``gave_up``, keeping the score earned so far. ``reason`` is
    free text and lands on the move span's outcome."""

    reason: str


@dataclass(frozen=True)
class Action:
    """Engine-true action: the engine action name plus optional row/col
    coordinate data. Not ``arcengine.ActionInput`` because that is exactly
    the seam the contract pins down: ``ActionInput`` carries a raw
    ``{'x', 'y'}`` dict (the recurring x/y-vs-row/col bug source) and
    validates nothing at construction. Agent code speaks (row, col); the
    one defined bijection ``{'x': col, 'y': row}`` lives in
    ``to_action_input``/``from_action_input`` and nowhere else. What this
    checks at construction is the SHAPE — a known action name, and
    coordinates present exactly on the actions that take them — so that
    class of mistake fails at the agent. Coordinate VALUES are not checked
    here: their bound is the game's grid size, which this type does not
    know, so the engine is what rejects an off-board click.
    Model-facing names (UP/SPACE/MOUSE, ...) are prompt-building, owned by
    the agent.
    """

    name: str
    row: int | None = None
    col: int | None = None

    def __post_init__(self) -> None:
        if self.name not in ENGINE_ACTIONS:
            raise ContractViolationError(f"unknown engine action {self.name!r}; valid: {ENGINE_ACTIONS}")
        has_coords = self.row is not None or self.col is not None
        if self.name in COORDINATE_ACTIONS:
            if self.row is None or self.col is None:
                raise ContractViolationError(f"{self.name} requires both row and col")
        elif has_coords:
            raise ContractViolationError(f"{self.name} is a simple action and must not carry row/col")

    def to_action_input(self) -> arcengine.ActionInput:
        """The engine-side form; the solver calls this, agents never do."""
        game_action = arcengine.GameAction[self.name]
        if self.name in COORDINATE_ACTIONS:
            return arcengine.ActionInput(id=game_action, data={"x": self.col, "y": self.row})
        return arcengine.ActionInput(id=game_action)

    @classmethod
    def from_action_input(cls, action_input: arcengine.ActionInput) -> Action:
        """Inverse of ``to_action_input``."""
        name = action_input.id.name
        if name in COORDINATE_ACTIONS:
            data = action_input.data or {}
            return cls(name, row=data["y"], col=data["x"])
        return cls(name)


def action_names(action_ids: list[int]) -> tuple[str, ...]:
    """Engine action ids -> names (``arcengine.GameAction(id)`` doesn't
    resolve; ``from_id`` is the supported lookup)."""
    return tuple(arcengine.GameAction.from_id(action_id).name for action_id in action_ids)


@dataclass(frozen=True)
class Observation:
    """Delivery envelope over ``taaf.game.GameState`` (§2). Raw engine data
    stays reachable via ``state.raw``. ``frame``/``animation_frames`` are
    the partitioned view: on a level transition one engine state yields two
    observations (the whale's rule — final frame becomes the new level's
    initial observation, everything before it belongs to the level-win
    observation), so they are not always ``state.frame``.

    ``level`` is the level this observation *belongs to*: the completed
    level for a ``level_won``/``game_won`` observation, the new level for
    the following initial one. ``index`` is the per-game global observation
    counter. ``caused_by`` is ``None`` for game/level initial states.

    Every observation is delivered through its own ``do_move`` call; a
    ``level_won`` observation's call is delivery-only — the returned
    action is discarded (a ``GiveUp`` is honored). A ``game_won``
    observation ends the game and gets no call.
    """

    state: taaf.game.GameState
    outcome: Outcome
    level: int
    index: int
    caused_by: Action | None
    frame: taaf.game.Frame
    animation_frames: tuple[taaf.game.Frame, ...]

    @classmethod
    def from_state(
        cls,
        state: taaf.game.GameState,
        *,
        outcome: Outcome,
        level: int,
        index: int,
        caused_by: Action | None,
        frame: taaf.game.Frame | None = None,
        animation_frames: tuple[taaf.game.Frame, ...] | None = None,
    ) -> Observation:
        """Build an observation over ``state``, defaulting the frame view
        to the state's own frames; the partitioning solver overrides them
        for split observations. New state-derived fields belong here, not
        at the call sites."""
        return cls(
            state=state,
            outcome=outcome,
            level=level,
            index=index,
            caused_by=caused_by,
            frame=state.frame if frame is None else frame,
            animation_frames=tuple(state.animation_frames) if animation_frames is None else animation_frames,
        )

    @property
    def available_actions(self) -> tuple[str, ...]:
        """Engine-true available action names (RESET always present);
        model-visible filtering is agent-side."""
        return action_names(self.state.available_actions)


@dataclass(frozen=True)
class Move:
    """One executed action plus every observation it produced (usually one;
    two at a level transition — split observations are part of the
    canonical record). The last observation is the settled state the next
    move starts from."""

    action: Action
    observations: tuple[Observation, ...]

    @property
    def observation(self) -> Observation:
        """The settled observation — what the next move starts from."""
        return self.observations[-1]


def partition(game_state: taaf.game.GameState, caused_by: Action, next_index: int) -> tuple[Observation, ...]:
    """Observations produced by one executed action (§2). On a level
    transition the engine state's final frame becomes the new level's
    initial observation and all preceding frames (animation and win
    screen) belong to the level-win observation — the whale's rule.
    Observations are never fused."""
    if game_state.won:
        return (
            Observation.from_state(
                game_state,
                outcome="game_won",
                level=game_state.levels_completed - 1,
                index=next_index,
                caused_by=caused_by,
            ),
        )
    if game_state.just_won_level:
        frames = game_state.all_frames
        win_observation = Observation.from_state(
            game_state,
            outcome="level_won",
            level=game_state.levels_completed - 1,
            index=next_index,
            caused_by=caused_by,
            frame=frames[-2],
            animation_frames=tuple(frames[:-2]),
        )
        initial_observation = Observation.from_state(
            game_state,
            outcome="playing",
            level=game_state.levels_completed,
            index=next_index + 1,
            caused_by=None,
            frame=frames[-1],
            animation_frames=(),
        )
        return (win_observation, initial_observation)
    outcome = "lost" if game_state.raw.state == arcengine.GameState.GAME_OVER else "playing"
    return (
        Observation.from_state(
            game_state,
            outcome=outcome,
            level=game_state.levels_completed,
            index=next_index,
            caused_by=caused_by,
        ),
    )


@dataclass(frozen=True)
class GameInfo:
    """What an agent knows about the game before playing it, mounted on the
    context for the whole run. ``hint`` is the game designer's prose, absent
    for most games; ``number_of_levels`` is ``None`` when the engine does not
    report one up front.

    ``pass_index`` is which pass copy of the game this is (TAAF R2.2, 0-based;
    0 outside a multi-pass benchmark). The copies are still played
    independently — no copy's play may depend on what another saw or did — but
    an agent MAY read this to tell them apart, which is what lets a single run
    cover several configurations by giving each pass its own. It is TAAF's
    stamped ``GameRun.pass_index``, never the entry's position in the run's
    list: shuffling and the heartbeat both permute that list, so a position
    encodes play order rather than pass.
    """

    game_id: str
    hint: str | None
    action_space: tuple[str, ...]
    number_of_levels: int | None
    pass_index: int = 0


@dataclass(frozen=True)
class GameResult:
    """How a game ended, handed to ``on_finish`` on every terminal path.
    ``score`` is ``None`` when the engine reports none."""

    state: Literal["won", "gave_up", "cancelled", "crashed"]
    levels_completed: int
    score: float | None


@dataclass
class RunConfig:
    """Hosting knobs; lives ON the agent, read by ``AgentSolver`` from the
    template once at setup (per-game copies carry it inertly). A plain
    mutable dataclass: tweak fields freely while composing an experiment;
    the solver reads the values at run start, so edits after that have no
    effect on a running solver.

    ``rng_seed``: ``None`` (default) draws each game's seed from OS
    entropy; a fixed value makes runs reproducible (per-game seed =
    ``rng_seed`` + the game's index in the run, I11/I13). Either way the
    per-game seed is a concrete integer, resolved once by
    ``AgentSolver._game_seed`` and recorded, so any finished run can be
    replayed with the seeds it actually used.

    ``shuffle_games`` / ``shuffle_seed``: play the run's (game, pass)
    entries in a random order rather than the benchmark's. The heartbeat
    sacrifices whichever entry lands first, so it is drawn from the shuffle
    too. ``shuffle_seed=None`` derives the seed from the current date
    (2026-07-01 -> 20260701), which is what lets the same notebook be
    submitted repeatedly and still name the order it played; the resolved
    value is printed at run start either way.

    ``play_priority``: per-game priority, highest started first, applied
    stably after any shuffle; see the field.

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
    # seconds), so the floor only slows the games; a tier running its band
    # instead wants 5 s.
    min_request_gap_s: float = 0.0
    # The share of ``max_concurrent`` that must hold a slot for that floor to
    # apply. The gap protects a pool at its admission line with the tier
    # carrying the queue; once most games have finished there is no queue to
    # protect and it only slows the games left. Decided per move from the
    # run's slot tally (``platypus.slots``); 0 paces always, and an unknown
    # tally keeps the floor.
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
    # "least_level_tokens" ranks by what a game has spent on the level it
    # is standing on, so the machine follows progress rather than arrival:
    # a game that just won a level is cheap again and holds its slot, one
    # grinding the same level slides behind the queue.
    #
    # "least_game_tokens" ranks by what a game has spent in all, so every
    # game gets an even share of the machine and none races ahead: a game
    # that wins levels cheaply does not keep its slot for it, and one stuck
    # on a level is not pushed to the back. The agent reports the spend
    # (``game_tokens``); one that does not ranks at zero. The level
    # weighting below does not apply to it.
    #
    # It only bites when games yield (``ctx.yield_slot``) and there are
    # more games than slots, which is a submission's shape and not a local
    # run's — hence the neutral default.
    slot_policy: Literal["queue_order", "least_level_tokens", "least_game_tokens"] = "queue_order"
    # How hard the ranking discounts a token by what its level is worth.
    # ARC gives level i (1-indexed) the share i/total_weight of a game's
    # score (total_weight = n(n+1)/2 for n levels), so a token there buys
    # score at a rate of i/total_weight and the ranking charges it
    #
    #     level_tokens * (total_weight / i) ** scheduler_level_weighting
    #
    # 0 charges every token alike — rank on raw spend. 1 is the full
    # price: on a 6-level game a level-1 token is charged 21 and a level-2
    # token 10.5, so the same standing buys more tokens deeper into a game
    # and on any level of a short one. Above 1 sharpens that further.
    # Continuous rather than a pair of named policies because the right
    # setting is an empirical question: full pricing assumes deeper levels
    # cost no more to clear than shallow ones, which holds on the official
    # set and is what a steeper hidden-set difficulty ramp would break, so
    # the useful values are likely between the two ends.
    scheduler_level_weighting: float = 0.0
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
    # agents do not read clocks (A2). Not a runtime limit: the game keeps
    # playing either way. A swap that is not free costs the game its
    # resident context, and with a host KV tier bringing it back is a load
    # rather than a recompute but still a wait. ``None`` retires the
    # fallback: only a free offer is ever taken, so a slot changes hands at
    # compactions alone and a game that never compacts keeps its slot until
    # it finishes. Off by default (2026-10-16): the duck and the seal
    # offer at every compaction, and the fallback is not needed beside them.
    slot_hold_s: float | None = None
    # I9: agents run in a spawned per-game child by default; "inprocess"
    # is the single-process debug mode. Identical agent code either way.
    process_mode: Literal["subprocess", "inprocess"] = "subprocess"
    subprocess_memory_limit_bytes: int = 32 * 2**30
    # I2: "level" snapshots every (game, pass) as each new level starts, into
    # <job_dir>/checkpoints (``platypus.checkpoints``), for starting branched
    # games from; "move" is not implemented and is rejected at setup. Off by
    # default: a checkpoint is a whole pickled agent per level per game, a
    # storage cost a run takes on only when it wants the branch points, and
    # "off" leaves agents free to forfeit snapshot cadences (A4) without any
    # caller having to configure anything.
    checkpointing: Literal["move", "level", "off"] = "off"
    # I4: a game started from a level checkpoint names it itself
    # (``platypus.checkpoints.CheckpointStart``) and gets the checkpoint's
    # agent under this run's RunConfig; this is what its ctx.rng starts from.
    # "fresh": its own seed, like any game, so the passes of one checkpoint
    # draw apart — what a study of how play varies from one point wants.
    # "resume": the source game's state at the checkpoint, so every pass
    # continues the source's draws exactly — what checking that a branch IS
    # the source's continuation wants.
    branch_rng: Literal["fresh", "resume"] = "fresh"
    # Uncaught agent exception: "crash_game" contains it (this game ends
    # "crashed", others continue), "raise" propagates — the run dies and a
    # debugger lands on the original frame. An agent that wants to forfeit
    # on internal failure catches its own exception and returns GiveUp.
    on_agent_error: Literal["crash_game", "raise"] = "crash_game"
    # Per-game wall-clock budget, enforced by infra (agents still never
    # read clocks): checked between moves, and the in-flight move is
    # CANCELLED at the deadline (asyncio.wait_for around do_move), so a
    # single unbounded move cannot overrun the cap. The game finishes
    # "gave_up", scored as-is — the original duck harness's
    # max_runtime_minutes behavior, hardened. None = unlimited.
    #
    # Unlimited by default, because a budget belongs to a deployment rather
    # than to the contract: it is one harness's operating point, and an
    # agent that inherits it silently gets its games cut off on someone
    # else's schedule. The duck asks for its own, and says at
    # ``duck.presets.GAME_RUNTIME_CAP_S`` what it is catching.
    max_runtime_s_per_game: float | None = None
    # Play the (game, pass) entries in a random order rather than the
    # benchmark's. None as the seed = derive from today's date (YYYYMMDD),
    # so an unattended submission still has a seed that can be named.
    shuffle_games: bool = False
    shuffle_seed: int | None = None
    # Which games start first: a priority per game, highest first, a game
    # not named ranking as 0. Keyed by the id a game plays under, which a
    # TAAF game learns only when it starts, so an unstarted one is named by
    # the id it will start under (``AgentSolver._planned_game_id``). Applied
    # after the shuffle and stably, so games of one priority keep the
    # shuffled order among themselves. Under ``slot_policy="queue_order"``
    # the play order is the start order, so a run that knows its long games
    # starts them first and does not end on a tail of them started late.
    # Empty by default: the order is the benchmark's, or the shuffle's.
    play_priority: Mapping[str, float] = field(default_factory=lambda: dict[str, float]())
    rng_seed: int | None = None
    rng_seed_per_game: Mapping[str, int] = field(default_factory=lambda: dict[str, int]())
    # Answer a replay from the record instead of the game (``platypus.replay_shadow``):
    # a RESET is served the board the level opened on, moves after it that
    # repeat a path already played on the level from its start are served the
    # states the game returned then, and the game is sent nothing until the
    # model leaves the recorded paths, when it is caught up first. The model sees
    # no difference; the game is charged only for actions that could teach the
    # model something. Off: it buys little and games break its premise
    # (settings.md, P-0V, has the reading).
    replay_shadow: bool = False
    # A catch-up the game does not reproduce: ``"crash"`` ends the game, which
    # is what a local run wants to hear about; ``"continue"`` switches the
    # shadow off for the game and plays on from its real state, which is what a
    # submission wants. Set ``"continue"`` where a Kaggle column is built.
    replay_shadow_mismatch: Literal["crash", "continue"] = "crash"
    # End a game stuck on a level (``platypus.attempts``): the same attempt --
    # the actions from the level's start or a RESET up to the next RESET,
    # ``[RESET]`` alone included -- played this many times on one level,
    # consecutively or not, or this many actions on one level. ``None`` turns
    # either off, and both are off: they save compute, not score (settings.md,
    # P-0V, has the reading and the values it was read at).
    max_attempt_repeats: int | None = None
    max_actions_per_level: int | None = None
    # Sacrifice the FIRST game to a random-move liveness loop run by infra
    # (``platypus.heartbeat``), so a watchdog that ends a run after a
    # stretch with no moves — Kaggle's arena — sees activity while the
    # agent is still on its first turn. The agent is handed the remaining
    # games and never learns this one existed, so it costs one game's score
    # and nothing else. Off by default; on for Kaggle presets.
    heartbeat_game: bool = False
    # Far inside any watchdog window we know of, and the cost is nil — the
    # game is being thrown away regardless.
    heartbeat_interval_s: float = 10.0
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
    "Action",
    "ContractViolationError",
    "GameInfo",
    "GameResult",
    "GiveUp",
    "GiveUpError",
    "Move",
    "Observation",
    "Outcome",
    "RunConfig",
    "action_names",
    "partition",
]
