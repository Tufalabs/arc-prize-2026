from __future__ import annotations

import random
import threading
from collections.abc import Mapping, Sequence

GAME_SAMPLING_MODES = ("balanced", "nonzero_rate")

# Process-wide weighted-sampling state. The sampler's call sites
# (training.inference) know only (games, slot_count, seed, schedule_index),
# while the per-game outcome history that drives "nonzero_rate" weights lives
# in training.cli's loop state -- so cli publishes the configured mode/floor
# once at startup and a fresh rate snapshot after every recorded window.
# publish replaces the snapshot dict wholesale (never mutates it in place),
# which keeps cross-thread reads safe under the GIL.
_STATE_LOCK = threading.Lock()
_DEFAULT_MODE = "balanced"
_DEFAULT_WEIGHT_FLOOR = 0.15
_DEFAULT_GAMES_PER_WINDOW = 0
_configured_mode: str = _DEFAULT_MODE
_configured_weight_floor: float = _DEFAULT_WEIGHT_FLOOR
_configured_games_per_window: int = _DEFAULT_GAMES_PER_WINDOW
_configured_nonzero_rates: dict[str, float] = {}


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in GAME_SAMPLING_MODES:
        raise ValueError(
            f"game sampling mode must be one of {sorted(GAME_SAMPLING_MODES)}; "
            f"got {mode!r}."
        )
    return normalized


def _validate_weight_floor(weight_floor: float) -> float:
    floor = float(weight_floor)
    # floor == 0 would let a never-scoring game's weight collapse to zero and
    # silently drop it from the curriculum; every configured game must keep a
    # positive draw probability.
    if not 0.0 < floor <= 1.0:
        raise ValueError(
            f"game sampling weight floor must be in (0, 1]; got {weight_floor!r}."
        )
    return floor


def configure_game_sampling(
    *, mode: str, weight_floor: float, games_per_window: int = 0
) -> None:
    """Set the process-wide sampling policy consulted by sample_training_games."""
    global _configured_mode, _configured_weight_floor, _configured_games_per_window
    normalized_mode = _validate_mode(mode)
    floor = _validate_weight_floor(weight_floor)
    per_window = int(games_per_window)
    if normalized_mode == "nonzero_rate" and per_window < 1:
        raise ValueError(
            "nonzero_rate sampling requires games_per_window >= 1; "
            f"got {games_per_window!r}."
        )
    with _STATE_LOCK:
        _configured_mode = normalized_mode
        _configured_weight_floor = floor
        _configured_games_per_window = per_window


def publish_game_nonzero_rates(rates: Mapping[str, float]) -> None:
    """Publish trailing nonzero-return rates for "nonzero_rate" weighting."""
    snapshot = {
        str(game_id).strip(): min(1.0, max(0.0, float(rate)))
        for game_id, rate in rates.items()
        if str(game_id).strip()
    }
    global _configured_nonzero_rates
    with _STATE_LOCK:
        _configured_nonzero_rates = snapshot


def reset_game_sampling() -> None:
    """Restore the balanced defaults (tests and fresh processes)."""
    global _configured_mode, _configured_weight_floor, _configured_nonzero_rates
    global _configured_games_per_window
    with _STATE_LOCK:
        _configured_mode = _DEFAULT_MODE
        _configured_weight_floor = _DEFAULT_WEIGHT_FLOOR
        _configured_games_per_window = _DEFAULT_GAMES_PER_WINDOW
        _configured_nonzero_rates = {}


def game_sampling_weights(
    game_ids: Sequence[str],
    *,
    weight_floor: float,
    nonzero_rates: Mapping[str, float],
) -> dict[str, float]:
    """Normalized per-game draw weights for "nonzero_rate" sampling.

    Un-normalized weight = floor + (1 - floor) * trailing nonzero-return
    rate. A game with no recorded rate draws at full weight 1.0 (an untried
    game must keep appearing until there is evidence), so a cold start is
    uniform; the floor keeps every configured game's weight positive.
    """
    games = sorted(
        {str(game_id).strip() for game_id in game_ids if str(game_id).strip()}
    )
    if not games:
        return {}
    floor = _validate_weight_floor(weight_floor)
    raw_weights: dict[str, float] = {}
    for game_id in games:
        rate = nonzero_rates.get(game_id)
        if rate is None:
            raw_weights[game_id] = 1.0
        else:
            raw_weights[game_id] = floor + (1.0 - floor) * min(
                1.0, max(0.0, float(rate))
            )
    total = sum(raw_weights.values())
    return {game_id: weight / total for game_id, weight in raw_weights.items()}


def sample_training_games(
    game_ids: Sequence[str],
    *,
    slot_count: int,
    seed: int,
    schedule_index: int,
    mode: str | None = None,
    weight_floor: float | None = None,
    games_per_window: int | None = None,
    nonzero_rates: Mapping[str, float] | None = None,
) -> list[str]:
    """Sample a rollout window's game multiset.

    mode / weight_floor / games_per_window / nonzero_rates default to the
    process-wide values set via configure_game_sampling and
    publish_game_nonzero_rates. Both modes emit a UNIFORM multiset (every
    game present in the window appears an equal number of times): the
    platypus worker structurally runs unique-games x n_passes and rejects
    uneven windows, and multi-server splitting needs whole balanced passes.
    """
    games = sorted(
        {str(game_id).strip() for game_id in game_ids if str(game_id).strip()}
    )
    if slot_count <= 0 or not games:
        return []

    with _STATE_LOCK:
        if mode is None:
            mode = _configured_mode
        if weight_floor is None:
            weight_floor = _configured_weight_floor
        if games_per_window is None:
            games_per_window = _configured_games_per_window
        if nonzero_rates is None:
            nonzero_rates = _configured_nonzero_rates
    mode = _validate_mode(mode)

    rng = random.Random(
        f"train-game-sample:{seed}:{schedule_index}:{slot_count}:{'|'.join(games)}"
    )
    if mode == "balanced":
        per_window = int(games_per_window)
        if 0 < per_window < len(games):
            # Uniform per-window subset: games_per_window DISTINCT games drawn
            # without replacement (no weighting), each run
            # slot_count/games_per_window passes. Lets a game list larger
            # than one server's slot capacity train uniformly across windows.
            if slot_count % per_window != 0:
                raise ValueError(
                    f"parallel_rollouts={slot_count} is not divisible by"
                    f" games_per_window={per_window}; a uniform window multiset is"
                    " impossible."
                )
            chosen = rng.sample(games, per_window)
            passes = slot_count // per_window
            window = [game_id for game_id in chosen for _ in range(passes)]
            rng.shuffle(window)
            return window
        # Balanced multiset, not independent draws: cycle the game list to
        # fill the window, then shuffle for position. When slot_count is not
        # a multiple of the game count the order is shuffled first, so the
        # extra copies land on random games each window, not the first ones.
        order = list(games)
        if slot_count % len(order):
            rng.shuffle(order)
        window = [order[index % len(order)] for index in range(slot_count)]
        rng.shuffle(window)
        return window

    # "nonzero_rate": weighting acts on WHICH games make up the window, not
    # on per-slot draws -- games_per_window distinct games are drawn without
    # replacement by weight, and each runs slot_count/games_per_window passes,
    # keeping the multiset uniform. Across windows, expected per-game
    # frequency follows the weights; the floor inside game_sampling_weights
    # guarantees every configured game keeps appearing.
    per_window = int(games_per_window)
    if per_window < 1:
        raise ValueError(
            "nonzero_rate sampling requires sampling.game_sampling_games_per_window"
            f" >= 1; got {games_per_window!r}."
        )
    if per_window > len(games):
        raise ValueError(
            f"games_per_window={per_window} exceeds the {len(games)} configured games."
        )
    if slot_count % per_window != 0:
        raise ValueError(
            f"parallel_rollouts={slot_count} is not divisible by"
            f" games_per_window={per_window}; a uniform window multiset is"
            " impossible."
        )
    weights = game_sampling_weights(
        games, weight_floor=weight_floor, nonzero_rates=nonzero_rates
    )
    pool = dict(weights)
    chosen: list[str] = []
    for _ in range(per_window):
        ids = sorted(pool)
        pick = rng.choices(ids, weights=[pool[g] for g in ids], k=1)[0]
        chosen.append(pick)
        del pool[pick]
    passes = slot_count // per_window
    window = [game_id for game_id in chosen for _ in range(passes)]
    rng.shuffle(window)
    return window
