"""Shared ARC3 SFT holdouts, including tagged reimplementations of held-out games."""

from collections.abc import Iterable

ARC3_SFT_EVAL_GAME_IDS = frozenset({
    "pathfinder-0001", "taps-0001", "cd82-fb555c5d", "dc22-fdcac232", "escape_velocity-0001",
})

# re-arc-3 tags this game official-reimplemented; CD82's game_dir_url names
# crane_dipper. The other tagged reimplementations have non-held-out originals.
ARC3_SFT_HOLDOUT_DERIVATIVES = {"crane_dipper-0001": "cd82-fb555c5d"}
ARC3_SFT_TRAINING_EXCLUSIONS = ARC3_SFT_EVAL_GAME_IDS | frozenset(ARC3_SFT_HOLDOUT_DERIVATIVES)


def _game_slug(game_id: str) -> str:
    """Treat unversioned IDs and different versions as the same game family."""
    return game_id.strip().lower().rsplit("-", 1)[0]


def validate_arc3_sft_training_games(game_ids: Iterable[str]) -> None:
    """Refuse held-out game families in training; validation may still load them."""
    originals = {_game_slug(game): game for game in ARC3_SFT_EVAL_GAME_IDS}
    originals.update({_game_slug(game): parent for game, parent in ARC3_SFT_HOLDOUT_DERIVATIVES.items()})
    blocked = {game: originals[_game_slug(game)] for game in game_ids if _game_slug(game) in originals}
    if blocked:
        details = ", ".join(f"{game} (holdout {parent})" for game, parent in sorted(blocked.items()))
        raise ValueError(f"ARC3 SFT training includes held-out games: {details}; exclude them from training")
