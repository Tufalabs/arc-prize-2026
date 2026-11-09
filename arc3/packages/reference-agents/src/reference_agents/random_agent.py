"""A random agent expressed in the contract (the logic mirrors TAAF's
``SolverRandom``)."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from platypus import Action, Agent, GiveUp, MoveContext, Observation


def pick_random_action(rng: random.Random, obs: Observation, click_on_color: bool) -> Action:
    """Uniform random available non-RESET action; shared by the random
    reference agents so their action distributions are identical."""
    choices = [name for name in obs.available_actions if name != "RESET"]
    name = rng.choice(choices) if choices else "RESET"
    if name == "ACTION6":
        row, col = _pick_click(rng, obs, click_on_color)
        return Action(name, row=row, col=col)
    return Action(name)


def _pick_click(rng: random.Random, obs: Observation, click_on_color: bool) -> tuple[int, int]:
    """A click target. Colour-weighted picks a colour present in the frame and
    then a pixel of it, so a small blob is as likely a target as the
    background; otherwise the pixel is uniform over the frame."""
    frame = obs.frame.data
    if click_on_color:
        color = rng.choice(np.unique(frame).tolist())
        rows, cols = np.where(frame == color)
        index = rng.randrange(len(cols))
        return int(rows[index]), int(cols[index])
    height, width = frame.shape
    return rng.randint(0, height - 1), rng.randint(0, width - 1)


@dataclass
class RandomAgent(Agent):
    """Plays a uniformly random available non-RESET action each move; after
    a loss, resets (an LLM-free agent decision — decision ledger item 8).

    ``max_moves``: give up after this many moves (``None`` = play until the
    game or the run deadline ends it). ``click_on_color``: a CLICK targets
    a uniformly chosen *color* present in the frame, then a uniform pixel
    of that color — a small interactable blob is as likely as the
    background, a stronger CLICK signal than a uniform pixel (default
    ``False`` = uniform pixel). All randomness from ``ctx.rng`` (A2), so a
    fixed ``run_config.rng_seed`` reproduces the whole run.
    """

    max_moves: int | None = 100
    click_on_color: bool = False

    # -- state --
    moves_made: int = 0

    async def _do_move(self, ctx: MoveContext, obs: Observation) -> Action | GiveUp:
        """One move: reset after a loss, give up at the move cap, else play a
        random available action."""
        if obs.outcome == "level_won":
            return Action("RESET")  # delivery-only call; the return is discarded
        if self.max_moves is not None and self.moves_made >= self.max_moves:
            return GiveUp(f"reached max_moves={self.max_moves}")
        self.moves_made += 1
        if obs.outcome == "lost":
            return Action("RESET")
        return pick_random_action(ctx.rng, obs, self.click_on_color)
