from __future__ import annotations

from ..core import DslAgent, iter_frame_layers, layer_to_grid

COLOR_BTN_ON = 14  # green — indicator button when patterns match


def _frame_has_match(observation) -> bool:
    """Return True if any pixel in the frame is COLOR_BTN_ON (green button)."""
    for layer in iter_frame_layers(observation):
        grid = layer_to_grid(layer)
        if hasattr(grid, "__iter__"):
            for row in grid:
                if COLOR_BTN_ON in row:
                    return True
    return False


class RotatePatternDslAgent(DslAgent):
    """
    Reactive solver: keep rotating CCW (ACTION3) until the patterns match
    (green button appears), then confirm with ACTION5.
    Works for any starting rotation and all 5 levels.
    """

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=5)

    def next_action(self, _env, observation):
        if _frame_has_match(observation):
            return 5, {}  # ACTION5: confirm match → advance level
        return 3, {}  # ACTION3: rotate CCW


AGENT_CLASS = RotatePatternDslAgent
