from __future__ import annotations

from re_arc.dsl.core import MOVE_ACTION_BY_DELTA, DslAgent

_PROBE_ACTIONS = [1, 2, 3, 4]
_DISPLAY_SIZE = 64
_MIN_TOKEN_PIXELS = 4  # 2x2
_MAX_TOKEN_PIXELS = 32  # 4x4 * 2 (tolerance)

# Cross pattern shape (used for detection)
_CROSS_SHAPE = {(1, 0), (2, 0), (0, 1), (3, 1), (0, 2), (3, 2), (1, 3), (2, 3)}


def _detect_cell_size(blob_span: int) -> int:
    if blob_span <= 3:
        return 2
    return 4


def _get_grid(observation):
    """Extract the pixel grid from an observation."""
    frame = getattr(observation, "frame", None)
    if frame is None:
        raise RuntimeError("identify_the_agent DSL requires an observation frame.")
    if isinstance(frame, list):
        if not frame:
            raise RuntimeError("identify_the_agent DSL received an empty frame stack.")
        return frame[-1]
    return frame


def _color_counts(grid) -> dict[int, int]:
    """Count pixels per color in the grid."""
    counts: dict[int, int] = {}
    height = len(grid)
    width = len(grid[0]) if height else 0
    for y in range(height):
        for x in range(width):
            color = int(grid[y][x])
            counts[color] = counts.get(color, 0) + 1
    return counts


def _bg_color(counts: dict[int, int]) -> int:
    """Background is the most common color."""
    return max(counts, key=lambda c: counts[c])


def _detect_blobs(observation) -> list[tuple[int, int, int, set[tuple[int, int]]]]:
    """Find blobs as (cell_x, cell_y, color, pixel_set) tuples. Auto-detects cell size."""
    grid = _get_grid(observation)
    height = len(grid)
    width = len(grid[0]) if height else 0
    counts = _color_counts(grid)
    if not counts:
        raise RuntimeError("identify_the_agent DSL received an empty grid.")

    bg = _bg_color(counts)

    # Iterate all non-background colors; filter per-blob not per-color,
    # because multiple entities can share a color (agent+target+cross = 40px).
    candidate_colors = [color for color in counts if color != bg]

    blobs: list[tuple[int, int, int, set[tuple[int, int]]]] = []

    for color in candidate_colors:
        visited: set[tuple[int, int]] = set()
        for y in range(height):
            for x in range(width):
                if int(grid[y][x]) != int(color) or (x, y) in visited:
                    continue
                # Flood-fill (8-connected) to find blob
                stack = [(x, y)]
                visited.add((x, y))
                min_x = max_x = x
                min_y = max_y = y
                pixels: set[tuple[int, int]] = {(x, y)}
                while stack:
                    cx, cy = stack.pop()
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)
                    for nx, ny in (
                        (cx + 1, cy),
                        (cx - 1, cy),
                        (cx, cy + 1),
                        (cx, cy - 1),
                        (cx + 1, cy + 1),
                        (cx + 1, cy - 1),
                        (cx - 1, cy + 1),
                        (cx - 1, cy - 1),
                    ):
                        if not (0 <= nx < width and 0 <= ny < height):
                            continue
                        if (nx, ny) in visited:
                            continue
                        if int(grid[ny][nx]) != int(color):
                            continue
                        visited.add((nx, ny))
                        stack.append((nx, ny))
                        pixels.add((nx, ny))

                span_x = max_x - min_x + 1
                span_y = max_y - min_y + 1
                cell_size = _detect_cell_size(max(span_x, span_y))
                center_x = (min_x + max_x) // 2
                center_y = (min_y + max_y) // 2
                cell_x = center_x // cell_size
                cell_y = center_y // cell_size

                # Filter individual blobs by pixel count
                if not (_MIN_TOKEN_PIXELS <= len(pixels) <= _MAX_TOKEN_PIXELS):
                    continue

                # Normalize pixel coords relative to blob origin for shape detection
                local_pixels = {(px - min_x, py - min_y) for px, py in pixels}
                blobs.append((cell_x, cell_y, color, local_pixels))

    return blobs


def _is_cross_shape(local_pixels: set[tuple[int, int]]) -> bool:
    """Check if the blob's local pixel shape matches the cross pattern."""
    return local_pixels == _CROSS_SHAPE


def _token_positions(observation) -> list[tuple[int, int]]:
    """Find non-cross token positions as (cell_x, cell_y) tuples."""
    blobs = _detect_blobs(observation)
    positions = []
    for cell_x, cell_y, _color, local_pixels in blobs:
        if not _is_cross_shape(local_pixels):
            positions.append((cell_x, cell_y))
    return sorted(set(positions))


def _find_cross(observation) -> tuple[int, int] | None:
    """Find the cross collectible position, or None if absent."""
    blobs = _detect_blobs(observation)
    for cell_x, cell_y, _color, local_pixels in blobs:
        if _is_cross_shape(local_pixels):
            return (cell_x, cell_y)
    return None


def _move_plan(start: tuple[int, int], goal: tuple[int, int]) -> list[int]:
    """Simple L-shaped path (no obstacles)."""
    sx, sy = start
    gx, gy = goal
    plan: list[int] = []
    while sx < gx:
        plan.append(MOVE_ACTION_BY_DELTA[(1, 0)])
        sx += 1
    while sx > gx:
        plan.append(MOVE_ACTION_BY_DELTA[(-1, 0)])
        sx -= 1
    while sy < gy:
        plan.append(MOVE_ACTION_BY_DELTA[(0, 1)])
        sy += 1
    while sy > gy:
        plan.append(MOVE_ACTION_BY_DELTA[(0, -1)])
        sy -= 1
    return plan


class IdentifyTheAgentDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=3)
        self._tokens: list[tuple[int, int]] = []
        self._pending_probe: int | None = None
        self._probe_index = 0
        self._agent_pos: tuple[int, int] | None = None
        self._target_pos: tuple[int, int] | None = None
        self._cross_pos: tuple[int, int] | None = None
        self._plan: list[int] = []
        self._current_level = 0

    def reset_episode(self):
        super().reset_episode()
        self._current_level = 0
        self._reset_level_state()

    def _reset_level_state(self):
        self._tokens = []
        self._pending_probe = None
        self._probe_index = 0
        self._agent_pos = None
        self._target_pos = None
        self._cross_pos = None
        self._plan = []

    def _build_plan(self) -> list[int]:
        """Build movement plan: to cross first (if present), then to target."""
        assert self._agent_pos is not None
        assert self._target_pos is not None

        waypoints: list[tuple[int, int]] = []
        if self._cross_pos is not None:
            waypoints.append(self._cross_pos)
        waypoints.append(self._target_pos)

        plan: list[int] = []
        pos = self._agent_pos
        for wp in waypoints:
            plan.extend(_move_plan(pos, wp))
            pos = wp
        return plan

    def next_action(self, _env, observation):
        # Detect level transition
        levels_completed = getattr(observation, "levels_completed", 0)
        if levels_completed is not None and int(levels_completed) > self._current_level:
            self._current_level = int(levels_completed)
            self._reset_level_state()

        tokens = _token_positions(observation)
        cross = _find_cross(observation)

        if cross is not None:
            self._cross_pos = cross

        if self._agent_pos is None:
            # Probing phase: issue a move and see what changes
            if self._pending_probe is not None and self._tokens:
                moved_from = sorted(set(self._tokens) - set(tokens))
                moved_to = sorted(set(tokens) - set(self._tokens))
                if moved_from and moved_to:
                    self._agent_pos = moved_to[0]
                    self._target_pos = next(pos for pos in tokens if pos != self._agent_pos)
                    self._plan = self._build_plan()
                self._pending_probe = None

            if self._agent_pos is None:
                if self._probe_index >= len(_PROBE_ACTIONS):
                    raise RuntimeError("identify_the_agent DSL could not discover the controllable token.")
                self._tokens = tokens
                action_id = int(_PROBE_ACTIONS[self._probe_index])
                self._probe_index += 1
                self._pending_probe = action_id
                return action_id, {}

        if not self._plan and self._agent_pos is not None and self._target_pos is not None:
            self._plan = self._build_plan()
        if not self._plan:
            raise RuntimeError("identify_the_agent DSL has no remaining plan before terminal resolution.")

        action_id = int(self._plan.pop(0))
        dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action_id]
        if self._agent_pos is not None:
            self._agent_pos = (self._agent_pos[0] + dx, self._agent_pos[1] + dy)
        return action_id, {}


AGENT_CLASS = IdentifyTheAgentDslAgent
