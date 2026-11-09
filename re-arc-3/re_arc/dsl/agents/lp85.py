from __future__ import annotations

from collections import deque

from re_arc.dsl.core import DslAgent, observation_level_index

_TILE_SIZE = 3
_DISPLAY_SIZE = 64
_MOVEABLE_TAGS = {"tile", "goal", "goal-o"}
_SKIP_TAGS = {"goal-indicator", "goal-indicator-o", "bghvgbtwcb", "fdgmtkfrxl"}


def _game_level_name(game: object) -> str:
    for attr in ("level_name", "ucybisahh"):
        val = getattr(game, attr, None)
        if isinstance(val, str):
            return val
    raise AttributeError("lp85: cannot find level_name attribute on game object")


def _game_processed_maps(game: object) -> dict:
    for attr in ("processed_maps", "uopmnplcnv"):
        val = getattr(game, attr, None)
        if isinstance(val, dict):
            return val
    raise AttributeError("lp85: cannot find processed_maps attribute on game object")


def _path_data_field(path_data: dict, readable: str, obfuscated: str) -> object:
    if readable in path_data:
        return path_data[readable]
    return path_data[obfuscated]


def _get_all_moves(level_name, code, forward, processed_maps):
    path_data = processed_maps[level_name][code]
    num_to_pos = _path_data_field(path_data, "num_to_pos", "qcmzcjocmj")
    max_num = _path_data_field(path_data, "max_num", "oxbwsencfv")
    if max_num <= 1:
        return []
    moves = []
    for cur, from_pos in num_to_pos.items():
        if forward:
            tgt = 1 if cur == max_num else cur + 1
        else:
            tgt = max_num if cur == 1 else cur - 1
        moves.append((from_pos, num_to_pos[tgt]))
    return moves


def _apply_button(positions, level_name, code, forward, processed_maps):
    moves = _get_all_moves(level_name, code, forward, processed_maps)
    pos_to_idx: dict[tuple[int, int], int] = {}
    for i, pos in enumerate(positions):
        if pos not in pos_to_idx:
            pos_to_idx[pos] = i

    new_pos = list(positions)
    plan = []
    for from_pos, to_pos in moves:
        fp = (from_pos.x * _TILE_SIZE, from_pos.y * _TILE_SIZE)
        tp = (to_pos.x * _TILE_SIZE, to_pos.y * _TILE_SIZE)
        if fp in pos_to_idx:
            plan.append((pos_to_idx[fp], tp))
    for idx, tp in plan:
        new_pos[idx] = tp
    return tuple(new_pos)


def _find_accessible_button_clusters(game):
    """Return click positions, each with the list of (code, forward) buttons stacked there.

    Multiple button sprites can share the same pixel position — a single click
    activates all of them simultaneously. The old dedup-by-position logic was
    discarding all but the first button in such a cluster.
    """
    level = game.current_level
    level_name = _game_level_name(game)
    processed_maps = _game_processed_maps(game)
    camera = game.camera

    clusters: dict[tuple[int, int], list[tuple[str, bool]]] = {}
    for sprite in level._sprites:
        if not sprite.tags:
            continue
        tag = sprite.tags[0]
        if not tag.startswith("button_"):
            continue
        if not (0 <= sprite.x < camera.width and 0 <= sprite.y < camera.height):
            continue
        parts = tag.split("_")
        if len(parts) != 3:
            continue
        code = parts[1]
        forward = parts[2] == "R"
        if level_name not in processed_maps or code not in processed_maps[level_name]:
            continue
        pos = (sprite.x, sprite.y)
        clusters.setdefault(pos, []).append((code, forward))
    return clusters


def _solve(game):
    level = game.current_level
    level_name = _game_level_name(game)
    processed_maps = _game_processed_maps(game)
    camera = game.camera

    # Build target map: position → required tag
    targets: dict[tuple[int, int], str] = {}
    for tag, required in [("goal-indicator", "goal"), ("bghvgbtwcb", "goal")]:
        for ind in level.get_sprites_by_tag(tag):
            targets[(ind.x + 1, ind.y + 1)] = required
    for tag, required in [("goal-indicator-o", "goal-o"), ("fdgmtkfrxl", "goal-o")]:
        for ind in level.get_sprites_by_tag(tag):
            targets[(ind.x + 1, ind.y + 1)] = required

    # Collect moveable sprites in stable order
    moveable = [s for s in level._sprites if s.tags and s.tags[0] in _MOVEABLE_TAGS]
    if not moveable:
        raise RuntimeError(f"lp85: no moveable sprites found for level '{level_name}'")

    # Which indices in `moveable` are goal sprites
    goal_indices: dict[int, str] = {
        i: s.tags[0] for i, s in enumerate(moveable) if s.tags and s.tags[0] in ("goal", "goal-o")
    }

    init = tuple((s.x, s.y) for s in moveable)

    def is_win(state: tuple) -> bool:
        # Build position→tag map for goal sprites only (to avoid O(n) per target)
        goal_positions: dict[tuple[int, int], str] = {}
        for idx, needed_tag in goal_indices.items():
            goal_positions[state[idx]] = needed_tag
        for pos, required in targets.items():
            if goal_positions.get(pos) != required:
                return False
        return True

    if is_win(init):
        return []

    goal_idx_list = sorted(goal_indices.keys())

    def goal_key(state: tuple) -> tuple:
        return tuple(state[i] for i in goal_idx_list)

    # BFS keyed on goal-sprite positions only — tiles don't affect the win
    # condition, so two full states with the same goal layout are equivalent
    # for search purposes. This shrinks the visited set from O(grid^N_sprites)
    # to O(grid^N_goals), typically a few hundred states vs millions.
    #
    # Each BFS action = one click position, which activates ALL button codes
    # stacked at that position simultaneously (cluster semantics).
    clusters = _find_accessible_button_clusters(game)
    if not clusters:
        raise RuntimeError(
            f"lp85: no accessible button clusters for level '{level_name}' "
            f"(camera is {camera.width}x{camera.height} — buttons may be off-screen)"
        )

    queue: deque[tuple[tuple, list]] = deque([(init, [])])
    visited: set[tuple] = {goal_key(init)}

    while queue:
        state, path = queue.popleft()
        for (bx, by), codes in clusters.items():
            nstate = state
            for code, forward in codes:
                nstate = _apply_button(nstate, level_name, code, forward, processed_maps)
            gk = goal_key(nstate)
            if gk in visited:
                continue
            visited.add(gk)
            npath = [*path, (bx, by)]
            if is_win(nstate):
                return npath
            queue.append((nstate, npath))

    raise RuntimeError(
        f"lp85: no solution found for level '{level_name}' with accessible button clusters {list(clusters.keys())}"
    )


def _build_click(bx, by, camera):
    scale = min(_DISPLAY_SIZE // camera.width, _DISPLAY_SIZE // camera.height)
    x_off = (_DISPLAY_SIZE - camera.width * scale) // 2
    y_off = (_DISPLAY_SIZE - camera.height * scale) // 2
    dx = bx * scale + x_off
    dy = by * scale + y_off
    return {"x": dx, "y": dy}


class Lp85DslAgent(DslAgent):
    """
    Simulation-based BFS solver for lp85.

    Finds accessible buttons (within the 16x16 camera viewport), simulates
    tile movements, and searches for the shortest button-press sequence that
    satisfies all goal conditions.

    NOTE: Several lp85 levels have buttons placed outside the camera viewport
    (camera is 16x16 but level grids can be up to 60x64). Those levels will
    raise RuntimeError. To fix, the game's camera should be enlarged to cover
    the full level grid.
    """

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=8)
        self._plan: list[tuple[int, dict]] = []
        self._plan_idx = 0
        self._last_level: int | None = None

    def reset_episode(self) -> None:
        super().reset_episode()
        self._plan = []
        self._plan_idx = 0
        self._last_level = None

    def next_action(self, env, observation):
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            raise RuntimeError("lp85: missing levels_completed in observation")

        if level_idx != self._last_level:
            self._last_level = level_idx
            self._plan_idx = 0
            self._plan = self._build_plan(env)

        if self._plan_idx >= len(self._plan):
            raise RuntimeError(f"lp85: plan exhausted at level {level_idx} ({len(self._plan)} actions)")

        action = self._plan[self._plan_idx]
        self._plan_idx += 1
        return action

    def _build_plan(self, env) -> list[tuple[int, dict]]:
        game = env._game
        camera = game.camera
        solution = _solve(game)
        return [(6, _build_click(bx, by, camera)) for bx, by in solution]


AGENT_CLASS = Lp85DslAgent
