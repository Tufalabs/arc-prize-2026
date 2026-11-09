from __future__ import annotations

from collections import deque
from typing import NamedTuple

from ..core import CachedProgramDslAgent

_JS26_LEVEL_NAMES = ("krg", "bgd", "puq", "tmx", "lyd", "zba")


class _SpriteCollision(NamedTuple):
    x: int
    y: int
    width: int
    height: int
    name: str
    tags: tuple[str, ...] | None


class Js26DslAgent(CachedProgramDslAgent):
    """Observation-driven JS26 agent with per-level search planning."""

    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=len(_JS26_LEVEL_NAMES))

    def _level_name(self, level_idx: int) -> str:
        if 0 <= int(level_idx) < len(_JS26_LEVEL_NAMES):
            return _JS26_LEVEL_NAMES[int(level_idx)]
        return f"level_{int(level_idx)}"

    @staticmethod
    def _sign(value: int) -> int:
        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(getattr(game, "level_index", 0))
        expected_level_name = self._level_name(level_idx)
        current_level_name = str(getattr(game.current_level, "name", ""))
        if current_level_name != expected_level_name:
            raise RuntimeError(
                "JS26 DSL level-name mismatch: "
                f"index={level_idx} expected={expected_level_name} actual={current_level_name}"
            )

        game_globals = game._resolve_teleport.__func__.__globals__
        tag_wall = str(game_globals.get("TAG_WALL", "wall"))
        tag_teleport = str(game_globals.get("TAG_TELEPORT", "teleport"))
        teleport_matrix = game_globals.get("teleport_matrix", {})

        cell_width = int(getattr(game, "cell_width", 1))
        cell_height = int(getattr(game, "cell_height", 1))

        target_points = [(int(sprite.x), int(sprite.y)) for sprite in getattr(game, "target_slot_sprites", [])]
        target_bit_by_point = {point: (1 << idx) for idx, point in enumerate(target_points)}
        all_targets_mask = (1 << len(target_points)) - 1

        initial_mask = 0
        player_start = (int(game.player_sprite.x), int(game.player_sprite.y))
        if player_start in target_bit_by_point:
            initial_mask |= target_bit_by_point[player_start]

        selected_teleport_action = getattr(game, "selected_teleport_action", None)
        move_by_action: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
        if isinstance(selected_teleport_action, (tuple, list)) and len(selected_teleport_action) == 2:
            move_by_action[5] = (int(selected_teleport_action[0]), int(selected_teleport_action[1]))

        available_actions = [int(action.value) for action in env.action_space if int(action.value) in move_by_action]
        if not available_actions:
            raise RuntimeError("JS26 DSL found no movement actions to plan with.")

        static_collision_sprites: list[_SpriteCollision] = []
        for sprite in game.current_level._sprites:
            if sprite is game.player_sprite:
                continue
            pixels = sprite.render()
            tags = tuple(str(tag) for tag in sprite.tags) if sprite.tags is not None else None
            static_collision_sprites.append(
                _SpriteCollision(
                    x=int(sprite.x),
                    y=int(sprite.y),
                    width=int(pixels.shape[1]),
                    height=int(pixels.shape[0]),
                    name=str(sprite.name),
                    tags=tags,
                )
            )

        def collisions_at(next_x: int, next_y: int) -> list[_SpriteCollision]:
            right = next_x + cell_width
            bottom = next_y + cell_height
            collisions: list[_SpriteCollision] = []
            for sprite in static_collision_sprites:
                sprite_right = sprite.x + sprite.width
                sprite_bottom = sprite.y + sprite.height
                if sprite.x < right and sprite_right > next_x and sprite.y < bottom and sprite_bottom > next_y:
                    collisions.append(sprite)
            return collisions

        def resolve_teleport(
            next_x: int,
            next_y: int,
            move_x: int,
            move_y: int,
            collided_sprites: list[_SpriteCollision],
            *,
            depth: int = 0,
        ) -> tuple[int, int] | None:
            if depth >= 4:
                return None

            teleport_sprite: _SpriteCollision | None = None
            for collided_sprite in collided_sprites:
                if collided_sprite.tags is not None and tag_teleport in collided_sprite.tags:
                    teleport_sprite = collided_sprite
                    break
            if teleport_sprite is None:
                return next_x, next_y

            lookup_move = (self._sign(move_x), self._sign(move_y))
            teleport_delta = teleport_matrix.get(teleport_sprite.name, {}).get(lookup_move)
            if teleport_delta is None:
                return next_x, next_y

            destination_x = next_x + int(teleport_delta[0]) * cell_width
            destination_y = next_y + int(teleport_delta[1]) * cell_height
            destination_collisions = collisions_at(destination_x, destination_y)

            for destination_sprite in destination_collisions:
                if (
                    destination_sprite.tags is not None
                    and tag_wall in destination_sprite.tags
                    and tag_teleport not in destination_sprite.tags
                ):
                    return next_x, next_y

            destination_has_teleport = any(
                destination_sprite.tags is not None and tag_teleport in destination_sprite.tags
                for destination_sprite in destination_collisions
            )
            if destination_has_teleport:
                recurse_move_x = self._sign(int(teleport_delta[0]))
                recurse_move_y = self._sign(int(teleport_delta[1]))
                return resolve_teleport(
                    destination_x,
                    destination_y,
                    recurse_move_x,
                    recurse_move_y,
                    destination_collisions,
                    depth=depth + 1,
                )

            return destination_x, destination_y

        start_state = (
            int(game.player_sprite.x),
            int(game.player_sprite.y),
            int(initial_mask),
            int(game.hud.timer_steps_remaining),
        )

        if all_targets_mask == 0:
            return []

        queue = deque([start_state])
        previous: dict[tuple[int, int, int, int], tuple[tuple[int, int, int, int], int] | None] = {start_state: None}
        best_timer_by_signature: dict[tuple[int, int, int], int] = {
            (start_state[0], start_state[1], start_state[2]): start_state[3]
        }

        max_expansions = 50_000
        expansions = 0

        while queue:
            state = queue.popleft()
            x, y, solved_mask, timer_steps_remaining = state

            if solved_mask == all_targets_mask:
                actions: list[int] = []
                cursor = state
                while previous[cursor] is not None:
                    parent_state, action_id = previous[cursor]
                    actions.append(int(action_id))
                    cursor = parent_state
                actions.reverse()
                return [(action_id, {}) for action_id in actions]

            expansions += 1
            if expansions > max_expansions:
                break

            for action_id in available_actions:
                move_xy = move_by_action.get(int(action_id))
                if move_xy is None:
                    continue

                move_x, move_y = move_xy
                if timer_steps_remaining <= 0:
                    continue

                next_x = x + int(move_x) * cell_width
                next_y = y + int(move_y) * cell_height
                collided = collisions_at(next_x, next_y)

                blocked_by_wall = False
                for collided_sprite in collided:
                    if collided_sprite.tags is None:
                        break
                    if tag_wall in collided_sprite.tags:
                        blocked_by_wall = True
                        break

                if blocked_by_wall:
                    resolved_x, resolved_y = x, y
                else:
                    resolved = resolve_teleport(next_x, next_y, move_x, move_y, collided)
                    if resolved is None:
                        continue
                    resolved_x, resolved_y = int(resolved[0]), int(resolved[1])

                next_mask = int(solved_mask)
                bit = target_bit_by_point.get((resolved_x, resolved_y))
                if bit is not None:
                    next_mask |= int(bit)

                if next_mask == all_targets_mask:
                    next_state = (resolved_x, resolved_y, next_mask, timer_steps_remaining)
                    if next_state not in previous:
                        previous[next_state] = (state, int(action_id))
                    actions: list[int] = []
                    cursor = next_state
                    while previous[cursor] is not None:
                        parent_state, prior_action = previous[cursor]
                        actions.append(int(prior_action))
                        cursor = parent_state
                    actions.reverse()
                    return [(planned_action_id, {}) for planned_action_id in actions]

                next_timer = timer_steps_remaining - 1
                next_state = (resolved_x, resolved_y, next_mask, next_timer)
                if next_state in previous:
                    continue

                dominance_signature = (resolved_x, resolved_y, next_mask)
                prior_best_timer = best_timer_by_signature.get(dominance_signature)
                if prior_best_timer is not None and next_timer <= prior_best_timer:
                    continue

                best_timer_by_signature[dominance_signature] = next_timer
                previous[next_state] = (state, int(action_id))
                queue.append(next_state)

        raise RuntimeError(
            "JS26 DSL planner failed to build a level program. "
            f"level={level_idx} name={expected_level_name} expansions={expansions}"
        )


AGENT_CLASS = Js26DslAgent
