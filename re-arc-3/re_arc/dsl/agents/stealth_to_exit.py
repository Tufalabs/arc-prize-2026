from __future__ import annotations

import copy
import heapq
import time
from typing import Any

from ..core import CachedProgramDslAgent, observation_level_index, unpack_step_result


class StealthToExitDslAgent(CachedProgramDslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=1)
        self._search_timeout_seconds = 45.0
        self._search_max_expansions = 120_000

    def _guard_signature(self, guard) -> tuple[Any, ...]:
        return (
            tuple(int(v) for v in guard.head),
            tuple(int(v) for v in guard.facing),
            str(guard.behavior),
            str(guard.mode),
            tuple(int(v) for v in guard.anchor_head),
            tuple(int(v) for v in guard.anchor_facing),
            None if guard.patrol_min is None else int(guard.patrol_min),
            None if guard.patrol_max is None else int(guard.patrol_max),
            None if guard.patrol_y is None else int(guard.patrol_y),
            int(guard.patrol_pause),
            int(guard.patrol_dir),
            int(guard.pause_left),
            int(bool(guard.reverse_after_pause)),
            None if guard.investigate_target is None else tuple(int(v) for v in guard.investigate_target),
            int(guard.investigate_steps_left),
        )

    def _state_key(self, game, observation) -> tuple[Any, ...]:
        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            level_idx = int(getattr(game, "level_index", 0))
        noise = getattr(game, "_noise_event", None)
        if isinstance(noise, dict):
            center = noise.get("center")
            if center is not None:
                center = tuple(int(v) for v in center)
            age = int(noise.get("age", 0))
            noise_key = (center, age)
        else:
            noise_key = None
        return (
            int(level_idx),
            tuple(int(v) for v in getattr(game, "_player", (0, 0))),
            int(bool(getattr(game, "_has_key", False))),
            int(bool(getattr(game, "_door_open", False))),
            int(getattr(game, "_door_opening_steps", 0)),
            tuple(sorted((int(x), int(y)) for x, y in getattr(game, "_keys", set()))),
            int(getattr(game, "_time_remaining", 0)),
            int(getattr(game, "_caught_anim_steps", 0)),
            noise_key,
            tuple(self._guard_signature(guard) for guard in getattr(game, "_guards", ())),
        )

    def _heuristic(self, game) -> int:
        px, py = (int(v) for v in getattr(game, "_player", (0, 0)))
        exits = [(int(x), int(y)) for x, y in getattr(game, "_exits", set())]
        exit_dist = min(abs(px - ex) + abs(py - ey) for ex, ey in exits) if exits else 0

        keys = [(int(x), int(y)) for x, y in getattr(game, "_keys", set())]
        need_key = bool(getattr(game, "_door_tiles", set())) and not bool(getattr(game, "_door_open", False))
        has_key = bool(getattr(game, "_has_key", False))
        key_penalty = 0
        if need_key and not has_key and keys and exits:
            key_penalty = min(abs(px - kx) + abs(py - ky) for kx, ky in keys)
            key_penalty += min(abs(kx - ex) + abs(ky - ey) for kx, ky in keys for ex, ey in exits)

        vision_penalty = 0
        player = (px, py)
        hidden = player in getattr(game, "_hiding", set())
        if not hidden and player in getattr(game, "_vision_tiles", set()):
            vision_penalty = 30

        return int(exit_dist + key_penalty + vision_penalty)

    def _action_obj(self, env, action_id: int):
        for action in env.action_space:
            if int(getattr(action, "value", -1)) == int(action_id):
                return action
        raise RuntimeError(f"stealth_to_exit action {action_id} is unavailable.")

    def _plan_level_program(self, env, observation, level_idx: int) -> list[int]:
        start = time.time()
        root_env = copy.deepcopy(env)
        root_obs = observation
        root_game = root_env._game

        nodes: list[tuple[int | None, int | None, Any, Any]] = [(None, None, root_env, root_obs)]
        queue: list[tuple[int, int, int, int]] = []
        counter = 0
        expansions = 0
        seen = {self._state_key(root_game, root_obs)}

        root_priority = self._heuristic(root_game)
        heapq.heappush(queue, (root_priority, 0, counter, 0))
        counter += 1

        while (
            queue and expansions < self._search_max_expansions and (time.time() - start) < self._search_timeout_seconds
        ):
            _priority, cost_so_far, _ord, node_id = heapq.heappop(queue)
            _parent, _action_id, node_env, node_obs = nodes[node_id]
            node_game = node_env._game
            node_state_name = str(getattr(getattr(node_obs, "state", None), "name", None))
            node_level_idx = observation_level_index(node_obs, self.total_levels)
            if node_level_idx is None:
                node_level_idx = int(getattr(node_game, "level_index", 0))

            if node_state_name == "WIN" or int(node_level_idx) > int(level_idx):
                actions: list[int] = []
                cursor = node_id
                while nodes[cursor][0] is not None:
                    parent_id, action_id, _env_obj, _obs_obj = nodes[cursor]
                    actions.append(int(action_id))
                    cursor = int(parent_id)
                actions.reverse()
                return actions

            if node_state_name != "NOT_FINISHED":
                continue

            expansions += 1
            for action_id in (1, 2, 3, 4, 5):
                child_env = copy.deepcopy(node_env)
                action = self._action_obj(child_env, action_id)
                child_obs, _reward, _done, _info = unpack_step_result(child_env.step(action, data={}))
                if child_obs is None:
                    continue
                child_state_name = str(getattr(getattr(child_obs, "state", None), "name", None))
                if child_state_name == "GAME_OVER":
                    continue

                child_key = self._state_key(child_env._game, child_obs)
                if child_key in seen:
                    continue
                seen.add(child_key)

                child_id = len(nodes)
                nodes.append((node_id, int(action_id), child_env, child_obs))
                child_cost = cost_so_far + 1
                child_priority = child_cost + self._heuristic(child_env._game)
                heapq.heappush(queue, (child_priority, child_cost, counter, child_id))
                counter += 1

        raise RuntimeError(
            "stealth_to_exit DSL planner failed to build a replay program for "
            f"level={level_idx}. expanded={expansions} "
            f"timeout={self._search_timeout_seconds:.1f}s."
        )

    def _build_level_program(self, env) -> list[tuple[int, dict[str, int]]]:
        game = env._game
        level_idx = int(getattr(game, "level_index", 0))
        observation = getattr(env, "_obs", None)
        if observation is None:
            observation = getattr(env, "_observation", None)
        if observation is None:
            raise RuntimeError("stealth_to_exit DSL requires current observation to plan.")
        actions = self._plan_level_program(env, observation, level_idx)
        if not actions:
            raise RuntimeError(f"stealth_to_exit DSL produced an empty program for level={level_idx}.")
        return [(int(action_id), {}) for action_id in actions]


AGENT_CLASS = StealthToExitDslAgent
