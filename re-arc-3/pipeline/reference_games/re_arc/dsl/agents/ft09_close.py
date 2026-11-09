from __future__ import annotations

import copy
from dataclasses import dataclass

from ..core import DslAgent, camera_grid_to_display, observation_level_index, resolve_action, unpack_step_result


@dataclass(frozen=True)
class _TileRef:
    x: int
    y: int
    prefer_tag: str


class Ft09CloseDslAgent(DslAgent):
    def __init__(self, game_id: str):
        super().__init__(game_id=game_id, total_levels=None)
        self._current_level_idx = None
        self._action_idx = 0
        self._programs: dict[int, list[tuple[int, dict[str, int]]]] = {}

    def reset_episode(self):
        super().reset_episode()
        self._current_level_idx = None
        self._action_idx = 0

    def _sync_progress(self, observation):
        total_levels = getattr(observation, "win_levels", None)
        if total_levels is not None:
            try:
                total = int(total_levels)
            except (TypeError, ValueError):
                total = None
            if total is not None and total > 0:
                self.total_levels = total

        level_idx = observation_level_index(observation, self.total_levels)
        if level_idx is None:
            return

        self.mark_levels_solved(level_idx)
        reset_level = bool(getattr(observation, "full_reset", False))
        if self._current_level_idx is None or level_idx != self._current_level_idx:
            self._current_level_idx = level_idx
            self._action_idx = 0
            return
        if reset_level and self._action_idx > 0:
            self._action_idx = 0

    def _collect_tiles(self, level) -> list[_TileRef]:
        hkx = sorted(level.get_sprites_by_tag("Hkx"), key=lambda sprite: (int(sprite.y), int(sprite.x), sprite.name))
        nti = sorted(level.get_sprites_by_tag("NTi"), key=lambda sprite: (int(sprite.y), int(sprite.x), sprite.name))

        out: list[_TileRef] = []
        for sprite in hkx:
            out.append(_TileRef(x=int(sprite.x), y=int(sprite.y), prefer_tag="Hkx"))
        for sprite in nti:
            out.append(_TileRef(x=int(sprite.x), y=int(sprite.y), prefer_tag="NTi"))
        return out

    def _get_tile_sprite(self, level, tile: _TileRef):
        if tile.prefer_tag == "Hkx":
            sprite = level.get_sprite_at(tile.x, tile.y, "Hkx")
            if sprite is None:
                sprite = level.get_sprite_at(tile.x, tile.y, "NTi")
            return sprite

        sprite = level.get_sprite_at(tile.x, tile.y, "NTi")
        if sprite is None:
            sprite = level.get_sprite_at(tile.x, tile.y, "Hkx")
        return sprite

    def _tile_key(self, sprite) -> tuple[str, int, int]:
        tag = "Hkx" if "Hkx" in getattr(sprite, "tags", []) else "NTi"
        return (tag, int(sprite.x), int(sprite.y))

    def _extract_color_indices(self, level, tiles: list[_TileRef], palette: list[int]) -> list[int]:
        out: list[int] = []
        for tile in tiles:
            sprite = self._get_tile_sprite(level, tile)
            if sprite is None:
                raise RuntimeError(f"FT09 Close DSL could not find tile sprite at ({tile.x}, {tile.y}).")
            color = int(sprite.pixels[1][1])
            if color not in palette:
                raise RuntimeError(
                    f"FT09 Close DSL found tile color outside level palette: color={color} palette={palette}"
                )
            out.append(palette.index(color))
        return out

    def _build_allowed_color_sets(self, level, tiles: list[_TileRef], palette: list[int]) -> list[set[int]]:
        k = len(palette)
        allowed = [set(range(k)) for _ in tiles]
        tile_index = {(tile.prefer_tag, int(tile.x), int(tile.y)): idx for idx, tile in enumerate(tiles)}
        neighbor_coords = [(0, 0), (1, 0), (2, 0), (0, 1), (2, 1), (0, 2), (1, 2), (2, 2)]

        for target in level.get_sprites_by_tag("bsT"):
            target_color = int(target.pixels[1][1])
            target_idx = palette.index(target_color) if target_color in palette else None

            for px, py in neighbor_coords:
                require_equal = int(target.pixels[py][px]) == 0
                tx = int(target.x) + (px - 1) * 4
                ty = int(target.y) + (py - 1) * 4
                tile_sprite = level.get_sprite_at(tx, ty, "Hkx")
                if tile_sprite is None:
                    tile_sprite = level.get_sprite_at(tx, ty, "NTi")
                if tile_sprite is None:
                    continue

                idx = tile_index.get(self._tile_key(tile_sprite))
                if idx is None:
                    continue

                if target_idx is None:
                    if require_equal:
                        allowed[idx].clear()
                    continue

                if require_equal:
                    allowed[idx].intersection_update({target_idx})
                else:
                    allowed[idx].difference_update({target_idx})

                if not allowed[idx]:
                    raise RuntimeError(
                        "FT09 Close DSL found contradictory color constraints for "
                        "tile="
                        f"({tiles[idx].x}, {tiles[idx].y}, {tiles[idx].prefer_tag})"
                    )

        return allowed

    def _measure_click_effects(
        self,
        env,
        level_idx: int,
        tiles: list[_TileRef],
        click_targets: list[tuple[int, int]],
        base_colors: list[int],
        palette: list[int],
    ) -> list[list[int]]:
        del level_idx
        effects: list[list[int]] = []
        k = len(palette)

        for gx, gy in click_targets:
            env_copy = copy.deepcopy(env)
            click_action = resolve_action(env_copy, 6)
            dx, dy = camera_grid_to_display(env_copy._game.camera, gx, gy)
            obs, _, _, _ = unpack_step_result(env_copy.step(click_action, data={"x": int(dx), "y": int(dy)}))
            if obs is None:
                raise RuntimeError("FT09 Close DSL failed to evaluate a candidate click.")

            current_level = env_copy._game.current_level
            new_colors = self._extract_color_indices(current_level, tiles, palette)
            deltas = [int((new_colors[idx] - base_colors[idx]) % k) for idx in range(len(tiles))]
            effects.append(deltas)

        return effects

    def _gaussian_elimination_mod2(self, rows: list[int], rhs: list[int], n_vars: int) -> list[int] | None:
        work_rows = list(rows)
        work_rhs = [int(v) & 1 for v in rhs]
        pivots: list[tuple[int, int]] = []
        pivot_row = 0

        for col in range(n_vars):
            found = None
            for row in range(pivot_row, len(work_rows)):
                if (work_rows[row] >> col) & 1:
                    found = row
                    break
            if found is None:
                continue

            work_rows[pivot_row], work_rows[found] = (work_rows[found], work_rows[pivot_row])
            work_rhs[pivot_row], work_rhs[found] = work_rhs[found], work_rhs[pivot_row]

            for row in range(len(work_rows)):
                if row == pivot_row:
                    continue
                if ((work_rows[row] >> col) & 1) == 0:
                    continue
                work_rows[row] ^= work_rows[pivot_row]
                work_rhs[row] ^= work_rhs[pivot_row]

            pivots.append((pivot_row, col))
            pivot_row += 1

        for row in range(len(work_rows)):
            if work_rows[row] == 0 and work_rhs[row] != 0:
                return None

        solution = [0] * n_vars
        for row, col in reversed(pivots):
            value = work_rhs[row]
            high = work_rows[row] & ~((1 << (col + 1)) - 1)
            while high:
                lsb = high & -high
                idx = lsb.bit_length() - 1
                value ^= solution[idx]
                high ^= lsb
            solution[col] = value & 1

        return solution

    def _apply_effects(
        self, base_colors: list[int], effects: list[list[int]], click_counts: list[int], modulo: int
    ) -> list[int]:
        out = list(base_colors)
        for action_idx, count in enumerate(click_counts):
            repeats = int(count) % modulo
            if repeats == 0:
                continue
            deltas = effects[action_idx]
            for tile_idx in range(len(out)):
                out[tile_idx] = (out[tile_idx] + repeats * deltas[tile_idx]) % modulo
        return out

    def _solve_click_counts(
        self, allowed: list[set[int]], effects: list[list[int]], base_colors: list[int], palette: list[int]
    ) -> list[int] | None:
        k = len(palette)
        action_count = len(effects)
        tile_count = len(base_colors)

        if k == 2:
            rows: list[int] = []
            rhs: list[int] = []
            for tile_idx, options in enumerate(allowed):
                if len(options) != 1:
                    continue
                target = next(iter(options))
                row = 0
                for action_idx in range(action_count):
                    if effects[action_idx][tile_idx] % 2:
                        row ^= 1 << action_idx
                rows.append(row)
                rhs.append((int(target) - int(base_colors[tile_idx])) & 1)

            solution = self._gaussian_elimination_mod2(rows, rhs, action_count)
            if solution is None:
                return None

            final_colors = self._apply_effects(base_colors, effects, solution, 2)
            if all(final_colors[idx] in allowed[idx] for idx in range(tile_count)):
                return solution
            return None

        # FT09 Close currently has one ternary level (k=3) where each click is an
        # independent color cycle on one tile.
        if k == 3:
            action_for_tile = [-1] * tile_count
            action_delta = [0] * tile_count

            for action_idx, deltas in enumerate(effects):
                non_zero = [idx for idx, delta in enumerate(deltas) if delta % k != 0]
                if len(non_zero) != 1:
                    continue
                idx = non_zero[0]
                action_for_tile[idx] = action_idx
                action_delta[idx] = deltas[idx] % k

            counts = [0] * action_count
            for tile_idx, options in enumerate(allowed):
                required_shift_options = sorted((int(target) - int(base_colors[tile_idx])) % k for target in options)
                action_idx = action_for_tile[tile_idx]
                delta = action_delta[tile_idx]

                if action_idx < 0 or delta == 0:
                    if 0 not in required_shift_options:
                        return None
                    continue

                best_count = None
                for shift in required_shift_options:
                    for c in range(k):
                        if (delta * c) % k == shift:
                            if best_count is None or c < best_count:
                                best_count = c
                            break
                if best_count is None:
                    return None
                counts[action_idx] = best_count

            final_colors = self._apply_effects(base_colors, effects, counts, k)
            if all(final_colors[idx] in allowed[idx] for idx in range(tile_count)):
                return counts
            return None

        return None

    def _validate_program(self, env, level_idx: int, program: list[tuple[int, dict[str, int]]]):
        env_copy = copy.deepcopy(env)
        obs = None
        for action_id, action_data in program:
            action = resolve_action(env_copy, action_id)
            obs, _, _, _ = unpack_step_result(env_copy.step(action, data=action_data))
            if obs is None:
                return False
            solved_levels = getattr(obs, "levels_completed", None)
            if solved_levels is not None and int(solved_levels) > int(level_idx):
                return True
            state = getattr(obs, "state", None)
            if getattr(state, "name", str(state)) == "WIN":
                return True
        return False

    def _build_level_program(self, env, level_idx: int):
        game = env._game
        level = game.current_level
        palette = [int(color) for color in (game.gqb or [])]
        if not palette:
            raise RuntimeError("FT09 Close DSL could not read the level color palette.")

        tiles = self._collect_tiles(level)
        if not tiles:
            raise RuntimeError("FT09 Close DSL found no clickable tiles in the level.")

        click_targets = sorted({(int(tile.x), int(tile.y)) for tile in tiles}, key=lambda point: (point[1], point[0]))
        base_colors = self._extract_color_indices(level, tiles, palette)
        allowed = self._build_allowed_color_sets(level, tiles, palette)
        effects = self._measure_click_effects(
            env=env,
            level_idx=level_idx,
            tiles=tiles,
            click_targets=click_targets,
            base_colors=base_colors,
            palette=palette,
        )
        click_counts = self._solve_click_counts(
            allowed=allowed, effects=effects, base_colors=base_colors, palette=palette
        )
        if click_counts is None:
            raise RuntimeError(f"FT09 Close DSL could not solve level {level_idx} ({level.name}).")

        program: list[tuple[int, dict[str, int]]] = []
        for action_idx, count in enumerate(click_counts):
            repeats = int(count)
            if repeats <= 0:
                continue
            gx, gy = click_targets[action_idx]
            dx, dy = camera_grid_to_display(game.camera, gx, gy)
            for _ in range(repeats):
                program.append((6, {"x": int(dx), "y": int(dy)}))

        if not self._validate_program(env, level_idx, program):
            raise RuntimeError(
                "FT09 Close DSL produced an invalid level program that did not advance "
                f"the level. level={level_idx} ({level.name})"
            )

        return program

    def next_action(self, env, observation):
        self._sync_progress(observation)
        if self._current_level_idx is None:
            raise RuntimeError("FT09 Close DSL is missing `levels_completed` in observation.")

        if self._current_level_idx not in self._programs:
            self._programs[self._current_level_idx] = self._build_level_program(env, self._current_level_idx)

        program = self._programs[self._current_level_idx]
        if self._action_idx >= len(program):
            raise RuntimeError(
                "FT09 Close DSL program exhausted before advancing to the next level. "
                f"level={self._current_level_idx} steps={len(program)}"
            )

        action_id, action_data = program[self._action_idx]
        self._action_idx += 1
        return action_id, action_data
