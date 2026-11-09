from __future__ import annotations

from arcengine import ARCBaseGame, Camera, GameAction, Level, Sprite

FLOOR_COLOR = 0
TARGET_ACTIVE_COLOR = 15
TARGET_INACTIVE_COLOR = 7

LEVEL_SPECS: list[tuple[str, int, int, int]] = [
    ("Level 1", 7, 7, 3),
    ("Level 2", 8, 8, 5),
    ("Level 3", 9, 9, 7),
    ("Level 4", 10, 10, 9),
    ("Level 5", 11, 11, 12),
    ("Level 6", 12, 12, 15),
    ("Level 7", 13, 13, 18),
    ("Level 8", 14, 14, 22),
]


def _select_targets(width: int, height: int, count: int) -> list[tuple[int, int]]:
    cells = [(x, y) for y in range(height) for x in range(width)]
    if not cells:
        return []
    step = max(1, len(cells) // (count + 1))
    positions: list[tuple[int, int]] = []
    idx = step
    while len(positions) < count and idx < len(cells):
        positions.append(cells[idx])
        idx += step
    if len(positions) < count:
        for pos in cells:
            if pos in positions:
                continue
            positions.append(pos)
            if len(positions) >= count:
                break
    return positions[:count]


def _build_level(spec: tuple[str, int, int, int]) -> Level:
    name, width, height, targets = spec
    target_positions = _select_targets(width, height, targets)
    center = (width // 2, height // 2)
    target_positions.sort(key=lambda pos: (abs(pos[0] - center[0]) + abs(pos[1] - center[1]), pos[1], pos[0]))
    return _level_from_targets(name, width, height, target_positions)


def _level_from_targets(name: str, width: int, height: int, target_positions: list[tuple[int, int]]) -> Level:
    if width <= 0 or height <= 0:
        raise ValueError("Grid must be non-empty.")

    floor_pixels = [[FLOOR_COLOR] * width for _ in range(height)]

    sprites: list[Sprite] = [Sprite(pixels=floor_pixels, name="floor", collidable=False, layer=-10)]

    for idx, (x, y) in enumerate(target_positions):
        sprites.append(
            Sprite(
                pixels=[[TARGET_ACTIVE_COLOR if idx == 0 else TARGET_INACTIVE_COLOR]],
                name=f"target_{idx}",
                x=x,
                y=y,
                collidable=False,
                layer=2,
                tags=["target"],
            )
        )

    return Level(name=name, sprites=sprites, grid_size=(width, height), data={"targets": target_positions})


class Taps(ARCBaseGame):
    def __init__(self) -> None:
        levels = [_build_level(spec) for spec in LEVEL_SPECS]
        first_size = levels[0].grid_size or (64, 64)
        camera = Camera(0, 0, first_size[0], first_size[1], 5, 5, [])
        super().__init__("taps", levels, camera=camera, debug=False, win_score=len(levels), available_actions=[6])

    def on_set_level(self, _level: Level) -> None:
        self._progress = 0
        self._reset_targets()

    def _reset_targets(self) -> None:
        for sprite in self.current_level.get_sprites_by_tag("target"):
            self.current_level.remove_sprite(sprite)
        targets = self.current_level.get_data("targets") or []
        for idx, (x, y) in enumerate(targets):
            self.current_level.add_sprite(
                Sprite(
                    pixels=[[TARGET_ACTIVE_COLOR if idx == 0 else TARGET_INACTIVE_COLOR]],
                    name=f"target_{idx}",
                    x=x,
                    y=y,
                    collidable=False,
                    layer=2,
                    tags=["target"],
                )
            )

    def _activate_next_target(self) -> None:
        for target in self.current_level.get_sprites_by_tag("target"):
            index = int(target.name.split("_")[-1])
            target.pixels[0][0] = TARGET_ACTIVE_COLOR if index == self._progress else TARGET_INACTIVE_COLOR

    def _try_click_target(self) -> None:
        data = self.action.data or {}
        display_x = int(data.get("x", -1))
        display_y = int(data.get("y", -1))
        grid_pos = self.camera.display_to_grid(display_x, display_y)
        if grid_pos is None:
            return

        targets = self.current_level.get_data("targets") or []
        if not targets:
            return
        expected = targets[self._progress] if self._progress < len(targets) else None
        if expected is None:
            return
        if grid_pos == expected:
            for target in self.current_level.get_sprites_by_tag("target"):
                if target.x == grid_pos[0] and target.y == grid_pos[1]:
                    self.current_level.remove_sprite(target)
                    break
            self._progress += 1
            if self._progress >= len(targets):
                self.next_level()
                return
            self._activate_next_target()
        else:
            self._progress = 0
            self._reset_targets()

    def step(self) -> None:
        if self.action.id == GameAction.ACTION6:
            self._try_click_target()
        self.complete_action()
