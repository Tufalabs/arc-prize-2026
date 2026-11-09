# ArcEngine Authoring Guide

This guide summarizes the ArcEngine lifecycle and the repo conventions used by generated `re_arc` environments. It is intentionally short: inspect the installed runtime when details matter, but follow these patterns unless the game needs something unusual.

## Lifecycle

### `__init__`

Build static level data, displays, camera, and the initial internal fields that `on_set_level()` will later overwrite.

Typical shape:

```python
class MyGame(ARCBaseGame):
    def __init__(self, seed: int = 0) -> None:
        self._hud = MyHud()
        self._level_state = ...
        camera = Camera(0, 0, 64, 64, background, letterbox, [self._hud])
        super().__init__("my_game", levels, camera, False, 1, [1, 2, 3, 4, 5, 6], seed=seed)
```

Create `Sprite`, `Level`, `Camera`, and `RenderableUserDisplay` instances here, but do not use `__init__` as the only place that initializes gameplay state. ArcEngine calls `on_set_level()` whenever the current level is entered or reset.

### `on_set_level(level)`

Rebuild all per-level runtime state from `level.get_data(...)` and from sprites in `self.current_level`.

Use it for:

- player position, selected item, carried color, remaining budget, timers
- references to sprites that were cloned with the level
- visual synchronization for a fresh level
- clearing transient animation or failure state

Reset ordering caveat: during a reset, ArcEngine calls `on_set_level()` before it changes `self._state` back to `NOT_FINISHED`. That means `self._state` may still be `WIN` or `GAME_OVER` while `on_set_level()` is rebuilding the fresh level.

In `on_set_level()`, do not read `self._state` to decide what HUD, banner, or visuals to draw. Always initialize fresh-level visuals explicitly as playable/non-terminal state. Otherwise a reset can show stale `WIN` or `GAME_OVER` visuals on the new level.

### `step()`

`step()` resolves the current action. After a player action, ArcEngine calls `step()` until the action is complete. For most generated games, handle the action in one `step()` call and finish by calling `complete_action()`, `next_level()`, or `lose()`.

Typical one-call shape:

```python
def step(self) -> None:
    if self.action.id == GameAction.RESET:
        self.complete_action()
        return

    # Apply gameplay effects for self.action.
    # Sync sprites and HUD.
    # Call next_level(), lose(), or complete_action().
```

Guard reset actions. ArcEngine performs reset handling before entering the normal `step()` loop, then still calls `step()` to render/complete the reset action. If your `step()` has autonomous simulation, timers, enemy movement, or budget decrement, ensure reset does not advance the simulation by one tick.

### `reset_attempt`

Some games need to reset only an attempt inside the same level: for example, return the runner to start while preserving moved blocks, or consume one retry after a failed launch. This is game-owned logic, not an ArcEngine API requirement. Keep it separate from ArcEngine level reset:

- attempt reset: restore part of the level state during `step()`
- level reset: ArcEngine clones the clean level and calls `on_set_level()`
- full reset: ArcEngine clones every level and returns to level 0

Name helper methods plainly, such as `_reset_attempt()`, `_restore_runner()`, or `_reset_after_failed_run()`.

## Actions And `complete_action()`

ArcEngine keeps calling `step()` for the current action until the action is marked complete.

Call `complete_action()` when:

- the one-tick action has finished
- an invalid/no-op action has been handled
- a short explicit animation has finished
- you call `lose()` and want the frame returned

Do not call `complete_action()` before all state and visuals for the action are synchronized.

If an action intentionally spans multiple rendered frames, store a countdown or queue, update one frame per `step()` call, and call `complete_action()` only when the animation is done. Keep these animations short; tests reject very slow steps.

### Game-Native Undo

Every game should implement undo as an explicit game action in the game itself. Do not rely on viewer, MakePlay, CLI, or replay-session undo for gameplay semantics.

Use a simple action id and keep the game `available_actions` and `manifest.json` in sync. In ArcEngine, action id `6` is complex click, so use a later id such as `ACTION7` for simple undo.

Before any accepted board-mutating action, push one snapshot of board-owned state: positions, selected actor, inventory, gates, trails, paint, carried objects, or similar fields. Do not snapshot non-refundable resources such as energy bars, timers, or turn counters unless the spec explicitly says undo refunds them.

Undo should pop one snapshot, restore it, clear transient animation/flash state, and complete the action. Do not push snapshots for reset or undo itself. For multi-frame actions, snapshot once when input is accepted, not per animation frame.

Clear undo history in `on_set_level()` and game-owned attempt resets. Add a regression test that performs a move, undoes it through the public action API, and verifies board state rewinds while energy/turn counters do not.

## Animation Pattern

ArcEngine has no built-in tweening. Animation is explicit game-owned state: after an input action is submitted, ArcEngine keeps calling `step()` and rendering frames until the game calls `complete_action()`, `next_level()`, or `lose()`.

For short animations, resolve the input once, queue the visual frame updates, then consume one queued frame per later `step()` call:

```python
class MyGame(ARCBaseGame):
    def on_set_level(self, level: Level) -> None:
        self.player = self.current_level.get_sprites_by_tag("player")[0]
        self.animation_frames: list[tuple[Sprite, tuple[int, int]]] = []
        self._reset_runtime_state()

    def _queue_slide(self, sprite: Sprite, cells: list[tuple[int, int]]) -> None:
        self.animation_frames = [(sprite, _cell_top_left(x, y)) for x, y in cells]

    def _play_animation_frame(self) -> None:
        sprite, pixel = self.animation_frames.pop(0)
        sprite.set_position(*pixel)
        if not self.animation_frames:
            self.complete_action()

    def step(self) -> None:
        if self.animation_frames:
            self._play_animation_frame()
            return

        if self.action.id == GameAction.RESET:
            self.complete_action()
            return

        # Apply the input action once. This may queue animation frames.
        self._resolve_action()

        if not self.animation_frames:
            self.complete_action()
```

The `ls20_close` reference game uses this structure for pushers: `_queue_pusher_animation()` stores `(sprite, pixel_position)` updates, and `_play_animation_frame()` pops one update per `step()`.

Keep animations short. Treat the submitted input as one action: spend budget, consume items, and record undo state when the action is accepted, not once per animation frame. Clear animation queues in `on_set_level()` and attempt resets. If an animation ends in a win or loss, synchronize the final visual state before calling `next_level()` or `lose()`.

Common mistakes: calling `complete_action()` immediately after queueing frames, forgetting to call it when the queue empties, spending budget inside the frame player, or leaving stale animation frames after reset.

## Level Progression, Score, And Reward

Use `next_level()` when the player solves a level. Do not mutate `self._score` or `self._win_score` directly.

`TransitionRewardEnv` derives reward from `levels_completed / win_levels`, which ArcEngine backs with the base score fields. A solved level must therefore increment level progress through `next_level()`.

Do not call `win()` for successful level completion in generated environments. The final level should also call `next_level()`; ArcEngine credits that solved level and turns it into a win on the last level.

Mechanical `WIN` is not enough for validation. The ArcEngine scorecard must also record a completed `WIN` run for the same game id with a score of `100.0`. If the game reaches `WIN` but the scorecard says there is no completed run, or reports `99.0` instead of `100.0`, the game usually has a reset, id, progression, or baseline-counting bug.

Keep these scorecard rules:

- The `game_id` in `metadata.json`, the `game_id` in `manifest.json`, and the value passed to `ARCBaseGame(...)` must match exactly.
- `baseline_actions` must match the scorecard-counted winning actions for each level, not just a hand-counted DSL list. If a reset/startup action is accidentally counted as gameplay, fix reset handling; do not hide the bug by inflating the baseline.
- Do not override `full_reset()` or `level_reset()` unless the game has a rare engine-level need. Put normal per-level rebuild logic in `on_set_level()`.
- The initial reset from `NOT_PLAYED` must remain a full ArcEngine reset. That full reset is what starts a normal scorecard run. Treating startup reset as a current-level reset can produce a mechanically winning game with no completed scorecard run.
- A manual reset during a level should reset only the current level, preserve `levels_completed`, restore that level's initial frame, and spend no action budget.
- A reset after `WIN` should be a full reset for a fresh episode.

For most games, do not implement manual reset yourself. Let ArcEngine's `handle_reset()` call `full_reset()` at episode start or after `WIN`, and `level_reset()` during play. Your `step()` should only guard the post-reset render tick:

```python
def step(self) -> None:
    if self.action.id == GameAction.RESET:
        # ArcEngine has already cloned the correct clean level and called
        # on_set_level(). Do not spend budget, move enemies, advance timers,
        # or run autonomous simulation for this render tick.
        self._sync_visuals()
        self.complete_action()
        return

    # Normal non-reset action logic begins here.
```

For non-final levels, `next_level()` is a deferred transition. ArcEngine will render another frame after applying the level advance, so the last frame returned by the solving action is often the next level's initial frame, not the old level's solved frame. When inspecting replay frames, treat the transition action's frame sequence as containing both the completion visual and the new-level arrival frame; do not label `obs.frame[-1]` as the solved old-level frame without checking.

Use `lose()` for terminal failure. If the game is meant to allow retrying within the same level, implement an attempt reset instead of calling `lose()`.

## Core Objects

### `Sprite`

A visible or invisible object made from integer color pixels. Important fields and methods include:

- `pixels`: 2D color grid, usually color IDs `0..15`; transparent cells are represented by negative values in many existing games
- `x`, `y`, `width`, `height`: grid-space position and size
- `tags`: semantic labels used to find sprites
- `visible`, `collidable`, `layer`: rendering/collision behavior
- `set_position(x, y)`, `set_visible(flag)`, `set_rotation(degrees)`, `color_remap(old, new)`
- `clone()` when a reusable sprite template must be placed into a level

Prefer multi-pixel, readable sprites over one-pixel dots for important objects.

### `Level`

A level contains sprites, grid size, and arbitrary data. Put static specifications in `data` so `on_set_level()` can rebuild runtime state from the clean cloned level.

Example:

```python
Level(
    sprites=[board, player, goal],
    grid_size=(64, 64),
    data={"spec": spec, "budget": 40},
    name="Level 1",
)
```

### `Camera`

The camera defines the visible region and background. For 64x64 games, use a 64x64 camera unless the game deliberately uses a logical board that the camera scales.

Attach HUDs through the `interfaces` list.

### `RenderableUserDisplay`

Use this for HUD overlays such as budgets, charge pips, timers, selected-color swatches, or warning borders. The display should render from explicit values you set during `_sync_hud()`.

Do not let stale terminal state or old budget values leak into a fresh level; configure the display from the current level runtime state whenever a level is entered or an action completes.

### `try_move_sprite`

Use `try_move_sprite` when the game can rely on ArcEngine collision checks for simple grid movement. It attempts to move a sprite and respects collidable sprites/blocking behavior. For custom movement rules such as portals, colored gates, one-way paths, push chains, or logical-cell movement, explicit coordinate checks are usually clearer.

## Practical Checklist

- Store clean level specs in `Level.data`; rebuild mutable state in `on_set_level()`.
- Complete every action path with `complete_action()`.
- Guard reset actions so reset does not consume budget or advance autonomous systems.
- Implement undo as a game-native action and document exactly which state is restored; by default, undo should not refund energy or turn counters.
- Call `next_level()` for solved levels.
- Keep score fields engine-owned.
- Render current state visibly: selected item, carried key, open locks, budget, timer, charges, active hazards.
- Run reset-contract tests after changes that affect level setup, terminal states, HUDs, or failure handling.
