Implement exactly one game and one DSL solver from the game spec below.

Requirements:
- Implement the full game behavior described in the spec. The mechanics, rules, and puzzle intent of the spec are binding.
- Before implementing, read `docs/arcengine_authoring_guide.md` in the worker repo. Treat it as repo-local authoring constraints.
- Inspect the copied ArcEngine source at `vendor/arcengine/` for API signatures and runtime behavior. Prefer this local copy over probing a virtualenv path.
- Inspect the copied reference games and their visual assets:
  - reference game code: `re_arc/environment_files/{ft09_close,ls20_close}/`
  - reference DSL agents: `re_arc/dsl/agents/{ft09_close,ls20_close}.py`
  - reference replay GIFs: `pipeline/reference_games/screenshots/`
  Use these for expected 64x64 visual density, sprite scale, first-frame readability, and HUD clarity. Do not copy their mechanics unless the spec asks for that mechanic.
- Before relying on assumptions about ArcEngine components, inspect `vendor/arcengine/` to understand the signatures and behavior of known objects/methods such as `ARCBaseGame`, `Level`, `Sprite`, `Camera`, `RenderableUserDisplay`, `try_move_sprite`, `complete_action`, `next_level`, `win`, and `lose`.
- The process is launched with its current working directory set to the isolated worker repo. Treat the current working directory as `<repo_root>` and make all edits there. Do not use `git rev-parse --show-toplevel` to find the repo root, because the worker repo may sit inside an outer git checkout and `git rev-parse` can incorrectly resolve to the parent project.
- Use `<repo_root>/.venv/bin/python` when it exists; otherwise use the active `python`. Python commands must still be run from the worker repo root so imports resolve against the worker repo files.
- Do not run `make prepare`; worker repos do not include a Makefile. Use the scoped checks listed below instead.
- The game must be understandable from the screen and solvable through the visible interface.
- Implement a DSL solver that can reach the win condition for the generated game.
- Author level layouts as readable ASCII logical grids in the game implementation. Prefer a `rows: tuple[str, ...]`,
  `layout: list[str]`, or equivalent string-grid field in each level spec, then parse the symbols into walls, floor,
  player starts, objects, targets, switches, gates, hazards, and other mechanics. Do not hard-code level geometry only as
  scattered coordinate lists when an ASCII grid can express it. This makes layouts reviewable and editable.
  Keep any extra per-object metadata separate only when ASCII alone would be ambiguous, for example orientation,
  color, timing, or object identity.
- Use the canonical ArcEngine action mapping exactly:
  - `GameAction.ACTION1` / id `1` = UP / W
  - `GameAction.ACTION2` / id `2` = DOWN / S
  - `GameAction.ACTION3` / id `3` = LEFT / A
  - `GameAction.ACTION4` / id `4` = RIGHT / D
  - `GameAction.ACTION5` / id `5` = SPACE
  - `GameAction.ACTION6` / id `6` = CLICK(x,y)
  - `GameAction.ACTION7` / id `7` = UNDO / Z
  Do not infer numeric IDs from the order in a game spec's action-space list. If the spec says `RIGHT` before `LEFT`, still implement `LEFT = 3` and `RIGHT = 4`.
- Every generated game must provide game-native single-step undo through `GameAction.ACTION7`, even when the game spec omits undo or assigns undo to another control. Include action id `7` in both the environment's `available_actions` and `manifest.json`. Implement the undo history and restoration behavior described in `docs/arcengine_authoring_guide.md`; `RESET`, viewer/session history, and replay controls are not substitutes for game-native undo. Keep `ACTION5` as SPACE/interact and do not overload it as the canonical undo control.
- Do not use sentinel clicks like `{"x":-1,"y":-1}`. Use only valid ArcEngine actions `1..7`; action `7` is always simple undo, and when action `6` is a click, target only valid coordinates within the allowed `64x64` click region.
- After implementing the game and DSL solver, generate your own worker-local visual review artifacts before finishing. The outer pipeline will generate final replay/GIF artifacts after Codex exits, but you cannot inspect those final pipeline artifacts during this run. Therefore, render and inspect your own representative frames, replay contact sheet, or temporary GIF from the DSL solver path.
- Your visual review must include more than first frames. Generate a worker-local replay contact sheet or temporary GIF from the DSL solver path, covering every level and enough steps to include each distinct action type and each major state transition. During implementation, and again before finishing, visually check any authored animation by inspecting the relevant frame sequence and confirming that it actually shows the intended cause, motion, timing, and final state. Inspect the review artifacts to verify that action effects match the spec, selection and movement controls match the canonical action mapping, success/failure feedback is visible, animations are short and behave as expected, and final post-action frames match the resulting game state. For non-final `next_level()` transitions, do not assume `obs.frame[-1]` is the solved old-level frame; ArcEngine often renders the next level's initial frame last, so inspect the transition action's frame sequence separately. If the review reveals a broken, ambiguous, or misleading visual/action result, fix it and rerun the relevant validation and visual review before finishing.
- Do not add permanent custom trace-generation code solely for review. Temporary scripts or one-off commands are fine.
- Exact coordinates and room shapes are provisional and must be changed if they create cramped, merged, or ambiguous visuals.
- Do not draw visible internal grid lines, lattice lines, checkerboard backgrounds, or per-cell borders just because the game uses logical cells. Logical grids are implementation structure, not visual decoration. Render cells as connected shapes and continuous regions with only the gameplay-relevant outlines, walls, paths, pads, handles, sockets, or board frame needed for readability. Draw an internal grid only if the spec explicitly requires visible grid lines as a gameplay cue.
- Once the game and dsl solver are done, render and inspect representative frames for every level.
  - If a level is visually cramped, if important objects visually merge, if the goal is not easy to infer from the first frame, or if the scene contains unnecessary internal grid lines, the layout is invalid even if the mechanics are correct. In that case revise the layout and rerender.
  - When there is tension between exact coordinates/layout details and first-frame readability, preserve the puzzle logic and revise the layout.
- Keep action animations sparse and readable. Simple one-cell movement or state changes should usually render at most one intermediate frame plus the final frame; do not add long multi-frame tweens or repeated pulses that make GIF/replay playback feel slow. Completion or invalid-action feedback should be brief, usually one or two frames.
- If you animate derived visuals such as reflected ghosts, shadows, projected copies, previews, trails, beams, or linked helper objects, preserve stable identity across frames. Do not sort old and new derived cell sets and zip them together, because that can pair unrelated cells and create teleporting or diagonal glitches. Either animate from explicit source identities such as `(object_id, source_cell, projection_kind)` or render derived visuals only in their final state during the parent object's animation.
- Do not trust or preserve exact action budgets, move limits, timer values, or budget HUD details from the spec. Once the game and DSL solver are implemented and the solver reaches every level's win condition, compute or estimate the winning action count for each level from the verified solver path. Then set that level's action budget to 6x the verified winning action count, and add a visible step/time bar that decreases as actions are taken and causes complete level failure when it expires. This derived bar is mandatory and overrides any budget written in the spec.

Allowed edit scope:
- `re_arc/environment_files/<slug>/<variant>/**`
- `re_arc/dsl/agents/<slug>.py`

- In the `metadata.json` file which needs to be created in `re_arc/environment_files/<slug>/<variant>/**` use the following structure:

{
  "game_id": "", // populate this
  "title": "", // populate this
  "default_fps": 8, // populate this
  "baseline_actions": [], // populate this with the verified winning action count for each level
  "tags": [], // choose one tag from the following: ["keys-only", "click-only", "click-keys"] depending on which actions are allowed in the game
  "class_name": "" // populate this
}

Run this worker-local validation command from the worker repo root before finishing:
- `cd <repo_root> && PY=.venv/bin/python; [ -x "$PY" ] || PY=python; $PY -m pipeline.validate_generated_game --game <game_id> --files-only`
- If validation fails, fix the implementation and rerun the command until it passes.
