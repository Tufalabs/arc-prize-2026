# jc26 — design spec (v1)

A click-only **tower-defense** puzzle. The public name is the opaque codename
`jc26` so the mechanics aren't given away; this document is the internal design.
The player toggles **blockers** to maze **movers** ("enemies") from their spawns
to a **target**, while pre-placed **turrets** on high ground thin them out. Open
every mover's path and a **round** plays automatically.

All files live in this folder (`re_arc/environment_files/jc26/0001/`). The CI
DSL-agent loader will expect an agent under `re_arc/dsl/agents/jc26.py`; that
shim is a later phase (not built yet).

## Board & rendering

- Grid of **21×21 tiles**, each **3×3 px** → 63×63 px in the 64×64 frame. The
  bottom pixel **row 63** holds the **energy bar** (column 63 is free margin).
- A **mandatory energy bar** spans row 63: white (`C_ENERGY`) fill over a dark-grey
  (`C_ENERGY_TRACK`) track, width `round(64 · energy / ENERGY_MAX)`. `ENERGY_MAX =
  256`; every action (click or SPACE, RESET excepted) drains 1, and hitting **0
  is a loss** (`lose()` → GAME_OVER). 256 is generous for these puzzles — the bar
  is required, not a real constraint. RESET refills it.
- Three static terrains: **low ground** (black, mover-traversable), **high
  ground** (dark grey, turret-placeable, walls movers), and **highest ground**
  (one shade lighter grey) which also walls movers but **cannot** hold a turret.
- A **mover** is a 3×3 light-blue body. Its **lives** are a **per-level**
  property (the `lives` key, default `DEFAULT_LIVES = 27`). One shot removes one
  life; the 9 body pixels are dropped evenly as lives are lost
  (`pixels_removed = 9·(lives−hp)//lives`), so any life count maps onto the 9
  pixels. Pixels vanish in a fixed **spiral** (outer ring clockwise from the
  top-left, **centre dies last**). The body is positioned by its **centre
  pixel**, so it can sit mid-tile as it moves (see the round).
- A **turret** is a 3×3 dark-red block with a red centre "barrel". There are two
  kinds: **fixed** and **movable** — drawn **identically** (level design makes
  clear which is which). Only movable turrets can be picked up; a **selected**
  turret gets a white outer ring. It fires a single red **laser line** (not a
  projectile) to its victim. A 1-px **dark-red range ring** (firing radius,
  `TURRET_RADIUS` tiles) plus a **red recolouring of the covered stretch of each
  route line** is drawn **only in the frame just after a turret is placed** (for
  that turret); it clears on the next action. The editor instead shows **every**
  turret's ring and coverage (design view).
- The **target** is a 3×3 green block.
- **Blockers** only go in **designated slots** (not every low-ground tile). A
  slot toggled **on** is a filled 3×3 purple block; an **open** slot shows just
  its **4 corner pixels** in purple (the "you may block here" hint).
- Each mover's planned route is drawn as a blue path line (`C_PATH = 9`; movers
  are the lighter blue `10`, and the palette has no true dark blue). A blocked
  mover draws its route to the reachable tile closest (Manhattan) to the target;
  a fully-walled mover draws nothing. The line is pushed **one pixel past its
  last tile centre** toward the target (preferring the heading it was already
  going) so it visibly ends **at** the blocking tile rather than a tile short.

## Control — click to plan, SPACE to start

Two actions: **CLICK** (action id 6, carries `{x, y}`) plans, **SPACE** (action
id 5) starts the round. RESET re-inits the level. Clicking never starts a round.

One click resolves against the clicked tile:

- **No turret selected:** clicking a **movable** turret **selects** it (fixed
  turrets ignore the click); clicking a **designated slot** toggles its blocker;
  clicking any other tile does nothing.
- **A turret selected:** clicking that turret **deselects**; clicking another
  movable turret **switches** selection; clicking empty high ground **moves** the
  selected turret there (then deselects); anything else **deselects**. While a
  turret is selected, every **free high-ground tile** shows a purple centre dot
  so the valid drop spots are obvious.

**SPACE** checks every mover for a path. If **all** movers can reach the target,
the round runs (below). If **any** mover is **stuck** (no route), the round does
NOT start — instead the stuck movers and their best-effort route lines **blink
red** (3× on/off, ~0.2 s each at ~20 fps) and the player stays in planning.

## Pathfinding (deterministic)

- A tile is **open** iff it is low ground and not currently blockered. Movers
  ignore each other and may overlap.
- `dist_field`: BFS distance-to-target over open tiles.
- A mover's path: greedy descent of `dist_field` from its spawn, breaking ties
  in a fixed neighbour order **up, down, left, right** (`NEIGH`). This yields a
  unique shortest path; the exact tie-break is intentionally arbitrary (levels
  don't depend on which shortest path is taken).
- A mover **has a path** iff its spawn's `dist_field` is finite.

## The round

Pressing **SPACE** with all movers pathed runs the round as one animation. Each
**tick** = one emitted frame:

1. **Fire.** Each turret shoots the in-range mover **closest to the target**
   (fewest pixels left on its path; ties by mover index). Range is **Euclidean ≤
   `TURRET_RADIUS` tiles** (currently 3). Turrets never miss; one shot removes one
   life. Multiple turrets may hit the same mover. A laser line is drawn
   turret→victim for that tick.
2. Remove movers at 0 HP. **All movers cleared → win.**
3. **Move.** Each survivor advances **one pixel** along its path (tile centres
   are `TILE = 3` px apart, so each tile-hop takes 3 frames). **Any mover
   reaching the target → loss.**

Cadence is 1 shot/frame and 1 px/frame by construction, so a mover dies over
~27 px of travel ≈ 9 tiles (values likely to be tuned).

## Win / loss

- **Win:** all movers destroyed → `self.next_level()` (`win_score = number of
  levels`; we never touch `self._score` / `self._win_score`).
- **Loss:** a mover reaches the target. **Not** GAME_OVER — the round animation
  plays, the screen **flashes red 3×** (~0.2 s on / 0.2 s off at the player's
  ~20 fps), then the game returns to planning with the **same** gate/turret setup
  (SPACE is the trigger, so nothing needs undoing) for the player to adjust and
  retry.

## Files in this folder

- `spec.md` — this document.
- `jc26.py` — the `ARCBaseGame` engine: pure pathfinding + round simulation
  (`dist_field` / `greedy_path` / `reachable_path` / `simulate_round`, all
  arcengine-free and unit-testable) + the click/state/render glue. Self-contained
  (the loader execs it with no package context); reads `levels.json`.
  `JC26(specs=[...])` bypasses the file.
- `levels.json` — list of levels, each `{name, grid, lives?}` (`lives` defaults
  to 27). `grid` is the 21×21 ASCII layout: `.` low ground, `#` high ground, `H`
  highest ground, `E` mover spawn, `X` target, `G` fixed turret, `M` movable
  turret (turrets on high ground), `o` blocker slot starting **on**, `*` blocker
  slot starting **open**. Only slots and movable turrets can be acted on.
  Malformed levels are skipped with a warning.
- `metadata.json` — game metadata (`game_id` `jc26-0001`, `class_name` `JC26`,
  tags, …).
- `editor_lib.py` / `editor.ipynb` — **single-grid** level editor. Unlike the
  talos editor (paint grid + separate preview), the editable grid **is** the
  official game render: one fixed-size `ipywidgets.Image` (mouse events captured
  via `ipyevents`). A click paints one tile; **press-drag-release paints a
  rectangle** of the current brush, and it re-renders through the real `JC26`.
  The notebook is a one-liner (`editor_lib.build_editor()`); the module's
  `load_levels` / `save_levels` / `render_frame` / `frame_to_png` / `preview_png`
  run headless. New / Delete / Save manage `levels.json`.
- `agent.py` — oracle DSL solver (later phase, not built yet).
