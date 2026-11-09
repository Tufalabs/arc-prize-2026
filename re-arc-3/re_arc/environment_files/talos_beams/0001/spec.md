# Talos Beams — design spec (v1)

A grid puzzle game in the spirit of *The Talos Principle*'s connector
puzzles. The player rewires laser networks from **sources** through movable
**connectors** to **targets**; charging a target opens a **door**; the goal
is to reach the **exit** through the doors that open.

All files for this game live in this folder for now
(`re_arc/environment_files/talos_beams/0001/`). The CI DSL-agent loader
expects the agent under `re_arc/dsl/agents/talos_beams.py`; when we
integrate we add a one-line shim there that re-exports the agent defined
here.

## Board & rendering

- The world is a grid of **tiles**, each rendered as a **5×5 pixel** block.
  Levels are always **12×12 tiles** (60 px), centred in the 64×64 frame; the
  surrounding margin is letter-boxed.
- Movement and all object positions snap to the tile grid.
- **Laser beams are pixel-precise:** node anchors are tile centres and beams
  are float64 line segments, rasterised per-pixel over the 64×64 frame. The
  whole frame is composed by the game into a single full-canvas sprite.

## Actions (keys + click)

`[1,2,3,4,5,6]`:

- **1–4** — move up / down / left / right.
- **5** (space) — pick up / put down a connector **on the player's current
  tile** (connectors are transparent to the player, so you stand on one to
  grab it, or drop one where you stand).
- **6** (click) — only while **holding** a connector: click a node
  (source / target / connector) to **toggle a wire** between the held
  connector and that node. Any node, any distance, no line of sight needed
  to *form* the wire.

## Tiles

- **Floor** — passable; transparent to beams.
- **White wall** — blocks the player **and** beams.
- **Grey wall** — blocks the player only; beams pass through.
- **Door** (rendered yellow when closed) — a togglable wall linked to one or
  more targets. Closed = blocks player + beams; open = floor. Opening/closing
  is a beam re-resolve trigger.

## Nodes

Single-tile objects, each with a **stable** ID built from the painted char +
an index: sources `r1/g1/b1…`, targets `R1/G1/B1…`, connectors `o1, o2…`
(doors are keyed by their grid digit). IDs are assigned at placement and
persisted in the level's `ids` map (`"tx,ty" → id`): deleting an entity never
renumbers the others, and a new placement reuses the lowest freed ID. The
editor labels each cell with its ID. Doors and initial wiring reference them.

A **placed connector or inverter blocks any beam it is not wired to** — its
**circle** (radius = half a tile) is a hard obstacle like a wall, so a beam
can't slip through its centre. Sources and targets are pass-through: a beam
not wired to them crosses their tile.

- **Source** — fixed, coloured (red / green / blue). Always powered; emits
  only along wired edges.
- **Connector** — pre-placed, repositionable. **Space** carries / drops it.
  While **held** a connector is **inert**: it conducts nothing and only
  draws grey lines to whatever it is wired to. While held, **click** toggles
  its wires. A placed connector **adopts the colour that powers it** and
  re-emits that colour on its own edges.
- **Inverter** — exactly like a connector (carry with space, wire by click,
  inert while held, blocks beams when placed) except it **swaps colour**:
  powered by **blue** it emits **red** and vice-versa; **green does nothing**
  (it stays inert). Drawn as a grey ring with **blue** at top/bottom-middle
  and **red** at left/right-middle; centre shows the emitted colour. Grid
  char `i`, ids `i1…`.
- **Target** — fixed, coloured, with a **rotation** (0–3 = down / left / up /
  right) choosing which edge it faces. A beam always connects to the target's
  connection **nub** — the pixel in the middle of the facing edge (integer, so
  beam and nub line up exactly) — drawn as a bar whose long axis is
  perpendicular to the facing direction. The designer
  rotates the target toward open space so an adjoining wall/door isn't in the
  way — beams never clip through walls. Rotation lives per target in
  `levels.json` (`rot` map) and is set in the editor with the **↻** brush.
  Charges only from **matching-colour** power.
  Its bar fills one notch per player move while powered and **empties
  immediately** the moment it is unpowered (must-maintain). Full → its door
  opens; losing power → empties → its door closes.

Sources and targets block the player; connectors, inverters and the exit do not.

## Beam resolution

Re-run whenever the wiring or geometry could change (pick up, put down,
wire toggle, door open, door close). The pattern is computed **under the
hood first**; only if it **differs** from the previously settled pattern do
we play the animation (below).

Resolution is a **tiered wavefront** by **hop count** (graph distance in
edges), not pixel distance:

- **Tier 0:** every source is powered with its colour.
- A node powered at tier *k* emits beams along its wired edges; those beams
  resolve at **tier *k+1***. A connector reached at tier *k* re-emits at
  tier *k+1*; a reached target charges.
- Process tier by tier until no new node is powered.

**Crossing rule (uniform for all colours).** At any point where two beams
cross, the **earliest-arriving** beam passes and the later beam is **cut**
there. Arrival time is continuous:

```
arrival(point) = (tier - 1) + (distance_along_beam_from_its_emitter / beam_length)
```

i.e. a beam traverses its *whole length* within its tier's time-slot, so a
longer ("far-connected") beam moves faster in pixels and can win a crossing
a shorter beam reaches at a larger fraction of its own length.

- **Lower tier always beats higher tier** (separate time-slots).
- Within a tier, the smaller **fraction-along-its-own-length** wins.
- **Exact arrival-time tie** → a **seeded-random** winner (reproducible from
  the game seed).
- **Head-on** — the *same* edge driven from both ends at the same tier
  (A→B and B→A) — both beams meet in the middle and **both stop**.

Beams are also cut by **white walls**, **closed doors**, and any **placed
connector they aren't wired to** (its circle) — grey walls and the player are
transparent.

A **connector delivered ≥2 distinct colours → not powered at all** (dead;
emits nothing). Multi-colour / feedback conflicts are resolved greedily in a
single tier-ordered pass with **no backtracking** — deterministic but we do
not design puzzles around the pathological cases.

**Rendering:** every wired connection is drawn as a **solid grey line** at all
times; live beams are then drawn **over** it in colour, from the emitter up to
their cut point (or target). So lit portions read in colour and
unlit / blocked / held portions stay grey (the "wire" indicator). Lines are
rasterised **4-connected** so diagonals read as solid, not dashed. A beam blocked by another beam (crossing / head-on)
gets a **white** pixel at its **furthest pixel from the source** (the block
point), drawn the moment its front reaches there — including mid-animation.
A connector killed by two colours keeps its grey ring with a **white
centre** (white = "interaction / conflict").

Shapes (5×5 tiles): source = filled circle, connector = grey ring whose
centre fills with its colour when powered, target = 5-wide × 3-high
rectangle with a 3-pixel inner bar that fills left-to-right with charge
(`CHARGE_FULL = 3`), player = filled purple diamond, goal/exit = orange
diamond, door = solid yellow.

## Animation

The continuous resolution is sampled for display only: when the beam
pattern changes, play it out **tier by tier, 10 frames per tier** — tier-1
beams race out and settle, then tier-2, etc. Beam fronts advance at
`length / 10` px per frame (longer beams visibly faster), getting cut at
crossings exactly as resolved. This doubles as the teaching signal for the
hop-distance rule. (Resolution itself is never quantised to frames.) If a
re-resolve produces **no** change, no animation is played. A single action
can produce a **cascade**: a door flip re-routes beams, which can flip
another door — each changed stage is animated in sequence (and a frame
reflecting the new door state is always emitted, so doors open/close within
the same action, not on the next one). The build animation also plays on
**level start / RESET** when there are beams.

## Charging, doors, win

- Each **move** action (1–4, whether or not the player actually moved) adds
  `+1` to every matching-colour-powered target (capped at `CHARGE_FULL`).
  Any unpowered target is set straight to `0` (on every re-resolve, not just
  moves), so power loss is immediate.
- A door opens iff **all** its listed targets are full. A door's value is a
  list of target IDs (e.g. `{"1": ["t1", "t3"]}`) — so multiple same-colour
  targets can drive different doors, and one door can need several targets —
  or a colour string (`{"1": "R"}`) meaning every target of that colour. For
  **OR** semantics wrap the list in `{"any": [...]}` (door opens if *any*
  listed target is full); `{"all": [...]}` is the explicit form of the default
  AND. A
  door open/close flips the blocking geometry → beam re-resolve, which can
  cascade into further door flips; each changed stage animates.
- **Win:** the player reaches the **exit** tile → `next_level()`.
- **Energy / loss:** every action (any of 1–6, RESET excepted) drains the
  **energy bar** by 1, starting at `ENERGY_MAX = 256` per level. Running it to
  **0** is a loss (`lose()` → `GAME_OVER`); RESET refills it. The bar is drawn
  as a horizontal strip across the bottom letterbox rows (the grid is 60 px in
  a 64 px frame, so rows 62–63 are free) — width `round(64 * energy /
  ENERGY_MAX)` px (full row at start) in a unique pink (`C_ENERGY`), with the
  depleted remainder in dark grey (`C_ENERGY_TRACK`).

## Score contract

`win_score = number of levels`; solving a level calls `self.next_level()`.
We never touch `self._score` / `self._win_score` directly.

## Files in this folder

- `spec.md` — this document (living spec).
- `levels.json` — **level data, split from code**: a list of levels, each
  `{name, grid, doors, charge_full, edges}`, all **12×12**. `grid` is the
  ASCII layout the editor paints (`#` white wall, `%` grey wall, `.` floor,
  `@` player, `X` exit, `r/g/b` sources, `R/G/B` targets, `o` connectors,
  `i` inverters, `1`-`9` door tiles). `doors` maps a door digit to a list of target IDs
  (all must be full), a colour string, or `{"any":[...]}` / `{"all":[...]}` for
  explicit OR / AND; `edges` are initial wires as
  `[id, id]` pairs. An `ids` map (`"tx,ty" → id`) holds the stable entity IDs
  (`r1`/`R1`/`o1`…); a level without one derives them in reading order. A `rot`
  map (`target id → 0..3`) holds each target's facing. The
  game loads it at construction via the `re_arc`
  package path (the env loader execs the game with no `__file__`), and
  **skips any malformed level** (e.g. mid-edit) with a warning rather than
  failing the whole game. Starts as `[]`; if empty/all-skipped the game shows
  a placeholder room. A reference level lives in `levels_example.json`.
- `talos_beams.py` — the `ARCBaseGame` engine: pure beam resolver +
  pixel renderer + tiered animation + step logic. Self-contained (the loader
  execs it with no package context, so no sibling imports); reads
  `levels.json`. `TalosBeams(specs=[...])` bypasses the file (used by the
  editor preview).
- `metadata.json` — game metadata (`game_id`, `class_name`, tags, …).
- `editor_lib.py` / `editor.ipynb` — ipywidgets level editor. The notebook
  is a one-liner (`editor_lib.build_editor()`); all logic + an
  `editor_lib`-level headless-testable API (load/save/preview) live in the
  module. Paints the grid (12×12), rotates targets (**↻** brush), live-previews
  via the real renderer, shows entity IDs + facings, and creates / saves /
  **deletes** levels in `levels.json`.
- `test_resolver.py` — unit tests for the pure resolver (crossings,
  head-on, two-colour-dead, walls, connector-blocking, tier ordering).
  `test_doors.py` — door logic (multi-target, same-colour→different doors,
  colour shortcut). Run with `uv run`.
- `agent.py` — oracle DSL solver (added in the agent phase).
